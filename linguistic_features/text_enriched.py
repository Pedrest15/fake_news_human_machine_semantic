#!/usr/bin/env python3
"""Baseline TF-IDF / BoW enriquecido com features linguisticas.

Concatena horizontalmente:
  - vetor TF-IDF ou Bag-of-Words sobre o texto (truncado pareado em runtime)
  - vetor de features linguisticas (NILC + LIWC + Enhanced UD + silabas + POS +
    SAGE), com Tier A+B do NILC e agregadores LIWC removidos por default.

A truncagem pareada acontece em runtime com o tokenizador BERTimbau
(`neuralmind/bert-base-portuguese-cased`), exatamente como nos experimentos
BERT: para cada par humano/LLM do mesmo filename, calcula-se
  min_len = min(len_h_tokens, len_l_tokens, max_length - 2)
em tokens WordPiece (subtracao reserva espaco para [CLS] e [SEP]), e ambas as
versoes sao cortadas ao mesmo comprimento antes da vetorizacao. Garante
comparabilidade direta entre baselines BERT, TF-IDF e BoW.

Cada documento (humano e LLM, uma linha cada) vira um vetor combinado:
  [tfidf_or_bow_text | standardized_linguistic_features]

A matriz combinada e' mantida esparsa (hstack de sparse + sparse(dense)) para
evitar materializacao de 10^4+ features × 8 mil docs em memoria densa.

Uso:
    python3 linguistic_features/text_enriched.py --vectorizer tfidf --classifier all
    python3 linguistic_features/text_enriched.py --vectorizer bow   --classifier svm
    python3 linguistic_features/text_enriched.py --vectorizer both  --classifier all
    python3 linguistic_features/text_enriched.py --vectorizer tfidf --classifier svm --max-length 256
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.sparse import csr_matrix, hstack as sparse_hstack
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from linguistic_features import (  # noqa: E402
    BERT_MODELS,
    CORPUS_DIRS_DEFAULT,
    LinguisticFeaturesPipeline,
    load_split_files,
    save_misclassified,
    truncate_pair_min_tokens,
)

RESULTS_DIR = SCRIPT_DIR / "results"

# Mesma configuracao do caminho BERT: 512 tokens WordPiece (- 2 reservados
# para [CLS] e [SEP], conforme convencao BERT).
DEFAULT_MAX_LENGTH = 512
DEFAULT_BERT_MODEL = BERT_MODELS["bertimbau"]  # neuralmind/bert-base-portuguese-cased


def build_corpus_dirs(corpus_dirs_str: Dict[str, Path]) -> Dict[Tuple[str, int], Path]:
    """Converte o dict do linguistic_features (chaves string) para o esquema
    (subset_corpus, label) usado por load_paired_truncated_texts."""
    return {
        ("fake_br",      0): corpus_dirs_str["fake_br_human"],
        ("fake_br",      1): corpus_dirs_str["fake_br_llm"],
        ("fake_true_br", 0): corpus_dirs_str["fake_true_human"],
        ("fake_true_br", 1): corpus_dirs_str["fake_true_llm"],
    }


def corpus_dirs_from_root(corpus_root: Path) -> Dict[str, Path]:
    """Reproduz a logica de override --corpus-root do linguistic_features.py.

    Mantem os mesmos paths internos (mesma versao de Fake.Br humano, mesmas
    pastas data/ dentro de fake-news-llm-ptbr-main) para garantir que os
    splits entre text_enriched.py e linguistic_features.py em modo combined
    fiquem identicos.
    """
    return {
        "fake_br_human":   corpus_root / "corpus" / "Fake.br-Corpus-master" / "full_texts" / "fake",
        "fake_br_llm":     corpus_root / "corpus" / "fake-news-llm-ptbr-main" / "fake-news-llm-ptbr-main" / "data" / "Fake.Br",
        "fake_true_human": corpus_root / "corpus" / "FakeTrue.Br-main" / "fake",
        "fake_true_llm":   corpus_root / "corpus" / "fake-news-llm-ptbr-main" / "fake-news-llm-ptbr-main" / "data" / "FakeTrueBR",
    }


# Default: alinhado com CORPUS_DIRS_DEFAULT de linguistic_features.py.
# Para override (ex.: layout do servidor), use --corpus-root na CLI.
ORIGINAL_CORPUS_DIRS = build_corpus_dirs(CORPUS_DIRS_DEFAULT)


# =============================================================================
# Configuracao dos classificadores
#
# Importante: GaussianNB e MultinomialNB ficam de fora porque exigem matrizes
# densas (GaussianNB) ou nao-negativas (MultinomialNB), ambos incompativeis com
# a matriz hstack(sparse_tfidf, sparse(standardized_linguistic)) que produzimos.
# =============================================================================
CLASSIFIERS_CONFIG: Dict[str, Dict] = {
    "svm": {
        "estimator": SVC(probability=True, random_state=42),
        "param_grid": {
            "kernel": ["linear", "rbf"],
            "C": [0.1, 1.0, 10.0],
            "gamma": ["scale"],
        },
    },
    "random_forest": {
        "estimator": RandomForestClassifier(random_state=42, n_jobs=-1),
        "param_grid": {
            "n_estimators": [100, 200],
            "max_depth": [10, 20, None],
            "min_samples_split": [2, 5],
        },
    },
    "logistic_regression": {
        "estimator": LogisticRegression(random_state=42, max_iter=1000),
        "param_grid": {
            "C": [0.01, 0.1, 1.0, 10.0],
            "penalty": ["l2"],
            "solver": ["lbfgs", "liblinear"],
        },
    },
    "mlp": {
        "estimator": MLPClassifier(random_state=42, max_iter=500, early_stopping=True),
        "param_grid": {
            "hidden_layer_sizes": [(256,), (512,), (256, 128)],
            "alpha": [0.0001, 0.001],
        },
    },
}

if XGBOOST_AVAILABLE:
    CLASSIFIERS_CONFIG["xgboost"] = {
        "estimator": XGBClassifier(random_state=42, eval_metric="logloss", n_jobs=-1),
        "param_grid": {
            "n_estimators": [100, 200],
            "max_depth": [3, 6],
            "learning_rate": [0.1, 0.3],
        },
    }


logger = logging.getLogger("text_enriched")


# =============================================================================
# Truncagem pareada em runtime (BERTimbau WordPiece)
# =============================================================================

def _read_text(path: Path) -> Optional[str]:
    try:
        text = path.read_text(encoding="utf-8").strip()
    except Exception as exc:
        logger.warning("erro lendo %s: %s", path, exc)
        return None
    return text or None


def load_paired_truncated_texts(
    keys: List[Tuple[str, str, int]],
    corpus_dirs: Dict[Tuple[str, int], Path],
    tokenizer: Any,
    max_length: int = DEFAULT_MAX_LENGTH,
) -> Tuple[List[str], List[int], Dict[str, int]]:
    """Le textos originais e trunca pareadamente em runtime via BERTimbau.

    Para cada (filename, subset_corpus), carrega tanto a versao humana
    (label=0) quanto a LLM (label=1), tokeniza com o tokenizador BERTimbau e
    calcula min_len = min(len_h_tokens, len_l_tokens, max_length - 2) em
    tokens WordPiece (-2 reserva [CLS] e [SEP]). Ambas as versoes do par
    sao cortadas ao mesmo comprimento e decodificadas de volta para texto,
    garantindo paridade exata com os experimentos BERT.

    Pares onde uma das versoes esta ausente ou vazia sao descartados
    inteiros (nao basta truncar um lado sem o outro).

    Retorna (texts, kept_indices, stats). `texts` esta na mesma ordem
    de `keys` para os indices em `kept_indices`.
    """
    # Pass 1: carrega ambas as metades de cada par em memoria
    pair_texts: Dict[Tuple[str, str], Dict[int, str]] = defaultdict(dict)
    for filename, subset_corpus, label in keys:
        dir_path = corpus_dirs.get((subset_corpus, label))
        if dir_path is None:
            continue
        text = _read_text(dir_path / filename)
        if text is None:
            continue
        pair_texts[(filename, subset_corpus)][label] = text

    # Pass 2: para cada par completo, trunca via BERTimbau (truncate_pair_min_tokens)
    # e ja armazena o texto truncado decodificado para cada label.
    pair_truncated: Dict[Tuple[str, str], Tuple[str, str, int]] = {}
    constraint_counts = {"human_shorter": 0, "llm_shorter": 0, "max_length_cap": 0}
    for pair_key, halves in pair_texts.items():
        if 0 not in halves or 1 not in halves:
            continue
        n_h = len(tokenizer.encode(halves[0], add_special_tokens=False))
        n_l = len(tokenizer.encode(halves[1], add_special_tokens=False))
        h_t, l_t, min_len = truncate_pair_min_tokens(
            halves[0], halves[1], tokenizer, max_length=max_length,
        )
        pair_truncated[pair_key] = (h_t, l_t, min_len)
        if min_len == max_length - 2 and min_len < min(n_h, n_l):
            constraint_counts["max_length_cap"] += 1
        elif n_h <= n_l:
            constraint_counts["human_shorter"] += 1
        else:
            constraint_counts["llm_shorter"] += 1

    # Pass 3: produz lista de textos truncados na ordem de keys
    out_texts: List[str] = []
    kept_indices: List[int] = []
    missing = 0
    for i, (filename, subset_corpus, label) in enumerate(keys):
        truncated = pair_truncated.get((filename, subset_corpus))
        if truncated is None:
            missing += 1
            continue
        out_texts.append(truncated[label])  # label 0 -> h_t, label 1 -> l_t
        kept_indices.append(i)

    stats = {
        "pairs_complete": int(len(pair_truncated)),
        "keys_kept": int(len(kept_indices)),
        "keys_discarded": int(missing),
        "max_length_cap": int(constraint_counts["max_length_cap"]),
        "human_shorter": int(constraint_counts["human_shorter"]),
        "llm_shorter": int(constraint_counts["llm_shorter"]),
    }
    return out_texts, kept_indices, stats


def make_vectorizer(kind: str, max_features: int, min_df: int):
    if kind == "tfidf":
        return TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 1),
            min_df=min_df,
            sublinear_tf=True,
            lowercase=True,
        )
    if kind == "bow":
        return CountVectorizer(
            max_features=max_features,
            ngram_range=(1, 1),
            min_df=min_df,
            lowercase=True,
        )
    raise ValueError(f"Vectorizer desconhecido: {kind!r}")


# =============================================================================
# Treino + avaliacao
# =============================================================================

def grid_combinations(grid: Dict) -> int:
    n = 1
    for v in grid.values():
        n *= len(v)
    return n


def train_with_grid(clf_type: str, X_train, y_train, cv: int):
    cfg = CLASSIFIERS_CONFIG[clf_type]
    n_combos = grid_combinations(cfg["param_grid"])
    logger.info("  GridSearchCV: %s, %d combinacoes, cv=%d", clf_type, n_combos, cv)
    grid = GridSearchCV(
        estimator=copy.deepcopy(cfg["estimator"]),
        param_grid=cfg["param_grid"],
        cv=cv,
        scoring="f1_weighted",
        n_jobs=-1,
        refit=True,
        verbose=0,
    )
    t0 = time.time()
    grid.fit(X_train, y_train)
    logger.info("  best params: %s", grid.best_params_)
    logger.info("  best CV score (f1_weighted): %.4f  (%.1fs)",
                grid.best_score_, time.time() - t0)
    return grid.best_estimator_, grid.best_params_, float(grid.best_score_)


def evaluate(clf, X_test, y_test) -> Tuple[Dict, np.ndarray, np.ndarray]:
    y_pred = clf.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred, average="weighted")),
        "precision": float(precision_score(y_test, y_pred, average="weighted")),
        "recall": float(recall_score(y_test, y_pred, average="weighted")),
    }
    cm = confusion_matrix(y_test, y_pred)
    return metrics, cm, y_pred


def extract_feature_importance(clf, feature_names: List[str], top_k: int = 30
                                ) -> Optional[List[Tuple[str, float]]]:
    """Extrai importancia de features para classifiers que expoem
    `feature_importances_` (tree-based: RF, XGBoost) ou `coef_` (lineares: LR,
    SVM linear). Retorna None para SVM RBF, MLP, NaiveBayes.

    Os pesos sao normalizados/absolutos conforme o tipo de modelo e a saida e'
    a lista [(name, value), ...] ordenada por importancia decrescente.
    """
    if hasattr(clf, "feature_importances_"):
        imp = np.asarray(clf.feature_importances_)
    elif hasattr(clf, "coef_"):
        coef = np.asarray(clf.coef_)
        imp = np.abs(coef[0]) if coef.ndim > 1 else np.abs(coef)
    else:
        return None
    if len(imp) != len(feature_names):
        logger.warning("feature_importance: %d valores vs %d nomes — pulando",
                       len(imp), len(feature_names))
        return None
    pairs = sorted(zip(feature_names, imp.tolist()),
                   key=lambda x: x[1], reverse=True)
    return [(name, float(val)) for name, val in pairs[:top_k]]


# =============================================================================
# Pipeline
# =============================================================================

def build_matrices(vectorizer_kind: str,
                   max_features: int,
                   min_df: int,
                   max_length: int,
                   tokenizer: Any,
                   feature_groups: List[str],
                   corpus_dirs: Dict[Tuple[str, int], Path] = ORIGINAL_CORPUS_DIRS,
                   ):
    """Monta as matrizes train e test combinando texto + features linguisticas.

    Os textos sao carregados dos diretorios originais (corpus_dirs) e
    truncados pareadamente em runtime via tokenizador BERTimbau, ate
    max_length - 2 tokens WordPiece (paridade com experimentos BERT).

    Retorna X_train, y_train, X_test, y_test, train_keys, test_keys, feature_names.
    """
    logger.info("Construindo pipeline linguistic-only (sem BERT)...")
    pipeline = LinguisticFeaturesPipeline(
        exp_dir=SCRIPT_DIR,
        feature_mode="linguistic",
        feature_groups=feature_groups,
        drop_nilc_absolute=True,
    )

    splits = load_split_files(SCRIPT_DIR)
    logger.info("Carregando matrizes linguisticas (assemble_matrices)...")
    (X_ling_train, y_train, X_ling_test, y_test,
     train_keys, test_keys, ling_names) = pipeline.assemble_matrices(splits)
    logger.info("  X_ling_train=%s X_ling_test=%s n_ling=%d",
                X_ling_train.shape, X_ling_test.shape, len(ling_names))

    logger.info("Truncagem pareada em runtime (BERTimbau, max_length=%d)...", max_length)
    train_texts, train_keep, train_stats = load_paired_truncated_texts(
        train_keys, corpus_dirs, tokenizer, max_length,
    )
    test_texts, test_keep, test_stats = load_paired_truncated_texts(
        test_keys, corpus_dirs, tokenizer, max_length,
    )
    logger.info("  train: pares completos=%d, keys mantidas=%d, descartadas=%d",
                train_stats["pairs_complete"], train_stats["keys_kept"],
                train_stats["keys_discarded"])
    logger.info("    truncagem ditada por: human_shorter=%d  llm_shorter=%d  cap=%d",
                train_stats["human_shorter"], train_stats["llm_shorter"],
                train_stats["max_length_cap"])
    logger.info("  test:  pares completos=%d, keys mantidas=%d, descartadas=%d",
                test_stats["pairs_complete"], test_stats["keys_kept"],
                test_stats["keys_discarded"])

    # Filtra matrizes linguisticas por kept_indices
    X_ling_train = X_ling_train[train_keep]
    y_train = y_train[train_keep]
    train_keys = [train_keys[i] for i in train_keep]
    X_ling_test = X_ling_test[test_keep]
    y_test = y_test[test_keep]
    test_keys = [test_keys[i] for i in test_keep]

    # StandardScaler nas features linguisticas
    scaler = StandardScaler()
    X_ling_train = scaler.fit_transform(X_ling_train)
    X_ling_test = scaler.transform(X_ling_test)

    # Vetorizar texto
    logger.info("Vetorizando texto (%s, max_features=%d, ngram=(1,1), min_df=%d)...",
                vectorizer_kind, max_features, min_df)
    vec = make_vectorizer(vectorizer_kind, max_features, min_df)
    X_text_train = vec.fit_transform(train_texts)
    X_text_test = vec.transform(test_texts)
    text_features = list(vec.get_feature_names_out())
    logger.info("  vocab=%d  X_text_train=%s", len(text_features), X_text_train.shape)

    # hstack: sparse text + sparse(dense linguistic)
    X_train = sparse_hstack(
        [X_text_train, csr_matrix(X_ling_train)],
        format="csr",
    )
    X_test = sparse_hstack(
        [X_text_test, csr_matrix(X_ling_test)],
        format="csr",
    )
    feature_names = [f"text__{w}" for w in text_features] + list(ling_names)

    logger.info("Matrizes combinadas:")
    logger.info("  X_train=%s (sparsity=%.3f)",
                X_train.shape, 1 - X_train.nnz / (X_train.shape[0] * X_train.shape[1]))
    logger.info("  X_test=%s", X_test.shape)
    logger.info("  feature_names: %d (text=%d, linguistic=%d)",
                len(feature_names), len(text_features), len(ling_names))

    return X_train, y_train, X_test, y_test, train_keys, test_keys, feature_names


def run_experiment(vectorizer_kind: str,
                   classifiers: List[str],
                   max_features: int,
                   min_df: int,
                   max_length: int,
                   tokenizer: Any,
                   cv_folds: int,
                   feature_groups: List[str],
                   out_dir: Path,
                   bert_model_name: str,
                   corpus_dirs: Dict[Tuple[str, int], Path] = ORIGINAL_CORPUS_DIRS,
                   ) -> Dict[str, Dict]:
    (X_train, y_train, X_test, y_test,
     train_keys, test_keys, feature_names) = build_matrices(
        vectorizer_kind, max_features, min_df, max_length, tokenizer,
        feature_groups, corpus_dirs,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    all_results: Dict[str, Dict] = {}

    for clf_type in classifiers:
        logger.info("=" * 70)
        logger.info("EXPERIMENTO: %s + linguistic  |  classifier=%s",
                    vectorizer_kind.upper(), clf_type)
        logger.info("=" * 70)

        best_clf, best_params, cv_score = train_with_grid(
            clf_type, X_train, y_train, cv_folds,
        )
        metrics, cm, y_pred = evaluate(best_clf, X_test, y_test)
        tn, fp, fn, tp = cm.ravel()

        logger.info("Test: F1=%.4f acc=%.4f prec=%.4f recall=%.4f",
                    metrics["f1"], metrics["accuracy"],
                    metrics["precision"], metrics["recall"])
        logger.info("CM: TN=%d FP=%d FN=%d TP=%d", tn, fp, fn, tp)

        fi = extract_feature_importance(best_clf, feature_names, top_k=30)
        if fi:
            logger.info("Top-5 features por importancia:")
            for name, val in fi[:5]:
                logger.info("  %-55s %.6f", name, val)

        result = {
            "experiment": f"{vectorizer_kind}+linguistic",
            "vectorizer": vectorizer_kind,
            "classifier": clf_type,
            "n_features_text": int(sum(1 for n in feature_names if n.startswith("text__"))),
            "n_features_linguistic": int(sum(1 for n in feature_names if not n.startswith("text__"))),
            "n_features_total": int(len(feature_names)),
            "vectorizer_config": {
                "max_features": max_features, "ngram_range": [1, 1], "min_df": min_df,
                "sublinear_tf": vectorizer_kind == "tfidf",
            },
            "truncation": {
                "max_length": int(max_length),
                "effective_min_len_cap": int(max_length - 2),
                "tokenizer": bert_model_name,
                "strategy": "pairwise_bertimbau_runtime",
            },
            "data_split": {
                "train_samples": int(X_train.shape[0]),
                "test_samples": int(X_test.shape[0]),
            },
            "grid_search": {
                "enabled": True, "cv_folds": cv_folds,
                "best_params": best_params, "cv_score": cv_score,
            },
            "test_results": metrics,
            "confusion_matrix": {
                "matrix": cm.tolist(),
                "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            },
            "feature_importance": fi,
        }

        out_path = out_dir / f"text_{vectorizer_kind}_linguistic_{clf_type}.json"
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2, ensure_ascii=False)
        logger.info("Resultado salvo: %s", out_path)

        # Salva lista de misclassified (mesmo formato do combined_*_errors.json)
        probs = None
        if hasattr(best_clf, "predict_proba"):
            try:
                probs = best_clf.predict_proba(X_test)
            except Exception as exc:
                logger.warning("predict_proba falhou para %s: %s", clf_type, exc)
        miscls_path = out_dir / "misclassified" / f"text_{vectorizer_kind}_linguistic_{clf_type}_errors.json"
        save_misclassified(
            keys=test_keys,
            y_true=y_test,
            y_pred=y_pred,
            output_path=miscls_path,
            experiment=f"{vectorizer_kind}+linguistic / {clf_type}",
            probs=probs,
        )

        all_results[clf_type] = result

    return all_results


def write_comparison(vectorizer_kind: str,
                     all_results: Dict[str, Dict],
                     out_dir: Path):
    summary = {}
    for clf, r in all_results.items():
        summary[clf] = {
            "cv_score": r["grid_search"]["cv_score"],
            "test_f1": r["test_results"]["f1"],
            "test_accuracy": r["test_results"]["accuracy"],
            "best_params": r["grid_search"]["best_params"],
            "n_features": r["n_features_total"],
        }
    out_path = out_dir / f"comparison_text_{vectorizer_kind}_linguistic.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    logger.info("Comparison: %s", out_path)

    # Tabela legivel no log
    logger.info("=" * 90)
    logger.info("COMPARACAO  %s + linguistic", vectorizer_kind.upper())
    logger.info("=" * 90)
    logger.info("%-25s %12s %12s %12s", "Classifier", "CV Score", "Test F1", "Test Acc")
    logger.info("-" * 90)
    for clf, s in summary.items():
        logger.info("%-25s %12.4f %12.4f %12.4f",
                    clf, s["cv_score"], s["test_f1"], s["test_accuracy"])
    logger.info("=" * 90)


# =============================================================================
# CLI
# =============================================================================

def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vectorizer", choices=["tfidf", "bow", "both"], default="both",
                   help="Vetorizacao do texto (default: both)")
    p.add_argument("--classifier", default="all",
                   choices=list(CLASSIFIERS_CONFIG.keys()) + ["all"],
                   help="Classificador (default: all)")
    p.add_argument("--max-features", type=int, default=10000)
    p.add_argument("--min-df", type=int, default=5)
    p.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH,
                   help=f"Limite de tokens WordPiece BERTimbau na truncagem pareada "
                        f"(default: {DEFAULT_MAX_LENGTH}; efetivo: {DEFAULT_MAX_LENGTH-2} apos [CLS]/[SEP])")
    p.add_argument("--bert-model", type=str, default=DEFAULT_BERT_MODEL,
                   help=f"Modelo BERT para o tokenizer (default: {DEFAULT_BERT_MODEL})")
    p.add_argument("--corpus-root", type=str, default=None,
                   help="Override do diretorio raiz do corpus (mesma semantica do "
                        "--corpus-root em linguistic_features.py). Se omitido, usa "
                        "CORPUS_DIRS_DEFAULT importado de linguistic_features.")
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--features", default="all",
                   help="Grupos linguisticos: nilcmetrics,liwc,enhanced_ud,syllables,"
                        "pos_tagger,parser_stats,sage_terms ou all")
    p.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    return p.parse_args(argv[1:])


def main(argv):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args = parse_args(argv)

    if args.features == "all":
        feature_groups = ["nilcmetrics", "liwc", "enhanced_ud",
                          "syllables", "pos_tagger", "parser_stats", "sage_terms"]
    else:
        feature_groups = [f.strip() for f in args.features.split(",")]

    if args.classifier == "all":
        classifiers = list(CLASSIFIERS_CONFIG.keys())
    else:
        classifiers = [args.classifier]

    vectorizers = ["tfidf", "bow"] if args.vectorizer == "both" else [args.vectorizer]

    # Resolve corpus_dirs: usa override --corpus-root se fornecido, caso
    # contrario CORPUS_DIRS_DEFAULT (importado de linguistic_features).
    if args.corpus_root:
        logger.info("Override --corpus-root: %s", args.corpus_root)
        corpus_dirs_str = corpus_dirs_from_root(Path(args.corpus_root))
        corpus_dirs = build_corpus_dirs(corpus_dirs_str)
    else:
        corpus_dirs = ORIGINAL_CORPUS_DIRS

    # Sanity check dos diretorios de corpus originais (a truncagem e' em runtime)
    missing_dirs = [str(p) for p in corpus_dirs.values() if not p.exists()]
    if missing_dirs:
        logger.error("Diretorios de corpus originais ausentes:\n  - %s",
                     "\n  - ".join(missing_dirs))
        logger.error("Passe --corpus-root <raiz> ou ajuste CORPUS_DIRS_DEFAULT "
                     "em linguistic_features.py para refletir o layout do servidor.")
        return 1

    # Tokenizer BERTimbau carregado uma vez e reusado entre vetorizadores
    logger.info("Carregando tokenizer '%s'...", args.bert_model)
    try:
        from transformers import AutoTokenizer
    except ImportError:
        logger.error("transformers nao instalado. Instale com 'pip install transformers' "
                     "para usar a truncagem BERTimbau dos experimentos BERT.")
        return 1
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model)
    logger.info("Tokenizer pronto.")

    for vec_kind in vectorizers:
        results = run_experiment(
            vectorizer_kind=vec_kind,
            classifiers=classifiers,
            max_features=args.max_features,
            min_df=args.min_df,
            max_length=args.max_length,
            tokenizer=tokenizer,
            cv_folds=args.cv_folds,
            feature_groups=feature_groups,
            out_dir=args.out_dir,
            bert_model_name=args.bert_model,
            corpus_dirs=corpus_dirs,
        )
        if len(results) > 1:
            write_comparison(vec_kind, results, args.out_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
