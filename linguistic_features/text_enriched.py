#!/usr/bin/env python3
"""Baseline TF-IDF / BoW enriquecido com features linguisticas.

Concatena horizontalmente:
  - vetor TF-IDF ou Bag-of-Words sobre o texto (em corpus_truncated/)
  - vetor de features linguisticas (NILC + LIWC + Enhanced UD + silabas + POS +
    SAGE), com Tier A do NILC ja removido por default.

Cada documento (humano e LLM, uma linha cada) vira um vetor combinado:
  [tfidf_or_bow_text | standardized_linguistic_features]

A matriz combinada e' mantida esparsa (hstack de sparse + sparse(dense)) para
evitar materializacao de 10^4+ features × 8 mil docs em memoria densa.

Uso:
    python3 linguistic_features/text_enriched.py --vectorizer tfidf --classifier all
    python3 linguistic_features/text_enriched.py --vectorizer bow   --classifier svm
    python3 linguistic_features/text_enriched.py --vectorizer both  --classifier all
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

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
    LinguisticFeaturesPipeline,
    load_split_files,
)

SEMANTICA_ROOT = SCRIPT_DIR.parent
CORPUS_TRUNCATED = SEMANTICA_ROOT / "corpus_truncated"
RESULTS_DIR = SCRIPT_DIR / "results"

# Mapeia (subset_corpus, label) → subdir em corpus_truncated/
SUBSET_DIR_MAP = {
    ("fake_br",      0): "fake_br_human",
    ("fake_br",      1): "fake_br_llm",
    ("fake_true_br", 0): "fake_true_human",
    ("fake_true_br", 1): "fake_true_llm",
}


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
# Carregamento de textos alinhados aos keys do pipeline linguistico
# =============================================================================

def load_texts_for_keys(keys: List[Tuple[str, str, int]],
                        corpus_root: Path,
                        ) -> Tuple[List[str], List[int]]:
    """Le texto truncado para cada key (filename, subset_corpus, label).

    Retorna (texts, kept_indices). Linhas onde o arquivo nao existe sao
    silenciosamente puladas (mas tambem precisamos remover a linha
    correspondente da matriz linguistica — usamos kept_indices p/ isso).
    """
    texts: List[str] = []
    kept_indices: List[int] = []
    missing = 0
    for i, (filename, subset_corpus, label) in enumerate(keys):
        subdir = SUBSET_DIR_MAP.get((subset_corpus, label))
        if subdir is None:
            missing += 1
            continue
        path = corpus_root / subdir / filename
        if not path.exists():
            missing += 1
            continue
        try:
            text = path.read_text(encoding="utf-8").strip()
        except Exception as exc:
            logger.warning("erro lendo %s: %s", path, exc)
            missing += 1
            continue
        if not text:
            missing += 1
            continue
        texts.append(text)
        kept_indices.append(i)
    if missing:
        logger.warning("  %d documentos sem texto truncado disponivel (descartados)", missing)
    return texts, kept_indices


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


# =============================================================================
# Pipeline
# =============================================================================

def build_matrices(vectorizer_kind: str,
                   max_features: int,
                   min_df: int,
                   feature_groups: List[str]
                   ):
    """Monta as matrizes train e test combinando texto + features linguisticas.

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

    logger.info("Carregando textos truncados alinhados aos keys...")
    train_texts, train_keep = load_texts_for_keys(train_keys, CORPUS_TRUNCATED)
    test_texts, test_keep = load_texts_for_keys(test_keys, CORPUS_TRUNCATED)
    logger.info("  textos: train=%d (descartados %d), test=%d (descartados %d)",
                len(train_texts), len(train_keys) - len(train_texts),
                len(test_texts), len(test_keys) - len(test_texts))

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
                   cv_folds: int,
                   feature_groups: List[str],
                   out_dir: Path) -> Dict[str, Dict]:
    (X_train, y_train, X_test, y_test,
     train_keys, test_keys, feature_names) = build_matrices(
        vectorizer_kind, max_features, min_df, feature_groups,
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
        }

        out_path = out_dir / f"text_{vectorizer_kind}_linguistic_{clf_type}.json"
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2, ensure_ascii=False)
        logger.info("Resultado salvo: %s", out_path)
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

    if not CORPUS_TRUNCATED.exists():
        logger.error("corpus_truncated/ nao encontrado em %s. "
                     "Rode corpus_prep/truncate_pairs.py primeiro.",
                     CORPUS_TRUNCATED)
        return 1

    for vec_kind in vectorizers:
        results = run_experiment(
            vectorizer_kind=vec_kind,
            classifiers=classifiers,
            max_features=args.max_features,
            min_df=args.min_df,
            cv_folds=args.cv_folds,
            feature_groups=feature_groups,
            out_dir=args.out_dir,
        )
        if len(results) > 1:
            write_comparison(vec_kind, results, args.out_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
