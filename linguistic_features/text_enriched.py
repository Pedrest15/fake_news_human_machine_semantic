#!/usr/bin/env python3
"""Baseline TF-IDF / BoW enriquecido com features linguisticas.

Concatena horizontalmente:
  - vetor TF-IDF ou Bag-of-Words sobre o texto (truncado pareado em runtime)
  - vetor de features linguisticas (NILC + LIWC + Enhanced UD + silabas + POS +
    SAGE), com Tier A+B do NILC e agregadores LIWC removidos por default.

A truncagem pareada acontece em runtime com o tokenizador do encoder escolhido
em --bert-model (default BERTimbau, `neuralmind/bert-base-portuguese-cased`),
exatamente como nos experimentos com embeddings: para cada par humano/LLM do
mesmo filename, calcula-se
  min_len = min(len_h_tokens, len_l_tokens, max_length - 2)
em tokens do proprio tokenizador (a subtracao reserva os dois tokens especiais
do inicio e do fim), e ambas as versoes sao cortadas ao mesmo comprimento antes
da vetorizacao. Garante comparabilidade direta entre os baselines de embedding,
TF-IDF e BoW, desde que o mesmo tokenizador seja usado nos dois lados.

Cada documento (humano e LLM, uma linha cada) vira um vetor combinado:
  [tfidf_or_bow_text | standardized_linguistic_features]

A matriz combinada e' mantida esparsa (hstack de sparse + sparse(dense)) para
evitar materializacao de 10^4+ features × 8 mil docs em memoria densa.

Uso:
    python3 linguistic_features/text_enriched.py --vectorizer tfidf --classifier all
    python3 linguistic_features/text_enriched.py --vectorizer bow   --classifier svm
    python3 linguistic_features/text_enriched.py --vectorizer both  --classifier all
    python3 linguistic_features/text_enriched.py --vectorizer tfidf --classifier all --no-linguistic
    python3 linguistic_features/text_enriched.py --vectorizer tfidf --classifier svm --max-length 256
    python3 linguistic_features/text_enriched.py --vectorizer tfidf --bert-model bert-multilingual
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score,
)
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
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
    DEFAULT_BERT_MODEL_KEY,
    DEFAULT_MAX_LENGTH,
    LinguisticFeaturesPipeline,
    load_split_files,
    save_misclassified,
    splits_to_keys,
    truncate_pair_min_tokens,
)

RESULTS_DIR = SCRIPT_DIR / "results"

# DEFAULT_MAX_LENGTH is imported from linguistic_features so both scripts agree
# on what "the default run" is: 512 tokens, minus the 2 reserved for the
# opening/closing special tokens, following the BERT convention.
DEFAULT_BERT_MODEL = BERT_MODELS[DEFAULT_BERT_MODEL_KEY]  # neuralmind/bert-base-portuguese-cased


def resolve_bert_model(value: str) -> str:
    """Aceita uma chave de BERT_MODELS (ex.: 'bert-multilingual') ou um id completo do HF."""
    return BERT_MODELS.get(value, value)


def bert_model_slug(bert_model_name: str) -> str:
    """Nome curto do tokenizador, para rotular resultados e arquivos."""
    for key, name in BERT_MODELS.items():
        if name == bert_model_name:
            return key
    return re.sub(r'[^a-zA-Z0-9]+', '_', bert_model_name.split('/')[-1]).strip('_').lower()


def experiment_prefix(vectorizer_kind: str, bert_model_name: str,
                      max_length: int = DEFAULT_MAX_LENGTH,
                      use_linguistic: bool = True) -> str:
    """Prefixo dos arquivos de resultado.

    Mantem os nomes historicos (`text_<vec>_linguistic`) para a configuracao
    padrao (BERTimbau + 512 tokens) e acrescenta um sufixo para cada parametro
    fora do default, para que uma bateria nova nao sobrescreva a anterior:
    `text_tfidf_linguistic_bert_multilingual`, `text_tfidf_linguistic_len256`.
    Sem features linguisticas (--no-linguistic) o sufixo e' `_only`:
    `text_tfidf_only`.
    """
    prefix = f"text_{vectorizer_kind}_" + ("linguistic" if use_linguistic else "only")
    if bert_model_name != DEFAULT_BERT_MODEL:
        prefix = f"{prefix}_{bert_model_slug(bert_model_name)}"
    if max_length != DEFAULT_MAX_LENGTH:
        prefix = f"{prefix}_len{max_length}"
    return prefix


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


# Default: aligned with CORPUS_DIRS_DEFAULT from linguistic_features.py.
# To override (e.g. a different server layout), use --corpus-root on the CLI.
ORIGINAL_CORPUS_DIRS = build_corpus_dirs(CORPUS_DIRS_DEFAULT)


# =============================================================================
# Classifier configuration
#
# Note: GaussianNB requires dense matrices, so it runs behind a densifying
# step (inside the CV fold, so the full matrix is never held dense twice) and
# with n_jobs=1 to bound memory. MultinomialNB is left out because it needs
# non-negative inputs, which the standardised linguistic features are not.
# =============================================================================
def _to_dense(X):
    return X.toarray() if hasattr(X, "toarray") else X


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
    "naive_bayes": {
        "estimator": Pipeline([
            ("dense", FunctionTransformer(_to_dense, accept_sparse=True)),
            ("nb", GaussianNB()),
        ]),
        "param_grid": {
            "nb__var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6, 1e-5],
        },
        "n_jobs": 1,
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
# Runtime paired truncation (tokenizer of the chosen encoder)
# =============================================================================

def _read_text(path: Path) -> Optional[str]:
    try:
        text = path.read_text(encoding="utf-8").strip()
    except Exception as exc:
        logger.warning("error reading %s: %s", path, exc)
        return None
    return text or None


def load_paired_truncated_texts(
    keys: List[Tuple[str, str, int]],
    corpus_dirs: Dict[Tuple[str, int], Path],
    tokenizer: Any,
    max_length: int = DEFAULT_MAX_LENGTH,
) -> Tuple[List[str], List[int], Dict[str, int]]:
    """Le textos originais e trunca pareadamente em runtime via tokenizador.

    Para cada (filename, subset_corpus), carrega tanto a versao humana
    (label=0) quanto a LLM (label=1), tokeniza com o tokenizador recebido e
    calcula min_len = min(len_h_tokens, len_l_tokens, max_length - 2) em
    tokens do tokenizador (-2 reserva os dois tokens especiais). Ambas as
    versoes do par sao cortadas ao mesmo comprimento e decodificadas de volta
    para texto, garantindo paridade exata com os experimentos de embedding.

    Pares onde uma das versoes esta ausente ou vazia sao descartados
    inteiros (nao basta truncar um lado sem o outro).

    Retorna (texts, kept_indices, stats). `texts` esta na mesma ordem
    de `keys` para os indices em `kept_indices`.
    """
    # Pass 1: load both halves of every pair into memory
    pair_texts: Dict[Tuple[str, str], Dict[int, str]] = defaultdict(dict)
    for filename, subset_corpus, label in keys:
        dir_path = corpus_dirs.get((subset_corpus, label))
        if dir_path is None:
            continue
        text = _read_text(dir_path / filename)
        if text is None:
            continue
        pair_texts[(filename, subset_corpus)][label] = text

    # Pass 2: for every complete pair, truncate with the tokenizer
    # (truncate_pair_min_tokens) and store the decoded truncated text per label.
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

    # Pass 3: build the list of truncated texts in the order of `keys`
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
    raise ValueError(f"Unknown vectorizer: {kind!r}")


TEXT_COLUMN = "text"


def make_preprocessor(vectorizer_kind: str, max_features: int, min_df: int,
                      ling_columns: List[str]) -> ColumnTransformer:
    """Vetorizador de texto + (imputer, scaler) das features linguisticas.

    Vive dentro do Pipeline do GridSearchCV: vocabulario, IDF, medianas, medias
    e desvios sao reajustados so com o treino de cada fold. A saida e sempre
    esparsa (sparse_threshold=1.0), na ordem texto -> linguisticas.
    """
    transformers = [(TEXT_COLUMN, make_vectorizer(vectorizer_kind, max_features, min_df), TEXT_COLUMN)]
    if ling_columns:
        transformers.append((
            "ling",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scaler", StandardScaler()),
            ]),
            ling_columns,
        ))
    return ColumnTransformer(transformers, sparse_threshold=1.0)


def fitted_feature_names(model: Pipeline, ling_names: List[str]) -> List[str]:
    """Nomes das colunas que chegam ao classificador: text__<token> e depois as linguisticas."""
    vec = model.named_steps["prep"].named_transformers_[TEXT_COLUMN]
    return [f"text__{w}" for w in vec.get_feature_names_out()] + list(ling_names)


# =============================================================================
# Training + evaluation
# =============================================================================

def grid_combinations(grid: Dict) -> int:
    n = 1
    for v in grid.values():
        n *= len(v)
    return n


def train_with_grid(clf_type: str, X_train, y_train, cv: int, groups, preprocessor):
    """GridSearchCV com GroupKFold: o par humano/LLM fica sempre no mesmo fold.

    O preprocessor entra no Pipeline, entao e reajustado a cada fold.
    """
    cfg = CLASSIFIERS_CONFIG[clf_type]
    n_combos = grid_combinations(cfg["param_grid"])
    n_splits = min(cv, len(set(groups)))
    logger.info("  GridSearchCV: %s, %d combinations, GroupKFold n_splits=%d (%d groups)",
                clf_type, n_combos, n_splits, len(set(groups)))
    grid = GridSearchCV(
        estimator=Pipeline([
            ("prep", clone(preprocessor)),
            ("clf", copy.deepcopy(cfg["estimator"])),
        ]),
        param_grid={f"clf__{k}": v for k, v in cfg["param_grid"].items()},
        cv=GroupKFold(n_splits=n_splits),
        scoring="f1_weighted",
        n_jobs=cfg.get("n_jobs", -1),
        refit=True,
        verbose=0,
    )
    t0 = time.time()
    grid.fit(X_train, y_train, groups=groups)
    best_params = {k.removeprefix("clf__"): v for k, v in grid.best_params_.items()}
    logger.info("  best params: %s", best_params)
    logger.info("  best CV score (f1_weighted): %.4f  (%.1fs)",
                grid.best_score_, time.time() - t0)
    return grid.best_estimator_, best_params, float(grid.best_score_)


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
        coef = clf.coef_.toarray() if sparse.issparse(clf.coef_) else np.asarray(clf.coef_)
        imp = np.abs(coef[0]) if coef.ndim > 1 else np.abs(coef)
    else:
        return None
    if len(imp) != len(feature_names):
        logger.warning("feature_importance: %d values vs %d names — skipping",
                       len(imp), len(feature_names))
        return None
    pairs = sorted(zip(feature_names, imp.tolist()),
                   key=lambda x: x[1], reverse=True)
    return [(name, float(val)) for name, val in pairs[:top_k]]


# =============================================================================
# Pipeline
# =============================================================================

def build_matrices(max_length: int,
                   tokenizer: Any,
                   feature_groups: List[str],
                   corpus_dirs: Dict[Tuple[str, int], Path] = ORIGINAL_CORPUS_DIRS,
                   use_linguistic: bool = True,
                   missing_features: str = "impute",
                   ):
    """Monta as entradas train e test: texto truncado + features linguisticas brutas.

    Nada e vetorizado nem padronizado aqui: X e um DataFrame com a coluna de
    texto e as colunas linguisticas (NaN onde um grupo nao cobre o documento),
    e o preprocessor do Pipeline aprende vocabulario, medianas e escala somente
    com o treino de cada fold.

    Com use_linguistic=False, o DataFrame contem apenas o texto
    (condicao sem features linguisticas); as chaves vem direto dos splits.

    Os textos sao carregados dos diretorios originais (corpus_dirs) e
    truncados pareadamente em runtime pelo tokenizador recebido, ate
    max_length - 2 tokens (paridade com os experimentos de embedding).

    Retorna X_train, y_train, X_test, y_test, train_keys, test_keys, ling_names.
    """
    splits = load_split_files(SCRIPT_DIR)
    if use_linguistic:
        logger.info("Building the linguistic-only pipeline (no BERT)...")
        pipeline = LinguisticFeaturesPipeline(
            exp_dir=SCRIPT_DIR,
            feature_mode="linguistic",
            feature_groups=feature_groups,
            drop_nilc_absolute=True,
            missing_features=missing_features,
        )
        logger.info("Loading the linguistic matrices (assemble_matrices)...")
        (X_ling_train, y_train, X_ling_test, y_test,
         train_keys, test_keys, ling_names) = pipeline.assemble_matrices(splits)
        logger.info("  X_ling_train=%s X_ling_test=%s n_ling=%d",
                    X_ling_train.shape, X_ling_test.shape, len(ling_names))
    else:
        logger.info("Text-only condition: no linguistic features.")
        train_keys, test_keys = splits_to_keys(splits)
        y_train = np.array([k[2] for k in train_keys])
        y_test = np.array([k[2] for k in test_keys])
        X_ling_train = X_ling_test = None
        ling_names = []

    logger.info("Runtime paired truncation (%s, max_length=%d)...",
                bert_model_slug(getattr(tokenizer, "name_or_path", "tokenizer")), max_length)
    train_texts, train_keep, train_stats = load_paired_truncated_texts(
        train_keys, corpus_dirs, tokenizer, max_length,
    )
    test_texts, test_keep, test_stats = load_paired_truncated_texts(
        test_keys, corpus_dirs, tokenizer, max_length,
    )
    logger.info("  train: complete pairs=%d, keys kept=%d, discarded=%d",
                train_stats["pairs_complete"], train_stats["keys_kept"],
                train_stats["keys_discarded"])
    logger.info("    truncation driven by: human_shorter=%d  llm_shorter=%d  cap=%d",
                train_stats["human_shorter"], train_stats["llm_shorter"],
                train_stats["max_length_cap"])
    logger.info("  test:  complete pairs=%d, keys kept=%d, discarded=%d",
                test_stats["pairs_complete"], test_stats["keys_kept"],
                test_stats["keys_discarded"])

    # Filter the linguistic matrices by kept_indices
    y_train = y_train[train_keep]
    train_keys = [train_keys[i] for i in train_keep]
    y_test = y_test[test_keep]
    test_keys = [test_keys[i] for i in test_keep]

    def _frame(texts: List[str], ling: Optional[np.ndarray]) -> pd.DataFrame:
        frame = pd.DataFrame({TEXT_COLUMN: texts})
        if ling is not None:
            frame = pd.concat([frame, pd.DataFrame(ling, columns=ling_names)], axis=1)
        return frame

    if use_linguistic:
        X_train = _frame(train_texts, X_ling_train[train_keep])
        X_test = _frame(test_texts, X_ling_test[test_keep])
    else:
        X_train = _frame(train_texts, None)
        X_test = _frame(test_texts, None)

    logger.info("Model inputs (raw; fitted per CV fold by the pipeline):")
    logger.info("  X_train=%s  X_test=%s  linguistic columns=%d (NaN cells: train=%d, test=%d)",
                X_train.shape, X_test.shape, len(ling_names),
                int(X_train[ling_names].isna().sum().sum()) if ling_names else 0,
                int(X_test[ling_names].isna().sum().sum()) if ling_names else 0)

    return X_train, y_train, X_test, y_test, train_keys, test_keys, ling_names


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
                   use_linguistic: bool = True,
                   missing_features: str = "impute",
                   ) -> Dict[str, Dict]:
    (X_train, y_train, X_test, y_test,
     train_keys, test_keys, ling_names) = build_matrices(
        max_length, tokenizer, feature_groups, corpus_dirs, use_linguistic,
        missing_features,
    )
    preprocessor = make_preprocessor(vectorizer_kind, max_features, min_df, ling_names)
    # Group id = (filename, subset): both versions of a pair share it, so
    # GroupKFold never splits a human/LLM pair across CV folds.
    train_groups = [f"{k[0]}__{k[1]}" for k in train_keys]
    cond = "linguistic" if use_linguistic else "only"
    cond_label = "+linguistic" if use_linguistic else " (text only)"

    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = experiment_prefix(vectorizer_kind, bert_model_name, max_length, use_linguistic)
    all_results: Dict[str, Dict] = {}

    for clf_type in classifiers:
        logger.info("=" * 70)
        logger.info("EXPERIMENT: %s%s  |  classifier=%s",
                    vectorizer_kind.upper(), cond_label, clf_type)
        logger.info("=" * 70)

        best_clf, best_params, cv_score = train_with_grid(
            clf_type, X_train, y_train, cv_folds, train_groups, preprocessor,
        )
        feature_names = fitted_feature_names(best_clf, ling_names)
        metrics, cm, y_pred = evaluate(best_clf, X_test, y_test)
        tn, fp, fn, tp = cm.ravel()

        logger.info("Test: F1=%.4f acc=%.4f prec=%.4f recall=%.4f",
                    metrics["f1"], metrics["accuracy"],
                    metrics["precision"], metrics["recall"])
        logger.info("CM: TN=%d FP=%d FN=%d TP=%d", tn, fp, fn, tp)

        fi = extract_feature_importance(best_clf.named_steps["clf"], feature_names, top_k=30)
        if fi:
            logger.info("Top-5 features by importance:")
            for name, val in fi[:5]:
                logger.info("  %-55s %.6f", name, val)

        result = {
            "experiment": f"{vectorizer_kind}{cond_label}",
            "condition": cond,
            "vectorizer": vectorizer_kind,
            "classifier": clf_type,
            "n_features_text": int(sum(1 for n in feature_names if n.startswith("text__"))),
            "n_features_linguistic": int(sum(1 for n in feature_names if not n.startswith("text__"))),
            "n_features_total": int(len(feature_names)),
            "preprocessing_scope": "vectorizer/imputer/scaler fitted inside each CV fold",
            "missing_features": missing_features if use_linguistic else None,
            "vectorizer_config": {
                "max_features": max_features, "ngram_range": [1, 1], "min_df": min_df,
                "sublinear_tf": vectorizer_kind == "tfidf",
            },
            "truncation": {
                "max_length": int(max_length),
                "effective_min_len_cap": int(max_length - 2),
                "tokenizer": bert_model_name,
                "strategy": f"pairwise_{bert_model_slug(bert_model_name)}_runtime",
            },
            "data_split": {
                "train_samples": int(X_train.shape[0]),
                "test_samples": int(X_test.shape[0]),
            },
            "grid_search": {
                "enabled": True, "cv_folds": cv_folds, "cv_scheme": "GroupKFold",
                "best_params": best_params, "cv_score": cv_score,
            },
            "test_results": metrics,
            "confusion_matrix": {
                "matrix": cm.tolist(),
                "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            },
            "feature_importance": fi,
        }

        out_path = out_dir / f"{prefix}_{clf_type}.json"
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2, ensure_ascii=False)
        logger.info("Result saved: %s", out_path)

        # Save the misclassified list (same format as combined_*_errors.json)
        probs = None
        if hasattr(best_clf, "predict_proba"):
            try:
                probs = best_clf.predict_proba(X_test)
            except Exception as exc:
                logger.warning("predict_proba failed for %s: %s", clf_type, exc)
        miscls_path = out_dir / "misclassified" / f"{prefix}_{clf_type}_errors.json"
        save_misclassified(
            keys=test_keys,
            y_true=y_test,
            y_pred=y_pred,
            output_path=miscls_path,
            experiment=f"{vectorizer_kind}{cond_label} / {clf_type}",
            probs=probs,
        )

        all_results[clf_type] = result

    return all_results


def write_comparison(vectorizer_kind: str,
                     all_results: Dict[str, Dict],
                     out_dir: Path,
                     bert_model_name: str = DEFAULT_BERT_MODEL,
                     max_length: int = DEFAULT_MAX_LENGTH,
                     use_linguistic: bool = True):
    summary = {}
    for clf, r in all_results.items():
        summary[clf] = {
            "cv_score": r["grid_search"]["cv_score"],
            "test_f1": r["test_results"]["f1"],
            "test_accuracy": r["test_results"]["accuracy"],
            "best_params": r["grid_search"]["best_params"],
            "n_features": r["n_features_total"],
        }
    out_path = out_dir / f"comparison_{experiment_prefix(vectorizer_kind, bert_model_name, max_length, use_linguistic)}.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    logger.info("Comparison: %s", out_path)

    # Human-readable table in the log
    logger.info("=" * 90)
    logger.info("COMPARISON  %s%s", vectorizer_kind.upper(),
                "+linguistic" if use_linguistic else " (text only)")
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
                   help="Text vectorisation (default: both)")
    p.add_argument("--classifier", default="all",
                   choices=list(CLASSIFIERS_CONFIG.keys()) + ["all"],
                   help="Classifier (default: all)")
    p.add_argument("--max-features", type=int, default=10000)
    p.add_argument("--min-df", type=int, default=5)
    p.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH,
                   help=f"Token cap used by the paired truncation "
                        f"(default: {DEFAULT_MAX_LENGTH}; effective: {DEFAULT_MAX_LENGTH-2} "
                        f"after the two special tokens)")
    p.add_argument("--bert-model", type=str, default=DEFAULT_BERT_MODEL,
                   help=f"Encoder whose tokenizer drives the paired truncation: a key of "
                        f"BERT_MODELS ({', '.join(BERT_MODELS)}) or a full Hugging Face id "
                        f"(default: {DEFAULT_BERT_MODEL})")
    p.add_argument("--corpus-root", type=str, default=None,
                   help="Override the corpus root directory (same semantics as "
                        "--corpus-root in linguistic_features.py). When omitted, "
                        "CORPUS_DIRS_DEFAULT from linguistic_features is used.")
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--features", default="all",
                   help="Linguistic groups: nilcmetrics,liwc,enhanced_ud,syllables,"
                        "pos_tagger,parser_stats,sage_terms or all")
    p.add_argument("--missing-features", choices=["impute", "drop"], default="impute",
                   help="Documents missing in some linguistic group: impute keeps them all "
                        "(median imputed inside each CV fold) so the test set matches the "
                        "text-only condition; drop keeps only documents present in every group "
                        "(default: impute)")
    p.add_argument("--no-linguistic", action="store_true",
                   help="Text-only condition: BoW/TF-IDF without linguistic features.")
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

    # Resolve corpus_dirs: use the --corpus-root override when given, otherwise
    # CORPUS_DIRS_DEFAULT (imported from linguistic_features).
    if args.corpus_root:
        logger.info("Override --corpus-root: %s", args.corpus_root)
        corpus_dirs_str = corpus_dirs_from_root(Path(args.corpus_root))
        corpus_dirs = build_corpus_dirs(corpus_dirs_str)
    else:
        corpus_dirs = ORIGINAL_CORPUS_DIRS

    # Sanity-check the original corpus directories (truncation happens at runtime)
    missing_dirs = [str(p) for p in corpus_dirs.values() if not p.exists()]
    if missing_dirs:
        logger.error("Original corpus directories are missing:\n  - %s",
                     "\n  - ".join(missing_dirs))
        logger.error("Pass --corpus-root <root> or adjust CORPUS_DIRS_DEFAULT in "
                     "linguistic_features.py to match the server layout.")
        return 1

    # The tokenizer is loaded once and reused across vectorisers
    bert_model_name = resolve_bert_model(args.bert_model)
    logger.info("Loading tokenizer '%s'...", bert_model_name)
    try:
        from transformers import AutoTokenizer
    except ImportError:
        logger.error("transformers is not installed. Run 'pip install transformers' "
                     "to use the paired truncation from the embedding experiments.")
        return 1
    tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
    logger.info("Tokenizer ready.")

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
            bert_model_name=bert_model_name,
            corpus_dirs=corpus_dirs,
            use_linguistic=not args.no_linguistic,
            missing_features=args.missing_features,
        )
        if len(results) > 1:
            write_comparison(vec_kind, results, args.out_dir, bert_model_name,
                             args.max_length, not args.no_linguistic)

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
