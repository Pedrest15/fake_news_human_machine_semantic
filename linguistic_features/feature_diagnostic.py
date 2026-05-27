#!/usr/bin/env python3
"""Diagnostico de features essenciais para o F1 perfeito do classificador.

Computa, para cada uma das ~839 features linguisticas (NILC + LIWC + Enhanced UD
+ silabas + POS + parser + SAGE) tres metricas estaticas e uma metrica
preditiva, depois faz uma busca incremental pelo menor conjunto de features que
reproduz F1=1.0 em um SVM linear C=0.1.

Saidas em linguistic_features/results/:
  - feature_diagnostic_ranking.csv  (todas as features, ranqueadas)
  - feature_diagnostic.json         (sumario estruturado)

Uso:
    python3 linguistic_features/feature_diagnostic.py
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
from scipy import stats as scipy_stats
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Permite executar como `python3 feature_diagnostic.py` ou como modulo.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from linguistic_features import (  # noqa: E402
    LinguisticFeaturesPipeline,
    load_split_files,
)

RESULTS_DIR = SCRIPT_DIR / "results"

# Grade inicial para a forward search.
DEFAULT_K_GRID = [1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 400, 600]

GROUP_PREFIXES = (
    ("nilc__", "nilc"),
    ("liwc__", "liwc"),
    ("ud__", "enhanced_ud"),
    ("syll__", "syllables"),
    ("pos__", "pos_tagger"),
    ("psr__", "parser_stats"),
    ("sage__", "sage_terms"),
)


logger = logging.getLogger("feature_diagnostic")


def group_from_name(name: str) -> str:
    for prefix, group in GROUP_PREFIXES:
        if name.startswith(prefix):
            return group
    return "unknown"


def group_counts(feature_names: List[str]) -> dict:
    out = {}
    for n in feature_names:
        g = group_from_name(n)
        out[g] = out.get(g, 0) + 1
    return out


def compute_static_metrics(X_train: np.ndarray, y_train: np.ndarray
                            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = X_train.shape[1]

    logger.info("Mutual information (k-NN estimator, random_state=42)...")
    t0 = time.time()
    mi = mutual_info_classif(X_train, y_train, random_state=42)
    logger.info("  done in %.1fs", time.time() - t0)

    logger.info("F-statistic ANOVA (univariada)...")
    f_stat, p_val = f_classif(X_train, y_train)
    f_stat = np.nan_to_num(f_stat, nan=0.0)
    p_val = np.nan_to_num(p_val, nan=1.0)

    logger.info("Pearson correlation sinalizada...")
    corr = np.zeros(n)
    for i in range(n):
        col = X_train[:, i]
        if col.std() > 0:
            corr[i], _ = scipy_stats.pearsonr(col, y_train)

    return mi, f_stat, p_val, corr


def univariate_f1(X_train: np.ndarray, y_train: np.ndarray,
                  X_test: np.ndarray, y_test: np.ndarray) -> np.ndarray:
    n = X_train.shape[1]
    out = np.zeros(n)
    logger.info("F1 univariado no teste para %d features...", n)
    t0 = time.time()
    for i in range(n):
        if X_train[:, i].std() == 0:
            continue
        clf = LogisticRegression(
            C=0.1, penalty="l2", solver="liblinear",
            max_iter=200, random_state=42,
        )
        clf.fit(X_train[:, i:i + 1], y_train)
        y_pred = clf.predict(X_test[:, i:i + 1])
        out[i] = f1_score(y_test, y_pred, average="weighted")
        if (i + 1) % 100 == 0:
            logger.info("  %d/%d features (%.0f%%) — %.1fs",
                        i + 1, n, 100 * (i + 1) / n, time.time() - t0)
    logger.info("  done in %.1fs", time.time() - t0)
    return out


def fit_svm_topk(X_train: np.ndarray, y_train: np.ndarray,
                 X_test: np.ndarray, y_test: np.ndarray,
                 idx: np.ndarray) -> Tuple[float, float]:
    clf = SVC(kernel="linear", C=0.1, random_state=42)
    clf.fit(X_train[:, idx], y_train)
    y_pred = clf.predict(X_test[:, idx])
    return (
        float(f1_score(y_test, y_pred, average="weighted")),
        float(accuracy_score(y_test, y_pred)),
    )


def forward_curve(X_train, y_train, X_test, y_test,
                  sorted_idx: np.ndarray, k_grid: List[int]) -> List[dict]:
    curve = []
    logger.info("Forward search (curva) sobre k_grid=%s", k_grid)
    n = len(sorted_idx)
    for k in k_grid:
        if k > n:
            continue
        t0 = time.time()
        f1, acc = fit_svm_topk(X_train, y_train, X_test, y_test, sorted_idx[:k])
        logger.info("  k=%-3d  F1=%.6f  acc=%.6f  (%.1fs)", k, f1, acc, time.time() - t0)
        curve.append({"k": int(k), "f1": f1, "accuracy": acc})
    return curve


def find_kmin(X_train, y_train, X_test, y_test,
              sorted_idx: np.ndarray, threshold: float, k_cap: int = 50
              ) -> Tuple[int, float]:
    """Procura linearmente k=1,2,... ate F1 >= threshold ou k>=k_cap."""
    logger.info("Refino k_min com threshold=%.4f (cap=%d)", threshold, k_cap)
    for k in range(1, min(k_cap, len(sorted_idx)) + 1):
        f1, acc = fit_svm_topk(X_train, y_train, X_test, y_test, sorted_idx[:k])
        logger.info("  k=%-3d  F1=%.6f  acc=%.6f", k, f1, acc)
        if f1 >= threshold:
            return k, f1
    return -1, -1.0


def make_pipeline(exp_dir: Path) -> LinguisticFeaturesPipeline:
    return LinguisticFeaturesPipeline(
        exp_dir=exp_dir,
        feature_mode="linguistic",
        feature_groups=[
            "nilcmetrics", "liwc", "enhanced_ud",
            "syllables", "pos_tagger", "parser_stats", "sage_terms",
        ],
        drop_nilc_absolute=True,
    )


def main(argv: List[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--threshold", type=float, default=0.9999,
                   help="F1 minimo para considerar 'reproduzir F1=1' (default: 0.9999)")
    p.add_argument("--k-cap", type=int, default=50,
                   help="Limite superior do refino linear de k_min (default: 50)")
    p.add_argument("--out-dir", type=Path, default=RESULTS_DIR)
    args = p.parse_args(argv[1:])

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    logger.info("Construindo pipeline linguistic (sem BERT)...")
    pipeline = make_pipeline(SCRIPT_DIR)

    splits = load_split_files(SCRIPT_DIR)
    for c, parts in splits.items():
        logger.info("  splits[%s] = train:%d test:%d",
                    c, len(parts["train"]), len(parts["test"]))

    logger.info("Carregando matrizes (assemble_matrices)...")
    X_train, y_train, X_test, y_test, _, _, feature_names = pipeline.assemble_matrices(splits)
    logger.info("  X_train=%s  X_test=%s  n_features=%d",
                X_train.shape, X_test.shape, len(feature_names))
    logger.info("  contagem por grupo: %s", group_counts(feature_names))

    logger.info("StandardScaler (fit em train, transform em test)...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    mi, f_stat, p_val, corr = compute_static_metrics(X_train, y_train)
    univ_f1 = univariate_f1(X_train, y_train, X_test, y_test)

    sorted_idx = np.argsort(-univ_f1)
    curve = forward_curve(X_train, y_train, X_test, y_test,
                          sorted_idx, DEFAULT_K_GRID)
    k_min, f1_at_kmin = find_kmin(X_train, y_train, X_test, y_test,
                                  sorted_idx, args.threshold, args.k_cap)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # CSV: todas as features, ranqueadas por univariate_f1
    csv_path = args.out_dir / "feature_diagnostic_ranking.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["rank_univariate_f1", "feature", "group",
                    "mi", "f_stat", "p_value", "pearson_corr", "univariate_f1"])
        for rank, idx in enumerate(sorted_idx, 1):
            name = feature_names[idx]
            w.writerow([
                rank,
                name,
                group_from_name(name),
                f"{mi[idx]:.6f}",
                f"{f_stat[idx]:.4f}",
                f"{p_val[idx]:.4e}",
                f"{corr[idx]:.4f}",
                f"{univ_f1[idx]:.6f}",
            ])
    logger.info("CSV: %s", csv_path)

    # JSON: sumario
    def topk(indices: List[int], k: int = 20) -> List[dict]:
        out = []
        for i in indices[:k]:
            out.append({
                "feature": feature_names[i],
                "group": group_from_name(feature_names[i]),
                "mi": float(mi[i]),
                "f_stat": float(f_stat[i]),
                "p_value": float(p_val[i]),
                "pearson_corr": float(corr[i]),
                "univariate_f1": float(univ_f1[i]),
            })
        return out

    top20_mi = sorted(range(len(mi)), key=lambda i: -mi[i])[:20]
    top20_f = sorted(range(len(f_stat)), key=lambda i: -f_stat[i])[:20]
    top20_univ = list(sorted_idx[:20])

    payload = {
        "n_features_total": len(feature_names),
        "features_by_group": group_counts(feature_names),
        "data_split": {"train": int(X_train.shape[0]), "test": int(X_test.shape[0])},
        "threshold_for_kmin": float(args.threshold),
        "top20_by_mutual_information": topk(top20_mi),
        "top20_by_f_statistic": topk(top20_f),
        "top20_by_univariate_f1": topk(top20_univ),
        "forward_search_curve": curve,
        "k_min": {
            "k": int(k_min),
            "f1": float(f1_at_kmin),
            "features": [feature_names[i] for i in sorted_idx[:k_min]] if k_min > 0 else None,
        },
    }
    json_path = args.out_dir / "feature_diagnostic.json"
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    logger.info("JSON: %s", json_path)

    logger.info("Concluido. k_min=%d, F1@k_min=%.6f", k_min, f1_at_kmin)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
