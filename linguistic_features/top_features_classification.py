#!/usr/bin/env python3
"""
Classificacao Humano vs LLM com features escolhidas a dedo (top-10 por modulo).

Roda dois experimentos independentes:
  - nilcmetrics_top10: 5 features top-discriminantes para humano + 5 para LLM
                       do NILC-Metrix.
  - liwc_top10:        5 features top-discriminantes para humano + 5 para LLM
                       do LIWC (inclui agregadores LIWC; o loader e chamado
                       com drop_absolute=False so neste experimento).

Reusa a mesma metodologia de linguistic_features.py (splits pareados,
GridSearchCV com GroupKFold, StandardScaler, etc.). A unica diferenca em
relacao a module_classification.py e o filtro por whitelist de colunas em vez
de selecionar grupos inteiros.

Saidas (em linguistic_features/results/top_features/):
  - top_<exp>_<classifier>.json           (resultado individual)
  - top_<exp>_<classifier>_cm.png         (matriz de confusao)
  - top_<exp>_<classifier>_errors.json    (erros)
  - comparison_top_features.json          (grade experimento x classificador)
  - comparison_top_features_best.json     (melhor classificador por experimento)

Uso:
    python top_features_classification.py                          # tudo
    python top_features_classification.py --experiments liwc_top10
    python top_features_classification.py --classifier svm
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from linguistic_features import (
    CLASSIFIERS_CONFIG,
    CORPUS_DIRS_DEFAULT,
    LinguisticFeaturesPipeline,
    load_liwc,
    load_nilcmetrics,
    logger,
    merge_feature_frames,
)


# Whitelists (5 top-humano + 5 top-LLM por modulo). Os nomes ja vem com o
# prefixo que os loaders aplicam (`nilc__`/`liwc__`) e, no caso do LIWC, com
# o sufixo descritivo entre parenteses que e o nome real da coluna no CSV.
NILC_HUMAN_TOP = [
    "nilc__punctuation_diversity",
    "nilc__idade_aquisicao_1_25_ratio",
    "nilc__imageabilidade_4_55_ratio",
    "nilc__imageabilidade_mean",
    "nilc__concretude_mean",
]
NILC_LLM_TOP = [
    "nilc__honore",
    "nilc__idade_aquisicao_mean",
    "nilc__lsa_span_mean",
    "nilc__idade_aquisicao_55_7_ratio",
    "nilc__lsa_givenness_mean",
]
LIWC_HUMAN_TOP = [
    "liwc__social (Social)",
    "liwc__function (Function Words)",
    "liwc__focuspast (Past Focus)",
    "liwc__auxverb (Auxiliary Verbs)",
    "liwc__focuspresent (Present Focus)",
]
LIWC_LLM_TOP = [
    "liwc__cogproc (Cognitive Processes)",
    "liwc__drives (Drives)",
    "liwc__affect (Affect)",
    "liwc__work (Work)",
    "liwc__space (Space)",
]


EXPERIMENTS: Dict[str, Dict] = {
    "nilcmetrics_top10": {
        "groups": ["nilcmetrics"],
        "features": NILC_HUMAN_TOP + NILC_LLM_TOP,
        "liwc_keep_aggregators": False,  # nao se aplica
    },
    "liwc_top10": {
        "groups": ["liwc"],
        "features": LIWC_HUMAN_TOP + LIWC_LLM_TOP,
        "liwc_keep_aggregators": True,
    },
}


class TopFeaturesPipeline(LinguisticFeaturesPipeline):
    """LinguisticFeaturesPipeline com filtro por whitelist de colunas.

    Tambem permite manter as categorias agregadoras do LIWC (descartadas por
    padrao no loader). So sobrescreve `_load_linguistic_frame`, herdando todo
    o resto (montagem das matrizes, grid search, avaliacao, persistencia).
    """

    def __init__(
        self,
        feature_whitelist: List[str],
        liwc_keep_aggregators: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.feature_whitelist = feature_whitelist
        self.liwc_keep_aggregators = liwc_keep_aggregators

    def _load_linguistic_frame(self) -> Optional[pd.DataFrame]:
        frames = []
        if "nilcmetrics" in self.feature_groups:
            df = load_nilcmetrics(
                self.nilc_human_csv, self.nilc_llm_csv,
                drop_absolute=self.drop_nilc_absolute,
            )
            if df is not None:
                frames.append(df)
        if "liwc" in self.feature_groups:
            df = load_liwc(
                self.liwc_csv,
                drop_absolute=not self.liwc_keep_aggregators,
            )
            if df is not None:
                frames.append(df)
        # Outros grupos nao sao necessarios para os whitelists atuais; se for
        # estender, copie os blocos correspondentes de linguistic_features.py.
        if not frames:
            return None

        merged = merge_feature_frames(frames)

        meta = ["filename", "subset_corpus", "label"]
        available = set(merged.columns)
        missing = [c for c in self.feature_whitelist if c not in available]
        if missing:
            logger.warning(
                f"  whitelist: {len(missing)} feature(s) ausente(s) nos dados: {missing}"
            )
        keep = [c for c in self.feature_whitelist if c in available]
        if not keep:
            raise RuntimeError(
                "Nenhuma feature do whitelist encontrada apos load. "
                f"Esperadas: {self.feature_whitelist}"
            )
        logger.info(f"  whitelist: usando {len(keep)}/{len(self.feature_whitelist)} features")
        return merged[meta + keep]


def run_experiment(
    name: str,
    spec: Dict,
    classifiers: List[str],
    exp_dir: Path,
    results_root: Path,
    cv_folds: int,
    drop_nilc_absolute: bool,
    use_grid_search: bool,
    normalize_features: bool,
) -> Dict[str, Dict]:
    logger.info("\n" + "#" * 70)
    logger.info(f"# EXPERIMENTO: {name}  ({len(spec['features'])} features alvo)")
    logger.info("#" * 70)

    pipeline = TopFeaturesPipeline(
        feature_whitelist=spec["features"],
        liwc_keep_aggregators=spec["liwc_keep_aggregators"],
        exp_dir=exp_dir,
        feature_mode="linguistic",
        feature_groups=spec["groups"],
        corpus_dirs=CORPUS_DIRS_DEFAULT,
        cv_folds=cv_folds,
        drop_nilc_absolute=drop_nilc_absolute,
        experiment_label=f"top_{name}",
        results_root=results_root,
    )

    results: Dict[str, Dict] = {}
    for clf in classifiers:
        try:
            results[clf] = pipeline.run(
                classifier_type=clf,
                use_grid_search=use_grid_search,
                normalize_features=normalize_features,
            )
        except Exception as exc:
            logger.error(f"  {name} / {clf} falhou: {exc}")
    return results


def print_comparison(all_results: Dict[str, Dict[str, Dict]]):
    classifiers = sorted({c for exp in all_results.values() for c in exp})
    logger.info("\n" + "=" * 100)
    logger.info("COMPARACAO POR EXPERIMENTO (Test F1)")
    logger.info("=" * 100)
    logger.info(f"{'Experimento':<22}" + "".join(f"{c:<16}" for c in classifiers))
    logger.info("-" * 100)
    for exp, clf_res in all_results.items():
        row = f"{exp:<22}"
        for c in classifiers:
            f1 = clf_res.get(c, {}).get("test_results", {}).get("f1")
            row += f"{f1:<16.4f}" if isinstance(f1, float) else f"{'N/A':<16}"
        logger.info(row)
    logger.info("=" * 100)


def main():
    p = argparse.ArgumentParser(
        description="Classificacao Humano vs LLM com features hand-picked (top-10)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--experiments", default="all",
        help=f"Experimentos separados por virgula ({','.join(EXPERIMENTS)}) ou all",
    )
    p.add_argument(
        "--classifier", default="all",
        choices=list(CLASSIFIERS_CONFIG.keys()) + ["all"],
    )
    p.add_argument("--no-normalize", action="store_true")
    p.add_argument("--no-grid-search", action="store_true")
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument(
        "--keep-nilc-absolute", action="store_true",
        help="Mantem features length-biased do NILC (ablacao)",
    )
    args = p.parse_args()

    if args.experiments == "all":
        experiments = list(EXPERIMENTS)
    else:
        experiments = [e.strip() for e in args.experiments.split(",")]
        invalid = [e for e in experiments if e not in EXPERIMENTS]
        if invalid:
            p.error(f"Experimentos invalidos: {invalid}. Validos: {list(EXPERIMENTS)}")

    classifiers = (
        list(CLASSIFIERS_CONFIG.keys()) if args.classifier == "all"
        else [args.classifier]
    )

    script_dir = Path(__file__).resolve().parent
    results_root = script_dir / "results" / "top_features"

    all_results: Dict[str, Dict[str, Dict]] = {}
    for name in experiments:
        all_results[name] = run_experiment(
            name=name,
            spec=EXPERIMENTS[name],
            classifiers=classifiers,
            exp_dir=script_dir,
            results_root=results_root,
            cv_folds=args.cv_folds,
            drop_nilc_absolute=not args.keep_nilc_absolute,
            use_grid_search=not args.no_grid_search,
            normalize_features=not args.no_normalize,
        )

    comparison = {
        exp: {
            clf: {
                "cv_score": r.get("grid_search", {}).get("cv_score"),
                "test_f1": r.get("test_results", {}).get("f1"),
                "test_accuracy": r.get("test_results", {}).get("accuracy"),
                "test_precision": r.get("test_results", {}).get("precision"),
                "test_recall": r.get("test_results", {}).get("recall"),
                "best_params": r.get("grid_search", {}).get("best_params"),
                "n_features": r.get("n_features"),
            }
            for clf, r in clf_res.items()
        }
        for exp, clf_res in all_results.items()
    }
    results_root.mkdir(parents=True, exist_ok=True)
    comp_file = results_root / "comparison_top_features.json"
    with open(comp_file, "w", encoding="utf-8") as fh:
        json.dump(comparison, fh, indent=2, ensure_ascii=False)
    logger.info(f"\nComparacao salva em: {comp_file}")

    best_per_exp = {}
    for exp, clf_res in all_results.items():
        scored = [
            (clf, r.get("test_results", {}).get("f1"))
            for clf, r in clf_res.items()
            if isinstance(r.get("test_results", {}).get("f1"), float)
        ]
        if not scored:
            continue
        best_clf, best_f1 = max(scored, key=lambda x: x[1])
        r = clf_res[best_clf]
        best_per_exp[exp] = {
            "best_classifier": best_clf,
            "feature_groups": EXPERIMENTS[exp]["groups"],
            "feature_whitelist": EXPERIMENTS[exp]["features"],
            "test_f1": best_f1,
            "test_accuracy": r.get("test_results", {}).get("accuracy"),
            "cv_score": r.get("grid_search", {}).get("cv_score"),
            "n_features": r.get("n_features"),
        }
    best_file = results_root / "comparison_top_features_best.json"
    with open(best_file, "w", encoding="utf-8") as fh:
        json.dump(best_per_exp, fh, indent=2, ensure_ascii=False)
    logger.info(f"Melhor classificador por experimento salvo em: {best_file}")

    print_comparison(all_results)

    logger.info("\nMelhor classificador por experimento (Test F1):")
    for exp, info in sorted(best_per_exp.items(), key=lambda x: x[1]["test_f1"], reverse=True):
        logger.info(
            f"  {exp:<22} {info['best_classifier']:<20} "
            f"F1={info['test_f1']:.4f}  (n_feat={info['n_features']})"
        )


if __name__ == "__main__":
    main()
