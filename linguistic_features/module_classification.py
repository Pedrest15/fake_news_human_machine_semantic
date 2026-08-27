#!/usr/bin/env python3
"""
Classificacao Humano vs LLM por MODULO de features linguisticas.

Diferente de linguistic_features.py (que concatena TODOS os grupos num unico
vetor), este pipeline treina/avalia cada MODULO de features ISOLADAMENTE:
roda a classificacao so com NILCmetrics, depois so com LIWC, depois so com
Enhanced-UD, depois so com Syntax, etc. O objetivo e medir o poder
discriminativo de cada modulo sozinho e compara-los entre si.

Um modulo pode ser um unico grupo de features (ex.: nilcmetrics) ou um
agrupamento de varios grupos (ex.: syntax = syllables + pos_tagger +
parser_stats + sage_terms).

Para cada modulo e cada classificador reusa toda a metodologia de
linguistic_features.py (mesmos splits pareados, GridSearchCV com GroupKFold
no treino, avaliacao no teste held-out, StandardScaler, etc.) — apenas o
conjunto de features muda.

Saidas (em linguistic_features/results/modules/):
  - module_<modulo>_<classifier>.json           (resultado individual)
  - module_<modulo>_<classifier>_cm.png         (matriz de confusao)
  - module_<modulo>_<classifier>_errors.json    (erros)
  - comparison_modules.json                      (grade modulo x classificador)
  - comparison_modules_best.json                 (melhor classificador por modulo)

Uso:
    python module_classification.py                       # todos modulos, todos clfs
    python module_classification.py --modules nilcmetrics,syntax
    python module_classification.py --classifier svm
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

from linguistic_features import (
    CLASSIFIERS_CONFIG,
    CORPUS_DIRS_DEFAULT,
    LinguisticFeaturesPipeline,
    logger,
)

# Linguistic feature modules: name -> feature groups it is made of.
# Every group must be a name accepted by LinguisticFeaturesPipeline.feature_groups.
#
# Each individual group is a module of its own. On top of that, 'syntax' is an
# aggregate module that merges the 4 syntactic groups (syllables + pos_tagger +
# parser_stats + sage_terms) into a single vector for one joint classification.
MODULES: Dict[str, List[str]] = {
    'nilcmetrics': ['nilcmetrics'],
    'liwc': ['liwc'],
    'enhanced_ud': ['enhanced_ud'],
    'syllables': ['syllables'],
    'pos_tagger': ['pos_tagger'],
    'parser_stats': ['parser_stats'],
    'sage_terms': ['sage_terms'],
    'syntax': ['syllables', 'pos_tagger', 'parser_stats', 'sage_terms'],
}


def run_module(
    module: str,
    feature_groups: List[str],
    classifiers: List[str],
    exp_dir: Path,
    results_root: Path,
    corpus_dirs: Dict[str, Path],
    ud_top_k: int,
    cv_folds: int,
    drop_nilc_absolute: bool,
    use_grid_search: bool,
    normalize_features: bool,
    top_k_features: int,
) -> Dict[str, Dict]:
    """Treina/avalia todos os classificadores usando apenas `module`."""
    logger.info("\n" + "#" * 70)
    logger.info(f"# MODULE: {module}  (groups: {', '.join(feature_groups)})")
    logger.info("#" * 70)

    pipeline = LinguisticFeaturesPipeline(
        exp_dir=exp_dir,
        feature_mode='linguistic',
        feature_groups=feature_groups,
        corpus_dirs=corpus_dirs,
        ud_top_k=ud_top_k,
        cv_folds=cv_folds,
        drop_nilc_absolute=drop_nilc_absolute,
        experiment_label=f"module_{module}",
        results_root=results_root,
    )

    module_results: Dict[str, Dict] = {}
    for clf in classifiers:
        try:
            module_results[clf] = pipeline.run(
                classifier_type=clf,
                use_grid_search=use_grid_search,
                normalize_features=normalize_features,
                top_k_features=top_k_features,
            )
        except Exception as exc:
            logger.error(f"  Module {module} / classifier {clf} failed: {exc}")
    return module_results


def print_module_comparison(all_results: Dict[str, Dict[str, Dict]]):
    """Imprime grade modulo x classificador (Test F1)."""
    classifiers = sorted({c for mod in all_results.values() for c in mod})
    logger.info("\n" + "=" * 100)
    logger.info("COMPARISON PER MODULE (Test F1)")
    logger.info("=" * 100)
    header = f"{'Module':<16}" + "".join(f"{c:<16}" for c in classifiers)
    logger.info(header)
    logger.info("-" * 100)
    for module, clf_res in all_results.items():
        row = f"{module:<16}"
        for c in classifiers:
            f1 = clf_res.get(c, {}).get('test_results', {}).get('f1')
            row += f"{f1:<16.4f}" if isinstance(f1, float) else f"{'N/A':<16}"
        logger.info(row)
    logger.info("=" * 100)


def main():
    p = argparse.ArgumentParser(
        description='Human vs LLM classification, one linguistic feature module at a time',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--modules', default='all',
                   help=('Comma-separated list of modules to evaluate '
                         f'({",".join(MODULES)}) or all'))
    p.add_argument('--classifier', default='all',
                   choices=list(CLASSIFIERS_CONFIG.keys()) + ['all'])
    p.add_argument('--ud-top-k', type=int, default=500)
    p.add_argument('--top-k', type=int, default=None,
                   help='SelectKBest over the module vector')
    p.add_argument('--no-normalize', action='store_true')
    p.add_argument('--no-grid-search', action='store_true')
    p.add_argument('--cv-folds', type=int, default=5)
    p.add_argument('--keep-nilc-absolute', action='store_true',
                   help='Keep the length-biased NILC features (ablation)')
    args = p.parse_args()

    if args.modules == 'all':
        modules = list(MODULES)
    else:
        modules = [m.strip() for m in args.modules.split(',')]
        invalid = [m for m in modules if m not in MODULES]
        if invalid:
            p.error(f"Invalid modules: {invalid}. Valid ones: {list(MODULES)}")

    classifiers = (list(CLASSIFIERS_CONFIG.keys())
                   if args.classifier == 'all' else [args.classifier])

    script_dir = Path(__file__).resolve().parent
    results_root = script_dir / "results" / "modules"

    all_results: Dict[str, Dict[str, Dict]] = {}
    for module in modules:
        all_results[module] = run_module(
            module=module,
            feature_groups=MODULES[module],
            classifiers=classifiers,
            exp_dir=script_dir,
            results_root=results_root,
            corpus_dirs=CORPUS_DIRS_DEFAULT,
            ud_top_k=args.ud_top_k,
            cv_folds=args.cv_folds,
            drop_nilc_absolute=not args.keep_nilc_absolute,
            use_grid_search=not args.no_grid_search,
            normalize_features=not args.no_normalize,
            top_k_features=args.top_k,
        )

    # ---- Module x classifier comparison ----
    comparison = {
        module: {
            clf: {
                'cv_score': r.get('grid_search', {}).get('cv_score'),
                'test_f1': r.get('test_results', {}).get('f1'),
                'test_accuracy': r.get('test_results', {}).get('accuracy'),
                'test_precision': r.get('test_results', {}).get('precision'),
                'test_recall': r.get('test_results', {}).get('recall'),
                'best_params': r.get('grid_search', {}).get('best_params'),
                'n_features': r.get('n_features'),
            }
            for clf, r in clf_res.items()
        }
        for module, clf_res in all_results.items()
    }

    results_root.mkdir(parents=True, exist_ok=True)
    comp_file = results_root / "comparison_modules.json"
    with open(comp_file, 'w', encoding='utf-8') as fh:
        json.dump(comparison, fh, indent=2, ensure_ascii=False)
    logger.info(f"\nPer-module comparison saved to: {comp_file}")

    # ---- Best classifier per module (by Test F1) ----
    best_per_module = {}
    for module, clf_res in all_results.items():
        scored = [
            (clf, r.get('test_results', {}).get('f1'))
            for clf, r in clf_res.items()
            if isinstance(r.get('test_results', {}).get('f1'), float)
        ]
        if not scored:
            continue
        best_clf, best_f1 = max(scored, key=lambda x: x[1])
        r = clf_res[best_clf]
        best_per_module[module] = {
            'best_classifier': best_clf,
            'feature_groups': MODULES[module],
            'test_f1': best_f1,
            'test_accuracy': r.get('test_results', {}).get('accuracy'),
            'cv_score': r.get('grid_search', {}).get('cv_score'),
            'n_features': r.get('n_features'),
        }

    best_file = results_root / "comparison_modules_best.json"
    with open(best_file, 'w', encoding='utf-8') as fh:
        json.dump(best_per_module, fh, indent=2, ensure_ascii=False)
    logger.info(f"Best classifier per module saved to: {best_file}")

    print_module_comparison(all_results)

    logger.info("\nBest classifier per module (Test F1):")
    for module, info in sorted(best_per_module.items(),
                               key=lambda x: x[1]['test_f1'], reverse=True):
        logger.info(f"  {module:<16} {info['best_classifier']:<20} "
                    f"F1={info['test_f1']:.4f}  (n_feat={info['n_features']})")


if __name__ == "__main__":
    main()
