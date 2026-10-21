#!/usr/bin/env python3
"""
Classificadores sobre os embeddings [CLS] de um BERT JA AJUSTADO (fine-tuned).

Complementa bert_finetuning.py: pega o melhor fold salvo por ele, extrai o
vetor [CLS] da ultima camada do encoder ajustado (sem a cabeca de
classificacao) e treina os seis classificadores do repositorio sobre esse
vetor, com o mesmo protocolo dos embeddings congelados (split 80/20 por par,
truncamento pareado, GridSearchCV com GroupKFold agrupado por par).

Atencao: o encoder ajustado ja viu os textos de treino do fold escolhido, entao
os embeddings de treino sao mais separaveis que os de teste. O teste de 20%
nunca foi visto pelo encoder, e e nele que se reporta o resultado.

Uso:
    python3 linguistic_features/finetuned_embeddings.py --bert-model bertimbau
    python3 linguistic_features/finetuned_embeddings.py --bert-model bert-multilingual --classifier svm
    python3 linguistic_features/finetuned_embeddings.py --model-dir path/to/final_model --label my_run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bert_finetuning import FINETUNE_MODELS, default_label  # noqa: E402
from linguistic_features import (  # noqa: E402
    CLASSIFIERS_CONFIG,
    DEFAULT_BERT_MODEL_KEY,
    DEFAULT_MAX_LENGTH,
    LinguisticFeaturesPipeline,
    print_comparison_table,
)

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent


def resolve_model_dir(bert_model: str, max_length: int, model_dir: Optional[str],
                      results_root: Path) -> Path:
    """Diretorio do encoder ajustado: --model-dir ou o melhor fold do JSON do fine-tuning."""
    if model_dir:
        path = Path(model_dir)
    else:
        results_json = results_root / f"{default_label(bert_model, max_length)}.json"
        if not results_json.exists():
            raise FileNotFoundError(
                f"{results_json} not found: run bert_finetuning.py first or pass --model-dir"
            )
        with open(results_json, encoding="utf-8") as fh:
            info = json.load(fh)
        best = info["cross_validation"]["best_fold"]
        path = next(Path(f["model_dir"]) for f in info["cross_validation"]["folds"] if f["fold"] == best)
        logger.info(f"  Using the best fold ({best}) from {results_json.name}")
    if not (path / "config.json").exists():
        raise FileNotFoundError(f"No fine-tuned model found in {path}")
    return path


def build_pipeline(model_dir: Path, bert_model: str, max_length: int, label: str,
                   cv_folds: int, results_root: Path, cache_dir: Path) -> LinguisticFeaturesPipeline:
    """Pipeline de embeddings que aponta para o encoder ajustado em vez do pre-treinado."""
    pipeline = LinguisticFeaturesPipeline(
        exp_dir=SCRIPT_DIR,
        feature_mode="embedding",
        bert_model=bert_model,
        pooling="cls",
        max_length=max_length,
        cv_folds=cv_folds,
        experiment_label=label,
        results_root=results_root,
        cache_dir=cache_dir,
    )
    # A local path works for AutoTokenizer/AutoModel; the tokenizer saved next to
    # the weights is the original one, so the paired truncation is unchanged.
    pipeline.bert_model_name = str(model_dir)
    return pipeline


def parse_args():
    p = argparse.ArgumentParser(
        description="Classifiers over the [CLS] embeddings of a fine-tuned BERT encoder",
    )
    p.add_argument("--bert-model", default=DEFAULT_BERT_MODEL_KEY, choices=FINETUNE_MODELS)
    p.add_argument("--classifier", default="all", choices=list(CLASSIFIERS_CONFIG.keys()) + ["all"])
    p.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--model-dir", type=str, default=None,
                   help="Fine-tuned model directory (default: best fold of the bert_finetuning.py results)")
    p.add_argument("--label", type=str, default=None,
                   help="Prefix of the result files (default: finetuned_embedding_<model>)")
    p.add_argument("--no-grid-search", action="store_true")
    p.add_argument("--results-root", type=Path, default=SCRIPT_DIR / "results")
    p.add_argument("--cache-dir", type=Path, default=SCRIPT_DIR / "cache")
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()

    model_dir = resolve_model_dir(args.bert_model, args.max_length, args.model_dir, args.results_root)
    label = args.label or f"finetuned_embedding_{args.bert_model.replace('-', '_')}"
    pipeline = build_pipeline(model_dir, args.bert_model, args.max_length, label,
                              args.cv_folds, args.results_root, args.cache_dir)

    classifiers = list(CLASSIFIERS_CONFIG.keys()) if args.classifier == "all" else [args.classifier]
    results = {
        clf: pipeline.run(classifier_type=clf, use_grid_search=not args.no_grid_search)
        for clf in classifiers
    }

    if len(classifiers) > 1:
        comparison = {
            name: {
                "cv_score": r.get("grid_search", {}).get("cv_score"),
                "test_f1": r["test_results"]["f1"],
                "test_accuracy": r["test_results"]["accuracy"],
                "best_params": r.get("grid_search", {}).get("best_params"),
            }
            for name, r in results.items()
        }
        out = args.results_root / f"comparison_{label}.json"
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(comparison, fh, indent=2, ensure_ascii=False)
        logger.info(f"Comparison saved to {out}")
        print_comparison_table(results, title=label.upper())


if __name__ == "__main__":
    main()
