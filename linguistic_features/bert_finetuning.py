#!/usr/bin/env python3
"""
Fine-tuning de BERT (BERTimbau / BERT multilingual) para deteccao de autoria
humano vs LLM.

Protocolo (o mesmo dos classificadores sobre embeddings congelados):
  - split 80/20 pre-definido em train_*.txt / test_*.txt; cada filename e um PAR
    (texto humano + texto LLM) e o par inteiro cai do mesmo lado do split;
  - truncamento pareado: os dois textos do par sao cortados no menor numero de
    tokens do par (ate --max-length), para o modelo nao aprender o tamanho;
  - validacao cruzada de --cv-folds folds SOBRE O TREINO, agrupada por par
    (humano e LLM do mesmo filename nunca ficam em lados diferentes de um fold);
  - o fold com maior F1 de validacao e avaliado uma unica vez no teste de 20%.

Uso:
    python3 linguistic_features/bert_finetuning.py --bert-model bertimbau
    python3 linguistic_features/bert_finetuning.py --bert-model bert-multilingual
    python3 linguistic_features/bert_finetuning.py --bert-model bertimbau --folds-to-run 1 --epochs 1
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from linguistic_features import (  # noqa: E402
    BERT_MODELS,
    CORPUS_DIRS_DEFAULT,
    DEFAULT_BERT_MODEL_KEY,
    DEFAULT_MAX_LENGTH,
    load_paired_texts,
    load_split_files,
    save_confusion_matrix,
    save_misclassified,
    truncate_pair_min_tokens,
)

logger = logging.getLogger(__name__)

# Fine-tuning is defined for the encoders with a trained [CLS]/pooler head.
FINETUNE_MODELS = ("bertimbau", "bert-multilingual")

# Hyperparameters carried over from the original fine-tuning pipeline.
DEFAULTS = {
    "epochs": 3,
    "train_batch_size": 8,
    "eval_batch_size": 16,
    "learning_rate": 5e-5,
    "weight_decay": 0.01,
    "warmup_steps": 500,
    "eval_steps": 500,
    "early_stopping_patience": 3,
    "seed": 42,
}

# =============================================================================
# DATA
# =============================================================================

def build_samples(pairs, tokenizer, max_length: int) -> List[Tuple[Tuple[str, str, int], str, str]]:
    """
    Expande pares (fn, corpus, humano, llm) em duas amostras com truncamento pareado.

    Retorna tuplas (key, group, text), key = (filename, corpus, label) e
    group = "filename__corpus", a unidade que nunca pode ser dividida.
    """
    samples = []
    for fn, corpus, human, llm in pairs:
        h_t, l_t, _ = truncate_pair_min_tokens(human, llm, tokenizer, max_length=max_length)
        group = f"{fn}__{corpus}"
        samples.append(((fn, corpus, 0), group, h_t))
        samples.append(((fn, corpus, 1), group, l_t))
    return samples


def grouped_folds(groups: List[str], n_folds: int, seed: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Folds deterministicos agrupados: embaralha os grupos unicos e os distribui
    em round-robin. Cada grupo (par) tem uma amostra humana e uma LLM, entao os
    folds saem balanceados por construcao.
    """
    unique = sorted(set(groups))
    random.Random(seed).shuffle(unique)
    fold_of = {g: i % n_folds for i, g in enumerate(unique)}
    fold_ids = np.array([fold_of[g] for g in groups])
    return [
        (np.where(fold_ids != f)[0], np.where(fold_ids == f)[0])
        for f in range(n_folds)
    ]


class EncodedDataset:
    """Dataset de input_ids ja tokenizados; o padding e feito por batch no collator."""

    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int):
        enc = tokenizer(texts, truncation=True, max_length=max_length)
        self.items = [
            {"input_ids": ids, "attention_mask": mask, "labels": int(label)}
            for ids, mask, label in zip(enc["input_ids"], enc["attention_mask"], labels)
        ]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def compute_metrics(eval_pred) -> Dict[str, float]:
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
    }


# =============================================================================
# TRAINING
# =============================================================================

def make_training_args(output_dir: Path, hp: Dict, run_name: str):
    from transformers import TrainingArguments

    return TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=hp["epochs"],
        per_device_train_batch_size=hp["train_batch_size"],
        per_device_eval_batch_size=hp["eval_batch_size"],
        learning_rate=hp["learning_rate"],
        weight_decay=hp["weight_decay"],
        warmup_steps=hp["warmup_steps"],
        logging_steps=100,
        eval_strategy="steps",
        eval_steps=hp["eval_steps"],
        save_strategy="steps",
        save_steps=hp["eval_steps"],
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=1,
        seed=hp["seed"],
        report_to="none",
        run_name=run_name,
        dataloader_pin_memory=False,
    )


def train_fold(model_name: str, tokenizer, train_ds, val_ds, output_dir: Path, hp: Dict,
               run_name: str) -> Tuple[Dict, Path]:
    """Treina um fold a partir dos pesos pre-treinados; devolve metricas de validacao e o modelo salvo."""
    from transformers import (
        AutoModelForSequenceClassification,
        DataCollatorWithPadding,
        EarlyStoppingCallback,
        Trainer,
    )

    # Fresh classification head (pooler + dropout + Linear(hidden, 2)) on top of
    # the pre-trained encoder; the encoder itself is fine-tuned, nothing is frozen.
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2, problem_type="single_label_classification"
    )
    trainer = Trainer(
        model=model,
        args=make_training_args(output_dir, hp, run_name),
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=hp["early_stopping_patience"])],
    )
    trainer.train()
    val_metrics = trainer.evaluate()

    final_dir = output_dir / "final_model"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    return {k.removeprefix("eval_"): v for k, v in val_metrics.items()}, final_dir


def evaluate_on_test(model_dir: Path, tokenizer, test_ds, hp: Dict, output_dir: Path):
    """Avalia o modelo escolhido no teste; devolve metricas, matriz de confusao, predicoes e probabilidades."""
    from sklearn.metrics import confusion_matrix
    from transformers import AutoModelForSequenceClassification, DataCollatorWithPadding, Trainer
    from transformers import TrainingArguments

    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    args = TrainingArguments(
        output_dir=str(output_dir / "test_eval"),
        per_device_eval_batch_size=hp["eval_batch_size"],
        report_to="none",
        dataloader_pin_memory=False,
    )
    trainer = Trainer(model=model, args=args, data_collator=DataCollatorWithPadding(tokenizer),
                      compute_metrics=compute_metrics)
    out = trainer.predict(test_ds)
    logits = out.predictions
    y_pred = np.argmax(logits, axis=1)
    y_true = out.label_ids
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp / exp.sum(axis=1, keepdims=True)
    metrics = {k.removeprefix("test_"): float(v) for k, v in out.metrics.items()
               if k.removeprefix("test_") in ("accuracy", "f1", "precision", "recall")}
    return metrics, confusion_matrix(y_true, y_pred), y_true, y_pred, probs


# =============================================================================
# PIPELINE
# =============================================================================

def default_label(bert_model: str, max_length: int) -> str:
    suffix = "" if max_length == DEFAULT_MAX_LENGTH else f"_len{max_length}"
    return f"finetune_{bert_model.replace('-', '_')}{suffix}"


def run(args) -> Dict:
    from transformers import AutoTokenizer, set_seed

    hp = {
        "epochs": args.epochs,
        "train_batch_size": args.train_batch_size,
        "eval_batch_size": args.eval_batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_steps": args.warmup_steps,
        "eval_steps": args.eval_steps,
        "early_stopping_patience": args.early_stopping_patience,
        "seed": args.seed,
    }
    set_seed(args.seed)

    script_dir = Path(__file__).resolve().parent
    model_name = BERT_MODELS[args.bert_model]
    label = args.experiment_label or default_label(args.bert_model, args.max_length)
    results_dir = script_dir / "results"
    models_dir = script_dir / "models" / label

    corpus_dirs = CORPUS_DIRS_DEFAULT
    if args.corpus_root:
        croot = Path(args.corpus_root)
        corpus_dirs = {
            "fake_br_human": croot / "corpus" / "Fake.br-Corpus-master" / "full_texts" / "fake",
            "fake_br_llm": croot / "corpus" / "fake-news-llm-ptbr-main" / "fake-news-llm-ptbr-main" / "data" / "Fake.Br",
            "fake_true_human": croot / "corpus" / "FakeTrue.Br-main" / "fake",
            "fake_true_llm": croot / "corpus" / "fake-news-llm-ptbr-main" / "fake-news-llm-ptbr-main" / "data" / "FakeTrueBR",
        }

    logger.info("=" * 70)
    logger.info(f"FINE-TUNING: {model_name} | label={label}")
    logger.info("=" * 70)

    splits = load_split_files(script_dir)
    paired = load_paired_texts(splits, corpus_dirs)
    if not paired["train"] or not paired["test"]:
        raise RuntimeError("No train/test pairs — check the splits and the corpus directories")
    if args.limit_pairs:
        paired = {k: v[:args.limit_pairs] for k, v in paired.items()}
        logger.warning(f"  --limit-pairs: using only {args.limit_pairs} pairs per split (smoke test)")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_samples = build_samples(paired["train"], tokenizer, args.max_length)
    test_samples = build_samples(paired["test"], tokenizer, args.max_length)
    logger.info(f"  Train: {len(train_samples)} texts ({len(paired['train'])} pairs) | "
                f"Test: {len(test_samples)} texts ({len(paired['test'])} pairs)")

    train_keys = [s[0] for s in train_samples]
    train_groups = [s[1] for s in train_samples]
    train_texts = [s[2] for s in train_samples]
    train_labels = [k[2] for k in train_keys]

    folds = grouped_folds(train_groups, args.cv_folds, args.seed)
    fold_results = []
    for fold, (tr_idx, va_idx) in enumerate(folds, start=1):
        if args.folds_to_run and fold > args.folds_to_run:
            break
        # Guard against the leak this design exists to prevent.
        assert not ({train_groups[i] for i in tr_idx} & {train_groups[i] for i in va_idx})
        logger.info(f"--- Fold {fold}/{args.cv_folds}: {len(tr_idx)} train / {len(va_idx)} val texts ---")
        train_ds = EncodedDataset([train_texts[i] for i in tr_idx], [train_labels[i] for i in tr_idx],
                                  tokenizer, args.max_length)
        val_ds = EncodedDataset([train_texts[i] for i in va_idx], [train_labels[i] for i in va_idx],
                                tokenizer, args.max_length)
        val_metrics, model_dir = train_fold(
            model_name, tokenizer, train_ds, val_ds, models_dir / f"fold_{fold}", hp,
            run_name=f"{label}/fold_{fold}",
        )
        logger.info(f"  Fold {fold}: val F1={val_metrics['f1']:.4f} acc={val_metrics['accuracy']:.4f}")
        fold_results.append({"fold": fold, "val_metrics": val_metrics, "model_dir": str(model_dir)})

    best = max(fold_results, key=lambda r: r["val_metrics"]["f1"])
    logger.info(f"Best fold: {best['fold']} (val F1={best['val_metrics']['f1']:.4f}); evaluating on the test set")

    test_ds = EncodedDataset([s[2] for s in test_samples], [s[0][2] for s in test_samples],
                             tokenizer, args.max_length)
    test_metrics, cm, y_true, y_pred, probs = evaluate_on_test(
        Path(best["model_dir"]), tokenizer, test_ds, hp, models_dir
    )
    tn, fp, fn, tp = cm.ravel()
    logger.info(f"  Test F1={test_metrics['f1']:.4f} Acc={test_metrics['accuracy']:.4f}")
    logger.info(f"  CM: TN={tn} FP={fp} FN={fn} TP={tp}")

    val_f1s = [r["val_metrics"]["f1"] for r in fold_results]
    results = {
        "experiment": label,
        "bert_model": model_name,
        "max_length": args.max_length,
        "truncation": "paired (shorter member of the human/LLM pair)",
        "classification_head": "AutoModelForSequenceClassification, num_labels=2 "
                               "(BERT pooler + dropout + Linear(hidden_size, 2), cross-entropy); "
                               "encoder fully fine-tuned",
        "hyperparameters": {**hp, "optimizer": "adamw_torch", "lr_scheduler": "linear",
                            "best_model_metric": "f1"},
        "data_split": {
            "protocol": "pre-defined 80/20 split by pair; grouped CV folds on the 80% train part",
            "train_pairs": len(paired["train"]), "test_pairs": len(paired["test"]),
            "train_samples": len(train_samples), "test_samples": len(test_samples),
            "cv_folds": args.cv_folds, "folds_run": len(fold_results),
        },
        "cross_validation": {
            "folds": fold_results,
            "val_f1_mean": float(np.mean(val_f1s)), "val_f1_std": float(np.std(val_f1s)),
            "best_fold": best["fold"],
        },
        "test_results": test_metrics,
        "confusion_matrix": {"matrix": cm.tolist(), "tn": int(tn), "fp": int(fp),
                             "fn": int(fn), "tp": int(tp)},
    }

    results_dir.mkdir(parents=True, exist_ok=True)
    out_json = results_dir / f"{label}.json"
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
    logger.info(f"  Results saved to {out_json}")

    save_confusion_matrix(cm, results_dir / "confusion_matrices" / f"{label}_cm.png", label.upper())
    save_misclassified(
        [s[0] for s in test_samples], y_true, y_pred,
        results_dir / "misclassified" / f"{label}_errors.json",
        experiment=label, probs=probs,
    )
    return results


def parse_args():
    p = argparse.ArgumentParser(
        description="Fine-tune BERT for human vs LLM detection with the pair-grouped 80/20 split",
    )
    p.add_argument("--bert-model", default=DEFAULT_BERT_MODEL_KEY, choices=FINETUNE_MODELS)
    p.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH,
                   help=f"Token cap of the paired truncation (default: {DEFAULT_MAX_LENGTH})")
    p.add_argument("--cv-folds", type=int, default=5)
    p.add_argument("--folds-to-run", type=int, default=None,
                   help="Train only the first N folds (default: all)")
    p.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    p.add_argument("--train-batch-size", type=int, default=DEFAULTS["train_batch_size"])
    p.add_argument("--eval-batch-size", type=int, default=DEFAULTS["eval_batch_size"])
    p.add_argument("--learning-rate", type=float, default=DEFAULTS["learning_rate"])
    p.add_argument("--weight-decay", type=float, default=DEFAULTS["weight_decay"])
    p.add_argument("--warmup-steps", type=int, default=DEFAULTS["warmup_steps"])
    p.add_argument("--eval-steps", type=int, default=DEFAULTS["eval_steps"])
    p.add_argument("--early-stopping-patience", type=int, default=DEFAULTS["early_stopping_patience"])
    p.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    p.add_argument("--experiment-label", type=str, default=None,
                   help="Prefix of the result files (default: finetune_<model>)")
    p.add_argument("--corpus-root", type=str, default=None,
                   help="Override the corpus root (default: linguistic_features/data/corpus)")
    p.add_argument("--limit-pairs", type=int, default=None,
                   help="Smoke test: keep only the first N pairs of each split")
    return p.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    run(parse_args())


if __name__ == "__main__":
    main()
