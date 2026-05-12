"""Aplica o NILC-Metrix em todos os textos do dataset e salva CSVs agregados.

Estrutura esperada do dataset:
    ~/bert/dataset/
    ├── human/
    │   ├── fake_true_human/*.txt
    │   └── fake_br_human/*.txt
    └── llm/
        ├── fake_true_llm/*.txt
        └── fake_br_llm/*.txt

Saída:
    ./results/human.csv
    ./results/llm.csv

Colunas: file, subset, <todas as métricas do nilcmetrix>
"""

from __future__ import annotations

import argparse
import csv
import sys
import traceback
from pathlib import Path

import text_metrics


def iter_texts(group_dir: Path):
    """Itera sobre (subset_name, txt_path) para todas as .txt do grupo."""
    for subset_dir in sorted(p for p in group_dir.iterdir() if p.is_dir()):
        for txt_path in sorted(subset_dir.glob("*.txt")):
            yield subset_dir.name, txt_path


def process_group(group_dir: Path, output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    writer = None
    fieldnames: list[str] | None = None
    f_out = output_csv.open("w", newline="", encoding="utf-8")
    n_ok = 0
    n_fail = 0
    failed: list[tuple[Path, str]] = []

    try:
        for subset, txt_path in iter_texts(group_dir):
            try:
                t = text_metrics.Text(filepath=str(txt_path))
                values = text_metrics.all_metrics.values_for_text(t)
            except Exception as exc:
                n_fail += 1
                failed.append((txt_path, repr(exc)))
                print(f"[FAIL] {txt_path}: {exc}", file=sys.stderr)
                continue

            row = {"file": txt_path.name, "subset": subset, **values}

            if writer is None:
                fieldnames = ["file", "subset"] + list(values.keys())
                writer = csv.DictWriter(f_out, fieldnames=fieldnames)
                writer.writeheader()
            else:
                extra = set(values.keys()) - set(fieldnames or [])
                if extra:
                    print(
                        f"[WARN] {txt_path.name} possui métricas extras ignoradas: {sorted(extra)}",
                        file=sys.stderr,
                    )
                row = {k: row.get(k, "") for k in fieldnames or []}

            writer.writerow(row)
            n_ok += 1
            if n_ok % 25 == 0:
                print(f"  ... {n_ok} textos processados em {group_dir.name}", flush=True)
    finally:
        f_out.close()

    print(f"[{group_dir.name}] ok={n_ok}  falhas={n_fail}  -> {output_csv}")
    if failed:
        log_path = output_csv.with_suffix(".failures.log")
        with log_path.open("w", encoding="utf-8") as logf:
            for p, err in failed:
                logf.write(f"{p}\t{err}\n")
        print(f"  log de falhas: {log_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path.home() / "bert" / "dataset",
        help="Diretório raiz do dataset (default: ~/bert/dataset)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "results",
        help="Diretório de saída dos CSVs (default: ./results)",
    )
    args = parser.parse_args()

    if not args.dataset.is_dir():
        print(f"Dataset não encontrado: {args.dataset}", file=sys.stderr)
        return 1

    for group in ("human", "llm"):
        group_dir = args.dataset / group
        if not group_dir.is_dir():
            print(f"[skip] grupo ausente: {group_dir}", file=sys.stderr)
            continue
        print(f"=== Processando grupo: {group} ===")
        try:
            process_group(group_dir, args.out / f"{group}.csv")
        except Exception:
            traceback.print_exc()
            return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
