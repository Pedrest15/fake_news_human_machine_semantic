"""Aplica NILC-Metrix em todos os .txt do dataset, dentro do container cohmetrix:focal.

Estrutura esperada (montada como /data dentro do container):
    /data/human/<subset>/*.txt
    /data/llm/<subset>/*.txt

Saída:
    /data/results/human.csv
    /data/results/llm.csv

Colunas: file, subset, <todas as métricas>.
"""

import csv
import sys
import traceback
from pathlib import Path

import text_metrics


DATA_ROOT = Path("/data")
OUT_ROOT = DATA_ROOT / "results"


def iter_texts(group_dir: Path):
    for subset_dir in sorted(p for p in group_dir.iterdir() if p.is_dir()):
        for txt_path in sorted(subset_dir.glob("*.txt")):
            yield subset_dir.name, txt_path


def process_group(group_dir: Path, output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    fieldnames = None
    n_ok = 0
    n_fail = 0
    failed = []

    with output_csv.open("w", newline="", encoding="utf-8") as f_out:
        for subset, txt_path in iter_texts(group_dir):
            try:
                text = txt_path.read_text(encoding="utf-8", errors="replace").strip()
                if not text:
                    raise ValueError("arquivo vazio")
                t = text_metrics.Text(text)
                values = text_metrics.nilc_metrics.values_for_text(t).as_flat_dict()
            except Exception as exc:
                n_fail += 1
                failed.append((txt_path, repr(exc)))
                print(f"[FAIL] {txt_path}: {exc}", file=sys.stderr, flush=True)
                continue

            row = {"file": txt_path.name, "subset": subset, **values}

            if writer is None:
                fieldnames = ["file", "subset"] + list(values.keys())
                writer = csv.DictWriter(f_out, fieldnames=fieldnames)
                writer.writeheader()
            else:
                row = {k: row.get(k, "") for k in fieldnames}

            writer.writerow(row)
            f_out.flush()
            n_ok += 1
            if n_ok % 10 == 0:
                print(f"  ... {n_ok} textos em {group_dir.name}", flush=True)

    print(f"[{group_dir.name}] ok={n_ok}  falhas={n_fail}  -> {output_csv}", flush=True)

    if failed:
        log_path = output_csv.with_suffix(".failures.log")
        with log_path.open("w", encoding="utf-8") as logf:
            for p, err in failed:
                logf.write(f"{p}\t{err}\n")
        print(f"  log de falhas: {log_path}", flush=True)


def main() -> int:
    if not DATA_ROOT.is_dir():
        print(f"Diretório do dataset não encontrado dentro do container: {DATA_ROOT}", file=sys.stderr)
        return 1

    for group in ("human", "llm"):
        group_dir = DATA_ROOT / group
        if not group_dir.is_dir():
            print(f"[skip] grupo ausente: {group_dir}", file=sys.stderr)
            continue
        print(f"=== Grupo: {group} ===", flush=True)
        try:
            process_group(group_dir, OUT_ROOT / f"{group}.csv")
        except Exception:
            traceback.print_exc()
            return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
