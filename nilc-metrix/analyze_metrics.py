"""Compara as métricas do NILC-Metrix entre os grupos human e llm.

Para cada métrica numérica computa:
    - média, mediana e desvio padrão por grupo
    - Cohen's d (tamanho de efeito)
    - Mann-Whitney U + p-value (não-paramétrico)
    - q-value (FDR via Benjamini-Hochberg)

Saída em ./analysis/:
    full_comparison.csv  — todas as métricas ordenadas por |Cohen's d|
    top_human.csv        — top 20 onde human > llm (significativas)
    top_llm.csv          — top 20 onde llm > human (significativas)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


META_COLS = {"file", "subset"}


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    na, nb = len(a), len(b)
    va, vb = a.var(ddof=1), b.var(ddof=1)
    pooled = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if pooled == 0:
        return np.nan
    return (a.mean() - b.mean()) / pooled


def benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    out = np.empty(n)
    out[order] = q
    return out


def numeric_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for c in df.columns:
        if c in META_COLS:
            continue
        series = pd.to_numeric(df[c], errors="coerce")
        if series.notna().sum() > 0:
            cols.append(c)
    return cols


def compare(human_csv: Path, llm_csv: Path, out_dir: Path, top: int = 20) -> None:
    h = pd.read_csv(human_csv)
    l = pd.read_csv(llm_csv)
    print(f"human: {len(h)} textos  |  llm: {len(l)} textos")

    metrics = sorted(set(numeric_columns(h)) & set(numeric_columns(l)))
    print(f"métricas numéricas em comum: {len(metrics)}")

    rows = []
    for m in metrics:
        a = pd.to_numeric(h[m], errors="coerce").to_numpy()
        b = pd.to_numeric(l[m], errors="coerce").to_numpy()
        a_v = a[~np.isnan(a)]
        b_v = b[~np.isnan(b)]
        if len(a_v) < 2 or len(b_v) < 2:
            continue

        d = cohens_d(a, b)
        try:
            _, pval = stats.mannwhitneyu(a_v, b_v, alternative="two-sided")
        except ValueError:
            pval = np.nan

        rows.append({
            "metric": m,
            "n_human": len(a_v),
            "n_llm": len(b_v),
            "mean_human": a_v.mean(),
            "mean_llm": b_v.mean(),
            "median_human": np.median(a_v),
            "median_llm": np.median(b_v),
            "std_human": a_v.std(ddof=1) if len(a_v) > 1 else np.nan,
            "std_llm": b_v.std(ddof=1) if len(b_v) > 1 else np.nan,
            "diff_means": a_v.mean() - b_v.mean(),
            "cohens_d": d,
            "abs_cohens_d": abs(d) if not np.isnan(d) else np.nan,
            "p_value": pval,
            "higher_in": "human" if d > 0 else ("llm" if d < 0 else "tie"),
        })

    df = pd.DataFrame(rows)
    mask = df["p_value"].notna()
    df.loc[mask, "q_value_fdr"] = benjamini_hochberg(df.loc[mask, "p_value"].to_numpy())
    df = df.sort_values("abs_cohens_d", ascending=False).reset_index(drop=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    full_path = out_dir / "full_comparison.csv"
    df.to_csv(full_path, index=False)

    sig = df[(df["q_value_fdr"] < 0.05) & df["cohens_d"].notna()]
    top_human = sig[sig["cohens_d"] > 0].head(top)
    top_llm = sig[sig["cohens_d"] < 0].assign(abs_d=lambda x: x["cohens_d"].abs()) \
                                       .sort_values("abs_d", ascending=False) \
                                       .drop(columns="abs_d") \
                                       .head(top)

    top_human.to_csv(out_dir / "top_human.csv", index=False)
    top_llm.to_csv(out_dir / "top_llm.csv", index=False)

    cols_show = ["metric", "mean_human", "mean_llm", "cohens_d", "q_value_fdr"]
    print(f"\n=== TOP {top} métricas predominantes em HUMAN (FDR<0.05) ===")
    print(top_human[cols_show].to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print(f"\n=== TOP {top} métricas predominantes em LLM (FDR<0.05) ===")
    print(top_llm[cols_show].to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    print(f"\nArquivos salvos em {out_dir}/")
    print(f"  - full_comparison.csv  ({len(df)} métricas)")
    print(f"  - top_human.csv        ({len(top_human)} métricas)")
    print(f"  - top_llm.csv          ({len(top_llm)} métricas)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--human", type=Path, default=Path(__file__).parent / "results" / "human.csv")
    parser.add_argument("--llm",   type=Path, default=Path(__file__).parent / "results" / "llm.csv")
    parser.add_argument("--out",   type=Path, default=Path(__file__).parent / "analysis")
    parser.add_argument("--top",   type=int,  default=20)
    args = parser.parse_args()

    if not args.human.is_file() or not args.llm.is_file():
        print(f"CSV não encontrado: {args.human} / {args.llm}")
        return 1

    compare(args.human, args.llm, args.out, args.top)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
