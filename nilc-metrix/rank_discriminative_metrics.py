"""Rankeia as métricas do NILC-Metrix por poder discriminativo entre human e llm.

Restringe a análise às métricas de complexidade textual, coesão e semântica
(exclui morfologia e sintaxe). Para cada métrica computa:
    - mediana e MAD por grupo (não-paramétricos, robustos a outliers)
    - Mann-Whitney U + p-value bilateral
    - Cliff's delta como tamanho de efeito (δ = 2·U₁/(n_h·n_l) − 1)
    - AUC do ranque (= (δ+1)/2, prob. de um texto humano ter valor maior que LLM)
    - q-value via Benjamini-Hochberg restrito às métricas filtradas

Magnitude de Cliff's δ segundo Romano et al. (2006):
    large       |δ| ≥ 0.474
    medium      0.330 ≤ |δ| < 0.474
    small       0.147 ≤ |δ| < 0.330
    negligible  |δ| < 0.147

Saída em ./analysis/:
    ranked_filtered.csv  — todas as métricas filtradas, ordenadas por |δ|
    strong_signals.csv   — |δ| ≥ 0.330 e q < 0.05
    table_human.tex      — tabela LaTeX das strong signals onde humano > LLM
    table_llm.tex        — tabela LaTeX das strong signals onde LLM > humano
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


METRICS_BY_CATEGORY: dict[str, list[str]] = {
    "complexity_descriptive": [
        "paragraphs", "sentences", "words",
        "sentences_per_paragraph", "words_per_sentence", "syllables_per_content_word",
        "sentence_length_max", "sentence_length_min", "sentence_length_standard_deviation",
        "subtitles",
    ],
    "complexity_simplicity": [
        "simple_word_ratio",
        "easy_conjunctions_ratio", "hard_conjunctions_ratio",
        "short_sentence_ratio", "medium_short_sentence_ratio", "medium_long_sentence_ratio",
        "dialog_pronoun_ratio",
    ],
    "complexity_lexical_diversity": [
        "ttr", "content_density",
        "content_word_diversity", "content_word_max", "content_word_min", "content_word_standard_deviation",
        "function_word_diversity",
        "adjective_diversity_ratio", "adverbs_diversity_ratio",
        "noun_diversity", "verb_diversity",
        "pronoun_diversity", "indefinite_pronouns_diversity", "relative_pronouns_diversity_ratio",
        "preposition_diversity", "punctuation_diversity",
    ],
    "complexity_frequency": [
        "cw_freq", "cw_freq_bra", "cw_freq_brwac",
        "freq_bra", "freq_brwac",
        "min_cw_freq", "min_cw_freq_bra", "min_cw_freq_brwac",
        "min_freq_bra", "min_freq_brwac",
    ],
    "complexity_readability": [
        "flesch", "gunning_fox", "dalechall_adapted", "brunet", "honore",
    ],
    "complexity_psycholinguistic": [
        "concretude_mean", "concretude_std",
        "concretude_1_25_ratio", "concretude_25_4_ratio",
        "concretude_4_55_ratio", "concretude_55_7_ratio",
        "familiaridade_mean", "familiaridade_std",
        "familiaridade_1_25_ratio", "familiaridade_25_4_ratio",
        "familiaridade_4_55_ratio", "familiaridade_55_7_ratio",
        "idade_aquisicao_mean", "idade_aquisicao_std",
        "idade_aquisicao_1_25_ratio", "idade_aquisicao_25_4_ratio",
        "idade_aquisicao_4_55_ratio", "idade_aquisicao_55_7_ratio",
        "imageabilidade_mean", "imageabilidade_std",
        "imageabilidade_1_25_ratio", "imageabilidade_25_4_ratio",
        "imageabilidade_4_55_ratio", "imageabilidade_55_7_ratio",
    ],
    "cohesion_referential": [
        "adj_arg_ovl", "arg_ovl", "adj_stem_ovl", "stem_ovl", "adj_cw_ovl",
        "adjacent_refs", "anaphoric_refs",
        "coreference_pronoun_ratio", "demonstrative_pronoun_ratio",
    ],
    "cohesion_semantic_lsa": [
        "cross_entropy",
        "lsa_adj_mean", "lsa_adj_std",
        "lsa_all_mean", "lsa_all_std",
        "lsa_paragraph_mean", "lsa_paragraph_std",
        "lsa_givenness_mean", "lsa_givenness_std",
        "lsa_span_mean", "lsa_span_std",
    ],
    "cohesion_connectives": [
        "conn_ratio",
        "add_pos_conn_ratio", "add_neg_conn_ratio",
        "cau_pos_conn_ratio", "cau_neg_conn_ratio",
        "log_pos_conn_ratio", "log_neg_conn_ratio",
        "tmp_pos_conn_ratio", "tmp_neg_conn_ratio",
        "logic_operators", "and_ratio", "or_ratio", "if_ratio",
        "negation_ratio",
    ],
    "semantics_lexical": [
        "abstract_nouns_ratio",
        "content_words_ambiguity",
        "adjectives_ambiguity", "adverbs_ambiguity",
        "nouns_ambiguity", "verbs_ambiguity",
        "hypernyms_verbs",
        "named_entity_ratio_sentence", "named_entity_ratio_text",
        "positive_words", "negative_words",
    ],
}


MAGNITUDE_THRESHOLDS = [
    (0.474, "large"),
    (0.330, "medium"),
    (0.147, "small"),
]


CATEGORY_LABELS: dict[str, str] = {
    "complexity_descriptive": "Descriptive",
    "complexity_simplicity": "Simplicity",
    "complexity_lexical_diversity": "Lexical div.",
    "complexity_frequency": "Frequency",
    "complexity_readability": "Readability",
    "complexity_psycholinguistic": "Psycholinguistic",
    "cohesion_referential": "Referential cohesion",
    "cohesion_semantic_lsa": "Semantic cohesion (LSA)",
    "cohesion_connectives": "Connectives",
    "semantics_lexical": "Lexical semantics",
}


def magnitude(delta: float) -> str:
    if np.isnan(delta):
        return "n/a"
    a = abs(delta)
    for thr, label in MAGNITUDE_THRESHOLDS:
        if a >= thr:
            return label
    return "negligible"


def cliffs_delta_from_u(u1: float, n_x: int, n_y: int) -> float:
    # scipy returns U1 = #(x>y) + 0.5·#(x==y), so δ = 2·U1/(n_x·n_y) − 1
    # measures P(x>y) − P(x<y) and is equivalent to 2·AUC − 1.
    if n_x == 0 or n_y == 0:
        return np.nan
    return 2.0 * u1 / (n_x * n_y) - 1.0


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


def build_metric_lookup() -> dict[str, str]:
    return {m: cat for cat, metrics in METRICS_BY_CATEGORY.items() for m in metrics}


def _latex_table(df: pd.DataFrame, group: str, caption: str, label: str) -> str:
    sub = df[df["higher_in"] == group].sort_values("abs_delta", ascending=False)
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\small",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Metric & Category & Med.\ human & Med.\ LLM & Cliff's $\delta$ \\",
        r"\midrule",
    ]
    for _, row in sub.iterrows():
        metric = row["metric"].replace("_", r"\_")
        cat = CATEGORY_LABELS.get(row["category"], row["category"])
        med_h = f"{row['median_human']:.4g}"
        med_l = f"{row['median_llm']:.4g}"
        delta = f"{row['cliffs_delta']:+.3f}"
        lines.append(rf"\texttt{{{metric}}} & {cat} & {med_h} & {med_l} & ${delta}$ \\")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines) + "\n"


def rank(human_csv: Path, llm_csv: Path, out_dir: Path) -> None:
    h = pd.read_csv(human_csv)
    l = pd.read_csv(llm_csv)
    print(f"human: {len(h)} texts  |  llm: {len(l)} texts")

    lookup = build_metric_lookup()
    whitelist = list(lookup)
    print(f"whitelist (complexity + cohesion + semantics): {len(whitelist)} metrics")

    missing = [m for m in whitelist if m not in h.columns or m not in l.columns]
    if missing:
        print(f"WARNING: {len(missing)} whitelisted metrics missing from the CSVs: {missing}")

    metrics = [m for m in whitelist if m in h.columns and m in l.columns]
    print(f"metrics analysed: {len(metrics)}")

    rows = []
    for m in metrics:
        a = pd.to_numeric(h[m], errors="coerce").to_numpy()
        b = pd.to_numeric(l[m], errors="coerce").to_numpy()
        a = a[~np.isnan(a)]
        b = b[~np.isnan(b)]
        if len(a) < 2 or len(b) < 2:
            continue

        try:
            result = stats.mannwhitneyu(a, b, alternative="two-sided")
            u1 = float(result.statistic)
            pval = float(result.pvalue)
        except ValueError:
            u1, pval = np.nan, np.nan

        delta = cliffs_delta_from_u(u1, len(a), len(b)) if not np.isnan(u1) else np.nan
        auc = (delta + 1.0) / 2.0 if not np.isnan(delta) else np.nan

        rows.append({
            "metric": m,
            "category": lookup[m],
            "n_human": len(a),
            "n_llm": len(b),
            "median_human": float(np.median(a)),
            "median_llm": float(np.median(b)),
            "mad_human": float(stats.median_abs_deviation(a)),
            "mad_llm": float(stats.median_abs_deviation(b)),
            "cliffs_delta": delta,
            "abs_delta": abs(delta) if not np.isnan(delta) else np.nan,
            "auc": auc,
            "p_value": pval,
            "magnitude": magnitude(delta),
            "higher_in": "human" if (not np.isnan(delta) and delta > 0)
                         else ("llm" if (not np.isnan(delta) and delta < 0) else "tie"),
        })

    df = pd.DataFrame(rows)
    df["q_value_fdr"] = np.nan
    mask = df["p_value"].notna()
    df.loc[mask, "q_value_fdr"] = benjamini_hochberg(df.loc[mask, "p_value"].to_numpy())
    df = df.sort_values("abs_delta", ascending=False).reset_index(drop=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    ranked_path = out_dir / "ranked_filtered.csv"
    df.to_csv(ranked_path, index=False)

    strong = df[df["abs_delta"].ge(0.330) & df["q_value_fdr"].lt(0.05)].reset_index(drop=True)
    strong_path = out_dir / "strong_signals.csv"
    strong.to_csv(strong_path, index=False)

    n_h = int((strong["higher_in"] == "human").sum())
    n_l = int((strong["higher_in"] == "llm").sum())
    human_tex = _latex_table(
        strong, "human",
        caption=(
            rf"Most significant metrics for identifying human-written text "
            rf"(n={n_h}): median higher in human than in LLM, with "
            rf"$|\delta| \geq 0.330$ and FDR $<$ 0.05. Sorted by decreasing $|\delta|$."
        ),
        label="tab:strong_human",
    )
    llm_tex = _latex_table(
        strong, "llm",
        caption=(
            rf"Most significant metrics for identifying LLM-generated text "
            rf"(n={n_l}): median higher in LLM than in human, with "
            rf"$|\delta| \geq 0.330$ and FDR $<$ 0.05. Sorted by decreasing $|\delta|$."
        ),
        label="tab:strong_llm",
    )
    (out_dir / "table_human.tex").write_text(human_tex, encoding="utf-8")
    (out_dir / "table_llm.tex").write_text(llm_tex, encoding="utf-8")

    cols_show = ["metric", "category", "median_human", "median_llm",
                 "cliffs_delta", "magnitude", "q_value_fdr", "higher_in"]
    print(f"\n=== TOP 30 metrics by |Cliff's δ| ===")
    print(df.head(30)[cols_show].to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    print(f"\n=== Distribution by magnitude ===")
    print(df["magnitude"].value_counts().to_string())

    print(f"\n=== Top 3 per category (largest |δ|) ===")
    best_per_cat = (df.sort_values("abs_delta", ascending=False)
                      .groupby("category", sort=False)
                      .head(3))
    print(best_per_cat[cols_show].to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    print(f"\nFiles saved to {out_dir}/")
    print(f"  - ranked_filtered.csv  ({len(df)} metrics)")
    print(f"  - strong_signals.csv   ({len(strong)} metrics, medium+ and FDR<0.05)")
    print(f"  - table_human.tex      ({n_h} rows)")
    print(f"  - table_llm.tex        ({n_l} rows)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--human", type=Path,
                        default=Path(__file__).parent / "results" / "human.csv")
    parser.add_argument("--llm", type=Path,
                        default=Path(__file__).parent / "results" / "llm.csv")
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).parent / "analysis")
    args = parser.parse_args()

    if not args.human.is_file() or not args.llm.is_file():
        print(f"CSV not found: {args.human} / {args.llm}")
        return 1

    rank(args.human, args.llm, args.out)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
