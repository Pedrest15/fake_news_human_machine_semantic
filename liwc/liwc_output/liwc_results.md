# LIWC Analysis Results: Human-Written vs LLM-Generated Fake News

## Corpus

| | Human | LLM | Total |
|---|---|---|---|
| FakeTrue.Br | 1,791 | 1,791 | 3,582 |
| Fake.Br | 3,600 | 3,600 | 7,200 |
| **Total** | **5,391** | **5,391** | **10,782** |

- **Dictionary**: Brazilian Portuguese LIWC2015 (73 categories, 11,823 exact words, 2,634 prefixes)
- **Dictionary coverage**: Human 69.4% | LLM 69.6% (no bias between groups)
- **Mean words per document**: Human 176.7 | LLM 329.3

## Overall Summary

Of the 74 categories analysed (73 LIWC categories + dictionary coverage):

- **66 significant categories** (p<0.05) — 89.2%
- **63 significant categories** (p<0.01) — 85.1%
- 44 more characteristic of human-written texts
- 30 more characteristic of LLM-generated texts

## Most Discriminative Categories

### More frequent in HUMAN texts

| # | Category | Mean Human | Mean LLM | Cohen's d | p-value |
|---|---|---|---|---|---|
| 1 | verb (Verbs) | 10.31% | 8.13% | +0.740 | 9.38e-272 |
| 2 | social (Social) | 7.36% | 5.79% | +0.517 | 7.76e-134 |
| 3 | focuspast (Past Focus) | 2.66% | 1.69% | +0.595 | 1.06e-136 |
| 4 | informal (Informal Language) | 0.77% | 0.24% | +0.595 | 2.67e-145 |
| 5 | hear (Hear) | 0.93% | 0.41% | +0.552 | 2.33e-103 |
| 6 | male (Male) | 1.42% | 0.90% | +0.456 | 5.93e-45 |
| 7 | auxverb (Auxiliary Verbs) | 4.43% | 3.67% | +0.395 | 7.47e-68 |
| 8 | percept (Perceptual Processes) | 2.36% | 1.78% | +0.357 | 1.66e-44 |
| 9 | netspeak (Netspeak) | 0.38% | 0.13% | +0.348 | 1.88e-35 |
| 10 | i (I) | 0.35% | 0.11% | +0.348 | 6.80e-28 |

### More frequent in LLM texts

| # | Category | Mean Human | Mean LLM | Cohen's d | p-value |
|---|---|---|---|---|---|
| 1 | adj (Adjectives) | 2.83% | 4.07% | -0.731 | 0.00e+00 |
| 2 | number (Numbers) | 2.56% | 3.44% | -0.585 | 4.93e-233 |
| 3 | posemo (Positive Emotions) | 1.70% | 2.34% | -0.472 | 1.42e-192 |
| 4 | affect (Affect) | 3.81% | 4.75% | -0.460 | 7.85e-169 |
| 5 | anx (Anx) | 0.27% | 0.53% | -0.449 | 0.00e+00 |
| 6 | insight (Insight) | 1.32% | 1.80% | -0.435 | 1.22e-166 |
| 7 | achieve (Achievement) | 1.11% | 1.46% | -0.342 | 2.22e-130 |
| 8 | cogproc (Cognitive Processes) | 9.30% | 10.35% | -0.325 | 5.41e-76 |
| 9 | drives (Drives) | 9.24% | 10.24% | -0.324 | 4.71e-77 |
| 10 | work (Work) | 3.38% | 4.09% | -0.317 | 2.74e-94 |

## Interpretation

### Linguistic profile of HUMAN texts

Human-written texts stand out for a **narrative, oral and personal** style:

- **More verbs and auxiliaries**: richer verbal structures and more varied conjugations, indicating active narration
- **Past focus**: accounts of events in the past tense, typical of news reporting facts that already happened
- **Social language**: more references to people, relationships and social interaction
- **Informal language**: slang, netspeak, non-fluencies and swearing — hallmarks of spontaneous, informal writing
- **Perceptual processes (hearing)**: verbs such as "disse", "afirmou", "declarou" — quotes attributed to sources
- **First-person pronoun and male references**: a more personal tone with direct references to male public figures

### Linguistic profile of LLM texts

LLM-generated texts show a **descriptive, formal and elaborate** style:

- **More adjectives** (largest effect: d=-0.731): more qualifying, descriptive language
- **More numbers**: more frequent use of figures, statistics and dates
- **Positive emotions and affect**: a more emotional, more positive tone — possibly an attempt at persuasion
- **Cognitive processes and insight**: more analytical language, with more causal connectives and explanations
- **Drives (power, achievement, work)**: vocabulary oriented towards motivation and accomplishment
- **Prepositions and spatiality**: more elaborate clause structures with more prepositional complements
- **Anxiety**: more words associated with worry and fear

## NON-Significant Categories (p>=0.05)

The following categories showed no significant difference between the groups:

| Category | Mean Human | Mean LLM | p-value |
|---|---|---|---|
| dictionary_coverage | 69.41% | 69.57% | 9.79e-02 |
| leisure (Leisure) | 1.07% | 0.84% | 1.31e-01 |
| differ (Differentiation) | 2.67% | 2.53% | 6.16e-01 |
| feel (Feel) | 0.55% | 0.41% | 8.45e-01 |
| affiliation (Affiliation) | 1.85% | 1.71% | 1.47e-01 |
| body (Body) | 0.32% | 0.20% | 9.01e-02 |
| article (Articles) | 15.45% | 15.40% | 5.70e-02 |
| conj (Conjunctions) | 7.74% | 7.72% | 4.64e-01 |

## Effect Size (Cohen's d)

| Classification | Range | Categories |
|---|---|---|
| Large (>=0.8) | - | 0 |
| Medium (0.5-0.8) | verb, adj, focuspast, informal, number, hear, social | 7 |
| Small (0.2-0.5) | posemo, affect, anx, male, insight, auxverb, achieve, percept, netspeak, i, cogproc, drives, work, motion, prep, space, risk, quant, certain, cause, negemo, power, nonflu | 23 |
| Negligible (<0.2) | remaining categories | 44 |

## Output Files

| File | Description |
|---|---|
| `liwc_discriminative_categories.csv` | Every category with the full statistics |
| `liwc_discriminative_categories.txt` | Human-readable report with the complete ranking |
| `liwc_per_document_scores.csv` | Per-document LIWC scores (%), ready for ML |
| `liwc_analysis_summary.txt` | Overall analysis summary |

## Methodology

- **Tokenisation**: regex `[\w]+` over lower-cased text
- **Categorisation**: exact match + prefix match (LIWC dictionary `*` wildcards)
- **Normalisation**: per-category count / total words x 100 (percentage)
- **Statistical test**: Mann-Whitney U (two-sided, non-parametric)
- **Effect size**: Cohen's d with pooled standard deviation
