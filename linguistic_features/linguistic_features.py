#!/usr/bin/env python3
"""
Classificacao Humano vs LLM combinando embeddings BERT e features linguisticas.

Modos de operacao (--feature-mode):
  - linguistic : apenas features pre-computadas (NILCmetrics + LIWC + Enhanced-UD)
  - embedding  : apenas embedding [CLS] do BERT (BERTimbau ou bert-multilingual)
  - combined   : concatenacao [BERT] + [features linguisticas] (modo principal)

Metodologia (identica ao pipeline BERT pareado):
  1. Splits pre-definidos (train_fake_br.txt etc. em linguistic_features/)
  2. Pares humano-LLM por filename, truncados pelo menor numero de tokens do par
  3. GridSearchCV no TREINO para tuning; avaliacao final no TESTE held-out
  4. Cache de embeddings BERT em .npy para nao recomputar

Fontes de features (configuraveis no topo do arquivo):
  - NILCmetrics: nilc-metrix/results/{human,llm}.csv
  - LIWC:        liwc/liwc_output/liwc_per_document_scores.csv
  - Enhanced-UD: ed_rules/rules_output_all/<subset>/*.rules.json (vetor por doc)

Uso:
    python linguistic_features.py --feature-mode combined --classifier all
    python linguistic_features.py --feature-mode linguistic --features nilcmetrics,liwc
    python linguistic_features.py --feature-mode embedding --bert-model bertimbau
"""

import argparse
import copy
import hashlib
import json
import logging
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

if not XGBOOST_AVAILABLE:
    logger.warning("XGBoost nao instalado. Use 'pip install xgboost' para habilitar.")


# =============================================================================
# CONFIGURACAO DE CAMINHOS (paths default no repo)
# =============================================================================

REPO_ROOT = Path(__file__).resolve().parent.parent
CARACT_ROOT = REPO_ROOT / "linguistic_features" / "data"

NILC_HUMAN_CSV_DEFAULT = REPO_ROOT / "nilc-metrix" / "results" / "human.csv"
NILC_LLM_CSV_DEFAULT = REPO_ROOT / "nilc-metrix" / "results" / "llm.csv"
LIWC_CSV_DEFAULT = REPO_ROOT / "liwc" / "liwc_output" / "liwc_per_document_scores.csv"
UD_RULES_DIR_DEFAULT = REPO_ROOT / "ed_rules" / "rules_output_all"

SYLLABLES_CSV_DEFAULT = CARACT_ROOT / "contagem_silabas" / "resultados" / "estatisticas_silabas_nltk.csv"
POS_TAGGER_ROOT_DEFAULT = CARACT_ROOT / "tagger" / "tagger_results"
PARSER_HUMAN_TXT_DEFAULT = CARACT_ROOT / "parser" / "aggregated_output" / "human" / "per_file_statistics.txt"
PARSER_LLM_TXT_DEFAULT = CARACT_ROOT / "parser" / "aggregated_output" / "llm" / "per_file_statistics.txt"
SAGE_CSV_DEFAULT = CARACT_ROOT / "sage" / "resultados" / "sage_termos_Humano_vs_Maquina.csv"

UPOS_TAGS = ['ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN',
             'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X']

SILABAS_DATASET_MAP = {
    'FakeBR_Human': ('fake_br', 0),
    'FakeBR_LLM': ('fake_br', 1),
    'FakeTrue_Human': ('fake_true_br', 0),
    'FakeTrue_LLM': ('fake_true_br', 1),
}

POS_TAGGER_FOLDER_MAP = {
    ('FakeHuman', 'FakeBR'): ('fake_br', 0),
    ('FakeLLM', 'Fake.Br'): ('fake_br', 1),
    ('FakeHuman', 'FakeTrueBR'): ('fake_true_br', 0),
    ('FakeLLM', 'FakeTrueBR'): ('fake_true_br', 1),
}

PARSER_PATH_MAP = {
    'fake_br_human': ('fake_br', 0),
    'fake_br_llm': ('fake_br', 1),
    'fake_true_human': ('fake_true_br', 0),
    'fake_true_llm': ('fake_true_br', 1),
}

CORPUS_DIRS_DEFAULT: Dict[str, Path] = {
    "fake_br_human": CARACT_ROOT / "corpus" / "Fake.br-Corpus-master" / "full_texts" / "fake",
    "fake_br_llm": CARACT_ROOT / "corpus" / "fake-news-llm-ptbr-main" / "fake-news-llm-ptbr-main" / "data" / "Fake.Br",
    "fake_true_human": CARACT_ROOT / "corpus" / "FakeTrue.Br-main" / "fake",
    "fake_true_llm": CARACT_ROOT / "corpus" / "fake-news-llm-ptbr-main" / "fake-news-llm-ptbr-main" / "data" / "FakeTrueBR",
}

BERT_MODELS = {
    "bertimbau": "neuralmind/bert-base-portuguese-cased",
    "bert-multilingual": "bert-base-multilingual-cased",
}


# =============================================================================
# CLASSIFICADORES
# =============================================================================

def get_classifiers_config() -> Dict:
    config = {
        'svm': {
            'estimator': SVC(probability=True, random_state=42),
            'param_grid': {
                'kernel': ['rbf', 'linear', 'poly'],
                'C': [0.1, 1.0, 10.0, 100.0],
                'gamma': ['scale', 'auto']
            }
        },
        'random_forest': {
            'estimator': RandomForestClassifier(random_state=42, n_jobs=-1),
            'param_grid': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        },
        'naive_bayes': {
            'estimator': GaussianNB(),
            'param_grid': {
                'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
            }
        },
        'logistic_regression': {
            'estimator': LogisticRegression(random_state=42, max_iter=1000),
            'param_grid': {
                'C': [0.01, 0.1, 1.0, 10.0, 100.0],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }
        },
        'mlp': {
            'estimator': MLPClassifier(random_state=42, max_iter=500, early_stopping=True),
            'param_grid': {
                'hidden_layer_sizes': [(128,), (256,), (128, 64), (256, 128)],
                'activation': ['relu', 'tanh'],
                'alpha': [0.0001, 0.001, 0.01]
            }
        }
    }

    if XGBOOST_AVAILABLE:
        config['xgboost'] = {
            'estimator': XGBClassifier(random_state=42, eval_metric='logloss', n_jobs=-1),
            'param_grid': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 10],
                'learning_rate': [0.01, 0.1, 0.3],
                'subsample': [0.8, 1.0]
            }
        }

    return config


CLASSIFIERS_CONFIG = get_classifiers_config()


# =============================================================================
# SPLITS
# =============================================================================

def load_split_files(split_dir: Path) -> Dict[str, Dict[str, List[str]]]:
    """
    Carrega train_/test_ {fake_br,fake_true_br}.txt em linguistic_features/.

    Retorna {'fake_br': {'train': [...], 'test': [...]}, 'fake_true_br': {...}}.
    """
    splits = {
        'fake_br': {'train': [], 'test': []},
        'fake_true_br': {'train': [], 'test': []}
    }
    files = {
        ('fake_br', 'train'): split_dir / 'train_fake_br.txt',
        ('fake_br', 'test'): split_dir / 'test_fake_br.txt',
        ('fake_true_br', 'train'): split_dir / 'train_fake_true_br.txt',
        ('fake_true_br', 'test'): split_dir / 'test_fake_true_br.txt',
    }
    for (corpus, kind), path in files.items():
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                splits[corpus][kind] = [line.strip() for line in f if line.strip()]
        else:
            logger.warning(f"Split nao encontrado: {path}")
    return splits


def splits_to_keys(splits: Dict) -> Tuple[List[Tuple[str, str, int]], List[Tuple[str, str, int]]]:
    """
    Converte splits em listas de chaves (filename, subset_corpus, label).

    Cada filename do split gera 2 chaves: uma humana (label=0) e uma LLM (label=1).
    """
    train, test = [], []
    for corpus, parts in splits.items():
        for fn in parts['train']:
            train.append((fn, corpus, 0))
            train.append((fn, corpus, 1))
        for fn in parts['test']:
            test.append((fn, corpus, 0))
            test.append((fn, corpus, 1))
    return train, test


# =============================================================================
# LOADERS DE FEATURES LINGUISTICAS
# =============================================================================

SUBSET_TO_CORPUS = {
    'fake_br_human': 'fake_br', 'fake_br_llm': 'fake_br',
    'fake_true_human': 'fake_true_br', 'fake_true_llm': 'fake_true_br',
}

# Tier A: contagens absolutas do NILC-Metrix.
# Codificam diretamente o tamanho do texto (words, sentences, paragraphs etc.).
NILC_TIER_A_ABSOLUTE = frozenset({
    "words",
    "sentences",
    "paragraphs",
    "subtitles",
    "logic_operators",
    "relative_clauses",
    "subordinate_clauses",
    "infinite_subordinate_clauses",
    "sentences_with_zero_clause",
    "sentences_with_one_clause",
    "sentences_with_two_clauses",
    "sentences_with_three_clauses",
    "sentences_with_four_clauses",
    "sentences_with_five_clauses",
    "sentences_with_six_clauses",
    "sentences_with_seven_more_clauses",
})

# Tier B: features formalmente normalizadas mas empiricamente length-biased.
# Quatro grupos:
#  - Riqueza vocabular (TTR-like, Tweedie & Baayen 1998): ttr, honore, brunet.
#  - Diversidades TTR-like por classe gramatical: *_diversity / *_diversity_ratio.
#  - Max/min/std de contagens por POS e de comprimento de unidades: o MAX
#    cresce com N, o STD escala com sqrt(N) para muitas distribuicoes.
#  - Std de similaridades LSA: variancia amostral depende do n. de pares
#    sentenca-a-sentenca disponivel, que escala com N.
NILC_TIER_B_LENGTH_CORRELATED = frozenset({
    # Riqueza vocabular
    "ttr",
    "honore",
    "brunet",
    # Diversidades TTR-like
    "content_word_diversity",
    "function_word_diversity",
    "noun_diversity",
    "verb_diversity",
    "preposition_diversity",
    "punctuation_diversity",
    "pronoun_diversity",
    "indefinite_pronouns_diversity",
    "adjective_diversity_ratio",
    "adverbs_diversity_ratio",
    "relative_pronouns_diversity_ratio",
    "verbal_time_moods_diversity",
    # max/min/std de contagens por POS
    "adjectives_max", "adjectives_min", "adjectives_standard_deviation",
    "adverbs_max", "adverbs_min", "adverbs_standard_deviation",
    "content_word_max", "content_word_min", "content_word_standard_deviation",
    "nouns_max", "nouns_min", "nouns_standard_deviation",
    "pronouns_max", "pronouns_min", "pronouns_standard_deviation",
    "verbs_max", "verbs_min", "verbs_standard_deviation",
    # max/min/std de comprimento de unidades
    "sentence_length_max",
    "sentence_length_min",
    "sentence_length_standard_deviation",
    "max_noun_phrase",
    "min_noun_phrase",
    "std_noun_phrase",
    # Std das similaridades LSA (mantemos os means: lsa_*_mean)
    "lsa_adj_std",
    "lsa_all_std",
    "lsa_paragraph_std",
    "lsa_givenness_std",
    "lsa_span_std",
})

NILC_LENGTH_BIASED = NILC_TIER_A_ABSOLUTE | NILC_TIER_B_LENGTH_CORRELATED


def load_nilcmetrics(human_csv: Path, llm_csv: Path,
                     drop_absolute: bool = True) -> Optional[pd.DataFrame]:
    """Concatena human/llm CSVs do NILC-Metrix e padroniza colunas.

    Quando drop_absolute=True (default), remove as features length-biased:
    Tier A (contagens absolutas) + Tier B (diversidades TTR-like, riqueza
    vocabular, max/min/std de contagens, std de coseno LSA).
    """
    if not human_csv.exists() or not llm_csv.exists():
        logger.warning(f"NILCmetrics nao encontrado: {human_csv}, {llm_csv}")
        return None

    df_h = pd.read_csv(human_csv)
    df_l = pd.read_csv(llm_csv)
    df_h['label'] = 0
    df_l['label'] = 1
    df = pd.concat([df_h, df_l], ignore_index=True)

    df = df.rename(columns={'file': 'filename'})
    df['subset_corpus'] = df['subset'].map(SUBSET_TO_CORPUS)
    df = df.drop(columns=['subset'])

    if drop_absolute:
        tier_a = [c for c in df.columns if c in NILC_TIER_A_ABSOLUTE]
        tier_b = [c for c in df.columns if c in NILC_TIER_B_LENGTH_CORRELATED]
        to_drop = tier_a + tier_b
        if to_drop:
            df = df.drop(columns=to_drop)
            logger.info(f"  NILCmetrics: removidas {len(tier_a)} features Tier A "
                        f"+ {len(tier_b)} features Tier B = {len(to_drop)} length-biased")

    feature_cols = [c for c in df.columns if c not in {'filename', 'subset_corpus', 'label'}]
    df = df[['filename', 'subset_corpus', 'label'] + feature_cols]
    df = df.add_prefix('nilc__').rename(columns={
        'nilc__filename': 'filename',
        'nilc__subset_corpus': 'subset_corpus',
        'nilc__label': 'label',
    })
    logger.info(f"  NILCmetrics: {len(df)} linhas, {len(df.columns) - 3} features")
    return df


# LIWC expoe 3 campos length-direct (contagens/ratio de metadado, nao
# categorias psicolinguisticas) que devem sair junto com a categoria.
LIWC_ABSOLUTE_FIELDS = frozenset({
    "word_count",
    "matched_words",
    "dictionary_coverage",
})

# LIWC tem hierarquia: categorias agregadoras subsumem subcategorias mais
# especificas. Cada palavra recebe os ids do agregador E dos filhos, criando
# colinearidade entre eles. Removemos os agregadores e mantemos as folhas
# (categorias mais especificas). Lista derivada da estrutura LIWC2015:
#  function   ⊃  pronoun, article, prep, auxverb, adverb, conj, negate
#  pronoun    ⊃  ppron, ipron
#  ppron      ⊃  i, we, you, shehe, they
#  affect     ⊃  posemo, negemo
#  negemo     ⊃  anx, anger, sad
#  social     ⊃  family, friend, female, male
#  cogproc    ⊃  insight, cause, discrep, tentat, certain, differ
#  percept    ⊃  see, hear, feel
#  bio        ⊃  body, health, sexual, ingest
#  drives     ⊃  affiliation, achieve, power, reward, risk
#  relativ    ⊃  motion, space, time
#  informal   ⊃  swear, netspeak, assent, nonflu, filler
LIWC_AGGREGATOR_CATEGORIES = frozenset({
    "function (Function Words)",
    "pronoun (Pronouns)",
    "ppron (Personal Pronouns)",
    "affect (Affect)",
    "negemo (Negative Emotions)",
    "social (Social)",
    "cogproc (Cognitive Processes)",
    "percept (Perceptual Processes)",
    "bio (Biological Processes)",
    "drives (Drives)",
    "relativ (Relativity)",
    "informal (Informal Language)",
})


def load_liwc(liwc_csv: Path, drop_absolute: bool = True) -> Optional[pd.DataFrame]:
    """Carrega liwc_per_document_scores.csv (humano+LLM em um arquivo).

    Quando drop_absolute=True (default), remove:
      - LIWC_ABSOLUTE_FIELDS (word_count, matched_words, dictionary_coverage)
      - LIWC_AGGREGATOR_CATEGORIES (12 categorias agregadoras na hierarquia
        LIWC2015, mantendo apenas as folhas especificas).
    """
    if not liwc_csv.exists():
        logger.warning(f"LIWC nao encontrado: {liwc_csv}")
        return None

    df = pd.read_csv(liwc_csv)
    df = df.rename(columns={'file_name': 'filename'})
    df['label'] = (df['author_type'] == 'llm').astype(int)

    def _subset_from_path(p: str) -> Optional[str]:
        if 'Fake.br-Corpus-master' in p or '/Fake.Br/' in p:
            return 'fake_br'
        if 'FakeTrue.Br-main' in p or '/FakeTrueBR/' in p:
            return 'fake_true_br'
        return None

    df['subset_corpus'] = df['file_path'].map(_subset_from_path)
    df = df.drop(columns=['file_path', 'author_type'])

    df = df.dropna(subset=['subset_corpus'])

    if drop_absolute:
        length_direct = [c for c in df.columns if c in LIWC_ABSOLUTE_FIELDS]
        aggregators = [c for c in df.columns if c in LIWC_AGGREGATOR_CATEGORIES]
        to_drop = length_direct + aggregators
        if to_drop:
            df = df.drop(columns=to_drop)
            logger.info(f"  LIWC: removidos {len(length_direct)} length-direct "
                        f"+ {len(aggregators)} categorias agregadoras = {len(to_drop)}")

    feature_cols = [c for c in df.columns if c not in {'filename', 'subset_corpus', 'label'}]
    df = df[['filename', 'subset_corpus', 'label'] + feature_cols]
    df = df.add_prefix('liwc__').rename(columns={
        'liwc__filename': 'filename',
        'liwc__subset_corpus': 'subset_corpus',
        'liwc__label': 'label',
    })
    logger.info(f"  LIWC: {len(df)} linhas, {len(df.columns) - 3} features")
    return df


def load_enhanced_ud(rules_root: Path, top_k: int = 500) -> Optional[pd.DataFrame]:
    """
    Constroi vetor por doc a partir de rules_output_all/<subset>/<file>.rules.json.

    Restringe ao top-K regras mais frequentes globalmente. Cada feature e a
    frequencia normalizada da regra (count / total_rules) no documento.
    """
    if not rules_root.exists():
        logger.warning(f"Enhanced-UD rules dir nao encontrado: {rules_root}")
        return None

    subsets = ['fake_br_human', 'fake_br_llm', 'fake_true_human', 'fake_true_llm']
    rule_doc_counts: Counter = Counter()
    per_doc: List[Tuple[str, str, int, Counter, int]] = []

    for subset in subsets:
        subset_dir = rules_root / subset
        if not subset_dir.exists():
            logger.warning(f"  subset ausente em rules_output_all: {subset}")
            continue
        label = 1 if subset.endswith('_llm') else 0
        corpus = SUBSET_TO_CORPUS[subset]
        for json_path in subset_dir.glob('*.rules.json'):
            try:
                with open(json_path, 'r', encoding='utf-8') as fh:
                    payload = json.load(fh)
            except Exception as exc:
                logger.warning(f"  ignorando {json_path}: {exc}")
                continue
            rules = payload.get('rules', [])
            total = payload.get('total_rules') or len(rules) or 1
            counts = Counter(rules)
            for rule in counts:
                rule_doc_counts[rule] += 1
            filename = payload.get('source_file', json_path.stem + '.txt')
            if filename.endswith('.conllu'):
                filename = filename[:-len('.conllu')] + '.txt'
            per_doc.append((filename, corpus, label, counts, total))

    if not per_doc:
        logger.warning("  Enhanced-UD: nenhum documento carregado")
        return None

    top_rules = [r for r, _ in rule_doc_counts.most_common(top_k)]
    rule_to_idx = {r: i for i, r in enumerate(top_rules)}

    n_docs = len(per_doc)
    matrix = np.zeros((n_docs, len(top_rules)), dtype=np.float32)
    meta = []
    for i, (filename, corpus, label, counts, total) in enumerate(per_doc):
        for rule, c in counts.items():
            j = rule_to_idx.get(rule)
            if j is not None:
                matrix[i, j] = c / total
        meta.append((filename, corpus, label))

    feature_cols = [f"ud__{r}" for r in top_rules]
    df = pd.DataFrame(matrix, columns=feature_cols)
    meta_df = pd.DataFrame(meta, columns=['filename', 'subset_corpus', 'label'])
    df = pd.concat([meta_df, df], axis=1)
    logger.info(f"  Enhanced-UD: {len(df)} docs, {len(top_rules)} features (top-{top_k} regras)")
    return df


def load_syllables(csv_path: Path) -> Optional[pd.DataFrame]:
    """Le contagem_silabas/.../estatisticas_silabas_nltk.csv e padroniza chaves."""
    if not csv_path.exists():
        logger.warning(f"Silabas CSV nao encontrado: {csv_path}")
        return None

    df = pd.read_csv(csv_path)
    df = df.rename(columns={'arquivo': 'filename'})

    mapping = df['dataset'].map(SILABAS_DATASET_MAP)
    df = df.assign(
        subset_corpus=mapping.map(lambda x: x[0] if isinstance(x, tuple) else None),
        label=mapping.map(lambda x: x[1] if isinstance(x, tuple) else None),
    )
    df = df.dropna(subset=['subset_corpus', 'label'])
    df['label'] = df['label'].astype(int)

    keep = ['media_silabas_por_sentenca', 'media_silabas_por_palavra']
    missing = [c for c in keep if c not in df.columns]
    if missing:
        logger.warning(f"  Silabas: colunas esperadas ausentes: {missing}")
        return None
    df = df[['filename', 'subset_corpus', 'label'] + keep]
    df = df.rename(columns={c: f"syll__{c}" for c in keep})
    logger.info(f"  Silabas: {len(df)} linhas, {len(keep)} features")
    return df


def _pos_counts_from_file(path: Path) -> Dict[str, int]:
    counts: Counter = Counter()
    with open(path, 'r', encoding='utf-8') as fh:
        for line in fh:
            line = line.rstrip()
            if not line or line.startswith('#'):
                continue
            parts = line.split('\t')
            if len(parts) >= 3 and parts[0].isdigit():
                counts[parts[2]] += 1
    return counts


def load_pos_tagger(root: Path) -> Optional[pd.DataFrame]:
    """Le tagger_results/*/* e produz frequencia normalizada por UPOS (vetor por doc)."""
    if not root.exists():
        logger.warning(f"POS-tagger root nao encontrado: {root}")
        return None

    rows = []
    for (top, mid), (corpus, label) in POS_TAGGER_FOLDER_MAP.items():
        folder = root / top / mid
        if not folder.exists():
            logger.warning(f"  POS-tagger ausente: {folder}")
            continue
        files = list(folder.rglob('*.txt'))
        for path in files:
            counts = _pos_counts_from_file(path)
            total = sum(counts.values())
            if total == 0:
                continue
            row = {'filename': path.name, 'subset_corpus': corpus, 'label': label}
            for tag in UPOS_TAGS:
                row[f"pos__{tag}"] = counts.get(tag, 0) / total
            rows.append(row)
        logger.info(f"  POS-tagger {top}/{mid}: {len(files)} arquivos")

    if not rows:
        return None
    df = pd.DataFrame(rows)
    logger.info(f"  POS-tagger total: {len(df)} docs, {len(UPOS_TAGS)} features")
    return df


_PARSER_BLOCK_RE = re.compile(
    r"\[\d+\]\s+(?P<file>\S+?)\.conllu\s*\n"
    r"-{3,}\s*\n"
    r"\s*Caminho:\s*(?P<subset>fake_[a-z_]+)\\[^\n]*\n"
    r"\s*Sentenças:\s*(?P<sentences>\d+)\s*\n"
    r"\s*Regras totais:\s*(?P<total>\d+)\s*\n"
    r"\s*Regras únicas:\s*(?P<unique>\d+)",
    re.UNICODE,
)


def _parse_parser_stats_file(path: Path) -> List[Dict]:
    if not path.exists():
        logger.warning(f"  parser stats ausente: {path}")
        return []
    text = path.read_text(encoding='utf-8', errors='replace')
    rows = []
    for m in _PARSER_BLOCK_RE.finditer(text):
        subset_raw = m.group('subset')
        mapped = PARSER_PATH_MAP.get(subset_raw)
        if mapped is None:
            continue
        corpus, label = mapped
        rows.append({
            'filename': f"{m.group('file')}.txt",
            'subset_corpus': corpus,
            'label': label,
            'psr__sentences': int(m.group('sentences')),
            'psr__total_rules': int(m.group('total')),
            'psr__unique_rules': int(m.group('unique')),
        })
    return rows


def load_parser_stats(human_txt: Path, llm_txt: Path) -> Optional[pd.DataFrame]:
    """Le parser/.../per_file_statistics.txt (humano e LLM) e extrai 3 stats por doc."""
    rows = _parse_parser_stats_file(human_txt) + _parse_parser_stats_file(llm_txt)
    if not rows:
        return None
    df = pd.DataFrame(rows)
    logger.info(f"  Parser stats: {len(df)} docs, 3 features")
    return df


_TOKEN_RE = re.compile(r"[a-záàâãéèêíïóôõöúçñ0-9]+", re.IGNORECASE | re.UNICODE)


def _tokens(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


def _count_bigrams_in_text(text: str, bigrams: List[Tuple[str, str]]) -> Dict[Tuple[str, str], int]:
    toks = _tokens(text)
    if len(toks) < 2:
        return {b: 0 for b in bigrams}
    pair_counts: Counter = Counter(zip(toks, toks[1:]))
    return {b: pair_counts.get(b, 0) for b in bigrams}


def load_sage_terms(csv_path: Path, corpus_dirs: Dict[str, Path]) -> Optional[pd.DataFrame]:
    """Conta ocorrencias dos bigramas distintivos do SAGE em cada doc do corpus."""
    if not csv_path.exists():
        logger.warning(f"SAGE CSV nao encontrado: {csv_path}")
        return None

    terms_df = pd.read_csv(csv_path, encoding='utf-8-sig')
    if 'termo' not in terms_df.columns:
        logger.warning(f"SAGE CSV sem coluna 'termo': {csv_path}")
        return None
    terms_raw = terms_df['termo'].astype(str).str.lower().str.strip().tolist()
    bigrams: List[Tuple[str, str]] = []
    seen = set()
    for term in terms_raw:
        toks = term.split()
        if len(toks) != 2:
            continue
        key = (toks[0], toks[1])
        if key in seen:
            continue
        seen.add(key)
        bigrams.append(key)
    if not bigrams:
        logger.warning("SAGE: nenhum bigrama valido encontrado")
        return None

    feature_names = [f"sage__{a}_{b}" for (a, b) in bigrams]

    rows = []
    for subset_name, dirpath in corpus_dirs.items():
        if not dirpath.exists():
            logger.warning(f"  SAGE: corpus dir ausente: {dirpath}")
            continue
        corpus = SUBSET_TO_CORPUS[subset_name]
        label = 1 if subset_name.endswith('_llm') else 0
        files = list(dirpath.glob('*.txt'))
        for path in files:
            try:
                text = path.read_text(encoding='utf-8', errors='replace')
            except Exception as exc:
                logger.warning(f"    erro lendo {path}: {exc}")
                continue
            counts = _count_bigrams_in_text(text, bigrams)
            row = {'filename': path.name, 'subset_corpus': corpus, 'label': label}
            for (a, b), name in zip(bigrams, feature_names):
                row[name] = counts[(a, b)]
            rows.append(row)
        logger.info(f"  SAGE scan {subset_name}: {len(files)} arquivos")

    if not rows:
        return None
    df = pd.DataFrame(rows)
    logger.info(f"  SAGE: {len(df)} docs, {len(feature_names)} features")
    return df


def merge_feature_frames(frames: List[pd.DataFrame]) -> pd.DataFrame:
    """Inner join sequencial pelas chaves (filename, subset_corpus, label)."""
    if not frames:
        raise ValueError("Nenhuma fonte de features fornecida")
    base = frames[0]
    for nxt in frames[1:]:
        before = len(base)
        base = base.merge(nxt, on=['filename', 'subset_corpus', 'label'], how='inner')
        logger.info(f"  Merge: {before} -> {len(base)} (apos juntar {nxt.shape[1] - 3} colunas)")
    return base


# =============================================================================
# BERT EMBEDDINGS (com cache)
# =============================================================================

class BERTEmbeddingExtractor:
    """Extrai embedding [CLS] do BERT com pesos congelados."""

    def __init__(self, model_name: str, device: Optional[str] = None):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name

        logger.info(f"  Carregando BERT '{model_name}' em {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def extract(self, texts: List[str], max_length: int = 512, batch_size: int = 16) -> np.ndarray:
        torch = self.torch
        out = []
        n_batches = (len(texts) + batch_size - 1) // batch_size
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = self.tokenizer(
                batch, padding=True, truncation=True,
                max_length=max_length, return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                cls = outputs.last_hidden_state[:, 0, :]
                out.append(cls.cpu().numpy())
            bn = i // batch_size + 1
            if bn % 20 == 0:
                logger.info(f"    Batch {bn}/{n_batches}")
        return np.vstack(out)


def truncate_pair_min_tokens(human: str, llm: str, tokenizer, max_length: int = 512) -> Tuple[str, str, int]:
    """Trunca um par humano/LLM pelo menor numero de tokens. Retorna (h_trunc, l_trunc, min_len)."""
    h_tokens = tokenizer.encode(human, add_special_tokens=False)
    l_tokens = tokenizer.encode(llm, add_special_tokens=False)
    min_len = min(len(h_tokens), len(l_tokens), max_length - 2)
    h_trunc = tokenizer.decode(h_tokens[:min_len], skip_special_tokens=True)
    l_trunc = tokenizer.decode(l_tokens[:min_len], skip_special_tokens=True)
    return h_trunc, l_trunc, min_len


def load_paired_texts(splits: Dict, corpus_dirs: Dict[str, Path]) -> Dict[str, List[Tuple[str, str, str, str]]]:
    """
    Para cada split, carrega tuplas (filename, subset_corpus, human_text, llm_text).

    Pula filenames cujos pares nao existem em disco.
    """
    result = {'train': [], 'test': []}
    for corpus, parts in splits.items():
        h_dir = corpus_dirs.get(f"{corpus}_human")
        if corpus == 'fake_true_br':
            l_dir = corpus_dirs.get('fake_true_llm')
        else:
            l_dir = corpus_dirs.get(f"{corpus}_llm")
        if h_dir is None or l_dir is None or not h_dir.exists() or not l_dir.exists():
            logger.warning(f"  Corpus dirs ausentes para {corpus}: {h_dir}, {l_dir}")
            continue
        for kind in ('train', 'test'):
            loaded, missing = 0, 0
            for fn in parts[kind]:
                hp, lp = h_dir / fn, l_dir / fn
                if not hp.exists() or not lp.exists():
                    missing += 1
                    continue
                try:
                    h = hp.read_text(encoding='utf-8').strip()
                    l = lp.read_text(encoding='utf-8').strip()
                except Exception as exc:
                    logger.warning(f"    erro lendo {fn} em {corpus}: {exc}")
                    missing += 1
                    continue
                if h and l:
                    result[kind].append((fn, corpus, h, l))
                    loaded += 1
                else:
                    missing += 1
            logger.info(f"  {corpus} {kind}: {loaded} pares carregados, {missing} ausentes")
    return result


def get_or_compute_embeddings(
    keys: List[Tuple[str, str, int]],
    texts: List[str],
    model_name: str,
    cache_dir: Path,
    max_length: int = 512,
    batch_size: int = 16,
) -> np.ndarray:
    """
    Calcula ou recupera embeddings BERT. Cache key inclui modelo + hash das chaves + max_length.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    sig = hashlib.sha1()
    sig.update(model_name.encode())
    sig.update(str(max_length).encode())
    for k in keys:
        sig.update(f"{k[0]}|{k[1]}|{k[2]}".encode())
    digest = sig.hexdigest()[:16]
    model_slug = re.sub(r'[^a-zA-Z0-9_-]', '_', model_name)
    cache_path = cache_dir / f"{model_slug}__{digest}.npy"

    if cache_path.exists():
        logger.info(f"  Cache hit: {cache_path.name}")
        arr = np.load(cache_path)
        if arr.shape[0] == len(texts):
            return arr
        logger.warning(f"  Cache invalido (shape mismatch), recomputando...")

    extractor = BERTEmbeddingExtractor(model_name)
    arr = extractor.extract(texts, max_length=max_length, batch_size=batch_size)
    np.save(cache_path, arr)
    logger.info(f"  Embeddings salvos em {cache_path.name} ({arr.shape})")
    return arr


# =============================================================================
# TREINAMENTO / AVALIACAO (genericos)
# =============================================================================

def count_grid_combinations(param_grid: Dict) -> int:
    n = 1
    for v in param_grid.values():
        n *= len(v)
    return n


def train_with_grid_search(clf_type: str, X, y, cv=5, scoring='f1_weighted', n_jobs=-1, groups=None):
    config = CLASSIFIERS_CONFIG[clf_type]
    n_comb = count_grid_combinations(config['param_grid'])
    if groups is not None:
        n_groups = len(set(groups))
        n_splits = min(cv, n_groups)
        cv_obj = GroupKFold(n_splits=n_splits)
        logger.info(f"  Grid Search {clf_type}: {n_comb} combinacoes, GroupKFold n_splits={n_splits} ({n_groups} grupos)")
    else:
        cv_obj = cv
        logger.info(f"  Grid Search {clf_type}: {n_comb} combinacoes, cv={cv}")
    gs = GridSearchCV(
        estimator=copy.deepcopy(config['estimator']),
        param_grid=config['param_grid'],
        cv=cv_obj, scoring=scoring, n_jobs=n_jobs, refit=True
    )
    gs.fit(X, y, groups=groups) if groups is not None else gs.fit(X, y)
    logger.info(f"  Melhores params: {gs.best_params_}")
    logger.info(f"  CV score: {gs.best_score_:.4f}")
    cv_results = {
        'best_score': float(gs.best_score_),
        'best_params': gs.best_params_,
        'cv_scores': gs.cv_results_['mean_test_score'].tolist()
    }
    return gs.best_estimator_, gs.best_params_, cv_results


def evaluate(clf, X, y):
    y_pred = clf.predict(X)
    metrics = {
        'accuracy': float(accuracy_score(y, y_pred)),
        'f1': float(f1_score(y, y_pred, average='weighted')),
        'precision': float(precision_score(y, y_pred, average='weighted')),
        'recall': float(recall_score(y, y_pred, average='weighted')),
    }
    return metrics, confusion_matrix(y, y_pred), y_pred


def save_confusion_matrix(cm: np.ndarray, output_path: Path, title: str):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Humano', 'LLM'], yticklabels=['Humano', 'LLM'])
    plt.xlabel('Predito'); plt.ylabel('Real'); plt.title(title)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150); plt.close()
    logger.info(f"  CM salva em {output_path}")


def save_misclassified(keys, y_true, y_pred, output_path: Path, experiment: str, probs=None) -> Dict:
    label_names = {0: 'Humano', 1: 'LLM'}
    errors = []
    for i, (key, t, p) in enumerate(zip(keys, y_true, y_pred)):
        if t == p:
            continue
        entry = {
            'index': i,
            'filename': key[0],
            'subset_corpus': key[1],
            'true_label': int(t),
            'true_label_name': label_names[int(t)],
            'predicted_label': int(p),
            'predicted_label_name': label_names[int(p)],
            'error_type': 'Falso Positivo' if p == 1 else 'Falso Negativo',
        }
        if probs is not None:
            entry['confidence'] = float(probs[i][p])
            entry['prob_human'] = float(probs[i][0])
            entry['prob_llm'] = float(probs[i][1])
        errors.append(entry)
    result = {
        'experiment': experiment,
        'total_samples': len(keys),
        'total_errors': len(errors),
        'false_positives': sum(1 for e in errors if e['error_type'] == 'Falso Positivo'),
        'false_negatives': sum(1 for e in errors if e['error_type'] == 'Falso Negativo'),
        'error_rate': len(errors) / len(keys) if keys else 0,
        'misclassified': errors,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)
    logger.info(f"  Erros salvos em {output_path} (n={len(errors)})")
    return result


def analyze_feature_importance(clf, feature_names: List[str], top_k: int = 20):
    if hasattr(clf, 'feature_importances_'):
        imp = clf.feature_importances_
    elif hasattr(clf, 'coef_'):
        coef = clf.coef_
        imp = np.abs(coef[0]) if coef.ndim > 1 else np.abs(coef)
    else:
        return None
    if len(imp) != len(feature_names):
        return None
    return sorted(zip(feature_names, imp), key=lambda x: x[1], reverse=True)[:top_k]


# =============================================================================
# PIPELINE
# =============================================================================

class LinguisticFeaturesPipeline:
    """Pipeline com 3 modos: linguistic, embedding, combined."""

    def __init__(
        self,
        exp_dir: Path,
        feature_mode: str = 'combined',
        feature_groups: Optional[List[str]] = None,
        bert_model: str = 'bertimbau',
        corpus_dirs: Optional[Dict[str, Path]] = None,
        nilc_human_csv: Path = NILC_HUMAN_CSV_DEFAULT,
        nilc_llm_csv: Path = NILC_LLM_CSV_DEFAULT,
        liwc_csv: Path = LIWC_CSV_DEFAULT,
        ud_rules_dir: Path = UD_RULES_DIR_DEFAULT,
        ud_top_k: int = 500,
        syllables_csv: Path = SYLLABLES_CSV_DEFAULT,
        pos_tagger_root: Path = POS_TAGGER_ROOT_DEFAULT,
        parser_human_txt: Path = PARSER_HUMAN_TXT_DEFAULT,
        parser_llm_txt: Path = PARSER_LLM_TXT_DEFAULT,
        sage_csv: Path = SAGE_CSV_DEFAULT,
        cv_folds: int = 5,
        cache_dir: Optional[Path] = None,
        drop_nilc_absolute: bool = True,
        experiment_label: Optional[str] = None,
        results_root: Optional[Path] = None,
    ):
        self.exp_dir = exp_dir
        self.feature_mode = feature_mode
        # Rotulo usado nos nomes de arquivo/experimento. Por padrao e o
        # feature_mode (combined/linguistic/embedding); a classificacao por
        # modulo sobrescreve com "module_<modulo>" para nao colidir os outputs.
        self.experiment_label = experiment_label or feature_mode
        self.results_root = results_root
        self.feature_groups = feature_groups or [
            'nilcmetrics', 'liwc', 'enhanced_ud',
            'syllables', 'pos_tagger', 'parser_stats', 'sage_terms',
        ]
        self.bert_model_key = bert_model
        self.bert_model_name = BERT_MODELS[bert_model]
        self.corpus_dirs = corpus_dirs or CORPUS_DIRS_DEFAULT
        self.nilc_human_csv = nilc_human_csv
        self.nilc_llm_csv = nilc_llm_csv
        self.liwc_csv = liwc_csv
        self.ud_rules_dir = ud_rules_dir
        self.ud_top_k = ud_top_k
        self.syllables_csv = syllables_csv
        self.drop_nilc_absolute = drop_nilc_absolute
        self.pos_tagger_root = pos_tagger_root
        self.parser_human_txt = parser_human_txt
        self.parser_llm_txt = parser_llm_txt
        self.sage_csv = sage_csv
        self.cv_folds = cv_folds
        self.cache_dir = cache_dir or (exp_dir / "cache")

        logger.info("Pipeline:")
        logger.info(f"  exp_dir={exp_dir}")
        logger.info(f"  feature_mode={feature_mode}")
        if feature_mode in ('linguistic', 'combined'):
            logger.info(f"  feature_groups={self.feature_groups}")
        if feature_mode in ('embedding', 'combined'):
            logger.info(f"  bert_model={bert_model} ({self.bert_model_name})")
        logger.info(f"  cv_folds={cv_folds}")

    # ---- Carregamento de features linguisticas ----

    def _load_linguistic_frame(self) -> Optional[pd.DataFrame]:
        frames = []
        if 'nilcmetrics' in self.feature_groups:
            df = load_nilcmetrics(self.nilc_human_csv, self.nilc_llm_csv,
                                  drop_absolute=self.drop_nilc_absolute)
            if df is not None:
                frames.append(df)
        if 'liwc' in self.feature_groups:
            df = load_liwc(self.liwc_csv)
            if df is not None:
                frames.append(df)
        if 'enhanced_ud' in self.feature_groups:
            df = load_enhanced_ud(self.ud_rules_dir, top_k=self.ud_top_k)
            if df is not None:
                frames.append(df)
        if 'syllables' in self.feature_groups:
            df = load_syllables(self.syllables_csv)
            if df is not None:
                frames.append(df)
        if 'pos_tagger' in self.feature_groups:
            df = load_pos_tagger(self.pos_tagger_root)
            if df is not None:
                frames.append(df)
        if 'parser_stats' in self.feature_groups:
            df = load_parser_stats(self.parser_human_txt, self.parser_llm_txt)
            if df is not None:
                frames.append(df)
        if 'sage_terms' in self.feature_groups:
            df = load_sage_terms(self.sage_csv, self.corpus_dirs)
            if df is not None:
                frames.append(df)
        if not frames:
            return None
        return merge_feature_frames(frames)

    # ---- Montagem das matrizes de treino/teste ----

    def assemble_matrices(self, splits: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                                       List[Tuple[str, str, int]],
                                                       List[Tuple[str, str, int]],
                                                       List[str]]:
        """
        Retorna X_train, y_train, X_test, y_test, train_keys, test_keys, feature_names.

        Cada key e (filename, subset_corpus, label). Linhas estao alinhadas com keys.
        """
        train_keys_all, test_keys_all = splits_to_keys(splits)
        logger.info(f"  Splits: {len(train_keys_all)} train, {len(test_keys_all)} test (instancias)")

        feat_df = None
        feat_cols: List[str] = []
        if self.feature_mode in ('linguistic', 'combined'):
            logger.info("  Carregando features linguisticas...")
            feat_df = self._load_linguistic_frame()
            if feat_df is None:
                raise RuntimeError("Nenhuma feature linguistica carregada")
            feat_df = feat_df.set_index(['filename', 'subset_corpus', 'label'])
            feat_cols = list(feat_df.columns)

        emb_train = emb_test = None
        emb_keys_train: List[Tuple[str, str, int]] = []
        emb_keys_test: List[Tuple[str, str, int]] = []
        if self.feature_mode in ('embedding', 'combined'):
            emb_train, emb_test, emb_keys_train, emb_keys_test = self._build_embedding_matrices(splits)

        def _select(keys_wanted: List[Tuple[str, str, int]]):
            xs, ys, kept_keys = [], [], []
            for key in keys_wanted:
                blocks: List[np.ndarray] = []
                if feat_df is not None:
                    if key not in feat_df.index:
                        continue
                    blocks.append(feat_df.loc[key].values.astype(np.float64))
                xs.append(blocks)
                ys.append(key[2])
                kept_keys.append(key)
            return xs, ys, kept_keys

        train_xs, train_ys, train_keys = _select(train_keys_all)
        test_xs, test_ys, test_keys = _select(test_keys_all)

        if self.feature_mode in ('embedding', 'combined'):
            emb_train_idx = {k: i for i, k in enumerate(emb_keys_train)}
            emb_test_idx = {k: i for i, k in enumerate(emb_keys_test)}

            def _join_emb(xs, ys, kept, idx_map, mat):
                out_x, out_y, out_keys = [], [], []
                for x, y, k in zip(xs, ys, kept):
                    if k not in idx_map:
                        continue
                    emb_vec = mat[idx_map[k]].astype(np.float64)
                    if self.feature_mode == 'embedding':
                        out_x.append(emb_vec)
                    else:
                        out_x.append(np.concatenate(x + [emb_vec]))
                    out_y.append(y); out_keys.append(k)
                return out_x, out_y, out_keys

            train_xs, train_ys, train_keys = _join_emb(train_xs, train_ys, train_keys, emb_train_idx, emb_train)
            test_xs, test_ys, test_keys = _join_emb(test_xs, test_ys, test_keys, emb_test_idx, emb_test)
        else:
            train_xs = [np.concatenate(x) for x in train_xs]
            test_xs = [np.concatenate(x) for x in test_xs]

        X_train = np.vstack(train_xs)
        X_test = np.vstack(test_xs)
        y_train = np.asarray(train_ys)
        y_test = np.asarray(test_ys)

        feature_names: List[str] = []
        if self.feature_mode in ('linguistic', 'combined'):
            feature_names.extend(feat_cols)
        if self.feature_mode in ('embedding', 'combined'):
            dim = emb_train.shape[1]
            feature_names.extend([f"bert_{self.bert_model_key}__{i}" for i in range(dim)])

        logger.info(f"  Matriz final: X_train={X_train.shape}, X_test={X_test.shape}")
        logger.info(f"  Feature names: {len(feature_names)} (esperado {X_train.shape[1]})")
        return X_train, y_train, X_test, y_test, train_keys, test_keys, feature_names

    # ---- Embeddings ----

    def _build_embedding_matrices(self, splits: Dict):
        """Carrega textos pareados, trunca, e extrai embeddings (com cache)."""
        from transformers import AutoTokenizer

        logger.info("  Carregando textos pareados do corpus...")
        paired = load_paired_texts(splits, self.corpus_dirs)
        logger.info(f"  Pares: train={len(paired['train'])}, test={len(paired['test'])}")
        if not paired['train'] or not paired['test']:
            raise RuntimeError("Sem pares de treino/teste — verifique splits e corpus_dirs")

        tok = AutoTokenizer.from_pretrained(self.bert_model_name)

        def _expand_and_truncate(pairs):
            keys, texts = [], []
            for fn, corpus, h, l in pairs:
                h_t, l_t, _ = truncate_pair_min_tokens(h, l, tok, max_length=512)
                keys.append((fn, corpus, 0)); texts.append(h_t)
                keys.append((fn, corpus, 1)); texts.append(l_t)
            return keys, texts

        train_keys, train_texts = _expand_and_truncate(paired['train'])
        test_keys, test_texts = _expand_and_truncate(paired['test'])
        logger.info(f"  Apos truncagem pareada: {len(train_texts)} train texts, {len(test_texts)} test texts")

        emb_train = get_or_compute_embeddings(
            train_keys, train_texts, self.bert_model_name, self.cache_dir
        )
        emb_test = get_or_compute_embeddings(
            test_keys, test_texts, self.bert_model_name, self.cache_dir
        )
        return emb_train, emb_test, train_keys, test_keys

    # ---- Pipeline principal ----

    def run(
        self,
        classifier_type: str,
        use_grid_search: bool = True,
        normalize_features: bool = True,
        top_k_features: Optional[int] = None,
    ) -> Dict:
        logger.info("=" * 70)
        logger.info(f"EXECUCAO: mode={self.feature_mode} | classifier={classifier_type}")
        logger.info("=" * 70)

        splits = load_split_files(self.exp_dir)
        for c, parts in splits.items():
            logger.info(f"  {c}: {len(parts['train'])} train fn, {len(parts['test'])} test fn")

        X_train, y_train, X_test, y_test, train_keys, test_keys, feature_names = self.assemble_matrices(splits)

        if normalize_features:
            logger.info("  Aplicando StandardScaler")
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        selected_features = feature_names
        if top_k_features and top_k_features < X_train.shape[1]:
            logger.info(f"  SelectKBest top-{top_k_features}")
            selector = SelectKBest(f_classif, k=top_k_features)
            X_train = selector.fit_transform(X_train, y_train)
            X_test = selector.transform(X_test)
            selected_features = [feature_names[i] for i in selector.get_support(indices=True)]

        if use_grid_search:
            train_groups = [f"{k[0]}__{k[1]}" for k in train_keys]
            best_clf, best_params, cv_results = train_with_grid_search(
                classifier_type, X_train, y_train, cv=self.cv_folds, groups=train_groups
            )
            cv_score = cv_results['best_score']
        else:
            best_clf = copy.deepcopy(CLASSIFIERS_CONFIG[classifier_type]['estimator'])
            best_clf.fit(X_train, y_train)
            best_params, cv_score = {}, None

        test_metrics, cm, y_pred = evaluate(best_clf, X_test, y_test)
        tn, fp, fn, tp = cm.ravel()
        logger.info(f"  Test F1={test_metrics['f1']:.4f} Acc={test_metrics['accuracy']:.4f}")
        logger.info(f"  CM: TN={tn} FP={fp} FN={fn} TP={tp}")

        importance = analyze_feature_importance(best_clf, selected_features, top_k=20)

        results = {
            'experiment': self.experiment_label,
            'classifier': classifier_type,
            'feature_mode': self.feature_mode,
            'feature_groups': self.feature_groups if self.feature_mode != 'embedding' else None,
            'bert_model': self.bert_model_name if self.feature_mode != 'linguistic' else None,
            'n_features': len(selected_features),
            'normalized': normalize_features,
            'top_k_selection': top_k_features,
            'data_split': {'train_samples': int(len(y_train)), 'test_samples': int(len(y_test))},
            'grid_search': {
                'enabled': use_grid_search,
                'cv_folds': self.cv_folds if use_grid_search else None,
                'best_params': best_params,
                'cv_score': cv_score,
            },
            'test_results': test_metrics,
            'confusion_matrix': {
                'matrix': cm.tolist(), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)
            },
            'feature_importance': [[f, float(v)] for f, v in importance] if importance else None,
        }

        results_dir = self.results_root or (self.exp_dir / "results")
        results_dir.mkdir(parents=True, exist_ok=True)
        tag = f"{self.experiment_label}_{classifier_type}"
        out_json = results_dir / f"{tag}.json"
        with open(out_json, 'w', encoding='utf-8') as fh:
            json.dump(results, fh, indent=2, ensure_ascii=False)
        logger.info(f"  Resultados salvos em {out_json}")

        cm_path = results_dir / "confusion_matrices" / f"{tag}_cm.png"
        save_confusion_matrix(cm, cm_path, f"{self.experiment_label.upper()} + {classifier_type.upper()}")

        probs = None
        if hasattr(best_clf, 'predict_proba'):
            try:
                probs = best_clf.predict_proba(X_test)
            except Exception:
                pass
        save_misclassified(
            test_keys, y_test, y_pred,
            results_dir / "misclassified" / f"{tag}_errors.json",
            experiment=tag, probs=probs,
        )

        return results


# =============================================================================
# COMPARACAO
# =============================================================================

def print_comparison_table(results: Dict, title: str = ""):
    logger.info("\n" + "=" * 85)
    logger.info(f"COMPARACAO {title}".strip())
    logger.info("=" * 85)
    logger.info(f"{'Classifier':<25} {'CV Score':<12} {'Test F1':<12} {'Test Acc':<12}")
    logger.info("-" * 85)
    for name, r in results.items():
        cv = r.get('grid_search', {}).get('cv_score')
        f1 = r.get('test_results', {}).get('f1')
        acc = r.get('test_results', {}).get('accuracy')
        cv_s = f"{cv:.4f}" if isinstance(cv, float) else "N/A"
        f1_s = f"{f1:.4f}" if isinstance(f1, float) else "N/A"
        acc_s = f"{acc:.4f}" if isinstance(acc, float) else "N/A"
        logger.info(f"{name:<25} {cv_s:<12} {f1_s:<12} {acc_s:<12}")
    logger.info("=" * 85)


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description='Linguistic features + BERT embeddings para deteccao humano vs LLM',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  python linguistic_features.py --feature-mode combined --classifier all
  python linguistic_features.py --feature-mode linguistic --features nilcmetrics,liwc
  python linguistic_features.py --feature-mode embedding --bert-model bertimbau --classifier svm
        """,
    )
    p.add_argument('--feature-mode', choices=['linguistic', 'embedding', 'combined'],
                   default='combined', help='Tipo de vetor de entrada (default: combined)')
    p.add_argument('--classifier', default='all',
                   choices=list(CLASSIFIERS_CONFIG.keys()) + ['all'])
    p.add_argument('--features', default='all',
                   help=('Grupos linguisticos: nilcmetrics,liwc,enhanced_ud,'
                         'syllables,pos_tagger,parser_stats,sage_terms ou all'))
    p.add_argument('--bert-model', default='bertimbau', choices=list(BERT_MODELS.keys()))
    p.add_argument('--corpus-root', type=str, default=None,
                   help='Override CARACT_ROOT (default: ../noticias_falsas_humano_maquina_caracterizacao)')
    p.add_argument('--ud-top-k', type=int, default=500,
                   help='Top-K regras Enhanced-UD mais frequentes (default: 500)')
    p.add_argument('--top-k', type=int, default=None,
                   help='SelectKBest sobre o vetor final')
    p.add_argument('--no-normalize', action='store_true')
    p.add_argument('--no-grid-search', action='store_true')
    p.add_argument('--cv-folds', type=int, default=5)
    p.add_argument('--keep-nilc-absolute', action='store_true',
                   help='Desativa o blacklist Tier A (mantem words/sentences/paragraphs/'
                        'subtitles/logic_operators/*_clauses no NILC). Use para ablacao.')
    return p.parse_args()


def main():
    args = parse_args()

    if args.features == 'all':
        feature_groups = [
            'nilcmetrics', 'liwc', 'enhanced_ud',
            'syllables', 'pos_tagger', 'parser_stats', 'sage_terms',
        ]
    else:
        feature_groups = [f.strip() for f in args.features.split(',')]

    corpus_dirs = CORPUS_DIRS_DEFAULT
    if args.corpus_root:
        croot = Path(args.corpus_root)
        corpus_dirs = {
            "fake_br_human": croot / "corpus" / "Fake.br-Corpus-master" / "full_texts" / "fake",
            "fake_br_llm": croot / "corpus" / "fake-news-llm-ptbr-main" / "fake-news-llm-ptbr-main" / "data" / "Fake.Br",
            "fake_true_human": croot / "corpus" / "FakeTrue.Br-main" / "fake",
            "fake_true_llm": croot / "corpus" / "fake-news-llm-ptbr-main" / "fake-news-llm-ptbr-main" / "data" / "FakeTrueBR",
        }

    script_dir = Path(__file__).resolve().parent
    pipeline = LinguisticFeaturesPipeline(
        exp_dir=script_dir,
        feature_mode=args.feature_mode,
        feature_groups=feature_groups,
        bert_model=args.bert_model,
        corpus_dirs=corpus_dirs,
        ud_top_k=args.ud_top_k,
        cv_folds=args.cv_folds,
        drop_nilc_absolute=not args.keep_nilc_absolute,
    )

    classifiers = list(CLASSIFIERS_CONFIG.keys()) if args.classifier == 'all' else [args.classifier]
    all_results = {}
    for clf in classifiers:
        all_results[clf] = pipeline.run(
            classifier_type=clf,
            use_grid_search=not args.no_grid_search,
            normalize_features=not args.no_normalize,
            top_k_features=args.top_k,
        )

    if len(classifiers) > 1:
        comp = {
            name: {
                'cv_score': r.get('grid_search', {}).get('cv_score'),
                'test_f1': r.get('test_results', {}).get('f1'),
                'test_accuracy': r.get('test_results', {}).get('accuracy'),
                'test_precision': r.get('test_results', {}).get('precision'),
                'test_recall': r.get('test_results', {}).get('recall'),
                'best_params': r.get('grid_search', {}).get('best_params'),
                'n_features': r.get('n_features'),
            }
            for name, r in all_results.items()
        }
        comp_file = script_dir / "results" / f"comparison_{args.feature_mode}.json"
        comp_file.parent.mkdir(parents=True, exist_ok=True)
        with open(comp_file, 'w', encoding='utf-8') as fh:
            json.dump(comp, fh, indent=2, ensure_ascii=False)
        logger.info(f"\nComparacao salva em: {comp_file}")
        print_comparison_table(all_results, title=args.feature_mode.upper())


if __name__ == "__main__":
    main()
