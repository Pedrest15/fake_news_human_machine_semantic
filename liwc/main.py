"""
Pipeline LIWC para Análise de Fake News: Human vs LLM

Executa a análise LIWC completa sobre os textos do corpus,
comparando categorias psicolinguísticas entre textos escritos
por humanos e gerados por LLMs.

Uso:
    python3 liwc/main.py [caminho_do_dicionario.dic]
"""

import sys
from pathlib import Path

import numpy as np

from .analyzer import LiwcAnalyzer


def main(dictionary_path: str | None = None):
    """Função principal para execução da análise LIWC."""

    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    # Path to the LIWC dictionary
    if dictionary_path is None:
        if len(sys.argv) > 1:
            dictionary_path = sys.argv[1]
        else:
            # Default: look inside the liwc/ folder
            default_paths = [
                PROJECT_ROOT / 'liwc' / 'Brazilian_Portuguese_LIWC2015_dictionary.dic',
                PROJECT_ROOT / 'liwc' / 'LIWC2015_Portuguese.dic',
                PROJECT_ROOT / 'liwc' / 'LIWC_Portuguese.dic',
                PROJECT_ROOT / 'liwc' / 'liwc.dic',
            ]
            for p in default_paths:
                if p.exists():
                    dictionary_path = str(p)
                    break

            if dictionary_path is None:
                print("ERROR: LIWC dictionary not found.")
                print("\nPlace the Portuguese LIWC .dic file in one of these paths:")
                for p in default_paths:
                    print(f"  - {p}")
                print("\nOr pass the path as an argument:")
                print("  python3 -m liwc.main /path/to/dictionary.dic")
                sys.exit(1)

    # Corpus directories holding the full texts
    CORPUS_DIR = PROJECT_ROOT.parent / 'noticias_falsas_humano_maquina_caracterizacao' / 'corpus'

    HUMAN_DIRS = [
        str(CORPUS_DIR / 'FakeTrue.Br-main' / 'fake'),
        str(CORPUS_DIR / 'Fake.br-Corpus-master' / 'full_texts' / 'fake_br_clean'),
    ]

    LLM_DIRS = [
        str(CORPUS_DIR / 'fake-news-llm-ptbr-main' / 'fake-news-llm-ptbr-main' / 'data' / 'FakeTrueBR'),
        str(CORPUS_DIR / 'fake-news-llm-ptbr-main' / 'fake-news-llm-ptbr-main' / 'data' / 'Fake.Br'),
    ]

    OUTPUT_DIR = str(PROJECT_ROOT / 'liwc' / 'liwc_output')

    # Banner
    print("=" * 80)
    print("LIWC ANALYSIS OF FAKE NEWS")
    print("Human vs LLM")
    print("=" * 80)

    print(f"\nDictionary: {dictionary_path}")
    print("\nHuman directories:")
    for d in HUMAN_DIRS:
        print(f"  - {d}")
    print("\nLLM directories:")
    for d in LLM_DIRS:
        print(f"  - {d}")

    # Build the analyser
    analyzer = LiwcAnalyzer(
        human_dirs=HUMAN_DIRS,
        llm_dirs=LLM_DIRS,
        dictionary_path=dictionary_path
    )

    # Pipeline
    analyzer.load_texts()

    if not analyzer.human_docs or not analyzer.llm_docs:
        print("\nERROR: could not load documents from both groups!")
        return None

    analyzer.compute_liwc_scores()
    analyzer.analyze_discriminative_categories()

    # Export the results
    analyzer.export_results(OUTPUT_DIR)

    # Preview
    print(f"\n{'='*80}")
    print("PREVIEW: TOP 10 DISCRIMINATIVE CATEGORIES")
    print("=" * 80)

    df = analyzer.discriminative_df
    sig_df = df[df['significant_005']]

    human_top = sig_df[sig_df['characteristic_of'] == 'human'].head(5)
    if len(human_top) > 0:
        print("\nMore characteristic of HUMAN:")
        for _, row in human_top.iterrows():
            print(f"  {row['category']:<35} (d={row['cohens_d']:+.3f}, p={row['p_value']:.2e})")

    llm_top = sig_df[sig_df['characteristic_of'] == 'llm'].head(5)
    if len(llm_top) > 0:
        print("\nMore characteristic of LLM:")
        for _, row in llm_top.iterrows():
            print(f"  {row['category']:<35} (d={row['cohens_d']:+.3f}, p={row['p_value']:.2e})")

    # Features for ML
    X, y, features = analyzer.get_features_for_ml()
    print(f"\n{'='*80}")
    print("FEATURES FOR MACHINE LEARNING")
    print("=" * 80)
    print(f"\nMatrix X: {X.shape}")
    print(f"Labels y: {y.shape} (0=human: {np.sum(y==0)}, 1=llm: {np.sum(y==1)})")
    print(f"Features: {len(features)} LIWC categories")

    print(f"\n{'='*80}")
    print("LIWC ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"\nResults saved to: {OUTPUT_DIR}/")

    return analyzer


if __name__ == "__main__":
    analyzer = main()
