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

    # Caminho do dicionário LIWC
    if dictionary_path is None:
        if len(sys.argv) > 1:
            dictionary_path = sys.argv[1]
        else:
            # Caminho padrão: procura na pasta liwc/
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
                print("ERRO: Dicionário LIWC não encontrado.")
                print("\nColoque o arquivo .dic do LIWC em português em um dos caminhos:")
                for p in default_paths:
                    print(f"  - {p}")
                print("\nOu passe o caminho como argumento:")
                print("  python3 -m liwc.main /caminho/para/dicionario.dic")
                sys.exit(1)

    # Diretórios do corpus com textos na íntegra
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
    print("ANÁLISE LIWC DE FAKE NEWS")
    print("Human vs LLM")
    print("=" * 80)

    print(f"\nDicionário: {dictionary_path}")
    print("\nDiretórios Human:")
    for d in HUMAN_DIRS:
        print(f"  - {d}")
    print("\nDiretórios LLM:")
    for d in LLM_DIRS:
        print(f"  - {d}")

    # Cria analisador
    analyzer = LiwcAnalyzer(
        human_dirs=HUMAN_DIRS,
        llm_dirs=LLM_DIRS,
        dictionary_path=dictionary_path
    )

    # Pipeline
    analyzer.load_texts()

    if not analyzer.human_docs or not analyzer.llm_docs:
        print("\nERRO: Não foi possível carregar documentos de ambos os grupos!")
        return None

    analyzer.compute_liwc_scores()
    analyzer.analyze_discriminative_categories()

    # Exporta resultados
    analyzer.export_results(OUTPUT_DIR)

    # Preview
    print(f"\n{'='*80}")
    print("PREVIEW: TOP 10 CATEGORIAS DISCRIMINATIVAS")
    print("=" * 80)

    df = analyzer.discriminative_df
    sig_df = df[df['significant_005']]

    human_top = sig_df[sig_df['characteristic_of'] == 'human'].head(5)
    if len(human_top) > 0:
        print("\nMais características de HUMAN:")
        for _, row in human_top.iterrows():
            print(f"  {row['category']:<35} (d={row['cohens_d']:+.3f}, p={row['p_value']:.2e})")

    llm_top = sig_df[sig_df['characteristic_of'] == 'llm'].head(5)
    if len(llm_top) > 0:
        print("\nMais características de LLM:")
        for _, row in llm_top.iterrows():
            print(f"  {row['category']:<35} (d={row['cohens_d']:+.3f}, p={row['p_value']:.2e})")

    # Features para ML
    X, y, features = analyzer.get_features_for_ml()
    print(f"\n{'='*80}")
    print("FEATURES PARA MACHINE LEARNING")
    print("=" * 80)
    print(f"\nMatriz X: {X.shape}")
    print(f"Labels y: {y.shape} (0=human: {np.sum(y==0)}, 1=llm: {np.sum(y==1)})")
    print(f"Features: {len(features)} categorias LIWC")

    print(f"\n{'='*80}")
    print("ANÁLISE LIWC CONCLUÍDA!")
    print("=" * 80)
    print(f"\nResultados salvos em: {OUTPUT_DIR}/")

    return analyzer


if __name__ == "__main__":
    analyzer = main()
