"""
TF-IDF de Regras Gramaticais de Dependência

Este módulo calcula TF-IDF das regras gramaticais extraídas de arquivos CoNLL-U,
permitindo análise comparativa entre textos escritos por humanos vs LLMs.

Abordagem:
- Cada arquivo .rules.json é um documento
- TF-IDF calculado sobre todo o corpus (human + llm)
- Análise discriminativa compara médias entre grupos
- Testes estatísticos avaliam significância das diferenças

Duas variantes:
1. Com repetição: TF reflete frequência real de uso da estrutura
2. Sem repetição (binária): presença/ausência da regra no documento
"""

import json
from pathlib import Path
import numpy as np
from scipy import stats as scipy_stats
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd


class RulesTfidfAnalyzer:
    """
    Analisador TF-IDF para regras gramaticais de dependência.

    Calcula TF-IDF considerando cada documento individualmente,
    depois agrega estatísticas por autor (human vs llm) para
    identificar regras discriminativas.
    """

    def __init__(self, human_dirs: list, llm_dirs: list, prefix: str = ""):
        """
        Args:
            human_dirs: Lista de diretórios com arquivos .rules.json de humanos
            llm_dirs: Lista de diretórios com arquivos .rules.json de LLMs
            prefix: Prefixo para nomes de arquivos de saída (ex: "ed_only_", "all_rules_")
        """
        self.human_dirs = human_dirs if isinstance(human_dirs, list) else [human_dirs]
        self.llm_dirs = llm_dirs if isinstance(llm_dirs, list) else [llm_dirs]
        self.prefix = prefix

        # Dados carregados
        self.human_docs = []  # Lista de dicts com info de cada documento
        self.llm_docs = []

        # Resultados das análises
        self.tfidf_results = {}
        self.discriminative_analysis = {}

    def load_rules_files(self):
        """Carrega todos os arquivos .rules.json dos diretórios especificados"""
        print("=" * 80)
        print("CARREGANDO ARQUIVOS DE REGRAS")
        print("=" * 80)

        # Carrega documentos humanos
        for dir_path in self.human_dirs:
            self._load_from_directory(dir_path, self.human_docs, 'human')

        # Carrega documentos LLM
        for dir_path in self.llm_dirs:
            self._load_from_directory(dir_path, self.llm_docs, 'llm')

        print(f"\nTotal de documentos humanos: {len(self.human_docs)}")
        print(f"Total de documentos LLM: {len(self.llm_docs)}")
        print(f"Total geral: {len(self.human_docs) + len(self.llm_docs)}")

    def _load_from_directory(self, dir_path, doc_list, author_type):
        """Carrega arquivos .rules.json de um diretório específico"""
        path = Path(dir_path)

        if not path.exists():
            print(f"  AVISO: Diretório não encontrado: {dir_path}")
            return

        files = sorted(list(path.glob('*.rules.json')))
        print(f"\n  {author_type.upper()}: {dir_path}")
        print(f"    Arquivos encontrados: {len(files)}")

        loaded = 0
        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                doc_list.append({
                    'file_path': str(file_path),
                    'file_name': file_path.name,
                    'doc_id': f"{author_type}_{file_path.stem}",
                    'source_file': data.get('source_file', ''),
                    'rules': data['rules'],
                    'unique_rules': list(set(data['rules'])),
                    'total_rules': len(data['rules']),
                    'num_unique': len(set(data['rules'])),
                    'author_type': author_type
                })
                loaded += 1
            except Exception as e:
                print(f"    ERRO ao carregar {file_path.name}: {e}")

        print(f"    Carregados com sucesso: {loaded}")

    def _get_rules_list(self, rules: list, with_repetition: bool = True) -> list:
        """
        Retorna lista de regras para uso com tokenizador customizado.

        Args:
            rules: Lista de regras gramaticais
            with_repetition: Se True, mantém todas as ocorrências;
                           Se False, usa apenas regras únicas (presença/ausência)

        Returns:
            Lista de regras (cada regra é um token completo)
        """
        if with_repetition:
            return rules
        else:
            return list(set(rules))

    def calculate_tfidf(self, with_repetition: bool = True):
        """
        Calcula TF-IDF sobre todos os documentos (cada arquivo = 1 documento).

        O IDF é calculado considerando em quantos documentos cada regra aparece,
        dando menor peso a regras ubíquas e maior peso a regras raras.

        Args:
            with_repetition: Se True, TF considera frequência real;
                           Se False, TF é binário (0 ou 1)

        Returns:
            dict com resultados da análise
        """
        mode = "com_repeticao" if with_repetition else "sem_repeticao"
        print(f"\n{'='*80}")
        print(f"CALCULANDO TF-IDF ({mode.upper().replace('_', ' ')})")
        print("=" * 80)

        # Combina todos os documentos mantendo a ordem (human primeiro, llm depois)
        all_docs = self.human_docs + self.llm_docs
        labels = ['human'] * len(self.human_docs) + ['llm'] * len(self.llm_docs)
        doc_ids = [doc['doc_id'] for doc in all_docs]

        # Prepara corpus como lista de listas de regras (cada regra é um token completo)
        # Isso evita o problema de tokenização incorreta quando regras contêm espaços
        corpus = [self._get_rules_list(doc['rules'], with_repetition) for doc in all_docs]

        print(f"\nCorpus: {len(corpus)} documentos")
        print(f"  - Human: {len(self.human_docs)}")
        print(f"  - LLM: {len(self.llm_docs)}")

        # Configura TfidfVectorizer com analisador customizado
        # O analyzer=lambda x: x indica que o documento já está tokenizado (lista de tokens)
        # Isso preserva regras completas como "VERB(NOUN/nsubj, *, NOUN/obj)"
        vectorizer = TfidfVectorizer(
            analyzer=lambda doc: doc,  # Documento já é lista de tokens
            use_idf=True,
            norm='l2',  # Normalização L2 por documento
            smooth_idf=True,  # Evita divisão por zero
            sublinear_tf=False  # TF linear (não logarítmico)
        )

        # Calcula matriz TF-IDF
        tfidf_matrix = vectorizer.fit_transform(corpus)
        feature_names = vectorizer.get_feature_names_out()

        print(f"\nMatriz TF-IDF calculada:")
        print(f"  - Dimensões: {tfidf_matrix.shape[0]} documentos × {tfidf_matrix.shape[1]} regras")
        print(f"  - Regras únicas (vocabulário): {len(feature_names)}")
        print(f"  - Elementos não-zero: {tfidf_matrix.nnz}")
        print(f"  - Esparsidade: {100 * (1 - tfidf_matrix.nnz / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1])):.2f}%")

        # Armazena resultados
        result = {
            'mode': mode,
            'with_repetition': with_repetition,
            'tfidf_matrix': tfidf_matrix,
            'feature_names': feature_names,
            'vectorizer': vectorizer,
            'labels': np.array(labels),
            'doc_ids': doc_ids,
            'documents': all_docs,
            'num_human': len(self.human_docs),
            'num_llm': len(self.llm_docs),
            'num_features': len(feature_names)
        }

        self.tfidf_results[mode] = result
        return result

    def analyze_discriminative_rules(self, mode: str = 'com_repeticao', min_docs: int = 10):
        """
        Analisa quais regras são mais discriminativas entre human e llm.

        Para cada regra, calcula:
        - Média de TF-IDF em documentos human vs llm
        - Diferença das médias (effect)
        - Teste estatístico (Mann-Whitney U) para significância
        - Tamanho do efeito (Cohen's d)

        Args:
            mode: 'com_repeticao' ou 'sem_repeticao'
            min_docs: Mínimo de documentos onde a regra deve aparecer para ser considerada

        Returns:
            DataFrame com análise discriminativa
        """
        if mode not in self.tfidf_results:
            print(f"Erro: Modo '{mode}' não encontrado. Execute calculate_tfidf() primeiro.")
            return None

        print(f"\n{'='*80}")
        print(f"ANÁLISE DISCRIMINATIVA ({mode.upper().replace('_', ' ')})")
        print("=" * 80)

        result = self.tfidf_results[mode]
        tfidf_matrix = result['tfidf_matrix'].toarray()
        feature_names = result['feature_names']
        labels = result['labels']

        # Separa índices por grupo
        human_idx = np.where(labels == 'human')[0]
        llm_idx = np.where(labels == 'llm')[0]

        print(f"\nAnalisando {len(feature_names)} regras...")
        print(f"  - Documentos human: {len(human_idx)}")
        print(f"  - Documentos LLM: {len(llm_idx)}")

        # Análise para cada regra
        analysis_rows = []

        for i, rule in enumerate(feature_names):
            human_scores = tfidf_matrix[human_idx, i]
            llm_scores = tfidf_matrix[llm_idx, i]

            # Conta em quantos documentos a regra aparece (TF-IDF > 0)
            human_presence = np.sum(human_scores > 0)
            llm_presence = np.sum(llm_scores > 0)
            total_presence = human_presence + llm_presence

            # Filtra regras muito raras
            if total_presence < min_docs:
                continue

            # Estatísticas básicas
            mean_human = np.mean(human_scores)
            mean_llm = np.mean(llm_scores)
            std_human = np.std(human_scores)
            std_llm = np.std(llm_scores)

            # Diferença de médias (positivo = mais característico de human)
            diff = mean_human - mean_llm

            # Teste Mann-Whitney U (não assume normalidade)
            # Alternativa: ttest_ind para teste t
            try:
                statistic, p_value = scipy_stats.mannwhitneyu(
                    human_scores, llm_scores, alternative='two-sided'
                )
            except ValueError:
                # Ocorre se todos os valores são iguais
                statistic, p_value = 0, 1.0

            # Cohen's d (tamanho do efeito)
            pooled_std = np.sqrt((std_human**2 + std_llm**2) / 2)
            if pooled_std > 0:
                cohens_d = diff / pooled_std
            else:
                cohens_d = 0

            analysis_rows.append({
                'rule': rule,
                'mean_human': mean_human,
                'mean_llm': mean_llm,
                'std_human': std_human,
                'std_llm': std_llm,
                'difference': diff,
                'abs_difference': abs(diff),
                'docs_human': human_presence,
                'docs_llm': llm_presence,
                'docs_total': total_presence,
                'pct_human': 100 * human_presence / len(human_idx),
                'pct_llm': 100 * llm_presence / len(llm_idx),
                'mann_whitney_u': statistic,
                'p_value': p_value,
                'cohens_d': cohens_d,
                'significant_005': p_value < 0.05,
                'significant_001': p_value < 0.01,
                'characteristic_of': 'human' if diff > 0 else 'llm' if diff < 0 else 'neutral'
            })

        # Cria DataFrame e ordena
        df = pd.DataFrame(analysis_rows)

        # Ordena por diferença absoluta (regras mais discriminativas primeiro)
        df = df.sort_values('abs_difference', ascending=False).reset_index(drop=True)

        print(f"\nRegras analisadas (com min_docs={min_docs}): {len(df)}")

        # Estatísticas resumidas
        sig_005 = df['significant_005'].sum()
        sig_001 = df['significant_001'].sum()
        human_char = (df['characteristic_of'] == 'human').sum()
        llm_char = (df['characteristic_of'] == 'llm').sum()

        print(f"\nResultados:")
        print(f"  - Regras significativas (p<0.05): {sig_005} ({100*sig_005/len(df):.1f}%)")
        print(f"  - Regras significativas (p<0.01): {sig_001} ({100*sig_001/len(df):.1f}%)")
        print(f"  - Mais características de human: {human_char}")
        print(f"  - Mais características de LLM: {llm_char}")

        # Armazena resultado
        self.discriminative_analysis[mode] = df

        return df

    def get_top_rules(self, mode: str = 'com_repeticao', top_n: int = 50,
                      only_significant: bool = True):
        """
        Retorna as top N regras mais discriminativas para cada grupo.

        Args:
            mode: 'com_repeticao' ou 'sem_repeticao'
            top_n: Número de regras top para retornar
            only_significant: Se True, filtra apenas regras com p<0.05

        Returns:
            dict com 'human_top' e 'llm_top' DataFrames
        """
        if mode not in self.discriminative_analysis:
            print(f"Execute analyze_discriminative_rules('{mode}') primeiro.")
            return None

        df = self.discriminative_analysis[mode].copy()

        if only_significant:
            df = df[df['significant_005']]

        # Top regras características de human (difference > 0)
        human_top = df[df['difference'] > 0].head(top_n)

        # Top regras características de llm (difference < 0)
        llm_top = df[df['difference'] < 0].head(top_n)

        return {
            'human_top': human_top,
            'llm_top': llm_top,
            'mode': mode,
            'only_significant': only_significant
        }

    def get_features_for_ml(self, mode: str = 'com_repeticao'):
        """
        Retorna features prontas para uso em classificador ML.

        Args:
            mode: 'com_repeticao' ou 'sem_repeticao'

        Returns:
            X: matriz de features (n_docs × n_features)
            y: labels (0=human, 1=llm)
            feature_names: nomes das features (regras)
        """
        if mode not in self.tfidf_results:
            print(f"Execute calculate_tfidf() com mode='{mode}' primeiro.")
            return None, None, None

        result = self.tfidf_results[mode]

        X = result['tfidf_matrix']
        y = np.array([0 if l == 'human' else 1 for l in result['labels']])
        feature_names = result['feature_names']

        return X, y, feature_names

    def export_results(self, output_dir: str):
        """Exporta todos os resultados para arquivos"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*80}")
        print("EXPORTANDO RESULTADOS")
        print("=" * 80)
        print(f"Diretório: {output_dir}")

        # Para cada modo analisado
        for mode in self.discriminative_analysis:
            df = self.discriminative_analysis[mode]

            p = self.prefix
            # 1. CSV completo com todas as regras
            csv_file = output_path / f'{p}discriminative_rules_{mode}.csv'
            df.to_csv(csv_file, index=False, encoding='utf-8')
            print(f"\n  >> {csv_file.name}")

            # 2. Relatório TXT legível
            txt_file = output_path / f'{p}discriminative_rules_{mode}.txt'
            self._write_report(txt_file, mode)
            print(f"  >> {txt_file.name}")

            # 3. Top rules separados
            top_rules = self.get_top_rules(mode, top_n=100, only_significant=True)
            if top_rules:
                # Human top
                human_csv = output_path / f'{p}top_rules_human_{mode}.csv'
                top_rules['human_top'].to_csv(human_csv, index=False, encoding='utf-8')
                print(f"  >> {human_csv.name}")

                # LLM top
                llm_csv = output_path / f'{p}top_rules_llm_{mode}.csv'
                top_rules['llm_top'].to_csv(llm_csv, index=False, encoding='utf-8')
                print(f"  >> {llm_csv.name}")

        # 4. Resumo geral
        summary_file = output_path / f'{self.prefix}analysis_summary.txt'
        self._write_summary(summary_file)
        print(f"\n  >> {summary_file.name}")

    def _write_report(self, output_file: Path, mode: str):
        """Escreve relatório detalhado em formato texto"""
        df = self.discriminative_analysis[mode]
        result = self.tfidf_results[mode]

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 100 + "\n")
            f.write(f"ANÁLISE DISCRIMINATIVA DE REGRAS GRAMATICAIS\n")
            f.write(f"Modo: {mode.upper().replace('_', ' ')}\n")
            f.write("=" * 100 + "\n\n")

            f.write("INTERPRETAÇÃO:\n")
            f.write("-" * 100 + "\n")
            f.write("  - mean_human/mean_llm: Média do TF-IDF da regra em cada grupo\n")
            f.write("  - difference: (mean_human - mean_llm)\n")
            f.write("      > 0: regra mais característica de textos HUMANOS\n")
            f.write("      < 0: regra mais característica de textos LLM\n")
            f.write("  - p_value: Significância estatística (Mann-Whitney U test)\n")
            f.write("  - cohens_d: Tamanho do efeito (>0.8=grande, 0.5=médio, 0.2=pequeno)\n")
            f.write("\n")

            f.write("ESTATÍSTICAS DO CORPUS:\n")
            f.write("-" * 100 + "\n")
            f.write(f"  Total de documentos: {result['num_human'] + result['num_llm']}\n")
            f.write(f"  Documentos human: {result['num_human']}\n")
            f.write(f"  Documentos LLM: {result['num_llm']}\n")
            f.write(f"  Total de regras únicas: {result['num_features']}\n")
            f.write(f"  Regras analisadas: {len(df)}\n")
            f.write("\n")

            f.write("RESUMO DA ANÁLISE:\n")
            f.write("-" * 100 + "\n")
            sig_005 = df['significant_005'].sum()
            sig_001 = df['significant_001'].sum()
            f.write(f"  Regras significativas (p<0.05): {sig_005} ({100*sig_005/len(df):.1f}%)\n")
            f.write(f"  Regras significativas (p<0.01): {sig_001} ({100*sig_001/len(df):.1f}%)\n")
            f.write(f"  Mais características de HUMAN: {(df['characteristic_of'] == 'human').sum()}\n")
            f.write(f"  Mais características de LLM: {(df['characteristic_of'] == 'llm').sum()}\n")
            f.write("\n")

            # Top 50 regras human
            f.write("=" * 140 + "\n")
            f.write("TOP 50 REGRAS MAIS CARACTERÍSTICAS DE HUMANOS\n")
            f.write("=" * 140 + "\n\n")

            human_df = df[df['difference'] > 0].head(50)
            f.write(f"{'#':<4}{'Regra':<95}{'Diff':>8}{'p-value':>10}{'Cohen d':>9}{'%H':>7}{'%L':>7}\n")
            f.write("-" * 140 + "\n")

            for idx, row in human_df.iterrows():
                rank = human_df.index.get_loc(idx) + 1
                sig = '*' if row['significant_005'] else ' '
                f.write(f"{rank:<4}{row['rule']:<95}{row['difference']:>+7.4f}{sig}"
                       f"{row['p_value']:>9.2e}{row['cohens_d']:>+8.3f}"
                       f"{row['pct_human']:>7.1f}{row['pct_llm']:>7.1f}\n")

            # Top 50 regras LLM
            f.write("\n" + "=" * 140 + "\n")
            f.write("TOP 50 REGRAS MAIS CARACTERÍSTICAS DE LLMs\n")
            f.write("=" * 140 + "\n\n")

            llm_df = df[df['difference'] < 0].head(50)
            f.write(f"{'#':<4}{'Regra':<95}{'Diff':>8}{'p-value':>10}{'Cohen d':>9}{'%H':>7}{'%L':>7}\n")
            f.write("-" * 140 + "\n")

            for idx, row in llm_df.iterrows():
                rank = llm_df.index.get_loc(idx) + 1
                sig = '*' if row['significant_005'] else ' '
                f.write(f"{rank:<4}{row['rule']:<95}{row['difference']:>+7.4f}{sig}"
                       f"{row['p_value']:>9.2e}{row['cohens_d']:>+8.3f}"
                       f"{row['pct_human']:>7.1f}{row['pct_llm']:>7.1f}\n")

            f.write("\n" + "=" * 100 + "\n")
            f.write("Legenda: * = significativo (p<0.05), %H = % docs human, %L = % docs LLM\n")
            f.write("=" * 100 + "\n")

    def _write_summary(self, output_file: Path):
        """Escreve resumo geral da análise"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("RESUMO DA ANÁLISE TF-IDF DE REGRAS GRAMATICAIS\n")
            f.write("=" * 80 + "\n\n")

            f.write("DADOS CARREGADOS:\n")
            f.write(f"  Documentos humanos: {len(self.human_docs)}\n")
            f.write(f"  Documentos LLM: {len(self.llm_docs)}\n")
            f.write(f"  Total: {len(self.human_docs) + len(self.llm_docs)}\n\n")

            f.write("ANÁLISES REALIZADAS:\n")
            for mode, result in self.tfidf_results.items():
                f.write(f"\n  [{mode.upper()}]\n")
                f.write(f"    Vocabulário (regras únicas): {result['num_features']}\n")
                f.write(f"    Dimensões matriz TF-IDF: {result['tfidf_matrix'].shape}\n")

                if mode in self.discriminative_analysis:
                    df = self.discriminative_analysis[mode]
                    sig = df['significant_005'].sum()
                    f.write(f"    Regras significativas (p<0.05): {sig}\n")

            f.write("\n" + "=" * 80 + "\n")


def main():
    """Função principal para execução da análise TF-IDF"""

    ED_RULES_DIR = Path(__file__).resolve().parent

    # CONFIGURAÇÃO
    HUMAN_DIRS = [
        str(ED_RULES_DIR / 'rules_output/fake_true_human'),
        str(ED_RULES_DIR / 'rules_output/fake_br_human'),
    ]

    LLM_DIRS = [
        str(ED_RULES_DIR / 'rules_output/fake_true_llm'),
        str(ED_RULES_DIR / 'rules_output/fake_br_llm'),
    ]

    OUTPUT_DIR = str(ED_RULES_DIR / 'tfidf_output')

    # Mínimo de documentos onde a regra deve aparecer para ser analisada
    # Valores menores incluem mais regras mas com menor confiança estatística
    # Sugestões: 10 (conservador), 5 (moderado), 1 (todas as regras)
    MIN_DOCS = 1

    # Banner
    print("=" * 80)
    print("ANÁLISE TF-IDF DE REGRAS GRAMATICAIS")
    print("Human vs LLM")
    print("=" * 80)

    print("\nDiretórios Human:")
    for d in HUMAN_DIRS:
        print(f"  - {d}")
    print("\nDiretórios LLM:")
    for d in LLM_DIRS:
        print(f"  - {d}")
    print(f"\nConfiguração:")
    print(f"  - MIN_DOCS: {MIN_DOCS} (regras em menos docs serão ignoradas)")

    # Cria analisador
    analyzer = RulesTfidfAnalyzer(
        human_dirs=HUMAN_DIRS,
        llm_dirs=LLM_DIRS
    )

    # Carrega dados
    analyzer.load_rules_files()

    if not analyzer.human_docs or not analyzer.llm_docs:
        print("\nERRO: Não foi possível carregar documentos de ambos os grupos!")
        return None

    # === ANÁLISE COM REPETIÇÃO ===
    print("\n" + "=" * 80)
    print("ANÁLISE 1: COM REPETIÇÃO DE REGRAS")
    print("(TF considera frequência real de cada regra no documento)")
    print("=" * 80)

    analyzer.calculate_tfidf(with_repetition=True)
    analyzer.analyze_discriminative_rules(mode='com_repeticao', min_docs=MIN_DOCS)

    # === ANÁLISE SEM REPETIÇÃO ===
    print("\n" + "=" * 80)
    print("ANÁLISE 2: SEM REPETIÇÃO (BINÁRIA)")
    print("(TF considera apenas presença/ausência da regra)")
    print("=" * 80)

    analyzer.calculate_tfidf(with_repetition=False)
    analyzer.analyze_discriminative_rules(mode='sem_repeticao', min_docs=MIN_DOCS)

    # Exporta resultados
    analyzer.export_results(OUTPUT_DIR)

    # Preview dos resultados
    print("\n" + "=" * 80)
    print("PREVIEW: TOP 10 REGRAS DISCRIMINATIVAS")
    print("=" * 80)

    for mode in ['com_repeticao', 'sem_repeticao']:
        top = analyzer.get_top_rules(mode, top_n=10, only_significant=True)

        if top and len(top['human_top']) > 0:
            print(f"\n--- {mode.upper()} ---")
            print("\nMais características de HUMAN:")
            for _, row in top['human_top'].head(5).iterrows():
                print(f"  {row['rule'][:50]:<50} (d={row['cohens_d']:+.3f}, p={row['p_value']:.2e})")

            print("\nMais características de LLM:")
            for _, row in top['llm_top'].head(5).iterrows():
                print(f"  {row['rule'][:50]:<50} (d={row['cohens_d']:+.3f}, p={row['p_value']:.2e})")

    # Features para ML
    print("\n" + "=" * 80)
    print("FEATURES PARA MACHINE LEARNING")
    print("=" * 80)

    X, y, features = analyzer.get_features_for_ml('com_repeticao')
    print(f"\nMatriz X: {X.shape}")
    print(f"Labels y: {y.shape} (0=human: {np.sum(y==0)}, 1=llm: {np.sum(y==1)})")
    print(f"Features: {len(features)} regras")

    print("\n" + "=" * 80)
    print("ANÁLISE CONCLUÍDA!")
    print("=" * 80)
    print(f"\nResultados salvos em: {OUTPUT_DIR}/")

    return analyzer


if __name__ == "__main__":
    analyzer = main()
