"""
Analisador LIWC para Comparação Human vs LLM

Calcula scores LIWC por documento e realiza análise discriminativa
entre textos escritos por humanos vs gerados por LLMs,
usando os mesmos testes estatísticos do módulo ed_rules.
"""

from pathlib import Path
import numpy as np
from scipy import stats as scipy_stats
import pandas as pd

from .dictionary import LiwcDictionary
from .text_extractor import TextExtractor


class LiwcAnalyzer:
    """
    Analisador LIWC para comparação entre textos humanos e LLM.

    Segue o mesmo padrão do RulesTfidfAnalyzer (ed_rules/tfidf_rules.py):
    carrega dados, calcula features, realiza testes estatísticos e exporta resultados.
    """

    def __init__(self, human_dirs: list, llm_dirs: list, dictionary_path: str):
        self.human_dirs = human_dirs if isinstance(human_dirs, list) else [human_dirs]
        self.llm_dirs = llm_dirs if isinstance(llm_dirs, list) else [llm_dirs]

        self.dictionary = LiwcDictionary(dictionary_path)
        self.category_names = self.dictionary.get_category_names()

        # Dados carregados
        self.human_docs: list[dict] = []
        self.llm_docs: list[dict] = []

        # Scores LIWC por documento
        self.human_scores: list[dict] = []
        self.llm_scores: list[dict] = []

        # Resultados
        self.discriminative_df: pd.DataFrame | None = None

    def load_texts(self):
        """Carrega textos de todos os diretórios."""
        print("=" * 80)
        print("CARREGANDO TEXTOS DO CORPUS")
        print("=" * 80)

        for dir_path in self.human_dirs:
            docs = TextExtractor.extract_texts_from_directory(dir_path)
            for doc in docs:
                doc['author_type'] = 'human'
            self.human_docs.extend(docs)
            print(f"  HUMAN: {dir_path} -> {len(docs)} documentos")

        for dir_path in self.llm_dirs:
            docs = TextExtractor.extract_texts_from_directory(dir_path)
            for doc in docs:
                doc['author_type'] = 'llm'
            self.llm_docs.extend(docs)
            print(f"  LLM:   {dir_path} -> {len(docs)} documentos")

        print(f"\nTotal de documentos humanos: {len(self.human_docs)}")
        print(f"Total de documentos LLM: {len(self.llm_docs)}")
        print(f"Total geral: {len(self.human_docs) + len(self.llm_docs)}")

    def compute_liwc_scores(self):
        """
        Calcula scores LIWC (%) para cada documento.

        Para cada documento:
        - Tokeniza o texto
        - Categoriza cada palavra via dicionário LIWC
        - Normaliza contagens para porcentagem (count / total_words * 100)
        """
        print(f"\n{'='*80}")
        print("CALCULANDO SCORES LIWC POR DOCUMENTO")
        print("=" * 80)

        self.human_scores = self._compute_scores_for_docs(self.human_docs, 'human')
        self.llm_scores = self._compute_scores_for_docs(self.llm_docs, 'llm')

        print(f"\n  Documentos processados: {len(self.human_scores) + len(self.llm_scores)}")
        print(f"  Categorias LIWC: {len(self.category_names)}")

    def _compute_scores_for_docs(self, docs: list[dict], label: str) -> list[dict]:
        """Calcula scores LIWC para uma lista de documentos."""
        scores = []

        for i, doc in enumerate(docs):
            counts, total_words, matched_words = self.dictionary.categorize_text(doc['text'])

            if total_words == 0:
                continue

            score = {
                'file_name': doc['file_name'],
                'file_path': doc['file_path'],
                'author_type': label,
                'word_count': total_words,
                'matched_words': matched_words,
                'dictionary_coverage': 100 * matched_words / total_words,
            }

            # Porcentagem por categoria
            for cat_name in self.category_names:
                score[cat_name] = 100 * counts.get(cat_name, 0) / total_words

            scores.append(score)

            if (i + 1) % 2000 == 0:
                print(f"    {label}: {i + 1}/{len(docs)} documentos processados...")

        print(f"  {label.upper()}: {len(scores)} documentos com scores calculados")
        return scores

    def analyze_discriminative_categories(self):
        """
        Analisa quais categorias LIWC são mais discriminativas entre human e LLM.

        Para cada categoria, calcula:
        - Média e desvio padrão em cada grupo
        - Mann-Whitney U test (bicaudal)
        - Cohen's d (tamanho do efeito)
        """
        print(f"\n{'='*80}")
        print("ANÁLISE DISCRIMINATIVA POR CATEGORIA LIWC")
        print("=" * 80)

        # Monta arrays por categoria
        all_categories = self.category_names + ['dictionary_coverage']

        analysis_rows = []

        for cat in all_categories:
            human_vals = np.array([s[cat] for s in self.human_scores])
            llm_vals = np.array([s[cat] for s in self.llm_scores])

            mean_human = np.mean(human_vals)
            mean_llm = np.mean(llm_vals)
            std_human = np.std(human_vals)
            std_llm = np.std(llm_vals)

            diff = mean_human - mean_llm

            # Mann-Whitney U test
            try:
                statistic, p_value = scipy_stats.mannwhitneyu(
                    human_vals, llm_vals, alternative='two-sided'
                )
            except ValueError:
                statistic, p_value = 0, 1.0

            # Cohen's d
            pooled_std = np.sqrt((std_human**2 + std_llm**2) / 2)
            cohens_d = diff / pooled_std if pooled_std > 0 else 0

            analysis_rows.append({
                'category': cat,
                'mean_human': mean_human,
                'mean_llm': mean_llm,
                'std_human': std_human,
                'std_llm': std_llm,
                'difference': diff,
                'abs_difference': abs(diff),
                'mann_whitney_u': statistic,
                'p_value': p_value,
                'cohens_d': cohens_d,
                'significant_005': p_value < 0.05,
                'significant_001': p_value < 0.01,
                'characteristic_of': 'human' if diff > 0 else 'llm' if diff < 0 else 'neutral'
            })

        self.discriminative_df = pd.DataFrame(analysis_rows)
        self.discriminative_df = self.discriminative_df.sort_values(
            'abs_difference', ascending=False
        ).reset_index(drop=True)

        # Resumo
        sig_005 = self.discriminative_df['significant_005'].sum()
        sig_001 = self.discriminative_df['significant_001'].sum()
        total = len(self.discriminative_df)

        print(f"\nCategorias analisadas: {total}")
        print(f"  Significativas (p<0.05): {sig_005} ({100*sig_005/total:.1f}%)")
        print(f"  Significativas (p<0.01): {sig_001} ({100*sig_001/total:.1f}%)")
        print(f"  Mais características de HUMAN: {(self.discriminative_df['characteristic_of'] == 'human').sum()}")
        print(f"  Mais características de LLM: {(self.discriminative_df['characteristic_of'] == 'llm').sum()}")

        return self.discriminative_df

    def get_features_for_ml(self) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """
        Retorna features prontas para uso em classificador ML.

        Returns:
            X: matriz de features (n_docs x n_categories)
            y: labels (0=human, 1=llm)
            feature_names: nomes das categorias LIWC
        """
        all_scores = self.human_scores + self.llm_scores
        feature_names = self.category_names

        X = np.array([[s[cat] for cat in feature_names] for s in all_scores])
        y = np.array([0] * len(self.human_scores) + [1] * len(self.llm_scores))

        return X, y, feature_names

    def export_results(self, output_dir: str):
        """Exporta todos os resultados para arquivos."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*80}")
        print("EXPORTANDO RESULTADOS LIWC")
        print("=" * 80)
        print(f"Diretório: {output_dir}")

        # 1. Categorias discriminativas (CSV)
        csv_file = output_path / 'liwc_discriminative_categories.csv'
        self.discriminative_df.to_csv(csv_file, index=False, encoding='utf-8')
        print(f"\n  >> {csv_file.name}")

        # 2. Relatório texto legível
        txt_file = output_path / 'liwc_discriminative_categories.txt'
        self._write_report(txt_file)
        print(f"  >> {txt_file.name}")

        # 3. Scores por documento (CSV)
        doc_csv = output_path / 'liwc_per_document_scores.csv'
        all_scores = self.human_scores + self.llm_scores
        pd.DataFrame(all_scores).to_csv(doc_csv, index=False, encoding='utf-8')
        print(f"  >> {doc_csv.name}")

        # 4. Resumo geral
        summary_file = output_path / 'liwc_analysis_summary.txt'
        self._write_summary(summary_file)
        print(f"  >> {summary_file.name}")

    def _write_report(self, output_file: Path):
        """Escreve relatório detalhado em formato texto."""
        df = self.discriminative_df

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 110 + "\n")
            f.write("ANÁLISE DISCRIMINATIVA LIWC: HUMAN vs LLM\n")
            f.write("=" * 110 + "\n\n")

            f.write("INTERPRETAÇÃO:\n")
            f.write("-" * 110 + "\n")
            f.write("  - mean_human/mean_llm: Média da % de palavras na categoria em cada grupo\n")
            f.write("  - difference: (mean_human - mean_llm)\n")
            f.write("      > 0: categoria mais presente em textos HUMANOS\n")
            f.write("      < 0: categoria mais presente em textos LLM\n")
            f.write("  - p_value: Significância estatística (Mann-Whitney U test)\n")
            f.write("  - cohens_d: Tamanho do efeito (>0.8=grande, 0.5=médio, 0.2=pequeno)\n\n")

            f.write("ESTATÍSTICAS DO CORPUS:\n")
            f.write("-" * 110 + "\n")
            f.write(f"  Documentos human: {len(self.human_scores)}\n")
            f.write(f"  Documentos LLM: {len(self.llm_scores)}\n")
            f.write(f"  Total: {len(self.human_scores) + len(self.llm_scores)}\n")
            f.write(f"  Categorias LIWC: {len(self.category_names)}\n\n")

            sig_005 = df['significant_005'].sum()
            sig_001 = df['significant_001'].sum()
            f.write("RESUMO DA ANÁLISE:\n")
            f.write("-" * 110 + "\n")
            f.write(f"  Categorias significativas (p<0.05): {sig_005}\n")
            f.write(f"  Categorias significativas (p<0.01): {sig_001}\n")
            f.write(f"  Mais características de HUMAN: {(df['characteristic_of'] == 'human').sum()}\n")
            f.write(f"  Mais características de LLM: {(df['characteristic_of'] == 'llm').sum()}\n\n")

            # Todas as categorias ordenadas
            f.write("=" * 110 + "\n")
            f.write("TODAS AS CATEGORIAS LIWC (ordenadas por diferença absoluta)\n")
            f.write("=" * 110 + "\n\n")

            f.write(f"{'#':<4}{'Categoria':<40}{'Mean_H':>8}{'Mean_L':>8}{'Diff':>9}{'p-value':>11}{'Cohen d':>9}{'Caract.':>10}\n")
            f.write("-" * 110 + "\n")

            for idx, row in df.iterrows():
                rank = df.index.get_loc(idx) + 1
                sig = '**' if row['significant_001'] else '* ' if row['significant_005'] else '  '
                f.write(
                    f"{rank:<4}{row['category']:<40}"
                    f"{row['mean_human']:>8.3f}{row['mean_llm']:>8.3f}"
                    f"{row['difference']:>+9.4f}{sig}"
                    f"{row['p_value']:>9.2e}{row['cohens_d']:>+9.3f}"
                    f"{row['characteristic_of']:>10}\n"
                )

            f.write("\n" + "=" * 110 + "\n")
            f.write("Legenda: ** = p<0.01, * = p<0.05\n")
            f.write("=" * 110 + "\n")

    def _write_summary(self, output_file: Path):
        """Escreve resumo geral da análise."""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("RESUMO DA ANÁLISE LIWC\n")
            f.write("=" * 80 + "\n\n")

            f.write("DADOS CARREGADOS:\n")
            f.write(f"  Documentos humanos: {len(self.human_scores)}\n")
            f.write(f"  Documentos LLM: {len(self.llm_scores)}\n")
            f.write(f"  Total: {len(self.human_scores) + len(self.llm_scores)}\n\n")

            f.write("DICIONÁRIO LIWC:\n")
            f.write(f"  Categorias: {len(self.category_names)}\n")
            f.write(f"  Palavras exatas: {len(self.dictionary.exact_matches)}\n")
            f.write(f"  Prefixos: {len(self.dictionary.prefix_matches)}\n\n")

            # Cobertura média do dicionário
            human_cov = np.mean([s['dictionary_coverage'] for s in self.human_scores])
            llm_cov = np.mean([s['dictionary_coverage'] for s in self.llm_scores])
            f.write("COBERTURA DO DICIONÁRIO:\n")
            f.write(f"  Média human: {human_cov:.1f}% das palavras reconhecidas\n")
            f.write(f"  Média LLM:   {llm_cov:.1f}% das palavras reconhecidas\n\n")

            # Word count médio
            human_wc = np.mean([s['word_count'] for s in self.human_scores])
            llm_wc = np.mean([s['word_count'] for s in self.llm_scores])
            f.write("CONTAGEM DE PALAVRAS:\n")
            f.write(f"  Média human: {human_wc:.1f} palavras/documento\n")
            f.write(f"  Média LLM:   {llm_wc:.1f} palavras/documento\n\n")

            if self.discriminative_df is not None:
                df = self.discriminative_df
                sig = df[df['significant_005']]

                f.write("CATEGORIAS SIGNIFICATIVAS (p<0.05):\n")
                f.write("-" * 80 + "\n")

                human_sig = sig[sig['characteristic_of'] == 'human'].head(10)
                if len(human_sig) > 0:
                    f.write("\n  Mais presentes em textos HUMANOS:\n")
                    for _, row in human_sig.iterrows():
                        f.write(f"    - {row['category']}: {row['mean_human']:.3f}% vs {row['mean_llm']:.3f}% "
                                f"(d={row['cohens_d']:+.3f})\n")

                llm_sig = sig[sig['characteristic_of'] == 'llm'].head(10)
                if len(llm_sig) > 0:
                    f.write("\n  Mais presentes em textos LLM:\n")
                    for _, row in llm_sig.iterrows():
                        f.write(f"    - {row['category']}: {row['mean_human']:.3f}% vs {row['mean_llm']:.3f}% "
                                f"(d={row['cohens_d']:+.3f})\n")

            f.write("\n" + "=" * 80 + "\n")
