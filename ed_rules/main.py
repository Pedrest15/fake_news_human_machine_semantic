"""
Script principal para extrair regras gramaticais de múltiplos arquivos CoNLL-U.

Este script processa todos os arquivos .conllu em diretórios especificados,
extrai suas regras gramaticais e calcula estatísticas agregadas.
"""
import json
from pathlib import Path
from collections import defaultdict
from extract_grammar_rules import (
    ConlluDependencyGrammar,
    realizar_testes_estatisticos,
    imprimir_resultados_estatisticos,
    exportar_resultados_csv
)
from eud_runner import apply_eud_batch
from tfidf_rules import RulesTfidfAnalyzer

EUD_STRATEGY = "eud_portuguese"


ED_RULES_DIR = Path(__file__).resolve().parent


def _eud_mirror(input_dirs):
    """Aplica as regras Grew do eud-portugues e devolve a lista de diretórios espelho.

    Para cada `portparser_results/<grupo>`, gera `ed_rules/portparser_results_eud/<grupo>`.
    """
    mirrored = []
    for d in input_dirs:
        src = Path(d)
        dst = ED_RULES_DIR / "portparser_results_eud" / src.name
        print(f"\n[EUD] {src} -> {dst}")
        apply_eud_batch(src, dst, strategy=EUD_STRATEGY)
        mirrored.append(str(dst))
    return mirrored


class GrammarBatchProcessor:
    """Processador em lote de arquivos CoNLL-U para extração de regras gramaticais"""

    def __init__(self, input_dirs, include_upos=True, include_deprel=False,
                 enhanced_only=False):
        """
        Args:
            input_dirs (list): Lista de diretórios para procurar arquivos .conllu
            include_upos (bool): Incluir UPOS nas regras
            include_deprel (bool): Incluir DEPREL nas regras
            enhanced_only (bool): Extrair apenas regras Enhanced (diferem das básicas)
        """
        dirs = input_dirs if isinstance(input_dirs, list) else [input_dirs]
        self.input_dirs = _eud_mirror(dirs)
        self.include_upos = include_upos
        self.include_deprel = include_deprel
        self.enhanced_only = enhanced_only
        self.conllu_files = []
        self.all_stats = []
        self.aggregated_stats = None

    def find_conllu_files(self):
        """Encontra todos os arquivos .conllu nos diretórios especificados"""
        print("\nProcurando arquivos .conllu...")

        for input_dir in self.input_dirs:
            input_path = Path(input_dir)

            if not input_path.exists():
                print(f"  AVISO: Diretório não encontrado: {input_dir}")
                continue

            # Busca não-recursiva por arquivos .conllu (apenas no diretório raiz)
            conllu_files = list(input_path.glob('*.conllu'))

            if conllu_files:
                print(f"  >> Encontrados {len(conllu_files)} arquivo(s) em: {input_dir}")
                self.conllu_files.extend(conllu_files)
            else:
                print(f"  AVISO: Nenhum arquivo .conllu encontrado em: {input_dir}")

        print(f"\nTotal de arquivos encontrados: {len(self.conllu_files)}\n")
        return self.conllu_files

    def process_file(self, file_path):
        """
        Processa um único arquivo CoNLL-U

        Args:
            file_path (Path): Caminho do arquivo

        Returns:
            dict: Estatísticas do arquivo ou None se houver erro
        """
        try:
            parser = ConlluDependencyGrammar(
                str(file_path),
                include_upos=self.include_upos,
                include_deprel=self.include_deprel,
                enhanced_only=self.enhanced_only,
            )

            parser.read_file()

            if not parser.sentences:
                print(f"  AVISO: Arquivo vazio ou sem sentenças: {file_path.name}")
                return None

            sentence_grammars = parser.extract_all_sentence_grammars()
            stats = parser.get_grammar_statistics(sentence_grammars)

            # Adiciona informação do arquivo
            stats['file_name'] = file_path.name
            stats['file_path'] = str(file_path)
            stats['file_id'] = str(file_path)  # Identificador único (caminho completo)

            return stats

        except Exception as e:
            print(f"  ERRO ao processar {file_path.name}: {e}")
            return None

    def process_all_files(self):
        """Processa todos os arquivos CoNLL-U encontrados"""
        if not self.conllu_files:
            print("Nenhum arquivo para processar!")
            return

        print("=" * 80)
        print(f"Processando {len(self.conllu_files)} arquivo(s)...")
        print("=" * 80)

        for i, file_path in enumerate(self.conllu_files, 1):
            print(f"\n[{i}/{len(self.conllu_files)}] Processando: {file_path.name}")

            stats = self.process_file(file_path)

            if stats:
                self.all_stats.append(stats)
                print(f"  Sentenças: {stats['total_sentences']}, "
                      f"Regras: {stats['total_rules']}, "
                      f"Únicas: {stats['unique_rules']}")

        print(f"\n{'='*80}")
        print(f"Processamento concluído: {len(self.all_stats)}/{len(self.conllu_files)} arquivo(s) processado(s) com sucesso")
        print("=" * 80)

    def calculate_aggregated_statistics(self):
        """Calcula estatísticas agregadas de todos os arquivos"""
        if not self.all_stats:
            print("Nenhuma estatística para agregar!")
            return

        print("\nCalculando estatísticas agregadas...")

        # Agrega frequências de regras de todos os arquivos
        aggregated_rule_freq = defaultdict(lambda: {'count': 0, 'is_root_rule': False, 'files': []})

        total_sentences = 0
        total_rules = 0

        for stats in self.all_stats:
            total_sentences += stats['total_sentences']
            total_rules += stats['total_rules']

            for rule, data in stats['rule_frequencies'].items():
                aggregated_rule_freq[rule]['count'] += data['count']
                # Verifica se a chave existe antes de acessar
                if 'is_root_rule' in data:
                    aggregated_rule_freq[rule]['is_root_rule'] = data['is_root_rule']

                # Adiciona info do arquivo usando file_id (caminho completo) para evitar duplicatas
                # Mesmo que arquivos tenham o mesmo nome em diretórios diferentes
                if stats['file_id'] not in aggregated_rule_freq[rule]['files']:
                    aggregated_rule_freq[rule]['files'].append(stats['file_id'])

        self.aggregated_stats = {
            'total_files': len(self.all_stats),
            'total_sentences': total_sentences,
            'total_rules': total_rules,
            'unique_rules': len(aggregated_rule_freq),
            'rule_frequencies': dict(aggregated_rule_freq),
            'per_file_stats': self.all_stats,
            'config': {
                'include_upos': self.include_upos,
                'include_deprel': self.include_deprel
            }
        }

        print(f"  Total de arquivos: {self.aggregated_stats['total_files']}")
        print(f"  Total de sentenças: {self.aggregated_stats['total_sentences']}")
        print(f"  Total de regras: {self.aggregated_stats['total_rules']}")
        print(f"  Regras únicas: {self.aggregated_stats['unique_rules']}")

        # Calcula médias
        avg_sentences = total_sentences / len(self.all_stats)
        avg_rules = total_rules / len(self.all_stats)
        avg_unique = sum(s['unique_rules'] for s in self.all_stats) / len(self.all_stats)

        self.aggregated_stats['averages'] = {
            'avg_sentences_per_file': avg_sentences,
            'avg_rules_per_file': avg_rules,
            'avg_unique_rules_per_file': avg_unique,
            'avg_rules_per_sentence': total_rules / total_sentences if total_sentences > 0 else 0
        }

        print(f"\n  Médias:")
        print(f"    - Sentenças por arquivo: {avg_sentences:.2f}")
        print(f"    - Regras por arquivo: {avg_rules:.2f}")
        print(f"    - Regras únicas por arquivo: {avg_unique:.2f}")
        print(f"    - Regras por sentença: {self.aggregated_stats['averages']['avg_rules_per_sentence']:.2f}")

    def export_aggregated_results(self, output_dir='aggregated_output'):
        """Exporta resultados agregados em múltiplos formatos"""
        if not self.aggregated_stats:
            print("Nenhuma estatística agregada para exportar!")
            return

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        print(f"\nExportando resultados para: {output_dir}/")

        # 1. Exporta JSON completo
        json_file = output_path / 'aggregated_grammar_statistics.json'
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(self.aggregated_stats, f, ensure_ascii=False, indent=2)
        print(f"  >> {json_file.name}")

        # 2. Exporta resumo em TXT
        txt_file = output_path / 'aggregated_grammar_summary.txt'
        self._write_text_summary(txt_file)
        print(f"  >> {txt_file.name}")

        # 3. Exporta apenas frequências de regras (ordenadas)
        freq_file = output_path / 'rule_frequencies.txt'
        self._write_rule_frequencies(freq_file)
        print(f"  >> {freq_file.name}")

        # 4. Exporta estatísticas por arquivo
        per_file_stats = output_path / 'per_file_statistics.txt'
        self._write_per_file_stats(per_file_stats)
        print(f"  >> {per_file_stats.name}")

    def _write_text_summary(self, output_file):
        """Escreve resumo em formato texto"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("RESUMO AGREGADO - EXTRAÇÃO DE REGRAS GRAMATICAIS\n")
            f.write("=" * 80 + "\n\n")

            f.write("CONFIGURAÇÃO:\n")
            f.write(f"  Include UPOS: {self.aggregated_stats['config']['include_upos']}\n")
            f.write(f"  Include DEPREL: {self.aggregated_stats['config']['include_deprel']}\n\n")

            f.write("ESTATÍSTICAS GERAIS:\n")
            f.write(f"  Total de arquivos processados: {self.aggregated_stats['total_files']}\n")
            f.write(f"  Total de sentenças: {self.aggregated_stats['total_sentences']}\n")
            f.write(f"  Total de regras: {self.aggregated_stats['total_rules']}\n")
            f.write(f"  Regras únicas: {self.aggregated_stats['unique_rules']}\n\n")

            f.write("MÉDIAS:\n")
            avg = self.aggregated_stats['averages']
            f.write(f"  Sentenças por arquivo: {avg['avg_sentences_per_file']:.2f}\n")
            f.write(f"  Regras por arquivo: {avg['avg_rules_per_file']:.2f}\n")
            f.write(f"  Regras únicas por arquivo: {avg['avg_unique_rules_per_file']:.2f}\n")
            f.write(f"  Regras por sentença: {avg['avg_rules_per_sentence']:.2f}\n\n")

            f.write("=" * 80 + "\n")

    def _write_rule_frequencies(self, output_file):
        """Escreve frequências de todas as regras"""
        total_sentences = self.aggregated_stats['total_sentences']

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 120 + "\n")
            f.write("FREQUÊNCIA DE REGRAS GRAMATICAIS (AGREGADAS)\n")
            f.write("=" * 120 + "\n")
            f.write(f"Total de regras únicas: {self.aggregated_stats['unique_rules']}\n")
            f.write("Ordenadas por frequência (decrescente)\n")
            f.write("=" * 120 + "\n\n")

            # Cabeçalho da tabela
            f.write(f"{'Regra':<70} {'Freq Total':>10}  {'Arquivos':>8}  {'Média/Sent':>11}\n")
            f.write("-" * 120 + "\n")

            sorted_rules = sorted(
                self.aggregated_stats['rule_frequencies'].items(),
                key=lambda x: x[1]['count'],
                reverse=True
            )

            for rule, data in sorted_rules:
                root_marker = " [ROOT]" if data['is_root_rule'] else ""
                num_files = len(data['files'])
                avg_per_sentence = data['count'] / total_sentences if total_sentences > 0 else 0
                f.write(f"{rule:<70} {data['count']:>10}  {num_files:>8}  {avg_per_sentence:>11.4f}{root_marker}\n")

            f.write("\n" + "=" * 120 + "\n")

    def _write_per_file_stats(self, output_file):
        """Escreve estatísticas individuais de cada arquivo"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("ESTATÍSTICAS POR ARQUIVO\n")
            f.write("=" * 80 + "\n\n")

            for i, stats in enumerate(self.all_stats, 1):
                f.write(f"\n[{i}] {stats['file_name']}\n")
                f.write("-" * 80 + "\n")
                f.write(f"  Sentenças: {stats['total_sentences']}\n")
                f.write(f"  Regras totais: {stats['total_rules']}\n")
                f.write(f"  Regras únicas: {stats['unique_rules']}\n")
                f.write(f"  Caminho: {stats['file_path']}\n")

            f.write("\n" + "=" * 80 + "\n")

    def process_and_save_individual_rules(self, output_dirs):
        """
        Processa arquivos CoNLL-U e salva as regras de cada arquivo individualmente.

        Para cada arquivo de entrada, cria um arquivo de saída correspondente
        contendo todas as regras extraídas daquele arquivo.

        Args:
            output_dirs (list or str): Lista de diretórios de saída correspondentes
                                       aos diretórios de entrada. Se for string única,
                                       usa o mesmo diretório para todos os inputs.

        Returns:
            dict: Resumo do processamento com arquivos criados
        """
        if not self.conllu_files:
            print("Nenhum arquivo para processar! Execute find_conllu_files() primeiro.")
            return None

        # Normaliza output_dirs para lista
        if isinstance(output_dirs, str):
            output_dirs = [output_dirs] * len(self.input_dirs)

        if len(output_dirs) != len(self.input_dirs):
            print(f"ERRO: Número de diretórios de saída ({len(output_dirs)}) "
                  f"deve ser igual ao de entrada ({len(self.input_dirs)})")
            return None

        # Cria mapeamento de diretório de entrada -> diretório de saída
        dir_mapping = {str(Path(inp).resolve()): out for inp, out in zip(self.input_dirs, output_dirs)}

        # Cria diretórios de saída se não existirem
        for out_dir in output_dirs:
            Path(out_dir).mkdir(parents=True, exist_ok=True)

        print("=" * 80)
        print("PROCESSAMENTO INDIVIDUAL DE ARQUIVOS")
        print("=" * 80)
        print(f"\nProcessando {len(self.conllu_files)} arquivo(s)...")
        print("Cada arquivo de entrada gerará um arquivo de saída correspondente.\n")

        processed_files = []
        failed_files = []

        for i, file_path in enumerate(self.conllu_files, 1):
            print(f"[{i}/{len(self.conllu_files)}] Processando: {file_path.name}")

            try:
                # Cria o parser e processa
                parser = ConlluDependencyGrammar(
                    str(file_path),
                    include_upos=self.include_upos,
                    include_deprel=self.include_deprel,
                    enhanced_only=self.enhanced_only,
                )
                parser.read_file()

                if not parser.sentences:
                    print(f"  AVISO: Arquivo vazio ou sem sentenças: {file_path.name}")
                    failed_files.append({'file': str(file_path), 'reason': 'empty'})
                    continue

                # Extrai gramáticas
                sentence_grammars = parser.extract_all_sentence_grammars()

                # Determina diretório de saída baseado no diretório de entrada
                input_parent = str(file_path.parent.resolve())
                output_dir = dir_mapping.get(input_parent, output_dirs[0])

                # Cria nome do arquivo de saída (mesmo nome, extensão .rules.json)
                output_filename = file_path.stem + '.rules.json'
                output_path = Path(output_dir) / output_filename

                # Extrai apenas as regras (lista de strings)
                all_rules = []
                for grammar in sentence_grammars:
                    for rule in grammar['rules']:
                        all_rules.append(rule['rule'])

                # Salva as regras em formato JSON
                output_data = {
                    'source_file': file_path.name,
                    'total_sentences': len(sentence_grammars),
                    'total_rules': len(all_rules),
                    'unique_rules': len(set(all_rules)),
                    'rules': all_rules,
                    'config': {
                        'include_upos': self.include_upos,
                        'include_deprel': self.include_deprel
                    }
                }

                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, ensure_ascii=False, indent=2)

                processed_files.append({
                    'input': str(file_path),
                    'output': str(output_path),
                    'sentences': len(sentence_grammars),
                    'rules': len(all_rules),
                    'unique_rules': len(set(all_rules))
                })

                print(f"  >> Salvo: {output_path}")
                print(f"     Sentenças: {len(sentence_grammars)}, "
                      f"Regras: {len(all_rules)}, "
                      f"Únicas: {len(set(all_rules))}")

            except Exception as e:
                print(f"  ERRO ao processar {file_path.name}: {e}")
                failed_files.append({'file': str(file_path), 'reason': str(e)})

        # Resumo final
        print(f"\n{'='*80}")
        print("RESUMO DO PROCESSAMENTO")
        print("=" * 80)
        print(f"Arquivos processados com sucesso: {len(processed_files)}")
        print(f"Arquivos com falha: {len(failed_files)}")

        if failed_files:
            print("\nArquivos com falha:")
            for f in failed_files:
                print(f"  - {f['file']}: {f['reason']}")

        return {
            'processed': processed_files,
            'failed': failed_files,
            'total_processed': len(processed_files),
            'total_failed': len(failed_files)
        }


def compare_human_vs_llm():
    """
    Compara estatísticas de regras gramaticais entre Human e LLM.
    Realiza testes estatísticos e exporta resultados.
    """
    # Diretório base (raiz do projeto)
    BASE_DIR = Path(__file__).resolve().parent.parent

    # CONFIGURAÇÃO
    HUMAN_DIRS = [
        BASE_DIR / 'portparser_results/fake_true_human',
        BASE_DIR / 'portparser_results/fake_br_human',
    ]

    LLM_DIRS = [
        BASE_DIR / 'portparser_results/fake_true_llm',
        BASE_DIR / 'portparser_results/fake_br_llm',
    ]

    OUTPUT_DIR = ED_RULES_DIR / 'comparacao_human_llm'
    INCLUDE_UPOS = True
    INCLUDE_DEPREL = True
    ENHANCED_ONLY = True

    # Banner
    print("=" * 80)
    print("COMPARACAO ESTATISTICA: HUMAN vs LLM")
    print("=" * 80)
    print(f"\nDiretorios Human: {[str(d) for d in HUMAN_DIRS]}")
    print(f"Diretorios LLM: {[str(d) for d in LLM_DIRS]}")

    # Processa grupo Human
    print("\n" + "=" * 80)
    print("PROCESSANDO GRUPO HUMAN")
    print("=" * 80)

    processor_human = GrammarBatchProcessor(
        input_dirs=[str(d) for d in HUMAN_DIRS],
        include_upos=INCLUDE_UPOS,
        include_deprel=INCLUDE_DEPREL,
        enhanced_only=ENHANCED_ONLY
    )
    processor_human.find_conllu_files()
    processor_human.process_all_files()

    if not processor_human.all_stats:
        print("ERRO: Nenhum arquivo Human processado!")
        return

    # Agrega estatísticas de todos os arquivos Human
    all_rules_per_sentence_human = []
    all_unique_rules_per_sentence_human = []

    for stats in processor_human.all_stats:
        if 'rules_per_sentence' in stats:
            all_rules_per_sentence_human.extend(stats['rules_per_sentence']['values'])
            all_unique_rules_per_sentence_human.extend(stats['unique_rules_per_sentence']['values'])

    # Processa grupo LLM
    print("\n" + "=" * 80)
    print("PROCESSANDO GRUPO LLM")
    print("=" * 80)

    processor_llm = GrammarBatchProcessor(
        input_dirs=[str(d) for d in LLM_DIRS],
        include_upos=INCLUDE_UPOS,
        include_deprel=INCLUDE_DEPREL,
        enhanced_only=ENHANCED_ONLY
    )
    processor_llm.find_conllu_files()
    processor_llm.process_all_files()

    if not processor_llm.all_stats:
        print("ERRO: Nenhum arquivo LLM processado!")
        return

    # Agrega estatísticas de todos os arquivos LLM
    all_rules_per_sentence_llm = []
    all_unique_rules_per_sentence_llm = []

    for stats in processor_llm.all_stats:
        if 'rules_per_sentence' in stats:
            all_rules_per_sentence_llm.extend(stats['rules_per_sentence']['values'])
            all_unique_rules_per_sentence_llm.extend(stats['unique_rules_per_sentence']['values'])

    # Cria estrutura de estatísticas agregadas para os testes
    import numpy as np

    stats_human_agregado = {
        'rules_per_sentence': {
            'mean': np.mean(all_rules_per_sentence_human) if all_rules_per_sentence_human else 0,
            'std': np.std(all_rules_per_sentence_human) if all_rules_per_sentence_human else 0,
            'values': all_rules_per_sentence_human
        },
        'unique_rules_per_sentence': {
            'mean': np.mean(all_unique_rules_per_sentence_human) if all_unique_rules_per_sentence_human else 0,
            'std': np.std(all_unique_rules_per_sentence_human) if all_unique_rules_per_sentence_human else 0,
            'values': all_unique_rules_per_sentence_human
        }
    }

    stats_llm_agregado = {
        'rules_per_sentence': {
            'mean': np.mean(all_rules_per_sentence_llm) if all_rules_per_sentence_llm else 0,
            'std': np.std(all_rules_per_sentence_llm) if all_rules_per_sentence_llm else 0,
            'values': all_rules_per_sentence_llm
        },
        'unique_rules_per_sentence': {
            'mean': np.mean(all_unique_rules_per_sentence_llm) if all_unique_rules_per_sentence_llm else 0,
            'std': np.std(all_unique_rules_per_sentence_llm) if all_unique_rules_per_sentence_llm else 0,
            'values': all_unique_rules_per_sentence_llm
        }
    }

    # Realiza testes estatísticos
    print("\n" + "=" * 80)
    print("TESTES ESTATISTICOS")
    print("=" * 80)

    resultados = realizar_testes_estatisticos(
        stats_human_agregado,
        stats_llm_agregado,
        "Human",
        "LLM"
    )

    # Imprime resultados
    imprimir_resultados_estatisticos(resultados)

    # Exporta resultados
    output_path = OUTPUT_DIR
    output_path.mkdir(parents=True, exist_ok=True)

    csv_file = output_path / 'testes_estatisticos_regras.csv'
    exportar_resultados_csv(resultados, str(csv_file))

    # Salva resumo geral
    summary_file = output_path / 'resumo_comparacao.txt'
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("RESUMO DA COMPARACAO HUMAN vs LLM\n")
        f.write("=" * 80 + "\n\n")

        f.write("DADOS PROCESSADOS:\n")
        f.write(f"  Arquivos Human: {len(processor_human.all_stats)}\n")
        f.write(f"  Arquivos LLM: {len(processor_llm.all_stats)}\n")
        f.write(f"  Sentencas Human: {len(all_rules_per_sentence_human)}\n")
        f.write(f"  Sentencas LLM: {len(all_rules_per_sentence_llm)}\n\n")

        f.write("METRICAS:\n")
        for metrica_key, dados in resultados['metricas'].items():
            f.write(f"\n  {dados['nome']}:\n")
            f.write(f"    Human: media = {dados['Human']['media']:.4f}, std = {dados['Human']['std']:.4f}\n")
            f.write(f"    LLM:   media = {dados['LLM']['media']:.4f}, std = {dados['LLM']['std']:.4f}\n")
            f.write(f"    Cohen's d: {dados['cohens_d']:.4f} ({dados['interpretacao_cohens_d']})\n")
            f.write(f"    p-value (t-test): {dados['t_test']['p_value']:.2e}\n")

        f.write("\n" + "=" * 80 + "\n")

    print(f"\nResultados salvos em: {str(OUTPUT_DIR)}/")
    print("  - testes_estatisticos_regras.csv")
    print("  - resumo_comparacao.txt")

    print("\n" + "=" * 80)
    print("COMPARACAO CONCLUIDA!")
    print("=" * 80)

    return resultados


def process_individual_files(enhanced_only=True, output_subdir="rules_output"):
    """
    Processa arquivos CoNLL-U individualmente, gerando um arquivo de regras
    para cada arquivo de entrada.

    Args:
        enhanced_only: se True, extrai só regras ED; se False, todas as regras
        output_subdir: nome do subdiretório de saída dentro de ed_rules/
    """

    BASE_DIR = Path(__file__).resolve().parent.parent

    # CONFIGURAÇÃO
    HUMAN_INPUT_DIRS = [
        str(BASE_DIR / 'portparser_results/fake_true_human'),
        str(BASE_DIR / 'portparser_results/fake_br_human'),
    ]

    LLM_INPUT_DIRS = [
        str(BASE_DIR / 'portparser_results/fake_true_llm'),
        str(BASE_DIR / 'portparser_results/fake_br_llm'),
    ]

    # Diretórios de saída correspondentes (um para cada input)
    HUMAN_OUTPUT_DIRS = [
        str(ED_RULES_DIR / output_subdir / 'fake_true_human'),
        str(ED_RULES_DIR / output_subdir / 'fake_br_human'),
    ]

    LLM_OUTPUT_DIRS = [
        str(ED_RULES_DIR / output_subdir / 'fake_true_llm'),
        str(ED_RULES_DIR / output_subdir / 'fake_br_llm'),
    ]

    INCLUDE_UPOS = True
    INCLUDE_DEPREL = True
    ENHANCED_ONLY = enhanced_only

    # Banner
    print("=" * 80)
    print("EXTRAÇÃO DE REGRAS INDIVIDUAIS - PROCESSAMENTO EM LOTE")
    print("=" * 80)
    print("\nConfiguração:")
    print(f"  Include UPOS: {INCLUDE_UPOS}")
    print(f"  Include DEPREL: {INCLUDE_DEPREL}")

    # Cria processador
    processor = GrammarBatchProcessor(
        input_dirs=HUMAN_INPUT_DIRS,
        include_upos=INCLUDE_UPOS,
        include_deprel=INCLUDE_DEPREL,
        enhanced_only=ENHANCED_ONLY
    )

    # Encontra arquivos
    files = processor.find_conllu_files()

    if not files:
        print("\nNenhum arquivo .conllu encontrado!")
        print("Verifique os diretórios de entrada e tente novamente.")
        return

    # Processa e salva regras individuais
    human_result = processor.process_and_save_individual_rules(HUMAN_OUTPUT_DIRS)

        # Cria processador
    processor = GrammarBatchProcessor(
        input_dirs=LLM_INPUT_DIRS,
        include_upos=INCLUDE_UPOS,
        include_deprel=INCLUDE_DEPREL,
        enhanced_only=ENHANCED_ONLY
    )

    # Encontra arquivos
    files = processor.find_conllu_files()

    if not files:
        print("\nNenhum arquivo .conllu encontrado!")
        print("Verifique os diretórios de entrada e tente novamente.")
        return

    # Processa e salva regras individuais
    llm_result = processor.process_and_save_individual_rules(LLM_OUTPUT_DIRS)

    if human_result and llm_result:
        print("\n" + "=" * 80)
        print("PROCESSAMENTO CONCLUÍDO!")
        print("=" * 80)
        print(f"\nArquivos de regras criados: {human_result['total_processed'] + llm_result['total_processed']}")


def main():
    """Função principal - processa e agrega estatísticas"""

    BASE_DIR = Path(__file__).resolve().parent.parent

    # CONFIGURAÇÃO
    HUMAN_INPUT_DIRS = [
        str(BASE_DIR / 'portparser_results/fake_true_human'),
        str(BASE_DIR / 'portparser_results/fake_br_human'),
    ]

    LLM_INPUT_DIRS = [
        str(BASE_DIR / 'portparser_results/fake_true_llm'),
        str(BASE_DIR / 'portparser_results/fake_br_llm'),
    ]

    HUMAN_OUTPUT_DIR = str(ED_RULES_DIR / 'comparacao_human_llm/human')
    LLM_OUTPUT_DIR = str(ED_RULES_DIR / 'comparacao_human_llm/llm')

    INCLUDE_UPOS = True
    INCLUDE_DEPREL = True
    ENHANCED_ONLY = True

    # Banner
    print("=" * 80)
    print("EXTRAÇÃO DE REGRAS GRAMATICAIS - PROCESSAMENTO EM LOTE")
    print("=" * 80)

    # Cria processador
    processor = GrammarBatchProcessor(
        input_dirs=HUMAN_INPUT_DIRS,
        include_upos=INCLUDE_UPOS,
        include_deprel=INCLUDE_DEPREL,
        enhanced_only=ENHANCED_ONLY
    )

    # Encontra arquivos
    files = processor.find_conllu_files()

    if not files:
        print("\nNenhum arquivo .conllu encontrado!")
        print("Verifique os diretórios de entrada e tente novamente.")
        return

    # Processa e salva regras individuais
    processor.process_all_files()

    # Calcula estatísticas agregadas
    processor.calculate_aggregated_statistics()

    # Exporta resultados
    processor.export_aggregated_results(HUMAN_OUTPUT_DIR)

    # Cria processador
    processor = GrammarBatchProcessor(
        input_dirs=LLM_INPUT_DIRS,
        include_upos=INCLUDE_UPOS,
        include_deprel=INCLUDE_DEPREL,
        enhanced_only=ENHANCED_ONLY
    )

    # Encontra arquivos
    files = processor.find_conllu_files()

    if not files:
        print("\nNenhum arquivo .conllu encontrado!")
        print("Verifique os diretórios de entrada e tente novamente.")
        return

    # Processa e salva regras individuais
    processor.process_all_files()

    if not processor.all_stats:
        print("\nNenhum arquivo foi processado com sucesso!")
        return

    # Calcula estatísticas agregadas
    processor.calculate_aggregated_statistics()

    # Exporta resultados
    processor.export_aggregated_results(LLM_OUTPUT_DIR)

    print("\n" + "=" * 80)
    print("PROCESSAMENTO CONCLUÍDO COM SUCESSO!")
    print("\n" + "=" * 80)


def run_tfidf_analysis(prefix="", rules_subdir="rules_output"):
    """Executa análise TF-IDF sobre os .rules.json gerados por process_individual_files().

    Args:
        prefix: prefixo para nomes de arquivos de saída (ex: "ed_only_", "all_rules_")
        rules_subdir: subdiretório onde estão os .rules.json
    """
    HUMAN_DIRS = [
        str(ED_RULES_DIR / rules_subdir / 'fake_true_human'),
        str(ED_RULES_DIR / rules_subdir / 'fake_br_human'),
    ]
    LLM_DIRS = [
        str(ED_RULES_DIR / rules_subdir / 'fake_true_llm'),
        str(ED_RULES_DIR / rules_subdir / 'fake_br_llm'),
    ]
    OUTPUT_DIR = str(ED_RULES_DIR / 'tfidf_output')
    MIN_DOCS = 1

    print("\n" + "=" * 80)
    print(f"ANÁLISE TF-IDF DE REGRAS GRAMATICAIS [{prefix.strip('_').upper() or 'DEFAULT'}]")
    print("Human vs LLM")
    print("=" * 80)

    analyzer = RulesTfidfAnalyzer(human_dirs=HUMAN_DIRS, llm_dirs=LLM_DIRS, prefix=prefix)
    analyzer.load_rules_files()

    if not analyzer.human_docs or not analyzer.llm_docs:
        print("\nERRO: Não foi possível carregar documentos de ambos os grupos!")
        return

    analyzer.calculate_tfidf(with_repetition=True)
    analyzer.analyze_discriminative_rules(mode='com_repeticao', min_docs=MIN_DOCS)

    analyzer.calculate_tfidf(with_repetition=False)
    analyzer.analyze_discriminative_rules(mode='sem_repeticao', min_docs=MIN_DOCS)

    analyzer.export_results(OUTPUT_DIR)

    print("\n" + "=" * 80)
    print(f"ANÁLISE TF-IDF [{prefix.strip('_').upper() or 'DEFAULT'}] CONCLUÍDA!")
    print(f"Resultados em: {OUTPUT_DIR}/")
    print("=" * 80)


if __name__ == "__main__":
    # 1a. Gera .rules.json individuais — só regras ED
    process_individual_files(enhanced_only=True, output_subdir="rules_output_ed")

    # 1b. Gera .rules.json individuais — todas as regras
    process_individual_files(enhanced_only=False, output_subdir="rules_output_all")

    # 2. Comparação estatística agregada (Human vs LLM)
    compare_human_vs_llm()

    # 3a. TF-IDF — só regras ED
    run_tfidf_analysis(prefix="ed_only_", rules_subdir="rules_output_ed")

    # 3b. TF-IDF — todas as regras
    run_tfidf_analysis(prefix="all_rules_", rules_subdir="rules_output_all")
