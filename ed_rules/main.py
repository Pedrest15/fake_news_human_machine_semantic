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
    run_statistical_tests,
    print_statistical_results,
    export_results_csv
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
        print("\nSearching for .conllu files...")

        for input_dir in self.input_dirs:
            input_path = Path(input_dir)

            if not input_path.exists():
                print(f"  WARNING: directory not found: {input_dir}")
                continue

            # Non-recursive search for .conllu files (root directory only)
            conllu_files = list(input_path.glob('*.conllu'))

            if conllu_files:
                print(f"  >> Found {len(conllu_files)} file(s) in: {input_dir}")
                self.conllu_files.extend(conllu_files)
            else:
                print(f"  WARNING: no .conllu file found in: {input_dir}")

        print(f"\nTotal files found: {len(self.conllu_files)}\n")
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
                print(f"  WARNING: empty file or no sentences: {file_path.name}")
                return None

            sentence_grammars = parser.extract_all_sentence_grammars()
            stats = parser.get_grammar_statistics(sentence_grammars)

            # Attach the file information
            stats['file_name'] = file_path.name
            stats['file_path'] = str(file_path)
            stats['file_id'] = str(file_path)  # Unique identifier (full path)

            return stats

        except Exception as e:
            print(f"  ERROR while processing {file_path.name}: {e}")
            return None

    def process_all_files(self):
        """Processa todos os arquivos CoNLL-U encontrados"""
        if not self.conllu_files:
            print("No files to process!")
            return

        print("=" * 80)
        print(f"Processing {len(self.conllu_files)} file(s)...")
        print("=" * 80)

        for i, file_path in enumerate(self.conllu_files, 1):
            print(f"\n[{i}/{len(self.conllu_files)}] Processing: {file_path.name}")

            stats = self.process_file(file_path)

            if stats:
                self.all_stats.append(stats)
                print(f"  Sentences: {stats['total_sentences']}, "
                      f"Rules: {stats['total_rules']}, "
                      f"Unique: {stats['unique_rules']}")

        print(f"\n{'='*80}")
        print(f"Processing finished: {len(self.all_stats)}/{len(self.conllu_files)} file(s) processed successfully")
        print("=" * 80)

    def calculate_aggregated_statistics(self):
        """Calcula estatísticas agregadas de todos os arquivos"""
        if not self.all_stats:
            print("No statistics to aggregate!")
            return

        print("\nComputing aggregated statistics...")

        # Aggregate the rule frequencies across every file
        aggregated_rule_freq = defaultdict(lambda: {'count': 0, 'is_root_rule': False, 'files': []})

        total_sentences = 0
        total_rules = 0

        for stats in self.all_stats:
            total_sentences += stats['total_sentences']
            total_rules += stats['total_rules']

            for rule, data in stats['rule_frequencies'].items():
                aggregated_rule_freq[rule]['count'] += data['count']
                # Make sure the key exists before reading it
                if 'is_root_rule' in data:
                    aggregated_rule_freq[rule]['is_root_rule'] = data['is_root_rule']

                # Track the file by file_id (full path) to avoid duplicates even
                # when two files share the same name in different directories
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

        print(f"  Total files: {self.aggregated_stats['total_files']}")
        print(f"  Total sentences: {self.aggregated_stats['total_sentences']}")
        print(f"  Total rules: {self.aggregated_stats['total_rules']}")
        print(f"  Unique rules: {self.aggregated_stats['unique_rules']}")

        # Compute the averages
        avg_sentences = total_sentences / len(self.all_stats)
        avg_rules = total_rules / len(self.all_stats)
        avg_unique = sum(s['unique_rules'] for s in self.all_stats) / len(self.all_stats)

        self.aggregated_stats['averages'] = {
            'avg_sentences_per_file': avg_sentences,
            'avg_rules_per_file': avg_rules,
            'avg_unique_rules_per_file': avg_unique,
            'avg_rules_per_sentence': total_rules / total_sentences if total_sentences > 0 else 0
        }

        print(f"\n  Averages:")
        print(f"    - Sentences per file: {avg_sentences:.2f}")
        print(f"    - Rules per file: {avg_rules:.2f}")
        print(f"    - Unique rules per file: {avg_unique:.2f}")
        print(f"    - Rules per sentence: {self.aggregated_stats['averages']['avg_rules_per_sentence']:.2f}")

    def export_aggregated_results(self, output_dir='aggregated_output'):
        """Exporta resultados agregados em múltiplos formatos"""
        if not self.aggregated_stats:
            print("No aggregated statistics to export!")
            return

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        print(f"\nExporting results to: {output_dir}/")

        # 1. Full JSON export
        json_file = output_path / 'aggregated_grammar_statistics.json'
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(self.aggregated_stats, f, ensure_ascii=False, indent=2)
        print(f"  >> {json_file.name}")

        # 2. Plain-text summary
        txt_file = output_path / 'aggregated_grammar_summary.txt'
        self._write_text_summary(txt_file)
        print(f"  >> {txt_file.name}")

        # 3. Rule frequencies only (sorted)
        freq_file = output_path / 'rule_frequencies.txt'
        self._write_rule_frequencies(freq_file)
        print(f"  >> {freq_file.name}")

        # 4. Per-file statistics
        per_file_stats = output_path / 'per_file_statistics.txt'
        self._write_per_file_stats(per_file_stats)
        print(f"  >> {per_file_stats.name}")

    def _write_text_summary(self, output_file):
        """Escreve resumo em formato texto"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("AGGREGATED SUMMARY - GRAMMAR RULE EXTRACTION\n")
            f.write("=" * 80 + "\n\n")

            f.write("CONFIGURATION:\n")
            f.write(f"  Include UPOS: {self.aggregated_stats['config']['include_upos']}\n")
            f.write(f"  Include DEPREL: {self.aggregated_stats['config']['include_deprel']}\n\n")

            f.write("OVERALL STATISTICS:\n")
            f.write(f"  Total files processed: {self.aggregated_stats['total_files']}\n")
            f.write(f"  Total sentences: {self.aggregated_stats['total_sentences']}\n")
            f.write(f"  Total rules: {self.aggregated_stats['total_rules']}\n")
            f.write(f"  Unique rules: {self.aggregated_stats['unique_rules']}\n\n")

            f.write("AVERAGES:\n")
            avg = self.aggregated_stats['averages']
            f.write(f"  Sentences per file: {avg['avg_sentences_per_file']:.2f}\n")
            f.write(f"  Rules per file: {avg['avg_rules_per_file']:.2f}\n")
            f.write(f"  Unique rules per file: {avg['avg_unique_rules_per_file']:.2f}\n")
            f.write(f"  Rules per sentence: {avg['avg_rules_per_sentence']:.2f}\n\n")

            f.write("=" * 80 + "\n")

    def _write_rule_frequencies(self, output_file):
        """Escreve frequências de todas as regras"""
        total_sentences = self.aggregated_stats['total_sentences']

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 120 + "\n")
            f.write("GRAMMAR RULE FREQUENCIES (AGGREGATED)\n")
            f.write("=" * 120 + "\n")
            f.write(f"Total unique rules: {self.aggregated_stats['unique_rules']}\n")
            f.write("Sorted by frequency (descending)\n")
            f.write("=" * 120 + "\n\n")

            # Table header
            f.write(f"{'Rule':<70} {'Total freq':>10}  {'Files':>8}  {'Mean/sent':>11}\n")
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
            f.write("PER-FILE STATISTICS\n")
            f.write("=" * 80 + "\n\n")

            for i, stats in enumerate(self.all_stats, 1):
                f.write(f"\n[{i}] {stats['file_name']}\n")
                f.write("-" * 80 + "\n")
                f.write(f"  Sentences: {stats['total_sentences']}\n")
                f.write(f"  Total rules: {stats['total_rules']}\n")
                f.write(f"  Unique rules: {stats['unique_rules']}\n")
                f.write(f"  Path: {stats['file_path']}\n")

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
            print("No files to process! Run find_conllu_files() first.")
            return None

        # Normalise output_dirs to a list
        if isinstance(output_dirs, str):
            output_dirs = [output_dirs] * len(self.input_dirs)

        if len(output_dirs) != len(self.input_dirs):
            print(f"ERROR: the number of output directories ({len(output_dirs)}) "
                  f"must match the number of input directories ({len(self.input_dirs)})")
            return None

        # Map each input directory to its output directory
        dir_mapping = {str(Path(inp).resolve()): out for inp, out in zip(self.input_dirs, output_dirs)}

        # Create the output directories when missing
        for out_dir in output_dirs:
            Path(out_dir).mkdir(parents=True, exist_ok=True)

        print("=" * 80)
        print("PER-FILE PROCESSING")
        print("=" * 80)
        print(f"\nProcessing {len(self.conllu_files)} file(s)...")
        print("Every input file produces one matching output file.\n")

        processed_files = []
        failed_files = []

        for i, file_path in enumerate(self.conllu_files, 1):
            print(f"[{i}/{len(self.conllu_files)}] Processing: {file_path.name}")

            try:
                # Build the parser and run it
                parser = ConlluDependencyGrammar(
                    str(file_path),
                    include_upos=self.include_upos,
                    include_deprel=self.include_deprel,
                    enhanced_only=self.enhanced_only,
                )
                parser.read_file()

                if not parser.sentences:
                    print(f"  WARNING: empty file or no sentences: {file_path.name}")
                    failed_files.append({'file': str(file_path), 'reason': 'empty'})
                    continue

                # Extract the grammars
                sentence_grammars = parser.extract_all_sentence_grammars()

                # Pick the output directory from the input directory
                input_parent = str(file_path.parent.resolve())
                output_dir = dir_mapping.get(input_parent, output_dirs[0])

                # Build the output file name (same stem, .rules.json extension)
                output_filename = file_path.stem + '.rules.json'
                output_path = Path(output_dir) / output_filename

                # Keep only the rules themselves (a list of strings)
                all_rules = []
                for grammar in sentence_grammars:
                    for rule in grammar['rules']:
                        all_rules.append(rule['rule'])

                # Write the rules out as JSON
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

                print(f"  >> Saved: {output_path}")
                print(f"     Sentences: {len(sentence_grammars)}, "
                      f"Rules: {len(all_rules)}, "
                      f"Unique: {len(set(all_rules))}")

            except Exception as e:
                print(f"  ERROR while processing {file_path.name}: {e}")
                failed_files.append({'file': str(file_path), 'reason': str(e)})

        # Final summary
        print(f"\n{'='*80}")
        print("PROCESSING SUMMARY")
        print("=" * 80)
        print(f"Files processed successfully: {len(processed_files)}")
        print(f"Files that failed: {len(failed_files)}")

        if failed_files:
            print("\nFailed files:")
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
    # Base directory (project root)
    BASE_DIR = Path(__file__).resolve().parent.parent

    # CONFIGURATION
    HUMAN_DIRS = [
        BASE_DIR / 'portparser_results/fake_true_human',
        BASE_DIR / 'portparser_results/fake_br_human',
    ]

    LLM_DIRS = [
        BASE_DIR / 'portparser_results/fake_true_llm',
        BASE_DIR / 'portparser_results/fake_br_llm',
    ]

    OUTPUT_DIR = ED_RULES_DIR / 'human_llm_comparison'
    INCLUDE_UPOS = True
    INCLUDE_DEPREL = True
    ENHANCED_ONLY = True

    # Banner
    print("=" * 80)
    print("STATISTICAL COMPARISON: HUMAN vs LLM")
    print("=" * 80)
    print(f"\nHuman directories: {[str(d) for d in HUMAN_DIRS]}")
    print(f"LLM directories: {[str(d) for d in LLM_DIRS]}")

    # Process the Human group
    print("\n" + "=" * 80)
    print("PROCESSING HUMAN GROUP")
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
        print("ERROR: no Human file was processed!")
        return

    # Aggregate the statistics across every Human file
    all_rules_per_sentence_human = []
    all_unique_rules_per_sentence_human = []

    for stats in processor_human.all_stats:
        if 'rules_per_sentence' in stats:
            all_rules_per_sentence_human.extend(stats['rules_per_sentence']['values'])
            all_unique_rules_per_sentence_human.extend(stats['unique_rules_per_sentence']['values'])

    # Process the LLM group
    print("\n" + "=" * 80)
    print("PROCESSING LLM GROUP")
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
        print("ERROR: no LLM file was processed!")
        return

    # Aggregate the statistics across every LLM file
    all_rules_per_sentence_llm = []
    all_unique_rules_per_sentence_llm = []

    for stats in processor_llm.all_stats:
        if 'rules_per_sentence' in stats:
            all_rules_per_sentence_llm.extend(stats['rules_per_sentence']['values'])
            all_unique_rules_per_sentence_llm.extend(stats['unique_rules_per_sentence']['values'])

    # Build the aggregated statistics structure consumed by the tests
    import numpy as np

    stats_human_aggregated = {
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

    stats_llm_aggregated = {
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

    # Run the statistical tests
    print("\n" + "=" * 80)
    print("STATISTICAL TESTS")
    print("=" * 80)

    results = run_statistical_tests(
        stats_human_aggregated,
        stats_llm_aggregated,
        "Human",
        "LLM"
    )

    # Print the results
    print_statistical_results(results)

    # Export the results
    output_path = OUTPUT_DIR
    output_path.mkdir(parents=True, exist_ok=True)

    csv_file = output_path / 'rule_statistical_tests.csv'
    export_results_csv(results, str(csv_file))

    # Write the overall summary
    summary_file = output_path / 'comparison_summary.txt'
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("HUMAN vs LLM COMPARISON SUMMARY\n")
        f.write("=" * 80 + "\n\n")

        f.write("DATA PROCESSED:\n")
        f.write(f"  Human files: {len(processor_human.all_stats)}\n")
        f.write(f"  LLM files: {len(processor_llm.all_stats)}\n")
        f.write(f"  Human sentences: {len(all_rules_per_sentence_human)}\n")
        f.write(f"  LLM sentences: {len(all_rules_per_sentence_llm)}\n\n")

        f.write("METRICS:\n")
        for metric_key, data in results['metrics'].items():
            f.write(f"\n  {data['name']}:\n")
            f.write(f"    Human: mean = {data['Human']['mean']:.4f}, std = {data['Human']['std']:.4f}\n")
            f.write(f"    LLM:   mean = {data['LLM']['mean']:.4f}, std = {data['LLM']['std']:.4f}\n")
            f.write(f"    Cohen's d: {data['cohens_d']:.4f} ({data['cohens_d_interpretation']})\n")
            f.write(f"    p-value (t-test): {data['t_test']['p_value']:.2e}\n")

        f.write("\n" + "=" * 80 + "\n")

    print(f"\nResults saved to: {str(OUTPUT_DIR)}/")
    print("  - rule_statistical_tests.csv")
    print("  - comparison_summary.txt")

    print("\n" + "=" * 80)
    print("COMPARISON COMPLETE!")
    print("=" * 80)

    return results


def process_individual_files(enhanced_only=True, output_subdir="rules_output"):
    """
    Processa arquivos CoNLL-U individualmente, gerando um arquivo de regras
    para cada arquivo de entrada.

    Args:
        enhanced_only: se True, extrai só regras ED; se False, todas as regras
        output_subdir: nome do subdiretório de saída dentro de ed_rules/
    """

    BASE_DIR = Path(__file__).resolve().parent.parent

    # CONFIGURATION
    HUMAN_INPUT_DIRS = [
        str(BASE_DIR / 'portparser_results/fake_true_human'),
        str(BASE_DIR / 'portparser_results/fake_br_human'),
    ]

    LLM_INPUT_DIRS = [
        str(BASE_DIR / 'portparser_results/fake_true_llm'),
        str(BASE_DIR / 'portparser_results/fake_br_llm'),
    ]

    # Matching output directories (one per input directory)
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
    print("PER-FILE RULE EXTRACTION - BATCH PROCESSING")
    print("=" * 80)
    print("\nConfiguration:")
    print(f"  Include UPOS: {INCLUDE_UPOS}")
    print(f"  Include DEPREL: {INCLUDE_DEPREL}")

    # Build the processor
    processor = GrammarBatchProcessor(
        input_dirs=HUMAN_INPUT_DIRS,
        include_upos=INCLUDE_UPOS,
        include_deprel=INCLUDE_DEPREL,
        enhanced_only=ENHANCED_ONLY
    )

    # Find the files
    files = processor.find_conllu_files()

    if not files:
        print("\nNo .conllu file found!")
        print("Check the input directories and try again.")
        return

    # Process and save the per-file rules
    human_result = processor.process_and_save_individual_rules(HUMAN_OUTPUT_DIRS)

        # Build the processor
    processor = GrammarBatchProcessor(
        input_dirs=LLM_INPUT_DIRS,
        include_upos=INCLUDE_UPOS,
        include_deprel=INCLUDE_DEPREL,
        enhanced_only=ENHANCED_ONLY
    )

    # Find the files
    files = processor.find_conllu_files()

    if not files:
        print("\nNo .conllu file found!")
        print("Check the input directories and try again.")
        return

    # Process and save the per-file rules
    llm_result = processor.process_and_save_individual_rules(LLM_OUTPUT_DIRS)

    if human_result and llm_result:
        print("\n" + "=" * 80)
        print("PROCESSING COMPLETE!")
        print("=" * 80)
        print(f"\nRule files created: {human_result['total_processed'] + llm_result['total_processed']}")


def main():
    """Função principal - processa e agrega estatísticas"""

    BASE_DIR = Path(__file__).resolve().parent.parent

    # CONFIGURATION
    HUMAN_INPUT_DIRS = [
        str(BASE_DIR / 'portparser_results/fake_true_human'),
        str(BASE_DIR / 'portparser_results/fake_br_human'),
    ]

    LLM_INPUT_DIRS = [
        str(BASE_DIR / 'portparser_results/fake_true_llm'),
        str(BASE_DIR / 'portparser_results/fake_br_llm'),
    ]

    HUMAN_OUTPUT_DIR = str(ED_RULES_DIR / 'human_llm_comparison/human')
    LLM_OUTPUT_DIR = str(ED_RULES_DIR / 'human_llm_comparison/llm')

    INCLUDE_UPOS = True
    INCLUDE_DEPREL = True
    ENHANCED_ONLY = True

    # Banner
    print("=" * 80)
    print("GRAMMAR RULE EXTRACTION - BATCH PROCESSING")
    print("=" * 80)

    # Build the processor
    processor = GrammarBatchProcessor(
        input_dirs=HUMAN_INPUT_DIRS,
        include_upos=INCLUDE_UPOS,
        include_deprel=INCLUDE_DEPREL,
        enhanced_only=ENHANCED_ONLY
    )

    # Find the files
    files = processor.find_conllu_files()

    if not files:
        print("\nNo .conllu file found!")
        print("Check the input directories and try again.")
        return

    # Process and save the per-file rules
    processor.process_all_files()

    # Compute the aggregated statistics
    processor.calculate_aggregated_statistics()

    # Export the results
    processor.export_aggregated_results(HUMAN_OUTPUT_DIR)

    # Build the processor
    processor = GrammarBatchProcessor(
        input_dirs=LLM_INPUT_DIRS,
        include_upos=INCLUDE_UPOS,
        include_deprel=INCLUDE_DEPREL,
        enhanced_only=ENHANCED_ONLY
    )

    # Find the files
    files = processor.find_conllu_files()

    if not files:
        print("\nNo .conllu file found!")
        print("Check the input directories and try again.")
        return

    # Process and save the per-file rules
    processor.process_all_files()

    if not processor.all_stats:
        print("\nNo file was processed successfully!")
        return

    # Compute the aggregated statistics
    processor.calculate_aggregated_statistics()

    # Export the results
    processor.export_aggregated_results(LLM_OUTPUT_DIR)

    print("\n" + "=" * 80)
    print("PROCESSING COMPLETED SUCCESSFULLY!")
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
    print(f"TF-IDF ANALYSIS OF GRAMMAR RULES [{prefix.strip('_').upper() or 'DEFAULT'}]")
    print("Human vs LLM")
    print("=" * 80)

    analyzer = RulesTfidfAnalyzer(human_dirs=HUMAN_DIRS, llm_dirs=LLM_DIRS, prefix=prefix)
    analyzer.load_rules_files()

    if not analyzer.human_docs or not analyzer.llm_docs:
        print("\nERROR: could not load documents from both groups!")
        return

    analyzer.calculate_tfidf(with_repetition=True)
    analyzer.analyze_discriminative_rules(mode='with_repetition', min_docs=MIN_DOCS)

    analyzer.calculate_tfidf(with_repetition=False)
    analyzer.analyze_discriminative_rules(mode='without_repetition', min_docs=MIN_DOCS)

    analyzer.export_results(OUTPUT_DIR)

    print("\n" + "=" * 80)
    print(f"TF-IDF ANALYSIS [{prefix.strip('_').upper() or 'DEFAULT'}] COMPLETE!")
    print(f"Results in: {OUTPUT_DIR}/")
    print("=" * 80)


if __name__ == "__main__":
    # 1a. Emit per-file .rules.json — ED rules only
    process_individual_files(enhanced_only=True, output_subdir="rules_output_ed")

    # 1b. Emit per-file .rules.json — every rule
    process_individual_files(enhanced_only=False, output_subdir="rules_output_all")

    # 2. Aggregated statistical comparison (Human vs LLM)
    compare_human_vs_llm()

    # 3a. TF-IDF — ED rules only
    run_tfidf_analysis(prefix="ed_only_", rules_subdir="rules_output_ed")

    # 3b. TF-IDF — every rule
    run_tfidf_analysis(prefix="all_rules_", rules_subdir="rules_output_all")
