"""
Script para exportar regras gramaticais por sentença de arquivos CoNLL-U.

Para cada arquivo .conllu no diretório de entrada, gera um arquivo .txt
com o formato:

Sentença: <texto da sentença>
Regras:
  - regra1
  - regra2
  ...

Uso:
    python export_sentence_rules.py
"""

from pathlib import Path
from extract_grammar_rules import ConlluDependencyGrammar


def export_rules_for_file(input_file, output_file, include_upos=True, include_deprel=True):
    """
    Exporta as regras de cada sentença de um arquivo CoNLL-U.

    Args:
        input_file (str): Caminho do arquivo .conllu
        output_file (str): Caminho do arquivo de saída .txt
        include_upos (bool): Incluir UPOS nas regras
        include_deprel (bool): Incluir DEPREL nas regras
    """
    parser = ConlluDependencyGrammar(
        str(input_file),
        include_upos=include_upos,
        include_deprel=include_deprel
    )
    parser.read_file()
    sentence_grammars = parser.extract_all_sentence_grammars()

    with open(output_file, 'w', encoding='utf-8') as f:
        for i, sent_grammar in enumerate(sentence_grammars, 1):
            sentence_text = sent_grammar['sentence_text']
            sentence_id = sent_grammar.get('sentence_id', f'S{i}')

            # Extrai apenas as regras únicas (sem repetição na mesma sentença)
            rules = sorted(set(rule['rule'] for rule in sent_grammar['rules']))

            f.write(f"[{sentence_id}] Sentença: {sentence_text}\n")
            f.write("Regras:\n")
            for rule in rules:
                f.write(f"  - {rule}\n")
            f.write("\n")

    return len(sentence_grammars)


def process_directory(input_dir, output_dir, include_upos=True, include_deprel=True):
    """
    Processa todos os arquivos .conllu de um diretório.

    Args:
        input_dir (str): Diretório com arquivos .conllu
        output_dir (str): Diretório de saída para os arquivos .txt
        include_upos (bool): Incluir UPOS nas regras
        include_deprel (bool): Incluir DEPREL nas regras
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    conllu_files = sorted(input_path.glob('*.conllu'))

    print("=" * 80)
    print("EXPORTANDO REGRAS POR SENTENÇA")
    print("=" * 80)
    print(f"\nDiretório de entrada: {input_dir}")
    print(f"Diretório de saída: {output_dir}")
    print(f"Arquivos encontrados: {len(conllu_files)}")
    print(f"\nConfiguração:")
    print(f"  Include UPOS: {include_upos}")
    print(f"  Include DEPREL: {include_deprel}")
    print("\n" + "=" * 80)
    print("Processando...")
    print("=" * 80 + "\n")

    total_sentences = 0
    processed = 0
    errors = []

    for i, file_path in enumerate(conllu_files, 1):
        try:
            output_file = output_path / f"{file_path.stem}.txt"
            num_sentences = export_rules_for_file(
                file_path, output_file, include_upos, include_deprel
            )
            total_sentences += num_sentences
            processed += 1
            print(f"[{i}/{len(conllu_files)}] {file_path.name} -> {output_file.name} ({num_sentences} sentenças)")

        except Exception as e:
            errors.append((file_path.name, str(e)))
            print(f"[{i}/{len(conllu_files)}] ERRO: {file_path.name} - {e}")

    print("\n" + "=" * 80)
    print("RESUMO")
    print("=" * 80)
    print(f"Arquivos processados: {processed}/{len(conllu_files)}")
    print(f"Total de sentenças: {total_sentences}")
    if errors:
        print(f"Erros: {len(errors)}")
        for filename, error in errors:
            print(f"  - {filename}: {error}")
