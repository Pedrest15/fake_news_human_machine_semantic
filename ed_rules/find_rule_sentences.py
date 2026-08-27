"""
Busca sentenças que contêm uma determinada regra ED (Enhanced Dependency).

Dado uma regra gramatical (ex: "NOUN(*, NOUN/nmod:de)"), varre os arquivos
.conllu em ed_rules/portparser_results_eud/ e reporta, para cada ocorrência,
a sentença e o arquivo onde a regra aparece.

Uso:
    python find_rule_sentences.py "NOUN(*, NOUN/nmod:de)"
    python find_rule_sentences.py "VERB(*, VERB/advcl:porque)" --group fake_br_human
    python find_rule_sentences.py "PRON(*)" --limit 20 --output resultado.txt

Por padrão usa a mesma configuração do pipeline ED:
    enhanced_only=True, include_upos=True, include_deprel=True
"""

import argparse
import json
import sys
from pathlib import Path

from extract_grammar_rules import ConlluDependencyGrammar


ED_RULES_DIR = Path(__file__).resolve().parent
DEFAULT_EUD_DIR = ED_RULES_DIR / "portparser_results_eud"
DEFAULT_GROUPS = ["fake_br_human", "fake_br_llm", "fake_true_human", "fake_true_llm"]


def find_rule_in_file(file_path, target_rule, include_upos, include_deprel, enhanced_only):
    """
    Retorna lista de dicts {sentence_id, sentence_text, matches} para sentenças
    do arquivo que contêm `target_rule`. `matches` é o número de ocorrências.
    """
    parser = ConlluDependencyGrammar(
        str(file_path),
        include_upos=include_upos,
        include_deprel=include_deprel,
        enhanced_only=enhanced_only,
    )
    parser.read_file()
    sentence_grammars = parser.extract_all_sentence_grammars()

    hits = []
    for grammar in sentence_grammars:
        matches = sum(1 for r in grammar["rules"] if r["rule"] == target_rule)
        if matches > 0:
            hits.append({
                "sentence_id": grammar["sentence_id"],
                "sentence_text": grammar["sentence_text"],
                "matches": matches,
            })
    return hits


def search_rule(target_rule, eud_dir, groups, include_upos, include_deprel,
                enhanced_only, limit=None):
    """Varre todos os arquivos .conllu nos grupos indicados e junta os hits."""
    all_results = []
    total_files_scanned = 0

    for group in groups:
        group_dir = Path(eud_dir) / group
        if not group_dir.is_dir():
            print(f"[WARNING] directory not found: {group_dir}", file=sys.stderr)
            continue

        conllu_files = sorted(group_dir.glob("*.conllu"))
        for file_path in conllu_files:
            total_files_scanned += 1
            try:
                hits = find_rule_in_file(
                    file_path, target_rule,
                    include_upos=include_upos,
                    include_deprel=include_deprel,
                    enhanced_only=enhanced_only,
                )
            except Exception as e:
                print(f"[ERROR] {file_path}: {e}", file=sys.stderr)
                continue

            for h in hits:
                all_results.append({
                    "group": group,
                    "file_path": str(file_path),
                    "file_name": file_path.name,
                    "sentence_id": h["sentence_id"],
                    "sentence_text": h["sentence_text"],
                    "matches": h["matches"],
                })
                if limit is not None and len(all_results) >= limit:
                    return all_results, total_files_scanned

    return all_results, total_files_scanned


def format_text_report(results, target_rule, files_scanned):
    lines = []
    lines.append("=" * 100)
    lines.append(f"Rule searched: {target_rule}")
    lines.append(f".conllu files scanned: {files_scanned}")
    lines.append(f"Sentences with at least one hit: {len(results)}")
    lines.append(f"Total occurrences: {sum(r['matches'] for r in results)}")
    lines.append("=" * 100)

    if not results:
        lines.append("\nNo sentence found.")
        return "\n".join(lines)

    by_group = {}
    for r in results:
        by_group.setdefault(r["group"], []).append(r)

    for group, items in by_group.items():
        lines.append(f"\n[{group}] {len(items)} sentence(s)")
        lines.append("-" * 100)
        for r in items:
            lines.append(f"File:    {r['file_path']}")
            lines.append(f"Sent ID: {r['sentence_id']}  (occurrences: {r['matches']})")
            lines.append(f"Text:    {r['sentence_text']}")
            lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Find the sentences and files where an ED rule occurs.",
    )
    parser.add_argument("rule", help='ED rule to search for, e.g. "NOUN(*, NOUN/nmod:de)"')
    parser.add_argument(
        "--eud-dir", default=str(DEFAULT_EUD_DIR),
        help=f"Directory holding the per-group subfolders (default: {DEFAULT_EUD_DIR})",
    )
    parser.add_argument(
        "--group", action="append", choices=DEFAULT_GROUPS,
        help="Restrict to one group (may be repeated). Default: every group.",
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap the total number of sentences returned.")
    parser.add_argument("--output", default=None,
                        help="Output file (default: stdout).")
    parser.add_argument("--json", action="store_true",
                        help="Export JSON instead of plain text.")
    parser.add_argument("--no-enhanced-only", action="store_true",
                        help="Use every rule, not just the enhanced-only ED ones.")
    parser.add_argument("--no-upos", action="store_true",
                        help="Disable include_upos.")
    parser.add_argument("--no-deprel", action="store_true",
                        help="Disable include_deprel.")
    args = parser.parse_args()

    groups = args.group if args.group else DEFAULT_GROUPS

    results, files_scanned = search_rule(
        target_rule=args.rule,
        eud_dir=args.eud_dir,
        groups=groups,
        include_upos=not args.no_upos,
        include_deprel=not args.no_deprel,
        enhanced_only=not args.no_enhanced_only,
        limit=args.limit,
    )

    if args.json:
        payload = {
            "rule": args.rule,
            "files_scanned": files_scanned,
            "total_sentences": len(results),
            "total_matches": sum(r["matches"] for r in results),
            "results": results,
        }
        output = json.dumps(payload, ensure_ascii=False, indent=2)
    else:
        output = format_text_report(results, args.rule, files_scanned)

    if args.output:
        Path(args.output).write_text(output, encoding="utf-8")
        print(f"Result saved to: {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
