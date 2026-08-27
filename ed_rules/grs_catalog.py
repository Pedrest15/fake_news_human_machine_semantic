"""
Cataloga as regras de um arquivo Grew (.grs) em JSON estruturado.

Extrai, para cada `rule` declarada (possivelmente dentro de `package`),
o nome, o pacote, e os blocos brutos de pattern/without/commands, junto
com o comentário (`% ...`) imediatamente acima e a linha de origem.

Uso:
    python -m ed_rules.grs_catalog <arquivo.grs> <saida.json>
"""
import json
import sys
from pathlib import Path


def _strip_line_comments(text: str) -> str:
    """Remove comentários `% ...` até o fim da linha, preservando '\\n'."""
    out = []
    for line in text.split("\n"):
        # Grew uses `%` for line comments. No rule in conjunto_regras_porttinari.grs
        # contains a `%` inside a string, so a plain split is enough.
        idx = line.find("%")
        if idx >= 0:
            line = line[:idx]
        out.append(line)
    return "\n".join(out)


def _find_matching_brace(text: str, open_idx: int) -> int:
    """Dado o índice de '{', retorna o índice do '}' correspondente."""
    assert text[open_idx] == "{"
    depth = 0
    for i in range(open_idx, len(text)):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i
    raise ValueError(f"Unclosed brace starting at {open_idx}")


def _line_of(text: str, idx: int) -> int:
    return text.count("\n", 0, idx) + 1


def _extract_blocks(body: str) -> dict:
    """Dentro do corpo de uma regra, extrai pattern/without/commands brutos."""
    blocks = {"pattern": None, "without": [], "commands": None}
    i = 0
    n = len(body)
    while i < n:
        # skip whitespace
        while i < n and body[i].isspace():
            i += 1
        if i >= n:
            break
        # read the keyword
        j = i
        while j < n and (body[j].isalpha() or body[j] == "_"):
            j += 1
        keyword = body[i:j]
        # find the '{' that follows the keyword
        k = body.find("{", j)
        if k == -1:
            break
        end = _find_matching_brace(body, k)
        block_text = body[k + 1 : end].strip()
        if keyword == "pattern":
            blocks["pattern"] = block_text
        elif keyword == "without":
            blocks["without"].append(block_text)
        elif keyword == "commands":
            blocks["commands"] = block_text
        # unknown blocks are ignored
        i = end + 1
    return blocks


def _comment_above(raw_text: str, decl_idx: int) -> str:
    """Retorna o bloco de comentário `% ...` imediatamente acima de decl_idx."""
    # walk backwards to the start of the declaration line
    line_start = raw_text.rfind("\n", 0, decl_idx) + 1
    # collect preceding lines that start with `%` (ignoring indentation)
    comments = []
    cursor = line_start - 1  # position of the previous '\n'
    while cursor > 0:
        prev_start = raw_text.rfind("\n", 0, cursor) + 1
        line = raw_text[prev_start:cursor]
        stripped = line.lstrip()
        if stripped.startswith("%"):
            comments.append(stripped.lstrip("%").strip())
            cursor = prev_start - 1
        else:
            break
    return "\n".join(reversed(comments)).strip()


def parse_grs(path: Path) -> list:
    """Parseia um .grs e retorna lista de dicts (uma entrada por regra)."""
    raw = path.read_text(encoding="utf-8")
    # Declarations are located on a comment-free copy, but the original text is
    # kept so the comment above each rule can still be extracted.
    clean = _strip_line_comments(raw)

    rules = []
    i = 0
    n = len(clean)
    package_stack = []  # list of (name, end_idx)

    while i < n:
        # close packages whose scope has ended
        while package_stack and i >= package_stack[-1][1]:
            package_stack.pop()

        # look for the next relevant keyword
        c = clean[i]
        if not (c.isalpha() or c == "_"):
            i += 1
            continue
        j = i
        while j < n and (clean[j].isalpha() or clean[j] == "_"):
            j += 1
        word = clean[i:j]

        if word == "package":
            # package NAME { ... }
            k = j
            while k < n and clean[k].isspace():
                k += 1
            name_start = k
            while k < n and (clean[k].isalnum() or clean[k] in "_-"):
                k += 1
            name = clean[name_start:k]
            brace = clean.find("{", k)
            if brace == -1:
                break
            end = _find_matching_brace(clean, brace)
            package_stack.append((name, end))
            i = brace + 1
            continue

        if word == "rule":
            k = j
            while k < n and clean[k].isspace():
                k += 1
            name_start = k
            while k < n and (clean[k].isalnum() or clean[k] in "_-"):
                k += 1
            name = clean[name_start:k]
            brace = clean.find("{", k)
            if brace == -1:
                break
            end = _find_matching_brace(clean, brace)
            body = clean[brace + 1 : end]
            blocks = _extract_blocks(body)
            rules.append(
                {
                    "name": name,
                    "package": package_stack[-1][0] if package_stack else None,
                    "pattern": blocks["pattern"],
                    "without": blocks["without"],
                    "commands": blocks["commands"],
                    "comment": _comment_above(raw, i),
                    "source_line": _line_of(raw, i),
                }
            )
            i = end + 1
            continue

        if word == "strat":
            # strat NAME { ... } — skipped
            brace = clean.find("{", j)
            if brace == -1:
                break
            end = _find_matching_brace(clean, brace)
            i = end + 1
            continue

        i = j

    return rules


def main(argv):
    if len(argv) != 3:
        print(__doc__)
        return 1
    src = Path(argv[1])
    dst = Path(argv[2])
    if not src.exists():
        print(f"ERROR: file not found: {src}")
        return 2
    rules = parse_grs(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(rules, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Catalogued {len(rules)} rules from {src} -> {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
