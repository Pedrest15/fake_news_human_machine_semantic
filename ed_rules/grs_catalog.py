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
        # Grew usa `%` para comentário de linha. Não há string com `%` nas regras
        # do conjunto_regras_porttinari.grs, então um split simples basta.
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
    raise ValueError(f"Brace não fechada a partir de {open_idx}")


def _line_of(text: str, idx: int) -> int:
    return text.count("\n", 0, idx) + 1


def _extract_blocks(body: str) -> dict:
    """Dentro do corpo de uma regra, extrai pattern/without/commands brutos."""
    blocks = {"pattern": None, "without": [], "commands": None}
    i = 0
    n = len(body)
    while i < n:
        # pula whitespace
        while i < n and body[i].isspace():
            i += 1
        if i >= n:
            break
        # lê palavra-chave
        j = i
        while j < n and (body[j].isalpha() or body[j] == "_"):
            j += 1
        keyword = body[i:j]
        # encontra '{' após a palavra-chave
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
        # blocos desconhecidos são ignorados
        i = end + 1
    return blocks


def _comment_above(raw_text: str, decl_idx: int) -> str:
    """Retorna o bloco de comentário `% ...` imediatamente acima de decl_idx."""
    # caminha para trás até começo da linha da declaração
    line_start = raw_text.rfind("\n", 0, decl_idx) + 1
    # coleta linhas anteriores que começam com `%` (ignorando indentação)
    comments = []
    cursor = line_start - 1  # posição do '\n' anterior
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
    # Para localizar declarações usamos uma versão sem comentários, mas mantemos
    # o texto original para extrair os comentários acima de cada regra.
    clean = _strip_line_comments(raw)

    rules = []
    i = 0
    n = len(clean)
    package_stack = []  # lista de (nome, end_idx)

    while i < n:
        # encerra packages cujo escopo já passou
        while package_stack and i >= package_stack[-1][1]:
            package_stack.pop()

        # procura próxima palavra-chave relevante
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
            # strat NAME { ... } — pulamos
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
        print(f"ERRO: arquivo não encontrado: {src}")
        return 2
    rules = parse_grs(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(rules, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Catalogadas {len(rules)} regras de {src} -> {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
