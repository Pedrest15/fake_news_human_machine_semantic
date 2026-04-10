"""
Aplica as regras Grew do eud-portugues a arquivos CoNLL-U.

Invoca `grew transform` diretamente (sem depender do Flask app.py).
Antes de rodar o Grew, limpa a coluna DEPS (col 9) de cada token.

Uso:
    python -m ed_rules.eud_runner <in_dir> <out_dir> [--strategy NAME]
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

DEFAULT_GRS = Path(__file__).resolve().parent / "conjunto_regras_porttinari.grs"
DEFAULT_STRATEGY = "eud_portuguese"


def _clean_deps_column(conllu_text: str) -> str:
    """Zera a coluna DEPS (col 9) para todos os tokens, como faz o app.py original."""
    lines = conllu_text.split("\n")
    for i, line in enumerate(lines):
        if "\t" in line:
            cols = line.split("\t")
            if len(cols) == 10:
                cols[8] = "_"
                lines[i] = "\t".join(cols)
    return "\n".join(lines)


def apply_eud(
    input_conllu: Path,
    output_conllu: Path,
    strategy: str = DEFAULT_STRATEGY,
    grs_path: Path = DEFAULT_GRS,
    timeout: int = 600,
) -> Path:
    """Aplica as regras EUD a um único .conllu, salvando em output_conllu."""
    input_conllu = Path(input_conllu)
    output_conllu = Path(output_conllu)
    if not input_conllu.exists():
        raise FileNotFoundError(input_conllu)
    if not grs_path.exists():
        raise FileNotFoundError(
            f"Arquivo .grs não encontrado: {grs_path}. "
            "Clone o eud-portugues em third_party/."
        )

    conllu_text = input_conllu.read_text(encoding="utf-8")
    cleaned = _clean_deps_column(conllu_text)

    output_conllu.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".conllu", delete=False, encoding="utf-8"
    ) as tmp_in:
        tmp_in.write(cleaned)
        tmp_in_path = tmp_in.name

    tmp_out_path = tmp_in_path.replace(".conllu", "_GREWED.conllu")
    try:
        cmd = (
            f"grew transform -config iwpt "
            f'-grs "{grs_path}" '
            f"-strat '{strategy}' "
            f"-i '{tmp_in_path}' "
            f"-o '{tmp_out_path}'"
        )
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            env=os.environ,
            timeout=timeout,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"grew transform falhou ({result.returncode}) em {input_conllu.name}:\n"
                f"STDERR:\n{result.stderr}"
            )
        if not os.path.exists(tmp_out_path):
            raise RuntimeError(
                f"grew transform não gerou saída para {input_conllu.name}.\n"
                f"STDOUT: {result.stdout[:500]}\nSTDERR: {result.stderr[:500]}"
            )
        out_text = Path(tmp_out_path).read_text(encoding="utf-8").strip() + "\n"
        if "\t" not in out_text:
            raise RuntimeError(
                f"Saída sem tokens para {input_conllu.name}.\n"
                f"Conteúdo (truncado): {out_text[:500]}"
            )
        output_conllu.write_text(out_text, encoding="utf-8")
    finally:
        for p in (tmp_in_path, tmp_out_path):
            if os.path.exists(p):
                os.remove(p)

    return output_conllu


def apply_eud_batch(
    input_dir: Path,
    output_dir: Path,
    strategy: str = DEFAULT_STRATEGY,
    grs_path: Path = DEFAULT_GRS,
    skip_existing: bool = True,
) -> list:
    """Processa todos os .conllu (não recursivo) de input_dir → output_dir."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    files = sorted(input_dir.glob("*.conllu"))
    if not files:
        print(f"AVISO: nenhum .conllu em {input_dir}")
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    error_log = output_dir / "_errors.log"
    processed = []
    errors = []
    for src in files:
        dst = output_dir / src.name
        if (
            skip_existing
            and dst.exists()
            and dst.stat().st_mtime >= src.stat().st_mtime
        ):
            print(f"  [skip] {src.name}")
            processed.append(dst)
            continue
        print(f"  [eud ] {src.name}")
        try:
            apply_eud(src, dst, strategy=strategy, grs_path=grs_path)
            processed.append(dst)
        except Exception as e:
            msg = f"{src.name}: {e}"
            print(f"  ERRO em {src.name}: {e}")
            errors.append(msg)

    if errors:
        error_log.write_text("\n".join(errors) + "\n", encoding="utf-8")
        print(f"\n  {len(errors)} erro(s) registrado(s) em {error_log}")

    return processed


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--strategy", default=DEFAULT_STRATEGY)
    parser.add_argument("--grs", type=Path, default=DEFAULT_GRS)
    parser.add_argument("--no-skip", action="store_true", help="Reprocessa tudo")
    args = parser.parse_args(argv[1:])

    if shutil.which("grew") is None:
        print(
            "AVISO: comando `grew` não encontrado no PATH. "
            "Instale via https://grew.fr/usage/install/ antes de prosseguir."
        )

    out = apply_eud_batch(
        args.input_dir,
        args.output_dir,
        strategy=args.strategy,
        grs_path=args.grs,
        skip_existing=not args.no_skip,
    )
    print(f"\n{len(out)} arquivo(s) processado(s) em {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
