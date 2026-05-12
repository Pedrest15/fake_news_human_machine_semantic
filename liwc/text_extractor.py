"""
Extrator de Texto

Lê arquivos .txt com texto bruto do corpus para uso na análise LIWC.
"""

from pathlib import Path


class TextExtractor:
    """Extrai texto bruto de arquivos .txt do corpus."""

    @staticmethod
    def extract_text_from_file(file_path: str | Path) -> dict:
        """
        Lê o conteúdo completo de um arquivo .txt.

        Args:
            file_path: Caminho para o arquivo .txt

        Returns:
            Dict com: file_name, file_path, text
        """
        file_path = Path(file_path)

        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()

        return {
            'file_name': file_path.name,
            'file_path': str(file_path),
            'text': text,
        }

    @staticmethod
    def extract_texts_from_directory(dir_path: str | Path) -> list[dict]:
        """
        Lê textos de todos os arquivos .txt em um diretório.

        Args:
            dir_path: Caminho do diretório com arquivos .txt

        Returns:
            Lista de dicts com info de cada documento
        """
        dir_path = Path(dir_path)

        if not dir_path.exists():
            print(f"  AVISO: Diretório não encontrado: {dir_path}")
            return []

        files = sorted(dir_path.glob('*.txt'))
        docs = []

        for file_path in files:
            doc = TextExtractor.extract_text_from_file(file_path)
            if doc['text']:
                docs.append(doc)

        return docs
