"""
Parser de Dicionários LIWC (formato .dic)

Parseia o formato padrão de dicionários LIWC e categoriza palavras/textos
nas categorias psicolinguísticas definidas no dicionário.
"""

import re
from pathlib import Path


class LiwcDictionary:
    """
    Parser e categorizador de dicionários LIWC no formato .dic.

    O formato .dic consiste em:
    - Seção de categorias (entre linhas '%'): número TAB nome_categoria
    - Seção de palavras (após segunda '%'): palavra TAB cat1 TAB cat2 ...
    - Palavras terminadas em '*' são prefixos (match parcial)
    """

    def __init__(self, dic_path: str):
        self.dic_path = Path(dic_path)
        self.categories: dict[int, str] = {}
        self.exact_matches: dict[str, list[int]] = {}
        self.prefix_matches: list[tuple[str, list[int]]] = []

        self._parse()

    def _parse(self):
        """Parseia o arquivo .dic"""
        if not self.dic_path.exists():
            raise FileNotFoundError(
                f"Dicionário LIWC não encontrado: {self.dic_path}\n"
                "Dicionários LIWC são proprietários. Obtenha o dicionário "
                "LIWC em português (.dic) em https://www.liwc.app/ e coloque "
                f"o arquivo em: {self.dic_path}"
            )

        with open(self.dic_path, 'r', encoding='utf-8-sig') as f:
            lines = f.readlines()

        # Encontra as linhas com '%' que delimitam as seções
        pct_indices = [i for i, line in enumerate(lines) if line.strip() == '%']

        if len(pct_indices) < 2:
            raise ValueError(
                "Formato inválido: esperado pelo menos 2 linhas '%' no arquivo .dic"
            )

        # Seção de categorias: entre primeira e segunda '%'
        # Linhas podem ter tabs de indentação para indicar hierarquia
        for line in lines[pct_indices[0] + 1:pct_indices[1]]:
            line = line.strip()
            if not line:
                continue
            # Extrai o primeiro número e o nome da categoria
            # Formato: [tabs]número[tab]nome (parenteses)
            match = re.match(r'(\d+)\s+(.+)', line)
            if match:
                cat_id = int(match.group(1))
                cat_name = match.group(2).strip()
                self.categories[cat_id] = cat_name

        # Seção de palavras: após segunda '%'
        for line in lines[pct_indices[1] + 1:]:
            line = line.strip()
            if not line or line == '%':
                continue
            parts = line.split('\t')
            if len(parts) < 2:
                # Tenta split por espaços múltiplos
                parts = line.split()
            if len(parts) < 2:
                continue

            word = parts[0].strip().lower()
            cat_ids = []
            for p in parts[1:]:
                p = p.strip()
                if p:
                    try:
                        cat_ids.append(int(p))
                    except ValueError:
                        continue

            if not cat_ids:
                continue

            if word.endswith('*'):
                prefix = word[:-1]
                if prefix:
                    self.prefix_matches.append((prefix, cat_ids))
            else:
                self.exact_matches[word] = cat_ids

        # Ordena prefixos do mais longo para o mais curto (match mais específico primeiro)
        self.prefix_matches.sort(key=lambda x: len(x[0]), reverse=True)

        print(f"Dicionário LIWC carregado: {self.dic_path.name}")
        print(f"  Categorias: {len(self.categories)}")
        print(f"  Palavras exatas: {len(self.exact_matches)}")
        print(f"  Prefixos: {len(self.prefix_matches)}")

    def get_category_names(self) -> list[str]:
        """Retorna lista ordenada de nomes de categorias"""
        return [self.categories[k] for k in sorted(self.categories.keys())]

    def categorize_word(self, word: str) -> list[str]:
        """
        Retorna as categorias LIWC de uma palavra.

        Args:
            word: Palavra a categorizar (será convertida para minúsculas)

        Returns:
            Lista de nomes de categorias (pode ser vazia)
        """
        word = word.lower()

        # Tenta match exato primeiro
        if word in self.exact_matches:
            cat_ids = self.exact_matches[word]
            return [self.categories[cid] for cid in cat_ids if cid in self.categories]

        # Tenta match por prefixo
        for prefix, cat_ids in self.prefix_matches:
            if word.startswith(prefix):
                return [self.categories[cid] for cid in cat_ids if cid in self.categories]

        return []

    def categorize_text(self, text: str) -> tuple[dict[str, int], int, int]:
        """
        Tokeniza um texto e conta palavras por categoria LIWC.

        Args:
            text: Texto a categorizar

        Returns:
            Tupla (counts, total_words, matched_words):
                - counts: dict categoria -> contagem de palavras
                - total_words: total de palavras no texto
                - matched_words: palavras que matcharam alguma categoria
        """
        tokens = re.findall(r'[\w]+', text.lower())
        total_words = len(tokens)

        counts: dict[str, int] = {}
        matched_words = 0

        for token in tokens:
            cats = self.categorize_word(token)
            if cats:
                matched_words += 1
                for cat in cats:
                    counts[cat] = counts.get(cat, 0) + 1

        return counts, total_words, matched_words
