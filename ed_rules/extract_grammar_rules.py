import numpy as np
from scipy import stats as scipy_stats


class ConlluDependencyGrammar:
    """
    Extrator de regras de gramatica de dependencias de arquivos CoNLL-U
    """

    def __init__(self, filepath: str, include_upos: bool = True, include_deprel: bool = True,
                 enhanced_only: bool = False) -> None:
        """
        Args:
            filepath (str): caminho do arquivo a ser processado.
            include_upos (bool): incluir UPOS nas regras com dependentes (default: True)
            include_deprel (bool): incluir DEPREL nas regras com dependentes (default: False)
            enhanced_only (bool): se True, extrai regras apenas das relações enhanced
                (col DEPS) que diferem das básicas (col HEAD/DEPREL). Requer arquivos
                processados com EUD (col 9 preenchida).

        Exemplos de formato conforme flags:
            - include_upos=True, include_deprel=True:  PROPN(*, PROPN/flat:name, PROPN/flat:name) + PROPN(*)
            - include_upos=True, include_deprel=False: PROPN(*, PROPN, PROPN) + PROPN(*)
            - include_upos=False, include_deprel=True: _(*, flat:name, flat:name) + _(*)
            - include_upos=False, include_deprel=False: _(*, _/_, _/_) + _(*)

        NOTA: As regras folha seguem o mesmo formato das regras com dependentes conforme as flags
        """
        self.filepath = filepath
        self.sentences = []
        self.include_upos = include_upos
        self.include_deprel = include_deprel
        self.enhanced_only = enhanced_only
        
    def read_file(self):
        """Le o arquivo CoNLL-U e separa as sentencas"""
        with open(self.filepath, 'r', encoding='utf-8') as f:
            current_sentence = []
            sentence_text = ""
            sentence_id = ""
            
            for line in f:
                line = line.strip()
                
                if line.startswith("# sent_id"):
                    sentence_id = line.split("=")[1].strip()
                elif line.startswith("# text"):
                    sentence_text = line.split("=")[1].strip()
                elif not line:
                    if current_sentence:
                        self.sentences.append({
                            'id': sentence_id,
                            'text': sentence_text,
                            'tokens': current_sentence
                        })
                        current_sentence = []
                        sentence_text = ""
                        sentence_id = ""
                # Ignora comentarios
                elif line.startswith("#"):
                    continue
                else:
                    # Ignora contracoes, multi-word tokens e empty nodes
                    token_id = line.split('\t')[0]
                    if '-' not in token_id and '.' not in token_id and '_' not in token_id:
                        current_sentence.append(self.parse_token(line))
            
            if current_sentence:
                self.sentences.append({
                    'id': sentence_id,
                    'text': sentence_text,
                    'tokens': current_sentence
                })
    
    def parse_token(self, line):
        """Parse de uma linha de token"""
        fields = line.split('\t')
        return {
            'id': int(fields[0]),
            'form': fields[1],
            'lemma': fields[2],
            'upos': fields[3],
            'xpos': fields[4],
            'feats': fields[5],
            'head': int(fields[6]),
            'deprel': fields[7],
            'deps': fields[8],
            'misc': fields[9]
        }
    
    def extract_all_sentence_grammars(self):
        """Extrai regras de todas as sentencas, mantendo-as agrupadas"""
        all_sentence_grammars = []
        
        for sentence in self.sentences:
            grammar = self.get_sentence_grammar(sentence)
            all_sentence_grammars.append(grammar)
        
        return all_sentence_grammars
    
    def _parse_enhanced_deps(self, deps_str):
        """Parseia a coluna DEPS (col 9) em lista de (head_id, deprel).

        Formato: '2:nsubj|6:nmod:de' -> [(2, 'nsubj'), (6, 'nmod:de')]
        """
        if not deps_str or deps_str == '_':
            return []
        result = []
        for part in deps_str.split('|'):
            head_str, *rel_parts = part.split(':')
            try:
                head_id = int(float(head_str))
            except ValueError:
                continue
            deprel = ':'.join(rel_parts) if rel_parts else '_'
            result.append((head_id, deprel))
        return result

    def _get_enhanced_only_edges(self, tokens):
        """Retorna apenas arestas enhanced que diferem das básicas.

        Para cada token, compara DEPS (col 9) com HEAD:DEPREL (cols 7-8).
        Uma aresta enhanced é considerada nova se:
          - aponta para um head diferente, OU
          - tem um deprel diferente (ex: 'nmod:de' vs 'nmod')
        """
        edges = []  # (dep_token, head_id, deprel)
        for token in tokens:
            basic_head = token['head']
            basic_deprel = token['deprel']
            enhanced = self._parse_enhanced_deps(token['deps'])
            for head_id, deprel in enhanced:
                # Aresta é "nova" se difere do básico
                if head_id != basic_head or deprel != basic_deprel:
                    edges.append((token, head_id, deprel))
        return edges

    def get_sentence_grammar(self, sentence):
        """Extrai a gramática completa de uma sentença"""
        tokens = sentence['tokens']
        rules = []
        root_token = None

        # Primeiro, identifica a raiz
        for token in tokens:
            if token['head'] == 0:
                root_token = token
                break

        if self.enhanced_only:
            return self._get_enhanced_grammar(sentence)

        # Para cada token, identifica seus dependentes
        for token in tokens:
            token_id = token['id']
            token_pos = token['upos']
            is_root = (token['head'] == 0)

            # Encontra todos os dependentes deste token
            dependents = []
            for dep_token in tokens:
                if dep_token['head'] == token_id:
                    dependents.append({
                        'id': dep_token['id'],
                        'pos': dep_token['upos'],
                        'deprel': dep_token['deprel'],
                        'form': dep_token['form']
                    })
            
            # Ordena dependentes por posição
            dependents.sort(key=lambda x: x['id'])
            
            # Separa dependentes à esquerda e à direita
            left_deps = [d for d in dependents if d['id'] < token_id]
            right_deps = [d for d in dependents if d['id'] > token_id]
            
            # Cria a regra base com os dependentes
            # Formata cada dependente conforme as flags include_upos e include_deprel
            def format_dependent(dep):
                """Formata um dependente conforme as flags de configuração"""
                upos_part = dep['pos'] if self.include_upos else '_'
                deprel_part = dep['deprel'] if self.include_deprel else '_'

                # Se ambas as flags estão desativadas, retorna apenas '_/_'
                if not self.include_upos and not self.include_deprel:
                    return '_/_'
                # Se apenas UPOS está ativo, retorna só o UPOS
                elif self.include_upos and not self.include_deprel:
                    return f"{upos_part}/_"
                # Se apenas DEPREL está ativo, retorna apenas 'deprel' (sem '_/')
                elif not self.include_upos and self.include_deprel:
                    return f"_/{deprel_part}"
                # Se ambas estão ativas, retorna 'UPOS/deprel'
                else:
                    return f"{upos_part}/{deprel_part}"

            left_pos = [format_dependent(d) for d in left_deps]
            right_pos = [format_dependent(d) for d in right_deps]

            rule_parts = left_pos + ['*'] + right_pos

            # Regra 1: Com dependentes (apenas se houver dependentes)
            if dependents:
                if self.include_upos and self.include_deprel:
                    rule_with_deps = f"{token_pos}({', '.join(rule_parts)})"
                elif not self.include_upos and self.include_deprel:
                    rule_with_deps = f"_({', '.join(rule_parts)})"
                elif self.include_upos and not self.include_deprel:
                    rule_with_deps = f"{token_pos}({', '.join(rule_parts)})"
                else:
                    rule_with_deps = f"_({', '.join(rule_parts)})"

                rules.append({
                    'rule': rule_with_deps,
                    'token': token['form'],
                    'lemma': token['lemma'],
                    'pos': token_pos,
                    'is_root': is_root,
                    'token_id': token_id,
                    'left_deps': left_deps,
                    'right_deps': right_deps,
                    'num_deps': len(dependents)
                })

            # Regra 2: Como folha (sempre gerada para todos os tokens)
            # Formato da regra folha depende das flags include_upos e include_deprel
            if is_root:
                # Raiz sem dependentes
                if self.include_upos and self.include_deprel:
                    # Ambos: *(VERB/root)
                    leaf_rule = f"*({token_pos})"
                elif not self.include_upos and self.include_deprel:
                    # Apenas DEPREL: *(root)
                    leaf_rule = "*(_)"
                elif self.include_upos and not self.include_deprel: # Apenas UPOS
                    leaf_rule = f"*({token_pos})"
                else:
                    leaf_rule = "*(_)"
            else:
                # Token folha (qualquer token visto como terminal)
                if self.include_upos and self.include_deprel:
                    leaf_rule = f"{token_pos}(*)"
                elif not self.include_upos and self.include_deprel:  # Apenas DEPREL
                    leaf_rule = "_(*)"
                elif self.include_upos and not self.include_deprel: # Apenas UPOS
                    leaf_rule = f"{token_pos}(*)"
                else:
                    leaf_rule = "_(*)"

            rules.append({
                'rule': leaf_rule,
                'token': token['form'],
                'lemma': token['lemma'],
                'pos': token_pos,
                'is_root': is_root,
                'token_id': token_id,
                'left_deps': [],
                'right_deps': [],
                'num_deps': 0
            })
        
        return {
            'sentence_id': sentence['id'],
            'sentence_text': sentence['text'],
            'rules': rules,
            'root': root_token
        }

    def _get_enhanced_grammar(self, sentence):
        """Extrai regras apenas das relações Enhanced (que diferem das básicas).

        Usa o mesmo formato do pipeline original:
          - Com dependentes: HEAD_POS(left_deps, *, right_deps)
          - Folha: POS(*)
        Mas a árvore é construída apenas com arestas enhanced-only.
        """
        tokens = sentence['tokens']
        token_map = {t['id']: t for t in tokens}
        edges = self._get_enhanced_only_edges(tokens)

        # Agrupa dependentes enhanced por head
        from collections import defaultdict
        head_to_deps = defaultdict(list)
        for dep_token, head_id, deprel in edges:
            head_to_deps[head_id].append({
                'id': dep_token['id'],
                'pos': dep_token['upos'],
                'deprel': deprel,
                'form': dep_token['form'],
            })

        rules = []

        def format_dependent(dep):
            upos_part = dep['pos'] if self.include_upos else '_'
            deprel_part = dep['deprel'] if self.include_deprel else '_'
            if not self.include_upos and not self.include_deprel:
                return '_/_'
            elif self.include_upos and not self.include_deprel:
                return f"{upos_part}/_"
            elif not self.include_upos and self.include_deprel:
                return f"_/{deprel_part}"
            else:
                return f"{upos_part}/{deprel_part}"

        # Gera regras para cada head que tem dependentes enhanced
        for head_id, dependents in head_to_deps.items():
            head_token = token_map.get(head_id)
            if head_token is None:
                continue

            token_pos = head_token['upos']
            dependents.sort(key=lambda x: x['id'])

            left_deps = [d for d in dependents if d['id'] < head_id]
            right_deps = [d for d in dependents if d['id'] > head_id]

            left_pos = [format_dependent(d) for d in left_deps]
            right_pos = [format_dependent(d) for d in right_deps]
            rule_parts = left_pos + ['*'] + right_pos

            if self.include_upos:
                rule_with_deps = f"{token_pos}({', '.join(rule_parts)})"
            else:
                rule_with_deps = f"_({', '.join(rule_parts)})"

            rules.append({
                'rule': rule_with_deps,
                'token': head_token['form'],
                'lemma': head_token['lemma'],
                'pos': token_pos,
                'is_root': head_token['head'] == 0,
                'token_id': head_id,
                'left_deps': left_deps,
                'right_deps': right_deps,
                'num_deps': len(dependents),
            })

        # Gera regras folha para cada dependente enhanced
        for dep_token, head_id, deprel in edges:
            token_pos = dep_token['upos']
            if self.include_upos:
                leaf_rule = f"{token_pos}(*)"
            else:
                leaf_rule = "_(*)"

            rules.append({
                'rule': leaf_rule,
                'token': dep_token['form'],
                'lemma': dep_token['lemma'],
                'pos': token_pos,
                'is_root': False,
                'token_id': dep_token['id'],
                'left_deps': [],
                'right_deps': [],
                'num_deps': 0,
            })

        root_token = None
        for t in tokens:
            if t['head'] == 0:
                root_token = t
                break

        return {
            'sentence_id': sentence['id'],
            'sentence_text': sentence['text'],
            'rules': rules,
            'root': root_token,
        }

    def _write_statistics_to_file(self, f, stats):
        """Método auxiliar para escrever estatísticas em arquivo de texto"""
        f.write("\n" + "=" * 80 + "\n")
        f.write("ESTATÍSTICAS DAS REGRAS GRAMATICAIS\n")
        f.write("=" * 80 + "\n")
        f.write(f"Total de sentenças: {stats['total_sentences']}\n")
        f.write(f"Total de regras: {stats['total_rules']}\n")
        f.write(f"Regras únicas: {stats['unique_rules']}\n")

        # Novas métricas com desvio padrão
        if 'rules_per_sentence' in stats:
            rps = stats['rules_per_sentence']
            f.write(f"\nRegras por sentença: média = {rps['mean']:.4f}, desvio padrão = {rps['std']:.4f}\n")

        if 'unique_rules_per_sentence' in stats:
            urps = stats['unique_rules_per_sentence']
            f.write(f"Regras únicas por sentença: média = {urps['mean']:.4f}, desvio padrão = {urps['std']:.4f}\n")

        f.write("\n" + "-" * 80 + "\n")
        f.write("FREQUÊNCIA DE TODAS AS REGRAS (ordenadas por frequência):\n")
        f.write("-" * 80 + "\n\n")

        sorted_rules = sorted(stats['rule_frequencies'].items(),
                             key=lambda x: x[1]['count'],
                             reverse=True)

        for rule, data in sorted_rules:
            f.write(f"{rule:<70} Freq: {data['count']:>4}\n")

        f.write("\n" + "=" * 80 + "\n")

    def export_sentence_grammars(self, sentence_grammars, output_file, stats=None):
        """Exporta gramáticas mantendo agrupamento por sentença"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("GRAMÁTICA DE DEPENDÊNCIAS POR SENTENÇA\n")
            f.write("=" * 80 + "\n")
            f.write("Formato: POS(dep_esquerda, ..., *, dep_direita, ...)\n")
            f.write("         *(POS) indica que POS é a raiz sem dependentes\n")
            f.write("         POS(*) indica que POS não tem dependentes (folha)\n")
            f.write("         [ROOT] marca o token que é raiz da sentença\n")
            f.write("=" * 80 + "\n\n")

            for i, grammar in enumerate(sentence_grammars, 1):
                f.write(f"\n{'='*80}\n")
                f.write(f"SENTENÇA {i}: {grammar['sentence_id']}\n")
                f.write(f"{'='*80}\n")
                f.write(f"Texto: {grammar['sentence_text']}\n")
                if grammar['root']:
                    f.write(f"Raiz: {grammar['root']['form']} ({grammar['root']['upos']})\n")
                f.write("\nRegras:\n")

                for rule in grammar['rules']:
                    root_marker = " [ROOT]" if rule['is_root'] else ""
                    f.write(f"  {rule['rule']:<45} # {rule['token']} ({rule['lemma']}){root_marker}\n")

                f.write("\n")

            # Adiciona estatísticas ao final
            if stats:
                self._write_statistics_to_file(f, stats)
    
    def export_compact_format(self, sentence_grammars, output_file, stats=None):
        """Exporta em formato compacto (apenas as regras)"""
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, grammar in enumerate(sentence_grammars, 1):
                f.write(f"# Sentença {i}: {grammar['sentence_text']}\n")
                if grammar['root']:
                    f.write(f"# Raiz: {grammar['root']['form']}\n")
                for rule in grammar['rules']:
                    root_marker = " # ROOT" if rule['is_root'] else ""
                    f.write(f"{rule['rule']}{root_marker}\n")
                f.write("\n")

            # Adiciona estatísticas ao final
            if stats:
                self._write_statistics_to_file(f, stats)
    
    def export_to_json(self, sentence_grammars, output_file, stats=None):
        """Exporta para JSON"""
        import json

        data = {
            'sentences': [],
            'statistics': None
        }

        for grammar in sentence_grammars:
            data['sentences'].append({
                'sentence_id': grammar['sentence_id'],
                'sentence_text': grammar['sentence_text'],
                'root': grammar['root']['form'] if grammar['root'] else None,
                'root_pos': grammar['root']['upos'] if grammar['root'] else None,
                'rules': [r['rule'] for r in grammar['rules']],
                'detailed_rules': [
                    {
                        'rule': r['rule'],
                        'token': r['token'],
                        'lemma': r['lemma'],
                        'pos': r['pos'],
                        'is_root': r['is_root'],
                        'num_dependents': r['num_deps']
                    } for r in grammar['rules']
                ]
            })

        # Adiciona estatísticas
        if stats:
            # Converte estatísticas para formato JSON
            sorted_rules = sorted(stats['rule_frequencies'].items(),
                                 key=lambda x: x[1]['count'],
                                 reverse=True)

            data['statistics'] = {
                'total_sentences': stats['total_sentences'],
                'total_rules': stats['total_rules'],
                'unique_rules': stats['unique_rules'],
                'rule_frequencies': [
                    {
                        'rule': rule,
                        'count': data_item['count'],
                    } for rule, data_item in sorted_rules  # TODAS as regras
                ]
            }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def get_grammar_statistics(self, sentence_grammars):
        """Calcula estatísticas sobre as gramáticas"""
        all_rules = []
        rule_freq = {}

        for grammar in sentence_grammars:
            for rule in grammar['rules']:
                all_rules.append(rule['rule'])
                rule_str = rule['rule']
                if rule_str not in rule_freq:
                    rule_freq[rule_str] = {
                        'count': 0,
                        'sentences': [],
                    }
                rule_freq[rule_str]['count'] += 1
                if len(rule_freq[rule_str]['sentences']) < 3:
                    rule_freq[rule_str]['sentences'].append({
                        'id': grammar['sentence_id'],
                        'text': grammar['sentence_text'],
                        'token': rule['token']
                    })

        # Calcula métricas por sentença para desvio padrão
        rules_per_sentence = [len(g['rules']) for g in sentence_grammars]
        unique_rules_per_sentence = [len(set(r['rule'] for r in g['rules'])) for g in sentence_grammars]

        return {
            'total_sentences': len(sentence_grammars),
            'total_rules': len(all_rules),
            'unique_rules': len(rule_freq),
            'rule_frequencies': rule_freq,
            # Novas métricas com desvio padrão
            'rules_per_sentence': {
                'mean': np.mean(rules_per_sentence) if rules_per_sentence else 0,
                'std': np.std(rules_per_sentence) if rules_per_sentence else 0,
                'values': rules_per_sentence
            },
            'unique_rules_per_sentence': {
                'mean': np.mean(unique_rules_per_sentence) if unique_rules_per_sentence else 0,
                'std': np.std(unique_rules_per_sentence) if unique_rules_per_sentence else 0,
                'values': unique_rules_per_sentence
            }
        }
    
    def sentence_grammar(self, sentence_idx=0):
        """Imprime a gramática de uma sentença específica"""
        if sentence_idx >= len(self.sentences):
            return

        return self.get_sentence_grammar(self.sentences[sentence_idx])


def realizar_testes_estatisticos(stats1: dict, stats2: dict, nome1: str, nome2: str):
    """
    Realiza testes estatísticos comparando métricas de regras gramaticais entre dois grupos.

    Args:
        stats1: Estatísticas do primeiro grupo (retorno de get_grammar_statistics)
        stats2: Estatísticas do segundo grupo (retorno de get_grammar_statistics)
        nome1: Nome do primeiro grupo (ex: "Human")
        nome2: Nome do segundo grupo (ex: "LLM")

    Returns:
        dict com resultados dos testes estatísticos
    """
    resultados = {
        'grupo1': nome1,
        'grupo2': nome2,
        'metricas': {}
    }

    # Métricas para comparar
    metricas = [
        ('rules_per_sentence', 'Regras por Sentenca'),
        ('unique_rules_per_sentence', 'Regras Unicas por Sentenca')
    ]

    for metrica_key, metrica_nome in metricas:
        valores1 = np.array(stats1[metrica_key]['values'])
        valores2 = np.array(stats2[metrica_key]['values'])

        # Estatísticas descritivas
        media1 = np.mean(valores1)
        media2 = np.mean(valores2)
        std1 = np.std(valores1)
        std2 = np.std(valores2)

        # Teste t independente
        t_stat, t_pvalue = scipy_stats.ttest_ind(valores1, valores2)

        # Mann-Whitney U (não assume normalidade)
        try:
            u_stat, u_pvalue = scipy_stats.mannwhitneyu(valores1, valores2, alternative='two-sided')
        except ValueError:
            u_stat, u_pvalue = 0, 1.0

        # Cohen's d (tamanho do efeito)
        pooled_std = np.sqrt((std1**2 + std2**2) / 2)
        if pooled_std > 0:
            cohens_d = (media1 - media2) / pooled_std
        else:
            cohens_d = 0

        # Interpretação do Cohen's d
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            interpretacao_d = "negligenciavel"
        elif abs_d < 0.5:
            interpretacao_d = "pequeno"
        elif abs_d < 0.8:
            interpretacao_d = "medio"
        else:
            interpretacao_d = "grande"

        resultados['metricas'][metrica_key] = {
            'nome': metrica_nome,
            nome1: {'media': media1, 'std': std1, 'n': len(valores1)},
            nome2: {'media': media2, 'std': std2, 'n': len(valores2)},
            'diferenca': media1 - media2,
            't_test': {'statistic': t_stat, 'p_value': t_pvalue},
            'mann_whitney': {'statistic': u_stat, 'p_value': u_pvalue},
            'cohens_d': cohens_d,
            'interpretacao_cohens_d': interpretacao_d,
            'significativo_005': t_pvalue < 0.05,
            'significativo_001': t_pvalue < 0.01
        }

    return resultados


def imprimir_resultados_estatisticos(resultados: dict):
    """
    Imprime os resultados dos testes estatísticos de forma formatada.

    Args:
        resultados: Retorno da função realizar_testes_estatisticos
    """
    print("\n" + "=" * 80)
    print("TESTES ESTATISTICOS: {} vs {}".format(resultados['grupo1'], resultados['grupo2']))
    print("=" * 80)

    for metrica_key, dados in resultados['metricas'].items():
        print("\n" + "-" * 60)
        print("Metrica: {}".format(dados['nome']))
        print("-" * 60)

        g1 = resultados['grupo1']
        g2 = resultados['grupo2']

        print("\n  Estatisticas Descritivas:")
        print("    {}: media = {:.4f}, std = {:.4f}, n = {}".format(
            g1, dados[g1]['media'], dados[g1]['std'], dados[g1]['n']))
        print("    {}: media = {:.4f}, std = {:.4f}, n = {}".format(
            g2, dados[g2]['media'], dados[g2]['std'], dados[g2]['n']))
        print("    Diferenca ({}  - {}): {:.4f}".format(g1, g2, dados['diferenca']))

        print("\n  Testes de Significancia:")
        print("    t-test: t = {:.4f}, p = {:.2e}".format(
            dados['t_test']['statistic'], dados['t_test']['p_value']))
        print("    Mann-Whitney U: U = {:.4f}, p = {:.2e}".format(
            dados['mann_whitney']['statistic'], dados['mann_whitney']['p_value']))

        sig_marker = ""
        if dados['significativo_001']:
            sig_marker = " **"
        elif dados['significativo_005']:
            sig_marker = " *"

        print("\n  Tamanho do Efeito:")
        print("    Cohen's d = {:.4f} ({}){}".format(
            dados['cohens_d'], dados['interpretacao_cohens_d'], sig_marker))

    print("\n" + "=" * 80)
    print("Legenda: * p < 0.05, ** p < 0.01")
    print("Cohen's d: <0.2=negligenciavel, 0.2-0.5=pequeno, 0.5-0.8=medio, >0.8=grande")
    print("=" * 80)


def exportar_resultados_csv(resultados: dict, output_file: str):
    """
    Exporta resultados dos testes estatísticos para CSV.

    Args:
        resultados: Retorno da função realizar_testes_estatisticos
        output_file: Caminho do arquivo CSV de saída
    """
    import csv

    g1 = resultados['grupo1']
    g2 = resultados['grupo2']

    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)

        # Cabeçalho
        writer.writerow([
            'metrica', 'nome_metrica',
            f'{g1}_media', f'{g1}_std', f'{g1}_n',
            f'{g2}_media', f'{g2}_std', f'{g2}_n',
            'diferenca', 't_statistic', 't_pvalue',
            'mann_whitney_u', 'mann_whitney_pvalue',
            'cohens_d', 'interpretacao', 'significativo_005', 'significativo_001'
        ])

        # Dados
        for metrica_key, dados in resultados['metricas'].items():
            writer.writerow([
                metrica_key, dados['nome'],
                dados[g1]['media'], dados[g1]['std'], dados[g1]['n'],
                dados[g2]['media'], dados[g2]['std'], dados[g2]['n'],
                dados['diferenca'],
                dados['t_test']['statistic'], dados['t_test']['p_value'],
                dados['mann_whitney']['statistic'], dados['mann_whitney']['p_value'],
                dados['cohens_d'], dados['interpretacao_cohens_d'],
                dados['significativo_005'], dados['significativo_001']
            ])

    print(f"\nResultados exportados para: {output_file}")
