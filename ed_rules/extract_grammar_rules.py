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
                # Skip comments
                elif line.startswith("#"):
                    continue
                else:
                    # Skip contractions, multi-word tokens and empty nodes
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
                # An edge is "new" if it differs from the basic one
                if head_id != basic_head or deprel != basic_deprel:
                    edges.append((token, head_id, deprel))
        return edges

    def get_sentence_grammar(self, sentence):
        """Extrai a gramática completa de uma sentença"""
        tokens = sentence['tokens']
        rules = []
        root_token = None

        # First, identify the root
        for token in tokens:
            if token['head'] == 0:
                root_token = token
                break

        if self.enhanced_only:
            return self._get_enhanced_grammar(sentence)

        # For each token, identify its dependents
        for token in tokens:
            token_id = token['id']
            token_pos = token['upos']
            is_root = (token['head'] == 0)

            # Find every dependent of this token
            dependents = []
            for dep_token in tokens:
                if dep_token['head'] == token_id:
                    dependents.append({
                        'id': dep_token['id'],
                        'pos': dep_token['upos'],
                        'deprel': dep_token['deprel'],
                        'form': dep_token['form']
                    })
            
            # Sort dependents by position
            dependents.sort(key=lambda x: x['id'])
            
            # Split dependents into left-side and right-side
            left_deps = [d for d in dependents if d['id'] < token_id]
            right_deps = [d for d in dependents if d['id'] > token_id]
            
            # Build the base rule with the dependents; each dependent is
            # formatted according to the include_upos / include_deprel flags
            def format_dependent(dep):
                """Formata um dependente conforme as flags de configuração"""
                upos_part = dep['pos'] if self.include_upos else '_'
                deprel_part = dep['deprel'] if self.include_deprel else '_'

                # Both flags off: return just '_/_'
                if not self.include_upos and not self.include_deprel:
                    return '_/_'
                # UPOS only: return just the UPOS
                elif self.include_upos and not self.include_deprel:
                    return f"{upos_part}/_"
                # DEPREL only: return just 'deprel' (without '_/')
                elif not self.include_upos and self.include_deprel:
                    return f"_/{deprel_part}"
                # Both flags on: return 'UPOS/deprel'
                else:
                    return f"{upos_part}/{deprel_part}"

            left_pos = [format_dependent(d) for d in left_deps]
            right_pos = [format_dependent(d) for d in right_deps]

            rule_parts = left_pos + ['*'] + right_pos

            # Rule 1: with dependents (only when the token has any)
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

            # Rule 2: as a leaf (always emitted, for every token)
            # The leaf rule format depends on the include_upos / include_deprel flags
            if is_root:
                # Root without dependents
                if self.include_upos and self.include_deprel:
                    # Both: *(VERB/root)
                    leaf_rule = f"*({token_pos})"
                elif not self.include_upos and self.include_deprel:
                    # DEPREL only: *(root)
                    leaf_rule = "*(_)"
                elif self.include_upos and not self.include_deprel:  # UPOS only
                    leaf_rule = f"*({token_pos})"
                else:
                    leaf_rule = "*(_)"
            else:
                # Leaf token (any token seen as a terminal)
                if self.include_upos and self.include_deprel:
                    leaf_rule = f"{token_pos}(*)"
                elif not self.include_upos and self.include_deprel:  # DEPREL only
                    leaf_rule = "_(*)"
                elif self.include_upos and not self.include_deprel:  # UPOS only
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

        # Group enhanced dependents by head
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

        # Emit a rule for every head that has enhanced dependents
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

        # Emit a leaf rule for every enhanced dependent
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
        f.write("GRAMMAR RULE STATISTICS\n")
        f.write("=" * 80 + "\n")
        f.write(f"Total sentences: {stats['total_sentences']}\n")
        f.write(f"Total rules: {stats['total_rules']}\n")
        f.write(f"Unique rules: {stats['unique_rules']}\n")

        # Per-sentence metrics with standard deviation
        if 'rules_per_sentence' in stats:
            rps = stats['rules_per_sentence']
            f.write(f"\nRules per sentence: mean = {rps['mean']:.4f}, std = {rps['std']:.4f}\n")

        if 'unique_rules_per_sentence' in stats:
            urps = stats['unique_rules_per_sentence']
            f.write(f"Unique rules per sentence: mean = {urps['mean']:.4f}, std = {urps['std']:.4f}\n")

        f.write("\n" + "-" * 80 + "\n")
        f.write("FREQUENCY OF ALL RULES (sorted by frequency):\n")
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
            f.write("DEPENDENCY GRAMMAR PER SENTENCE\n")
            f.write("=" * 80 + "\n")
            f.write("Format: POS(left_dep, ..., *, right_dep, ...)\n")
            f.write("         *(POS) means POS is the root and has no dependents\n")
            f.write("         POS(*) means POS has no dependents (leaf)\n")
            f.write("         [ROOT] marks the token that is the sentence root\n")
            f.write("=" * 80 + "\n\n")

            for i, grammar in enumerate(sentence_grammars, 1):
                f.write(f"\n{'='*80}\n")
                f.write(f"SENTENCE {i}: {grammar['sentence_id']}\n")
                f.write(f"{'='*80}\n")
                f.write(f"Text: {grammar['sentence_text']}\n")
                if grammar['root']:
                    f.write(f"Root: {grammar['root']['form']} ({grammar['root']['upos']})\n")
                f.write("\nRules:\n")

                for rule in grammar['rules']:
                    root_marker = " [ROOT]" if rule['is_root'] else ""
                    f.write(f"  {rule['rule']:<45} # {rule['token']} ({rule['lemma']}){root_marker}\n")

                f.write("\n")

            # Append the statistics block at the end
            if stats:
                self._write_statistics_to_file(f, stats)
    
    def export_compact_format(self, sentence_grammars, output_file, stats=None):
        """Exporta em formato compacto (apenas as regras)"""
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, grammar in enumerate(sentence_grammars, 1):
                f.write(f"# Sentence {i}: {grammar['sentence_text']}\n")
                if grammar['root']:
                    f.write(f"# Root: {grammar['root']['form']}\n")
                for rule in grammar['rules']:
                    root_marker = " # ROOT" if rule['is_root'] else ""
                    f.write(f"{rule['rule']}{root_marker}\n")
                f.write("\n")

            # Append the statistics block at the end
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

        # Append the statistics block
        if stats:
            # Convert the statistics to a JSON-friendly shape
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
                    } for rule, data_item in sorted_rules  # ALL rules
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

        # Per-sentence metrics, used for the standard deviation
        rules_per_sentence = [len(g['rules']) for g in sentence_grammars]
        unique_rules_per_sentence = [len(set(r['rule'] for r in g['rules'])) for g in sentence_grammars]

        return {
            'total_sentences': len(sentence_grammars),
            'total_rules': len(all_rules),
            'unique_rules': len(rule_freq),
            'rule_frequencies': rule_freq,
            # Per-sentence metrics with standard deviation
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


def run_statistical_tests(stats1: dict, stats2: dict, name1: str, name2: str):
    """
    Realiza testes estatísticos comparando métricas de regras gramaticais entre dois grupos.

    Args:
        stats1: Estatísticas do primeiro grupo (retorno de get_grammar_statistics)
        stats2: Estatísticas do segundo grupo (retorno de get_grammar_statistics)
        name1: Nome do primeiro grupo (ex: "Human")
        name2: Nome do segundo grupo (ex: "LLM")

    Returns:
        dict com resultados dos testes estatísticos
    """
    results = {
        'group1': name1,
        'group2': name2,
        'metrics': {}
    }

    # Metrics to compare
    metrics = [
        ('rules_per_sentence', 'Rules per Sentence'),
        ('unique_rules_per_sentence', 'Unique Rules per Sentence')
    ]

    for metric_key, metric_name in metrics:
        values1 = np.array(stats1[metric_key]['values'])
        values2 = np.array(stats2[metric_key]['values'])

        # Descriptive statistics
        mean1 = np.mean(values1)
        mean2 = np.mean(values2)
        std1 = np.std(values1)
        std2 = np.std(values2)

        # Independent t-test
        t_stat, t_pvalue = scipy_stats.ttest_ind(values1, values2)

        # Mann-Whitney U (does not assume normality)
        try:
            u_stat, u_pvalue = scipy_stats.mannwhitneyu(values1, values2, alternative='two-sided')
        except ValueError:
            u_stat, u_pvalue = 0, 1.0

        # Cohen's d (effect size)
        pooled_std = np.sqrt((std1**2 + std2**2) / 2)
        if pooled_std > 0:
            cohens_d = (mean1 - mean2) / pooled_std
        else:
            cohens_d = 0

        # Cohen's d interpretation
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            interpretation_d = "negligible"
        elif abs_d < 0.5:
            interpretation_d = "small"
        elif abs_d < 0.8:
            interpretation_d = "medium"
        else:
            interpretation_d = "large"

        results['metrics'][metric_key] = {
            'name': metric_name,
            name1: {'mean': mean1, 'std': std1, 'n': len(values1)},
            name2: {'mean': mean2, 'std': std2, 'n': len(values2)},
            'difference': mean1 - mean2,
            't_test': {'statistic': t_stat, 'p_value': t_pvalue},
            'mann_whitney': {'statistic': u_stat, 'p_value': u_pvalue},
            'cohens_d': cohens_d,
            'cohens_d_interpretation': interpretation_d,
            'significant_005': t_pvalue < 0.05,
            'significant_001': t_pvalue < 0.01
        }

    return results


def print_statistical_results(results: dict):
    """
    Imprime os resultados dos testes estatísticos de forma formatada.

    Args:
        results: Retorno da função run_statistical_tests
    """
    print("\n" + "=" * 80)
    print("STATISTICAL TESTS: {} vs {}".format(results['group1'], results['group2']))
    print("=" * 80)

    for metric_key, data in results['metrics'].items():
        print("\n" + "-" * 60)
        print("Metric: {}".format(data['name']))
        print("-" * 60)

        g1 = results['group1']
        g2 = results['group2']

        print("\n  Descriptive statistics:")
        print("    {}: mean = {:.4f}, std = {:.4f}, n = {}".format(
            g1, data[g1]['mean'], data[g1]['std'], data[g1]['n']))
        print("    {}: mean = {:.4f}, std = {:.4f}, n = {}".format(
            g2, data[g2]['mean'], data[g2]['std'], data[g2]['n']))
        print("    Difference ({} - {}): {:.4f}".format(g1, g2, data['difference']))

        print("\n  Significance tests:")
        print("    t-test: t = {:.4f}, p = {:.2e}".format(
            data['t_test']['statistic'], data['t_test']['p_value']))
        print("    Mann-Whitney U: U = {:.4f}, p = {:.2e}".format(
            data['mann_whitney']['statistic'], data['mann_whitney']['p_value']))

        sig_marker = ""
        if data['significant_001']:
            sig_marker = " **"
        elif data['significant_005']:
            sig_marker = " *"

        print("\n  Effect size:")
        print("    Cohen's d = {:.4f} ({}){}".format(
            data['cohens_d'], data['cohens_d_interpretation'], sig_marker))

    print("\n" + "=" * 80)
    print("Legend: * p < 0.05, ** p < 0.01")
    print("Cohen's d: <0.2=negligible, 0.2-0.5=small, 0.5-0.8=medium, >0.8=large")
    print("=" * 80)


def export_results_csv(results: dict, output_file: str):
    """
    Exporta resultados dos testes estatísticos para CSV.

    Args:
        results: Retorno da função run_statistical_tests
        output_file: Caminho do arquivo CSV de saída
    """
    import csv

    g1 = results['group1']
    g2 = results['group2']

    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)

        # Header
        writer.writerow([
            'metric', 'metric_name',
            f'{g1}_mean', f'{g1}_std', f'{g1}_n',
            f'{g2}_mean', f'{g2}_std', f'{g2}_n',
            'difference', 't_statistic', 't_pvalue',
            'mann_whitney_u', 'mann_whitney_pvalue',
            'cohens_d', 'interpretation', 'significant_005', 'significant_001'
        ])

        # Rows
        for metric_key, data in results['metrics'].items():
            writer.writerow([
                metric_key, data['name'],
                data[g1]['mean'], data[g1]['std'], data[g1]['n'],
                data[g2]['mean'], data[g2]['std'], data[g2]['n'],
                data['difference'],
                data['t_test']['statistic'], data['t_test']['p_value'],
                data['mann_whitney']['statistic'], data['mann_whitney']['p_value'],
                data['cohens_d'], data['cohens_d_interpretation'],
                data['significant_005'], data['significant_001']
            ])

    print(f"\nResults exported to: {output_file}")
