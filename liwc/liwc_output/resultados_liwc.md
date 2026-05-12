# Resultados da Analise LIWC: Fake News Humanas vs Geradas por LLM

## Corpus

| | Human | LLM | Total |
|---|---|---|---|
| FakeTrue.Br | 1.791 | 1.791 | 3.582 |
| Fake.Br | 3.600 | 3.600 | 7.200 |
| **Total** | **5.391** | **5.391** | **10.782** |

- **Dicionario**: Brazilian Portuguese LIWC2015 (73 categorias, 11.823 palavras exatas, 2.634 prefixos)
- **Cobertura do dicionario**: Human 69,4% | LLM 69,6% (sem vies entre grupos)
- **Media de palavras/documento**: Human 176,7 | LLM 329,3

## Resumo Geral

Das 74 categorias analisadas (73 LIWC + cobertura do dicionario):

- **66 categorias significativas** (p<0,05) — 89,2%
- **63 categorias significativas** (p<0,01) — 85,1%
- 44 mais caracteristicas de textos humanos
- 30 mais caracteristicas de textos LLM

## Categorias mais Discriminativas

### Mais presentes em textos HUMANOS

| # | Categoria | Mean Human | Mean LLM | Cohen's d | p-value |
|---|---|---|---|---|---|
| 1 | verb (Verbs) | 10,31% | 8,13% | +0,740 | 9,38e-272 |
| 2 | social (Social) | 7,36% | 5,79% | +0,517 | 7,76e-134 |
| 3 | focuspast (Past Focus) | 2,66% | 1,69% | +0,595 | 1,06e-136 |
| 4 | informal (Informal Language) | 0,77% | 0,24% | +0,595 | 2,67e-145 |
| 5 | hear (Hear) | 0,93% | 0,41% | +0,552 | 2,33e-103 |
| 6 | male (Male) | 1,42% | 0,90% | +0,456 | 5,93e-45 |
| 7 | auxverb (Auxiliary Verbs) | 4,43% | 3,67% | +0,395 | 7,47e-68 |
| 8 | percept (Perceptual Processes) | 2,36% | 1,78% | +0,357 | 1,66e-44 |
| 9 | netspeak (Netspeak) | 0,38% | 0,13% | +0,348 | 1,88e-35 |
| 10 | i (I) | 0,35% | 0,11% | +0,348 | 6,80e-28 |

### Mais presentes em textos LLM

| # | Categoria | Mean Human | Mean LLM | Cohen's d | p-value |
|---|---|---|---|---|---|
| 1 | adj (Adjectives) | 2,83% | 4,07% | -0,731 | 0,00e+00 |
| 2 | number (Numbers) | 2,56% | 3,44% | -0,585 | 4,93e-233 |
| 3 | posemo (Positive Emotions) | 1,70% | 2,34% | -0,472 | 1,42e-192 |
| 4 | affect (Affect) | 3,81% | 4,75% | -0,460 | 7,85e-169 |
| 5 | anx (Anx) | 0,27% | 0,53% | -0,449 | 0,00e+00 |
| 6 | insight (Insight) | 1,32% | 1,80% | -0,435 | 1,22e-166 |
| 7 | achieve (Achievement) | 1,11% | 1,46% | -0,342 | 2,22e-130 |
| 8 | cogproc (Cognitive Processes) | 9,30% | 10,35% | -0,325 | 5,41e-76 |
| 9 | drives (Drives) | 9,24% | 10,24% | -0,324 | 4,71e-77 |
| 10 | work (Work) | 3,38% | 4,09% | -0,317 | 2,74e-94 |

## Interpretacao dos Resultados

### Perfil linguistico dos textos HUMANOS

Textos humanos se destacam por um estilo **narrativo, oral e pessoal**:

- **Mais verbos e auxiliares**: estruturas verbais mais complexas e conjugacoes variadas, indicando narrativa ativa
- **Foco no passado**: relatos de eventos em tempo passado, tipico de noticias que descrevem fatos ocorridos
- **Linguagem social**: mais referencias a pessoas, relacoes e interacoes sociais
- **Linguagem informal**: presenca de girias, netspeak, nao-fluencias e palavroes — marcas de texto espontaneo e informal
- **Processos perceptuais (ouvir)**: uso de verbos como "disse", "afirmou", "declarou" — atribuicao de falas a fontes
- **Pronome "eu" e referencias masculinas**: tom mais pessoal e referencias diretas a figuras publicas masculinas

### Perfil linguistico dos textos LLM

Textos gerados por LLM apresentam um estilo **descritivo, formal e elaborado**:

- **Mais adjetivos** (maior efeito: d=-0,731): linguagem mais qualificadora e descritiva
- **Mais numeros**: uso mais frequente de dados numericos, estatisticas e datas
- **Emocoes positivas e afeto**: tom emocional mais presente e positivo, possivel tentativa de persuasao
- **Processos cognitivos e insight**: linguagem mais analitica, com mais conectivos causais e explicacoes
- **Drives (poder, conquista, trabalho)**: vocabulario mais orientado a motivacoes e realizacoes
- **Preposicoes e espacialidade**: construcoes frasais mais elaboradas com mais complementos preposicionados
- **Ansiedade**: mais palavras associadas a preocupacao e medo

## Categorias NAO Significativas (p>=0,05)

As seguintes categorias nao apresentaram diferenca significativa entre os grupos:

| Categoria | Mean Human | Mean LLM | p-value |
|---|---|---|---|
| dictionary_coverage | 69,41% | 69,57% | 9,79e-02 |
| leisure (Leisure) | 1,07% | 0,84% | 1,31e-01 |
| differ (Differentiation) | 2,67% | 2,53% | 6,16e-01 |
| feel (Feel) | 0,55% | 0,41% | 8,45e-01 |
| affiliation (Affiliation) | 1,85% | 1,71% | 1,47e-01 |
| body (Body) | 0,32% | 0,20% | 9,01e-02 |
| article (Articles) | 15,45% | 15,40% | 5,70e-02 |
| conj (Conjunctions) | 7,74% | 7,72% | 4,64e-01 |

## Tamanho do Efeito (Cohen's d)

| Classificacao | Faixa | Categorias |
|---|---|---|
| Grande (>=0,8) | - | 0 |
| Medio (0,5-0,8) | verb, adj, focuspast, informal, number, hear, social | 7 |
| Pequeno (0,2-0,5) | posemo, affect, anx, male, insight, auxverb, achieve, percept, netspeak, i, cogproc, drives, work, motion, prep, space, risk, quant, certain, cause, negemo, power, nonflu | 23 |
| Negligivel (<0,2) | Demais categorias | 44 |

## Arquivos de Saida

| Arquivo | Descricao |
|---|---|
| `liwc_discriminative_categories.csv` | Todas as categorias com estatisticas completas |
| `liwc_discriminative_categories.txt` | Relatorio legivel com ranking completo |
| `liwc_per_document_scores.csv` | Scores LIWC (%) por documento (util para ML) |
| `liwc_analysis_summary.txt` | Resumo geral da analise |

## Metodologia

- **Tokenizacao**: regex `[\w]+` sobre texto em minusculas
- **Categorizacao**: match exato + match por prefixo (wildcards `*` do dicionario LIWC)
- **Normalizacao**: contagem por categoria / total de palavras x 100 (porcentagem)
- **Teste estatistico**: Mann-Whitney U (bicaudal, nao-parametrico)
- **Tamanho do efeito**: Cohen's d com desvio padrao agrupado
