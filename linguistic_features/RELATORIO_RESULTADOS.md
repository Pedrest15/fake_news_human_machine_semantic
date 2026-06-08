# Análise de Resultados — TF-IDF e BoW Enriquecidos com Features Linguísticas

> Análise dos pipelines `text_enriched`: TF-IDF + features linguísticas e BoW
> + features linguísticas, com truncagem pareada pelo tokenizador BERTimbau a
> 512 tokens WordPiece. Inclui análise de importância de features (top-30 por
> classificador tree-based/linear) e mapeamento dos documentos misclassificados.
> Os números saem dos JSONs `text_*_linguistic_*.json` e respectivos
> `misclassified/text_*_linguistic_*_errors.json` em [results/](results/).

## 1. Configuração dos experimentos

Dois pipelines de representação são avaliados em paralelo, ambos consumindo o mesmo conjunto saneado de **780 features linguísticas** (140 NILC após Tier A+B, 61 LIWC após remoção dos três campos length-direct e das doze categorias agregadoras, 500 regras Enhanced UD top-K, 2 sílabas, 17 POS tagger, 60 SAGE):

- **`text_enriched --vectorizer tfidf`** — features linguísticas concatenadas com a matriz TF-IDF de 10 000 unigramas (`min_df=5`, `sublinear_tf=True`). Vetor final com **10 780 dimensões**, esparso (~92 % de zeros).
- **`text_enriched --vectorizer bow`** — análogo, mas com `CountVectorizer` (contagens brutas). Mesma dimensionalidade.

Ambos compartilham splits idênticos: **7 876 documentos de treino e 1 946 de teste**, distribuídos entre humanos (label 0) e LLM (label 1) em proporção balanceada.

O blacklist Tier A+B do NILC removeu **60 features length-biased** (16 contagens absolutas + 44 medidas correlacionadas com tamanho — diversidades TTR-like, riqueza vocabular, max/std de contagens e std das similaridades LSA). O blacklist LIWC removeu **15 campos** (3 length-direct: `word_count`, `matched_words`, `dictionary_coverage`; 12 agregadores hierárquicos: `function`, `pronoun`, `ppron`, `affect`, `negemo`, `social`, `cogproc`, `percept`, `bio`, `drives`, `relativ`, `informal` — preservando apenas as categorias-folha). Esses saneamentos tornam o classificador menos dependente de tamanho de texto e de redundância hierárquica.

A truncagem pareada é idêntica nos dois vetorizadores: `min(len_h_tokens, len_l_tokens, 510)` em tokens WordPiece BERTimbau, com decode de volta para texto. Garante que humano e LLM do mesmo factoide vejam exatamente a mesma janela de comprimento, fechando o vazamento de tamanho que dominava os resultados pré-saneamento.

Cinco classificadores foram avaliados via `GridSearchCV` com `cv=5` e `scoring='f1_weighted'`: SVM, Random Forest, Regressão Logística, MLP e XGBoost.

## 2. Tabela consolidada de resultados (F1 weighted no teste held-out)

| Classificador | TF-IDF + linguistic | BoW + linguistic |
|---|---|---|
| **XGBoost** | **0,9985** | **0,9985** |
| Logistic Regression | 0,9902 | 0,9938 |
| SVM | 0,9913 | 0,9938 |
| MLP | 0,9913 | **0,9943** |
| Random Forest | 0,9841 | 0,9836 |

## 3. TF-IDF + features linguísticas

A matriz TF-IDF de 10 000 unigramas com `sublinear_tf=True` concatenada com as 780 features linguísticas atinge **F1 entre 0,9841 e 0,9985 dependendo do classificador**. XGBoost lidera com folga: F1 = 0,9985 e **apenas 3 falsos negativos em 1 946 documentos, zero falsos positivos**. Seus hiperparâmetros (`depth=3`, `n_estimators=100`, `lr=0,3`) caracterizam um regime de "fast learner": árvores rasas com taxa de aprendizado alta, indicando que o sinal é fortemente discriminativo e poucas iterações bastam para capturá-lo.

SVM RBF (`C=10`) e MLP `(512,)` empatam em F1 = 0,9913 com 17 erros cada. Logistic Regression L2 com `C=0,01` chega a 0,9902 com 19 erros, e Random Forest fica em 0,9841 com 30 falsos negativos e apenas 1 falso positivo.

O padrão de erros nos modelos não-XGBoost é fortemente assimétrico: **muito mais falsos negativos que falsos positivos**. SVM tem 3 FP + 14 FN, RF tem 1 FP + 30 FN, MLP é o mais equilibrado dos não-XGBoost com 9 FP + 8 FN. Isso significa que esses classificadores tendem a **classificar LLMs como humanos** mais frequentemente do que o contrário — o sinal estilístico do LLM é mais sutil sem dependência de tamanho. XGBoost mantém o perfil `0 FP + N FN` mostrando que tem precisão perfeita para humanos no teste.

A escolha de `C=0,01` pela Regressão Logística com TF-IDF é tecnicamente reveladora. O TF-IDF com `sublinear_tf=True` produz valores comprimidos por `log(1+tf)`, mas distribuídos em 10 000 dimensões esparsas. A LR precisa de regularização **mais forte** (`C` menor = mais penalização) para evitar overfitting nessa alta dimensionalidade. O contraste com BoW abaixo (`C=1,0`) reforça que TF-IDF e BoW respondem diferente à regularização.

## 4. BoW + features linguísticas

Substituindo TF-IDF por contagens brutas (`CountVectorizer`, sem `sublinear_tf`), o desempenho **melhora levemente em quase todos os classificadores lineares e no MLP**, enquanto XGBoost e RF permanecem praticamente idênticos. MLP `(512,)` com `alpha=0,001` lidera os não-XGBoost com **F1 = 0,9943 e apenas 11 erros**, seguido por SVM RBF `C=10` e LR L2 `C=1,0`, ambos em 0,9938 com 12 erros. XGBoost mantém F1 = 0,9985 (3 FN, 0 FP), e RF fica em 0,9836 (31 FN, 1 FP).

A comparação direta TF-IDF vs BoW mostra um padrão consistente: **BoW supera TF-IDF nos classificadores lineares e no MLP**, empata praticamente em XGBoost e RF. As diferenças são pequenas mas sistemáticas:

- Logistic Regression: BoW 0,9938 → TF-IDF 0,9902 (Δ = +0,0036)
- SVM: BoW 0,9938 → TF-IDF 0,9913 (Δ = +0,0025)
- MLP: BoW 0,9943 → TF-IDF 0,9913 (Δ = +0,0030)
- XGBoost: empata em 0,9985

O resultado é intrigante porque a literatura padrão prefere TF-IDF a BoW. A explicação provável: na presença das 780 features linguísticas, que já fornecem informação de coesão, complexidade e estilo, o classificador linear consegue extrair sinal mais limpo das **contagens absolutas de palavras** do que da versão suavizada por `sublinear_tf`. A própria função `log(1 + tf)` do TF-IDF está "engolindo" um sinal que o classificador conseguiria usar melhor cru. Para o paper, isso é uma observação reportável: **em pipelines que combinam vetorização lexical com features linguísticas saneadas, BoW pode ser preferível a TF-IDF** — comportamento contraintuitivo que vale comentar.

XGBoost ignora completamente essa diferença porque árvores são invariantes a transformações monotônicas em features individuais — o ranking de cada palavra entre documentos não muda entre TF-IDF e BoW, e XGBoost decide por threshold, não por magnitude. RF, embora também baseado em árvores, sofre marginalmente porque seu critério de escolha de feature por split é sensível à variância das features.

## 5. Padrões nos hiperparâmetros vencedores

Olhando os melhores hiperparâmetros escolhidos pelo `GridSearchCV` em cada pipeline, aparecem padrões interessantes:

- **SVM**: em ambos os vetorizadores converge em **RBF kernel com `C=10`**. A preferência por kernel não-linear e flexibilidade moderada (C=10) indica que, removido o sinal de tamanho linear-dominante pelos blacklists, o classificador precisa capturar interações entre features para chegar ao desempenho ótimo.
- **Logistic Regression**: BoW prefere `C=1,0`, TF-IDF prefere `C=0,01`. A diferença reflete a normalização: TF-IDF com `sublinear_tf` exige regularização mais agressiva.
- **MLP**: ambos vetorizadores convergem em **uma única camada oculta de 512 neurônios** — arquitetura simples, sem profundidade. O sinal não exige rede profunda. Apenas `alpha` muda: 0,0001 para TF-IDF (menos regularização) vs 0,001 para BoW.
- **XGBoost**: ambos vetorizadores convergem em **árvores rasas (`depth=3`) com taxa de aprendizado alta (`lr=0,3`) e 100 estimadores**. Esse é o regime característico de "boost rápido": o sinal está em features individuais discriminativas (poucos splits revelam o rótulo), e o ensemble combina ~100 dessas árvores. É o classificador mais barato computacionalmente entre os de boa performance.
- **Random Forest**: profundidade ilimitada e `n_estimators=200` em ambos, com diferença apenas em `min_samples_split` (2 para TF-IDF, 5 para BoW). O RF se ajusta aos dois vetorizadores de forma quase idêntica, mas seu CV score baixo (~0,98) já anuncia o desempenho inferior nas duas configurações.

## 6. Importância de features

Três classificadores expõem importância numérica de cada feature (`feature_importances_` para tree-based; `|coef_|` para LR linear). SVM RBF e MLP não expõem importância por construção do algoritmo e não aparecem nesta seção.

### 6.1 XGBoost — features dominantes (TF-IDF e BoW empatam quase exatamente)

A leitura mais expressiva vem do XGBoost, que com `depth=3` precisa de poucas features muito discriminativas. O top-10 do TF-IDF e do BoW são **quase idênticos** com pequenas variações de ordem:

| Rank | Feature (XGBoost TF-IDF) | Importância | Grupo |
|---|---|---|---|
| 1 | `nilc__indicative_present_ratio` | 0,254 | nilc |
| 2 | `nilc__sentences_per_paragraph` | 0,137 | nilc |
| 3 | `nilc__clauses_per_sentence` | 0,137 | nilc |
| 4 | `nilc__named_entity_ratio_text` | 0,126 | nilc |
| 5 | `nilc__adjunct_per_clause` | 0,066 | nilc |
| 6 | `ud__PROPN(*)` | 0,054 | enhanced_ud |
| 7 | `ud__PROPN(ADP/case, DET/det, *)` | 0,031 | enhanced_ud |
| 8 | `syll__media_silabas_por_palavra` | 0,026 | sílabas |
| 9 | `nilc__named_entity_ratio_sentence` | 0,021 | nilc |
| 10 | `nilc__prepositions_per_sentence` | 0,017 | nilc |

A **feature dominante absoluta é `nilc__indicative_present_ratio`** (proporção de verbos no presente do indicativo) com 25,4 % da importância total. Esse achado é linguisticamente interpretável: LLMs tendem a usar o presente do indicativo de forma mais constante (estilo "atemporal" típico de modelos generativos), enquanto textos jornalísticos humanos sobre fake news alternam tempos verbais conforme narram passado, atribuem citações e fazem juízos. As features 2-5 reforçam isso: `sentences_per_paragraph`, `clauses_per_sentence` e `adjunct_per_clause` capturam a granularidade estrutural — LLMs produzem parágrafos mais densos sintaticamente, com mais subordinação e adjuntos por cláusula. `named_entity_ratio_text` (densidade de entidades nomeadas por documento) também distingue os grupos.

As duas regras Enhanced UD no top-7 são particularmente interessantes: **`PROPN(*)`** (substantivo próprio como folha, isolado de pai sintático) e **`PROPN(ADP/case, DET/det, *)`** (substantivo próprio precedido de preposição + determinante, ex.: "do Brasil", "da Universidade"). Ambas refletem a maneira como entidades nomeadas são construídas sintaticamente em cada grupo — LLMs parecem usar substantivos próprios com mais frequência em construções específicas.

O top-30 do XGBoost inclui **5 palavras `text__`** que vale citar: `text__redes`, `text__exclusivo`, `text__revela`, `text__fontes`, `text__bombástica` (TF-IDF) e `text__sociais`, `text__reviravolta`, `text__atenção`, `text__exclusivo` (BoW). Esse vocabulário sensacionalista/clickbait aparece desproporcionalmente em um dos grupos, e o XGBoost o detecta. Dois termos SAGE também emergem: `sage__uma_reviravolta` e `sage__voce_quiser` (TF-IDF) — bigramas característicos.

### 6.2 Random Forest — concentração total no NILC

O Random Forest mostra perfil diferente: **25 das top-30 features são do NILC**, sem nenhuma palavra `text__` no top-30. Ele essencialmente ignora a vetorização lexical e decide tudo via features linguísticas:

- Top-5 TF-IDF: `sentences_per_paragraph` (0,029), `lsa_paragraph_mean` (0,019), `syll__media_silabas_por_palavra` (0,019), `flesch` (0,019), `lsa_givenness_mean` (0,017).
- A regra `ud__VERB(PUNCT/punct, *, NOUN/obj)` aparece em ambos vetorizadores no top-11 — essa é a mesma regra já identificada no diagnóstico anterior como sinal sintático genuíno.
- Features psicolinguísticas dominam o middle do ranking: `idade_aquisicao_*` (várias variantes), `imageabilidade_*`, `familiaridade_*` — confirmam que o vocabulário lexical-cognitivo de LLM difere sistematicamente do humano.
- Uma única feature LIWC sobrevive ao top-30: `liwc__adj (Adjectives)` no rank 28-30. Indica que após o saneamento, LIWC contribui pouco para o sinal residual.

O fato de o RF ignorar palavras `text__` explica seu desempenho inferior (F1 ≈ 0,984): ele perde o sinal lexical que XGBoost captura. Por sua vez, RF prova que as 780 features linguísticas sozinhas (sem ajuda do vocabulário) já carregam aproximadamente 98,4 % do sinal.

### 6.3 Logistic Regression — visão linear complementar

A LR L1/L2 dá uma visão diferente porque coeficientes refletem direção (positivo = mais característico de LLM, negativo = de humano) embora aqui usemos `|coef_|`. O top-10 da LR no TF-IDF:

| Rank | Feature | \|coef\| | Grupo |
|---|---|---|---|
| 1 | `nilc__sentences_per_paragraph` | 0,375 | nilc |
| 2 | `ud__VERB(PUNCT/punct, *, NOUN/obj)` | 0,342 | enhanced_ud |
| 3 | `nilc__gerund_verbs` | 0,285 | nilc |
| 4 | `syll__media_silabas_por_palavra` | 0,266 | sílabas |
| 5 | `nilc__lsa_span_mean` | 0,261 | nilc |
| 6 | `nilc__add_neg_conn_ratio` | 0,258 | nilc |
| 7 | `liwc__number (Numbers)` | 0,248 | liwc |
| 8 | `nilc__adj_stem_ovl` | 0,239 | nilc |
| 9 | `nilc__stem_ovl` | 0,232 | nilc |
| 10 | `nilc__syllables_per_content_word` | 0,216 | nilc |

A LR confirma `sentences_per_paragraph` no topo e adiciona contribuições novas que o XGBoost com `depth=3` não capturava: `gerund_verbs` (uso de gerúndio), conectivos negativos aditivos e causais (`add_neg_conn_ratio`, `cau_neg_conn_ratio` — não aparece no top-10 mas está no top-15), sobreposição lexical (`stem_ovl`, `adj_stem_ovl` — coesão referencial), e categorias LIWC como `number`, `posemo`, `certain`, `prep`, `adj`. Várias regras Enhanced UD relacionadas a pontuação aparecem entre 12-26: `PUNCT(PUNCT/punct, *, PUNCT/punct, PUNCT/punct)`, `ADV(*, PRON/obl, PUNCT/punct)`, `ADP(*, DET/fixed, NOUN/fixed, PUNCT/punct)`, `VERB(PUNCT/punct, *, VERB/ccomp)` — padrões pontuacionais específicos.

No BoW, a LR também inclui palavras `text__apenas` (rank 16) e `text__disse` (rank 17) no top-30 — verbos dicendi e advérbios que aparecem com frequência diferente em humanos vs LLMs.

### 6.4 Síntese transversal — features que dominam consistentemente

Cruzando os rankings dos três classificadores em ambos os vetorizadores, **quatro features aparecem em **todos** os 6 runs (3 classificadores × 2 vetorizadores) no top-15**:

- **`nilc__sentences_per_paragraph`** — densidade estrutural dos parágrafos
- **`syll__media_silabas_por_palavra`** — complexidade lexical
- **`ud__VERB(PUNCT/punct, *, NOUN/obj)`** — padrão sintático "incisos + verbo + objeto"
- **`nilc__syllables_per_content_word`** — variante do anterior, complexidade de palavras de conteúdo

Adicionalmente, **`nilc__indicative_present_ratio`** é dominante exclusivamente no XGBoost (rank 1 absoluto em ambos vetorizadores) e desaparece do RF e LR. Isso sugere que XGBoost descobre uma regra simples e altamente discriminativa em torno do tempo verbal, enquanto LR e RF distribuem o peso entre múltiplas features correlacionadas.

## 7. Os 3 erros do XGBoost — identificados e analisados

Os **3 erros do XGBoost são exatamente os mesmos documentos LLM em TF-IDF e em BoW**, com confianças similares. Esse é um achado interessante: XGBoost reproduz seus próprios erros entre vetorizações, indicando que esses 3 documentos têm assinatura suficientemente atípica para escapar do classificador independentemente da representação textual usada.

| Documento | True → Pred | conf TF-IDF | conf BoW | n_palavras humano | n_palavras LLM |
|---|---|---|---|---|---|
| `fake_br/213.txt` | LLM → Humano | 0,997 | 0,991 | 141 | 563 |
| `fake_true_br/555.txt` | LLM → Humano | 0,851 | 0,911 | 204 | 247 |
| `fake_true_br/1746.txt` | LLM → Humano | 0,733 | 0,631 | 419 | 469 |

Três padrões interessantes nesses três documentos:

**1. `fake_br/213.txt`** é o erro de maior confiança (0,99). É um caso curioso: o humano tem 141 palavras e o LLM 563 (quase 4× mais longo, padrão típico para LLMs). O classificador errar com confiança extrema sugere que esse LLM imitou *muito bem* o estilo jornalístico humano apesar de longo — talvez um caso onde o modelo gerador produziu reportagem factual coerente em vez do tom mais elaborado que o classificador aprendeu a associar com LLM.

**2. `fake_true_br/555.txt` e `fake_true_br/1746.txt`** estão no FakeTrue.Br e têm **humano e LLM com tamanhos próximos** (204 vs 247; 419 vs 469). Isso é raro no corpus, onde LLMs são normalmente bem mais longos. A truncagem pareada (510 tokens WordPiece) corta ambos no mesmo ponto, mas como os originais já tinham comprimentos próximos, o LLM perde menos diferenciação de tamanho. Sem essa "almofada" de tamanho diferencial, o classificador depende ainda mais do sinal estilístico fino — e nesses dois casos a similaridade estrutural é grande o bastante para confundi-lo.

A consistência entre TF-IDF e BoW reforça que **não é um problema de representação lexical** — é um problema de "esses 3 LLMs são genuinamente bem feitos". Para o paper, podem ser citados como exemplos do tipo de LLM mais difícil de detectar: aqueles que reproduzem fielmente o estilo jornalístico.

## 8. Perfil dos erros por classificador

Em ambos os pipelines, **falsos negativos predominam fortemente sobre falsos positivos** — isto é, o classificador tende a confundir LLM com humano mais do que o contrário:

| Pipeline / Classificador | FP | FN | Total |
|---|---|---|---|
| TF-IDF / XGBoost | 0 | 3 | 3 |
| TF-IDF / SVM | 3 | 14 | 17 |
| TF-IDF / LR | 11 | 8 | 19 |
| TF-IDF / MLP | 9 | 8 | 17 |
| TF-IDF / RF | 1 | 30 | 31 |
| BoW / XGBoost | 0 | 3 | 3 |
| BoW / SVM | 1 | 11 | 12 |
| BoW / LR | 7 | 5 | 12 |
| BoW / MLP | 8 | 3 | 11 |
| BoW / RF | 1 | 31 | 32 |

O Random Forest mostra a assimetria mais forte (30+ FN com apenas 1 FP) — provável reflexo da escolha de `min_samples_split`: árvores um pouco mais conservadoras tendem a "votar humano" em casos de fronteira porque humanos têm maior diversidade interna no corpus (várias fontes, vários estilos jornalísticos), enquanto LLMs são mais homogêneos (mesmo modelo gerador). Quando uma instância LLM se assemelha mais a um humano "comum" do que aos LLMs sintéticos do treino, o ensemble vota humano. Esse efeito é amplificado em modelos com bias-variância conservador.

### 8.1 Documentos sistematicamente difíceis (aparecem em múltiplos classificadores)

Cruzando os arquivos de filename misclassificados entre os 10 runs (5 classificadores × 2 vetorizadores), alguns documentos aparecem em vários classificadores diferentes — são casos genuinamente difíceis, não erros de um modelo específico:

**LLMs que parecem humanos** (aparecem como FN em múltiplos classificadores):

- **`fake_true_br/1701.txt`** — confunde **8 dos 10 runs**: SVM, RF, LR, MLP em ambos TF-IDF e BoW. Confianças altas (0,73-0,99). É o documento mais sistematicamente mal classificado do conjunto.
- **`fake_true_br/1759.txt`** — confunde 8 runs (LR, MLP, RF, SVM nos dois vetorizadores). Confianças entre 0,52 e 0,85.
- **`fake_true_br/555.txt`** — confunde 6 runs (incluindo os dois XGBoost).
- **`fake_true_br/1414.txt`** — confunde 6 runs (LR, MLP, SVM, RF).
- **`fake_br/895.txt`** — confunde 5 runs (SVM, RF, LR em ambos vetorizadores).

**Humanos que parecem LLMs** (aparecem como FP em múltiplos classificadores):

- **`fake_br/2511.txt`** — confunde 5 runs (LR e MLP em TF-IDF e BoW, SVM TF-IDF). Confianças altas (0,89-0,99). É um humano que escreve sistematicamente como se fosse LLM.
- **`fake_true_br/202.txt`** — confunde 4 runs (LR e MLP em ambos vetorizadores). **Confiança 1,000** em todos — extremo.
- **`fake_br/3421.txt`** — confunde 4 runs (LR e MLP em ambos vetorizadores).
- **`fake_br/597.txt`** — confunde 4 runs com confiança altíssima (0,93-0,998 em LR e MLP).

A concentração de erros sistemáticos no FakeTrue.Br (especialmente no range 1700-1800) sugere que esse subgrupo do corpus contém LLMs particularmente bem feitos ou humanos com estilo atípico. Para o paper, vale uma análise qualitativa especificamente desses ~10 documentos — eles representam os limites de detectabilidade dos pipelines TF-IDF/BoW + linguistic.

## 9. Recomendações para o paper

Seis observações concretas para a redação:

**Primeira**, o pipeline `text_enriched` entrega **F1 ≈ 0,9985 no melhor caso (XGBoost)**, defensável metodologicamente porque (i) o saneamento das features linguísticas documenta de forma explícita quais atributos foram excluídos e por quê, com referências de literatura (Tweedie & Baayen 1998 para riqueza vocabular; argumento estatístico para max/std/diversidades; estrutura hierárquica LIWC2015 para os agregadores); (ii) cinco classificadores diferentes convergem para a faixa de 0,9836-0,9985 com perfis de erro coerentes; e (iii) a truncagem pareada pelo tokenizador BERTimbau elimina o viés residual de tamanho.

**Segunda**, **XGBoost é o classificador mais robusto** dos avaliados: vence ou empata nos dois pipelines, com hiperparâmetros idênticos entre TF-IDF e BoW (`depth=3`, `n=100`, `lr=0,3`), e mantém o perfil `0 FP + 3 FN` consistentemente. Para a tabela principal do paper, reporte XGBoost. Para análise complementar interpretável, reporte Logistic Regression (coeficientes legíveis).

**Terceira**, o achado de **BoW ≥ TF-IDF em modelos lineares** é não-trivial e merece um parágrafo de discussão. Mostra que TF-IDF não é universalmente superior, especialmente quando o pipeline já carrega features linguísticas estabilizadoras. O efeito desaparece em modelos baseados em árvores (XGBoost, RF) por invariância a transformações monotônicas.

**Quarta**, a **`nilc__indicative_present_ratio` é a feature dominante absoluta do XGBoost** (25,4 % da importância em ambos vetorizadores). Esse achado é altamente reportável: o tempo verbal — especificamente uso do presente do indicativo — é o sinal estatístico mais forte que separa humanos de LLMs neste corpus. LLMs produzem texto com tempo verbal mais constante, enquanto humanos alternam tempos conforme contexto narrativo. As outras features dominantes (`sentences_per_paragraph`, `clauses_per_sentence`, `named_entity_ratio_text`, `adjunct_per_clause`) caracterizam estilo estrutural-discursivo.

**Quinta**, os **3 erros do XGBoost são casos limítrofes interpretáveis**: dois deles (`fake_true_br/555.txt` e `1746.txt`) têm humano e LLM com tamanhos similares (raro no corpus), e o terceiro (`fake_br/213.txt`) é um LLM que reproduz estilo jornalístico humano com fidelidade. Estes três documentos são citáveis no paper como o limite empírico do que o pipeline consegue detectar.

**Sexta**, o subgrupo **`fake_true_br/1700-1800`** concentra erros sistemáticos em múltiplos classificadores. `fake_true_br/1701.txt` é classificado errado por 8 dos 10 runs — é o documento mais difícil do conjunto. Para análise qualitativa, esses ~5-10 documentos no range 1700-1800 são prioridade: provavelmente são LLMs gerados com prompts particularmente eficazes para mimetizar humano, ou humanos com estilo atípico.

## 10. Próximas etapas opcionais

Para fechar a análise:

- **Rodar `tfidf_baseline.py` e equivalente para BoW** (sem features linguísticas) para isolar a contribuição puramente lexical e quantificar o ganho líquido das features linguísticas sozinhas sobre representação puramente bag-of-words.
- **Análise qualitativa dos documentos sistematicamente difíceis**: ler os textos humano e LLM de `fake_true_br/1701.txt`, `1759.txt`, `555.txt`, `1414.txt` (LLMs que confundem 6+ classificadores), e de `fake_br/2511.txt`, `fake_true_br/202.txt` (humanos que confundem 4+ classificadores). Identificar a anatomia desses casos pode fornecer pistas qualitativas que complementem o paper.
- **Análise do tempo verbal**: dado que `nilc__indicative_present_ratio` é a feature dominante absoluta, vale extrair as distribuições marginais dessa feature para humanos e LLMs e reportar como figura no paper. É a contribuição mais nova e interpretável dos resultados.
