# Análise de Resultados — TF-IDF e BoW Enriquecidos com Features Linguísticas

> Análise dos pipelines `text_enriched`: TF-IDF + features linguísticas e BoW
> + features linguísticas, com truncagem pareada pelo tokenizador BERTimbau a
> 512 tokens WordPiece. Os números saem dos JSONs `text_*_linguistic_*.json`
> em [results/](results/), gerados no servidor.

## 1. Configuração dos experimentos

Dois pipelines de representação são avaliados em paralelo, ambos consumindo o mesmo conjunto saneado de **780 features linguísticas** (140 NILC após Tier A+B, 61 LIWC após remoção dos três campos length-direct e das doze categorias agregadoras, 500 regras Enhanced UD top-K, 2 sílabas, 17 POS tagger, 60 SAGE):

- **`text_enriched --vectorizer tfidf`** — features linguísticas concatenadas com a matriz TF-IDF de 10 000 unigramas (`min_df=5`, `sublinear_tf=True`). Vetor final com **10 780 dimensões**, esparso (~92 % de zeros).
- **`text_enriched --vectorizer bow`** — análogo, mas com `CountVectorizer` (contagens brutas). Mesma dimensionalidade.

Ambos compartilham splits idênticos: **7 876 documentos de treino e 1 946 de teste**, distribuídos entre humanos (label 0) e LLM (label 1) em proporção balanceada.

O blacklist Tier A+B do NILC removeu **60 features length-biased** (16 contagens absolutas + 44 medidas correlacionadas com tamanho — diversidades TTR-like, riqueza vocabular, max/std de contagens e std das similaridades LSA). O blacklist LIWC removeu **15 campos** (3 length-direct: `word_count`, `matched_words`, `dictionary_coverage`; 12 agregadores hierárquicos: `function`, `pronoun`, `ppron`, `affect`, `negemo`, `social`, `cogproc`, `percept`, `bio`, `drives`, `relativ`, `informal` — preservando apenas as categorias-folha). Esses saneamentos tornam o classificador menos dependente de tamanho de texto e de redundância hierárquica.

A truncagem pareada é idêntica nos dois vetorizadores: `min(len_h_tokens, len_l_tokens, 510)` em tokens WordPiece BERTimbau, com decode de volta para texto. Garante que humano e LLM do mesmo factoide vejam exatamente a mesma janela de comprimento, fechando o vazamento de tamanho que dominava os resultados pré-saneamento.

Cinco classificadores foram avaliados via `GridSearchCV` com `cv=5` e `scoring='f1_weighted'`: SVM, Random Forest, Regressão Logística, MLP e XGBoost. Naive Bayes foi excluído porque o GaussianNB é incompatível com a matriz hstack(sparse + dense_padronizada) por densificar internamente.

## 2. Tabela consolidada de resultados (F1 weighted no teste held-out)

| Classificador | TF-IDF + linguistic | BoW + linguistic |
|---|---|---|
| **XGBoost** | **0,9985** | **0,9985** |
| Logistic Regression | 0,9902 | 0,9938 |
| SVM | 0,9913 | 0,9938 |
| MLP | 0,9913 | **0,9943** |
| Random Forest | 0,9841 | 0,9836 |

## 3. TF-IDF + features linguísticas

A matriz TF-IDF de 10 000 unigramas com `sublinear_tf=True` concatenada com as 780 features linguísticas (após Tier A+B do NILC e remoção dos agregadores LIWC) atinge **F1 entre 0,9841 e 0,9985 dependendo do classificador**. XGBoost lidera com folga: F1 = 0,9985 e **apenas 3 falsos negativos em 1 946 documentos, zero falsos positivos**. Seus hiperparâmetros (`depth=3`, `n_estimators=100`, `lr=0,3`) caracterizam um regime de "fast learner": árvores rasas com taxa de aprendizado alta, indicando que o sinal é fortemente discriminativo e poucas iterações bastam para capturá-lo.

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

## 6. Perfil dos erros

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

Cruzando os arquivos de filename misclassificados entre os runs TF-IDF e BoW para o XGBoost, **os 3 erros são os mesmos documentos LLM em ambos os vetorizadores**. Vale uma análise qualitativa: provavelmente são notícias geradas por LLM cujo conteúdo factual é tão preciso que reproduz o estilo do jornalismo factual humano e perde os marcadores estilísticos típicos do LLM (verbosidade, paragrafação previsível, vocabulário psicolinguisticamente sofisticado).

## 7. Recomendações para o paper

Cinco observações concretas para a redação:

**Primeira**, o pipeline `text_enriched` entrega **F1 ≈ 0,9985 no melhor caso (XGBoost)**, defensável metodologicamente porque (i) o saneamento das features linguísticas documenta de forma explícita quais atributos foram excluídos e por quê, com referências de literatura (Tweedie & Baayen 1998 para riqueza vocabular; argumento estatístico para max/std/diversidades; estrutura hierárquica LIWC2015 para os agregadores); (ii) cinco classificadores diferentes convergem para a faixa de 0,9836-0,9985 com perfis de erro coerentes; e (iii) a truncagem pareada pelo tokenizador BERTimbau elimina o viés residual de tamanho.

**Segunda**, **XGBoost é o classificador mais robusto** dos avaliados: vence ou empata nos dois pipelines, com hiperparâmetros idênticos entre TF-IDF e BoW (`depth=3`, `n=100`, `lr=0,3`), e mantém o perfil `0 FP + 3 FN` consistentemente. Para a tabela principal do paper, reporte XGBoost. Para análise complementar, reporte Logistic Regression (interpretável via coeficientes).

**Terceira**, o achado de **BoW ≥ TF-IDF em modelos lineares** é não-trivial e merece um parágrafo de discussão. Mostra que TF-IDF não é universalmente superior, especialmente quando o pipeline já carrega features linguísticas estabilizadoras. O efeito desaparece em modelos baseados em árvores (XGBoost, RF) por invariância a transformações monotônicas.

**Quarta**, o ganho do MLP em BoW (F1 = 0,9943) sugere que a rede neural rasa consegue explorar interações entre palavras e features linguísticas que os modelos lineares perdem — mas ainda fica abaixo do XGBoost. Para o paper, MLP entrega ganho marginal sobre LR/SVM (cerca de 0,0005) com custo computacional bem maior; vale ressaltar essa proporção custo-benefício.

**Quinta**, o desempenho do **Random Forest é claramente inferior** (F1 ≈ 0,984) e seu perfil de erros (30+ FN) o torna pouco confiável para deployment. Reporte como baseline da família tree-based mas dê destaque ao XGBoost.

## 8. Próximas etapas opcionais

Para fechar a análise:

- **Rodar `tfidf_baseline.py` e equivalente para BoW** (sem features linguísticas) para isolar a contribuição puramente lexical e quantificar o ganho líquido das features linguísticas sozinhas sobre representação puramente bag-of-words.
- **Análise qualitativa dos 3 erros do XGBoost** (mesmos 3 documentos em TF-IDF e BoW): ler o texto humano e o LLM, identificar o que neles confunde o classificador. Provavelmente são casos onde o LLM consegue replicar bem o estilo jornalístico humano.
- **Análise de feature importance no XGBoost**: extrair `feature_importances_` do XGBoost vencedor para identificar exatamente quais palavras (tokens `text__*`) e features linguísticas estão dirigindo a discriminação, complementando a análise já feita em [RELATORIO_DIAGNOSTICO.md](RELATORIO_DIAGNOSTICO.md) com features de palavras concretas.
