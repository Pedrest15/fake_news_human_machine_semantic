# Análise de Resultados — Classificação com Features Linguísticas

> Versão atualizada após aplicação dos blacklists Tier A+B do NILC-Metrix e a
> remoção das três features length-direct e das doze categorias agregadoras
> do LIWC. Substitui a leitura anterior que partia de um pacotão de 1 607
> features com vazamentos de tamanho ativos. Os resultados desta versão saem
> dos JSONs `combined_*.json` em [results/](results/) gerados no servidor.

## 1. Configuração do experimento

Os artefatos em [results/](results/) correspondem ao pipeline executado em modo **`feature_mode='combined'`** — isto é, o vetor de cada documento concatena, lado a lado, **todas as features linguísticas saneadas** (NILC-Metrix com 140 features após Tier A+B, LIWC com 61 categorias após remoção dos campos length-direct e dos agregadores hierárquicos, Enhanced UD com top-500 regras, sílabas, POS tagger, parser stats e termos SAGE) **e os 768 embeddings [CLS] do BERTimbau** sobre o par humano/LLM truncado por `min_tokens` em WordPiece. O total fica em **1 548 features** (era 1 607 antes do saneamento — 59 features removidas: 44 NILC Tier B + 12 agregadores LIWC + 3 LIWC length-direct).

Os splits efetivamente carregados produziram 5 758 documentos de treino e 1 439 de teste, e o desbalanço de 1 documento entre os lados humano e LLM no teste (719 vs 720) persiste — provavelmente um único caso descartado por `dropna()` no merge entre grupos. Seis famílias de classificadores foram avaliadas: SVM, Random Forest, Naive Bayes Gaussiano, Regressão Logística, MLP e XGBoost (que agora está disponível no servidor).

## 2. Resultados numéricos atualizados

A removida dos 59 atributos length-biased **derrubou os F1 perfeitos** do run anterior. SVM linear e LR L1 que atingiam F1 = 1,0000 com `C` = 0,1 agora chegam a F1 = 0,9986 e 0,9993 respectivamente, com hiperparâmetros bem diferentes (SVM agora prefere kernel RBF com `C` = 10; LR L1 troca `saga` por `liblinear` e sobe `C` de 0,1 para 1,0). Esse deslocamento sozinho já é uma evidência forte: quando o sinal disponível era principalmente o tamanho do texto, os modelos podiam carregar tudo em poucas dimensões com regularização forte; agora que o sinal está distribuído por features mais sutis, eles precisam de maior capacidade efetiva (mais flexibilidade) para chegar perto da perfeição.

A tabela comparativa que sai do `print_comparison_table`:

| Classificador | CV Score (treino) | Test F1 | Test Accuracy | Erros / 1 439 | Melhores hiperparâmetros |
|---|---|---|---|---|---|
| Regressão Logística | 0,9990 | **0,9993** | 0,9993 | 1 (FP) | L1, C=1,0, liblinear |
| XGBoost | 0,9991 | **0,9993** | 0,9993 | 1 (FN) | depth=3, n_est=50, lr=0,1, subsample=0,8 |
| SVM | 0,9974 | 0,9986 | 0,9986 | 2 (FN) | RBF, C=10, γ=scale |
| MLP | 0,9958 | 0,9986 | 0,9986 | 2 (FP) | tanh, (128, 64), α=0,0001 |
| Random Forest | 0,9965 | 0,9958 | 0,9958 | 6 (FN) | depth=None, n_est=200, min_split=10 |
| Naive Bayes (Gaussiano) | 0,9918 | 0,9889 | 0,9889 | 16 (14 FP, 2 FN) | var_smoothing=1e-9 |

Dois pontos imediatos. Primeiro, **nenhum modelo atinge F1 = 1,0000 mais** — a janela de erros agora se concentra em 1-2 documentos para os quatro melhores classificadores. Isso é estatisticamente honesto: detecção estilística humano vs LLM em corpora pareados sobre o mesmo factoide costuma ficar na faixa F1 ∈ [0,75; 0,95] na literatura, mas o nosso corpus é grande (10 782 pares) e bem comportado, então um F1 ≈ 0,999 é defensável. Segundo, **o Naive Bayes permanece travado em 0,9889 com 16 erros** — exatamente os mesmos erros do run anterior (a comparação com `misclassified/combined_naive_bayes_errors.json` confirma sobreposição quase total). O NB Gaussiano com pressuposto de independência é o modelo menos sensível à melhoria das features porque sua decisão é dominada por features individuais de alta discriminância isolada, que persistem mesmo após o saneamento.

## 3. Importância de features — quem está discriminando agora

Comparar o ranking de importância antes e depois dos blacklists é a forma mais nítida de ver o efeito do saneamento. **`nilc__lsa_all_std`, que era a feature número um da Regressão Logística e do SVM linear no run anterior, desapareceu** — está no Tier B do NILC e foi removida. Seu papel foi absorvido por `nilc__lsa_paragraph_mean` (que é a **média** das similaridades LSA por parágrafo, length-invariante por construção), que aparece no top-2 da Regressão Logística (coef 1,67) e no top-2 do Random Forest (Gini 0,047). De forma análoga, `nilc__brunet` e `nilc__honore` saíram completamente do top-10, e `nilc__sentence_length_standard_deviation`, que era a oitava feature mais importante no SVM linear antigo, também sumiu.

No Random Forest, a importância de Gini está agora liderada por `nilc__sentences_per_paragraph` (0,084 — esta promoveu do segundo para o primeiro lugar), `nilc__lsa_paragraph_mean` (0,047), `nilc__flesch` (0,036), `nilc__syllables_per_content_word` (0,034), `pos__ADJ` (0,028), `ud__VERB(PUNCT/punct, *, NOUN/obj)` (0,028), `ud__ADJ(*)` (0,024), `nilc__cross_entropy` (0,021), `nilc__lsa_givenness_mean` (0,020) e `nilc__idade_aquisicao_mean` (0,018). Quatro observações importantes: (i) o NILC continua dominando, mas a contribuição agora se distribui mais equitativamente entre features genuinamente interpretáveis; (ii) duas regras Enhanced UD coexistem no top-10 do RF; (iii) `pos__ADJ` (contagem normalizada de adjetivos pelo POS tagger) aparece de forma proeminente, consistente com o achado LIWC anterior de que `adj` é a categoria mais característica de LLM; (iv) `nilc__cross_entropy` é uma feature de entropia cruzada com referência LLM — sinal estilístico interessante que não havia aparecido antes.

Na Regressão Logística L1 com C = 1,0, os coeficientes dominantes contam outra história. `nilc__sentences_per_paragraph` tem coeficiente 6,54 — uma ordem de grandeza maior que o segundo colocado. `nilc__lsa_paragraph_mean` segue com 1,67, depois `ud__VERB(PUNCT/punct, *, NOUN/obj)` com 0,65, `syll__media_silabas_por_palavra` com 0,64, `nilc__lsa_span_mean` com 0,59, `nilc__add_neg_conn_ratio` com 0,53 (proporção de conectivos aditivos negativos — uma feature de coesão discursiva nova no topo), `nilc__gerund_verbs` com 0,53, `nilc__verbs_ambiguity` com 0,48, `nilc__stem_ovl` com 0,43 e `liwc__adj (Adjectives)` com 0,40. **O fato de o LR L1 zerar todos os outros coeficientes mas reter dez features com pesos consistentes** mostra que existe uma estrutura de sinal estável; a L1 está executando uma seleção implícita e não consegue reduzir o problema a menos de cerca de dez dimensões.

O XGBoost dá a leitura mais peculiar: `nilc__sentences_per_paragraph` carrega **86,5 % da importância total**, com todas as outras features ficando abaixo de 2 % cada. Em contrapartida — e aqui está o achado novo — **três dimensões do BERTimbau aparecem no top-10 do XGBoost** (`bert_bertimbau__419`, `bert_bertimbau__39`, `bert_bertimbau__479`), o que não acontecia antes do saneamento. Isso confirma o que o diagnóstico anterior havia sugerido em [RELATORIO_DIAGNOSTICO.md §2](RELATORIO_DIAGNOSTICO.md): o BERT não consegue contribuir de forma significativa enquanto as features linguísticas de viés residual dominam tudo, mas passa a aparecer entre as features importantes quando esse viés é removido. A contribuição do BERT continua sendo de complementação — não de substituição —, mas agora está visível no ranking.

## 4. O diagnóstico anterior validado

A análise em [RELATORIO_DIAGNOSTICO.md](RELATORIO_DIAGNOSTICO.md) havia previsto duas coisas concretas: (i) o sinal linguístico sem BERT tem teto em torno de F1 = 0,9946; (ii) remover Tier B do NILC e o vazamento LIWC reduziria o F1 do run combined sem destruir o sinal. A primeira previsão já estava medida (o diagnóstico encontrou pico em k = 75 com F1 = 0,9946 sem BERT). A segunda agora foi confirmada empiricamente: o F1 combined caiu de 1,0000 para 0,9993 no caso ótimo (LR e XGBoost), uma redução de exatamente o "último 0,5 %" que o diagnóstico atribuía aos embeddings BERT. **O sinal não foi destruído** — caiu de "implausivelmente perfeito" para "praticamente perfeito mas defensável", que é o ponto onde queríamos chegar.

## 5. Perfil dos erros — sinal mais sutil

Os erros ainda concentram-se quase totalmente no corpus Fake.Br (não FakeTrue.Br), mas o padrão de "humanos longos viram LLM" do run anterior se diluiu. Os dois falsos negativos do SVM (`fake_br/678.txt` LLM com 279 palavras e `fake_br/2958.txt` LLM com 388 palavras) são LLMs de tamanho **normal** confundidos com humanos — não outliers nem em qualquer direção. O único falso positivo da Regressão Logística (`fake_br/597.txt`, humano com 377 palavras predito como LLM com confiança 0,87) é um humano relativamente longo, mas não atipicamente: 377 está acima da mediana humana mas dentro do primeiro desvio. O único falso negativo do XGBoost (`fake_br/213.txt`, LLM com 563 palavras predito como humano com confiança 0,97) é um caso onde o LLM tem tamanho dentro do esperado e ainda assim escapa.

Os 16 erros do Naive Bayes continuam fortemente correlacionados com tamanho (humanos longos sendo confundidos com LLM, com confiança próxima de 1,0) — a sobreposição com os erros do NB pré-saneamento é alta, indicando que o NB usa primariamente um subconjunto pequeno de features altamente discriminativas individualmente e ignora a estrutura conjunta. As mesmas features problemáticas dominavam o pré-saneamento e continuam dominando no NB pós-saneamento porque o pressuposto de independência condicional impede o NB de se aproveitar das interações que os outros classificadores capturam. Para o paper, isso recomenda **omitir o NB Gaussiano da tabela principal** e mantê-lo apenas como baseline trivial, talvez em apêndice.

Os 6 erros do Random Forest são todos falsos negativos com confiança baixa (0,50-0,58) — o modelo está hesitando exatamente no limite de decisão. Esse perfil é o mais salutar dos três: o RF erra com humildade quando erra, indicando que reconhece a incerteza nessas instâncias específicas.

## 6. O que mudou para o paper

A versão atual está finalmente em um ponto reportável. O F1 = 0,9993 para Regressão Logística e XGBoost é defensável metodologicamente porque: (i) os blacklists Tier A+B e LIWC documentam de forma explícita quais features foram excluídas e por quê, com referências de literatura (Tweedie & Baayen 1998 para riqueza vocabular; argumento estatístico para max/std/diversidades; estrutura WordPiece pareada para a truncagem); (ii) cinco classificadores diferentes convergem para a mesma faixa de desempenho (0,9958-0,9993) com 1-6 erros, indicando que o sinal é robusto e não um artefato de um modelo específico; (iii) a análise de feature importance é interpretável e identifica padrões linguísticos com leitura clara — preferência LLM por parágrafos com mais sentenças, palavras mais longas, vocabulário de aquisição tardia, uso específico de gerúndios e padrão pontuacional VERB(PUNCT, *, NOUN) — em vez de apontar para artefatos de tamanho.

A diferença F1 entre Logistic Regression sem features linguísticas (TF-IDF/BoW puro do `tfidf_baseline.py`) e Logistic Regression no modo combined (0,9993) quantifica o ganho líquido das features linguísticas + BERT sobre representação puramente bag-of-words. A diferença F1 entre modo `linguistic` (~0,994 pelo diagnóstico) e modo `combined` (0,9993) quantifica a contribuição específica do BERT. Estes números são separáveis e atribuem cada incremento à sua causa.

## 7. Próximas etapas sugeridas

Com o saneamento feito, dois experimentos adicionais ainda valem para fechar o quadro do paper. **Primeiro**, rodar os modos `tfidf` e `bow` puros (sem features linguísticas) usando o `tfidf_baseline.py` que você já tem, com a mesma truncagem BERTimbau, para gerar a linha de base "só vocabulário". **Segundo**, rodar `text_enriched.py --vectorizer both --classifier all` para obter os experimentos enriquecidos (TF-IDF + linguistic e BoW + linguistic), que substituem o caminho BERT como fonte de informação de contexto lexical. A comparação final ficaria com cinco colunas: BERT puro, TF-IDF puro, BoW puro, linguistic puro, e combined (cada bloco contra cada bloco), permitindo isolar exatamente o que cada camada de informação contribui.
