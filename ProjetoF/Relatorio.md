# A Jornada de Melhoria da Recuperação em RAG — Cadeia de Custódia da Prova Pericial

**Disciplina:** RAG & CAG Aplicados a Direito e Segurança Pública (MBA IBMEC)
**Autor:** Everaldo
**Ponto de partida:** aplicação do Projeto Final (`aula12/projeto_final/`), copiada para `ProjetoF/`

> Rascunho em construção — preenchido fase a fase, conforme os experimentos são medidos. Seções marcadas como `[PENDENTE]` ainda não têm dados.

---

## 1. A fonte de dados e por quê

Fonte escolhida: **"Cadeia de Custódia da Prova Pericial: uma análise da Lei 13.964/2019"**
(GIACOMOLLI; AMARAL, 2020) — artigo acadêmico em PDF nativo (com camada de texto,
não escaneado), tratando da cadeia de custódia de vestígios/provas periciais no
processo penal brasileiro, a partir da Lei 13.964/2019 (Pacote Anticrime).

Escolhido por ser diretamente relacionado ao tema da disciplina (Direito e Segurança
Pública), ter estrutura mista — texto corrido, seções tituladas, referência a artigos
de lei e a jurisprudência (Corte Interamericana de Direitos Humanos) — o que permite
testar diferentes técnicas de chunking e recuperação ao longo da jornada.

Motivação prática: o autor integra uma equipe de **Peritos Criminais**, para quem a
cadeia de custódia da prova pericial é um tema central da rotina profissional (rastreio
e integridade de vestígios do local de crime até o laudo). O caso de uso real de um
sistema RAG sobre esse tema seria apoiar peritos e operadores do direito a consultar
rapidamente a legislação e a doutrina sobre cadeia de custódia — o que também orienta
o tipo de pergunta incluído no gabarito (Seção 2): dúvidas práticas de procedimento
(lacre, embalagem, central de custódia), não só teóricas.

## 2. O dataset de avaliação (gabarito)

`avaliacao/dataset.json` — construído manualmente sobre o artigo:

- **18 documentos (D01–D18)**: segmentos temáticos do artigo, distribuídos ao longo
  de todo o texto (não são os chunks reais gerados pela ingestão — servem de unidade
  de gabarito).
- **30 perguntas (`queries_benchmark`)**, em 3 tipos (Seção 5 do `Roteiro_Final.md`):
  - **factual** (Q01–Q12): resposta está em 1 segmento.
  - **reformulável** (Q13–Q22): mesmo conteúdo, mas em linguagem coloquial/leiga —
    testa o ganho de multi-query / step-back.
  - **multihop** (Q23–Q30): resposta exige combinar 2–3 segmentos — testa o ganho de
    RAG-Fusion / grafo.
- Perguntas escritas em linguagem natural, evitando copiar termos exatos do texto
  (para não inflar artificialmente a baseline — lição da Seção 5 do roteiro).
- `relevancia`: nota 2 (muito relevante) / 1 (relevante) por documento, para NDCG
  graduado.

**Como a recuperação é avaliada sem `id_original` estável:** o pipeline do projeto
não grava um id de chunk rastreável até o documento de origem. A avaliação identifica
cada chunk recuperado **pelo conteúdo**, procurando "marcadores" (trechos distintivos)
de cada D0x no texto do chunk (ver `avaliacao/README.md` para o detalhe). É uma
aproximação documentada, revisável via `avaliacao/detalhe_<exp>.json`.

## 3. Metodologia

- **Métricas de recuperação:** Hit@5, Recall@5, MRR, NDCG@10 (`avaliacao/avaliar_recuperacao.py`).
- **Métricas de geração (RAGAS):** Faithfulness, Answer Relevancy, Context Precision/Recall —
  `[PENDENTE]`, rodam ao final (Fase 8), na baseline e na melhor configuração.
- **top_k:** 10 chunks recuperados por consulta (cobre Hit@5/Recall@5 e NDCG@10 na
  mesma rodada).
- **Experimento controlado:** uma variável por vez. Cada fase reindexa o OpenSearch
  fixando tudo o que não está sendo testado (mesmo embedding, mesma técnica de busca,
  mesmo top_k) e o script de cada fase restaura o índice para o estado da baseline
  ao final, para a próxima fase não herdar configuração errada.
- **Registro:** uma linha por experimento em `avaliacao/resultados.csv` (`exp`, `fase`,
  `mudança`, métricas, observação) + `avaliacao/detalhe_<exp>.json` com os chunks
  recuperados por pergunta (auditoria).
- **Infra:** OpenSearch local (Docker), embeddings via Ollama (`nomic-embed-text`,
  768d), geração via Groq (LLM agnóstico, `llama-3.3-70b-versatile`).

## 4. Baseline (Fase 0)

Projeto rodado como veio: ingestão do PDF forçando `estrategia=opensearch` (a
heurística automática do projeto rotearia para o grafo/LightRAG por causa da
densidade de entidades nomeadas do texto — decisão documentada, ver Seção 8.2),
chunking `auto` (a heurística escolheu `hierárquico`, por o texto ter ≥3 títulos),
embedding `nomic-embed-text`, busca `baseline` (sem reescrita de query), `top_k=10`.
143 chunks indexados.

| Métrica | Valor |
|---|---|
| Hit@5 | 0,633 |
| Recall@5 | 0,422 |
| MRR | 0,351 |
| NDCG@10 | 0,338 |
| Latência média | 2,72 s |

Ponto de partida: cobre a maioria das perguntas no top-5 (Hit@5 alto), mas o relevante
raramente vem em 1º lugar (MRR baixo) e o ranking fino ainda é fraco (NDCG@10 baixo) —
espaço claro de melhoria nas próximas fases.

## 5. Experimentos por fase

### Fase 1 — Extração (`exp02_extracao_pymupdf`)

**Hipótese:** texto mal extraído limita tudo a jusante (Docling, com sua camada de
markdown/estrutura, deveria extrair melhor que um fallback de texto puro).

**Mudança:** reextração do mesmo PDF via PyMuPDF (texto puro, sem markdown/OCR),
mantendo o chunking **fixo** em `hierárquico` (igual ao baseline) para isolar a
extração como única variável (`avaliacao/rodar_fase1_extracao.py`).

**Resultado:**

| Métrica | exp01 (Docling) | exp02 (PyMuPDF) | Δ |
|---|---|---|---|
| Hit@5 | 0,633 | 0,700 | +0,067 |
| Recall@5 | 0,422 | 0,489 | +0,067 |
| MRR | 0,351 | 0,436 | +0,085 |
| NDCG@10 | 0,338 | 0,415 | +0,077 |
| Chunks gerados | 143 | 134 | — |

**Análise:** resultado contraintuitivo — o extrator "mais simples" (PyMuPDF) venceu
o Docling em **todas** as métricas de recuperação. Hipótese para o motivo: a marcação
markdown do Docling (`#`, `##`, `|` de tabela) introduz tokens que não aparecem nas
perguntas em linguagem natural dos usuários, diluindo a similaridade semântica dos
embeddings; além disso, como o `HierarchicalDocumentSplitter` conta palavras para
decidir os limites de chunk, os símbolos de markdown deslocam esses limites — daí os
143 chunks do Docling vs 134 do PyMuPDF para o mesmo texto-fonte. Amostras salvas em
`avaliacao/amostras_extracao/{docling,pymupdf}.txt` para inspeção manual.

Achado relevante para o relatório final: o Docling costuma ganhar em documentos
complexos (múltiplas colunas, tabelas, PDFs escaneados) — aqui, num PDF acadêmico de
coluna única e bem formatado, a extração "burra" saiu na frente. Vale frisar isso
como *trade-off*, não como regra geral.

*(As próximas fases usam o índice restaurado ao estado do Docling/baseline — o ganho
do PyMuPDF nesta fase ainda não foi "adotado" como configuração corrente; decisão
sobre qual extração vira a base das fases seguintes fica para a Seção 7 deste
relatório, ao consolidar tudo.)*

### Fase 2 — Chunking (`exp03`–`exp06`)

**Hipótese:** a granularidade do chunk muda o que é recuperável.

**Mudança:** com a extração fixa em Docling (a mesma do baseline/exp01), testadas as
outras 4 técnicas de chunking do projeto — `fixo`, `recursivo`, `sentença_janela`,
`semântico` — cada uma reindexando do zero (`avaliacao/rodar_fase2_chunking.py`).
`hierárquico` não foi reindexado de novo: é exatamente a configuração do exp01.

**Resultado:**

| Técnica | Hit@5 | Recall@5 | MRR | NDCG@10 | Chunks |
|---|---|---|---|---|---|
| hierárquico (exp01) | 0,633 | 0,422 | 0,351 | 0,338 | 143 |
| fixo (exp03) | 0,467 | 0,306 | 0,287 | 0,312 | 72 |
| recursivo (exp04) | 0,600 | 0,372 | **0,418** | **0,426** | 81 |
| sentença_janela (exp05) | 0,533 | 0,383 | 0,338 | 0,416 | 240 |
| semântico (exp06) | 0,400 | 0,317 | 0,336 | 0,358 | 96 |

**Análise:** nenhuma técnica domina em tudo — há um trade-off entre *cobertura* e
*qualidade do ranking*. `hierárquico` continua sendo o melhor em Hit@5/Recall@5 (traz
o relevante no top-5 com mais frequência), mas `recursivo` tem o melhor MRR e NDCG@10
(quando acerta, acerta mais perto do topo do ranking) — apesar de gerar só 81 chunks
(bem menos que os 143 do hierárquico), sugerindo que blocos de ~200 palavras
respeitando frase/parágrafo capturam o suficiente de contexto sem diluir a
similaridade com blocos grandes demais. `fixo` (chunks cegos de 200 palavras, sem
respeitar limites de frase) e `semântico` (corte por mudança de tópico, 96 chunks)
saíram piores em todas as métricas — no caso do `semântico`, possivelmente porque o
artigo é curto e coeso o bastante para o corte por tópico não trazer benefício real,
só reduzir a granularidade. `sentença_janela` gerou muito mais chunks (240, blocos de
3 sentenças) e teve o 2º melhor NDCG@10, mas ficou atrás em Hit@5/Recall@5.

Como o roteiro pede otimizar recuperação (cobertura) antes de geração, e Hit@5/Recall@5
são as métricas de cobertura, `hierárquico` segue como chunking de referência para as
próximas fases — mas `recursivo` fica marcado como candidato forte a reavaliar quando
somarmos reranking (Fase 6), já que reranking tende a amplificar o benefício de um MRR
melhor.

*(Índice restaurado para `hierárquico` ao final da fase, para a Fase 3 partir do
mesmo estado do baseline.)*

### Fase 3 — Modelo de embedding (`exp07`–`exp11`)

**Hipótese:** o modelo de embedding define o teto da busca densa.

**Mudança:** com extração (Docling) e chunking (`hierárquico`) fixos, testados todos
os modelos de embedding baixados no Ollama local além do baseline
(`nomic-embed-text`): `bge-m3`, `mxbai-embed-large` e as 3 variantes de tamanho do
`qwen3-embedding` (0.6b/4b/8b). Dimensão de cada modelo medida em runtime — os Qwen3
não estavam mapeados em `app/config.py::DIMENSAO_EMBEDDING`
(`avaliacao/rodar_fase3_embedding.py`).

**Resultado:**

| Modelo | Dim. | Hit@5 | Recall@5 | MRR | NDCG@10 |
|---|---|---|---|---|---|
| nomic-embed-text (exp01, baseline) | 768 | 0,633 | 0,422 | 0,351 | 0,338 |
| bge-m3 (exp07) | 1024 | 0,733 | 0,572 | 0,505 | 0,618 |
| mxbai-embed-large (exp08) | 1024 | 0,467 | 0,328 | 0,269 | 0,253 |
| qwen3-embedding:0.6b (exp09) | 1024 | 0,833 | 0,639 | 0,527 | 0,618 |
| qwen3-embedding:4b (exp10) | 2560 | **0,933** | 0,700 | **0,661** | 0,775 |
| qwen3-embedding:8b "latest" (exp11) | 4096 | **0,933** | **0,733** | 0,615 | **0,775** |

**Análise:** de longe a fase com maior ganho da jornada até aqui. A família
Qwen3-Embedding varre o baseline em todas as métricas, mesmo na menor variante
(0.6b): Hit@5 salta de 0,633 para 0,833–0,933, e o MRR praticamente dobra. Dentro da
família Qwen3, o ganho de 0.6b → 4b é grande (MRR 0,527 → 0,661), mas de 4b → 8b o
ganho é marginal e misto (8b vence em Recall@5, mas *perde* em MRR para o 4b) — mais
que dobrar a dimensão (2560 → 4096) e o tamanho do modelo não se traduziu em ganho
proporcional. Custo-benefício: **qwen3-embedding:4b** parece o melhor equilíbrio
(dimensão bem menor que o 8b, latência parecida, métricas equivalentes ou melhores).

`bge-m3` também superou o baseline com folga (2º melhor colocado no geral) — resultado
esperado, é um modelo multilíngue forte e mais recente que o `nomic-embed-text`.
`mxbai-embed-large`, por outro lado, ficou **abaixo do baseline** em todas as
métricas — resultado negativo relevante: nem todo modelo "maior"/mais recente
supera o `nomic-embed-text` neste corpus; dimensão (1024) não é garantia de
qualidade — vale registrar como contraexemplo no relatório final.

`qwen3-embedding:4b` é o forte candidato a entrar na configuração final (Seção 7);
a decisão de **quando** passar a usá-lo como base fixa das próximas fases (em vez de
seguir isolando tudo contra o `nomic-embed-text` do exp01) está registrada em
"8.1 Decisões técnicas relevantes já tomadas".

*(Nota técnica: `qwen3-embedding:0.6b` falhou na primeira tentativa por um erro de
conexão com o Ollama local — resolvido rodando o script de novo, que é resumível e
tentou de novo só esse modelo.)*

### Fase 4 — Recuperação base (top_k e busca híbrida) (`exp12`–`exp16`)

**Hipótese:** combinar busca lexical (BM25) com a densa deveria aumentar a cobertura
(vestígios de termos exatos da lei que a busca puramente semântica pode não priorizar);
variar `top_k` também deveria afetar o ranking fino medido pelo NDCG@10.

**Mudança:** a partir desta fase, a base fixa passa a ser Docling + `hierárquico` +
`qwen3-embedding:4b` (jornada progressiva, ver Seção 8.1). Testada a nova técnica
`hibrida` (BM25 + densa via RRF, `OpenSearchHybridRetriever` — novidade em
`app/busca_avancada.py`) contra a busca densa pura, em `top_k` 5/10/20
(`avaliacao/rodar_fase4_hibrida_topk.py`). `densa/top_k=10` não foi remedido: é
exatamente o `exp10` da Fase 3, já na mesma base.

**Resultado:**

| Combinação | Hit@5 | Recall@5 | MRR | NDCG@10 |
|---|---|---|---|---|
| densa top_k=5 (exp12) | 0,933 | 0,700 | 0,657 | 0,674 |
| densa top_k=10 (exp10, referência) | 0,933 | 0,700 | 0,661 | 0,775 |
| densa top_k=20 (exp13) | 0,933 | 0,700 | 0,661 | 0,775 |
| híbrida top_k=5 (exp14) | 0,800 | 0,633 | 0,548 | 0,599 |
| híbrida top_k=10 (exp15) | 0,900 | 0,683 | 0,587 | 0,682 |
| híbrida top_k=20 (exp16) | 0,800 | 0,600 | 0,569 | 0,660 |

**Análise:** resultado contraintuitivo de novo — a busca híbrida (BM25+densa via RRF)
ficou **abaixo** da busca densa pura em todas as métricas, em qualquer `top_k` testado.
A hipótese mais provável: com `qwen3-embedding:4b` a busca densa já está muito forte
neste corpus (Hit@5=0,933), então o BM25 tem pouco a contribuir de complementar — ao
contrário, ele traz para o ranking fundido chunks lexicamente parecidos mas
semanticamente menos relevantes (termos jurídicos repetidos em várias seções do
artigo), e o RRF acaba empurrando para baixo alguns acertos fortes da busca densa.
Dentro da própria família híbrida, `top_k=10` (exp15) foi consistentemente o melhor
dos três, sugerindo que `top_k` muito baixo (5) corta cedo demais um ranking já mais
ruidoso, e `top_k` muito alto (20) dilui ainda mais com chunks de cauda longa do BM25 —
mas mesmo o melhor ponto da híbrida não alcançou a densa pura.

Sobre `top_k` isolado (mantendo densa): `top_k=5` já mantém Hit@5/Recall@5 idênticos a
`top_k=10/20` (o relevante já está garantido nas top-5 posições, se está), mas o
NDCG@10 despenca (0,674 vs 0,775) — efeito mecânico, não de qualidade de busca: com
`top_k=5` só existem 5 documentos para calcular um NDCG que olha até a posição 10, o
que já penaliza o score independente da real relevância. `top_k=10` e `top_k=20` deram
resultado idêntico em todas as métricas — ampliar a janela além de 10 não trouxe nem
tirou nada do ranking dentro das primeiras 10 posições, ou seja, não há chunk relevante
"escondido" entre as posições 11-20 nesta base. Conclusão prática: `top_k=10` segue
sendo o ponto de equilíbrio (não perde nada de `top_k=20` e evita o corte artificial de
`top_k=5`), e a busca densa pura segue como configuração de recuperação de referência
até aqui — a híbrida fica descartada nesta base, mas registrada como resultado negativo
relevante para a análise crítica (Seção 8).

*(Nota técnica: a primeira tentativa desta fase teve o OpenSearch caindo por ~8s no
meio da rodada do `exp15_hibrida_top10` — 23 das 30 perguntas falharam por erro de
conexão. A linha malformada (`n_queries=7`) foi removida do `resultados.csv` e o
script foi rodado de novo — resumível, então só refez o `exp15`; os demais
experimentos (que já tinham completado 30/30) foram pulados.)*

*(Índice **não** foi restaurado ao final desta fase — fica na base
Docling+hierárquico+qwen3-embedding:4b para a Fase 5 em diante, por causa da mudança
de metodologia isolada→progressiva, ver Seção 8.1.)*

### Fase 5 — Query enhancement (`exp17`–`exp19`)

**Hipótese:** reescrever a pergunta original (gerar variações ou uma versão mais
genérica) deveria ajudar a recuperar trechos que a formulação original, sozinha, não
acha — sobretudo nas perguntas `reformulável` (linguagem coloquial) e `multihop`
(Seção 2).

**Mudança:** testadas as 3 técnicas de query enhancement já implementadas em
`app/busca_avancada.py` — `multi_query` (LLM gera variações, funde por dedup),
`rag_fusion` (mesma ideia, funde por RRF) e `step_back` (LLM gera uma pergunta mais
geral, busca [específica + geral]) — contra a busca densa pura (`exp10`), todas em
`top_k=10` (ponto de equilíbrio confirmado na Fase 4). Nenhuma reindexação: a base
segue Docling + `hierárquico` + `qwen3-embedding:4b`, herdada da Fase 4
(`avaliacao/rodar_fase5_query_enhancement.py`).

**Resultado:**

| Combinação | Hit@5 | Recall@5 | MRR | NDCG@10 |
|---|---|---|---|---|
| densa top_k=10 (exp10, referência) | 0,933 | 0,700 | 0,661 | 0,775 |
| multi_query (exp17) | 0,933 | 0,722 | 0,616 | 0,705 |
| rag_fusion (exp18) | 0,933 | 0,706 | 0,579 | 0,723 |
| step_back (exp19) | 0,900 | 0,706 | 0,624 | 0,756 |

**Análise:** o padrão da Fase 4 se repete nas 3 técnicas — todas ficaram **abaixo**
da busca densa pura no MRR, e 2 das 3 (`multi_query`, `rag_fusion`) também ficaram
abaixo no NDCG@10, com ganho só marginal em Recall@5 (entre +0,006 e +0,022).
Hipótese: com a busca densa já tão forte (`qwen3-embedding:4b`), somar consultas
extras (variações do `multi_query`/`rag_fusion`, a pergunta genérica do `step_back`)
traz para o ranking documentos mais diversos mas em média menos precisos, diluindo a
posição dos acertos certeiros da consulta original — o mesmo mecanismo observado na
busca híbrida (Fase 4). Confirma-se também a leitura de que "mais consultas fundidas
dilui mais o ranking": `step_back` (2 consultas, dedup) teve o melhor MRR e NDCG@10
das 3 técnicas; `multi_query` (5 consultas, dedup) teve o melhor Recall@5 mas o pior
NDCG@10; `rag_fusion` (5 consultas, RRF) ficou no meio em Recall@5/NDCG@10 mas teve o
pior MRR de todos — a fusão por RRF, apesar de teoricamente mais robusta que dedup,
não superou o dedup simples neste corpus pequeno e coeso.

Conclusão da fase: nenhuma das 3 técnicas de query enhancement superou a busca densa
pura de referência no ranking fino (MRR/NDCG@10) — mesmo padrão da Fase 4 (busca
híbrida também perdeu para a densa pura). A busca densa pura com `qwen3-embedding:4b`
segue como configuração de recuperação de referência até aqui.

*(Nota técnica: `multi_query` (`exp17`) falhou duas vezes antes de completar 30/30 —
1ª tentativa esgotou a cota diária de tokens do Groq (TPD) no meio da rodada (11/30
antes de travar); numa 2ª tentativa, já com uma chave de API nova, só 2/30 perguntas
passaram (latência média de 30s sugeria timeout/retentativas). Ambas as linhas
malformadas foram descartadas do `resultados.csv` antes da rodada final, que
completou 30/30 sem falhas — a causa exata das 2 tentativas anteriores não foi
diagnosticada a fundo, mas não se repetiu na rodada final.)*

### Fase 6 — Reranking

`[PENDENTE]`

### Fase 7 — Técnica avançada

`[PENDENTE]`

### Fase 8 — RAGAS (avaliação da geração)

`[PENDENTE]`

## 6. Tabela consolidada

| exp | fase | mudança | Hit@5 | Recall@5 | MRR | NDCG@10 | latência(s) |
|---|---|---|---|---|---|---|---|
| exp01_baseline | Fase 0 | baseline (Docling, hierárquico, nomic-embed-text, busca=baseline) | 0,633 | 0,422 | 0,351 | 0,338 | 2,72 |
| exp02_extracao_pymupdf | Fase 1 | extração=PyMuPDF (vs Docling), chunking fixo=hierárquico | 0,700 | 0,489 | 0,436 | 0,415 | 2,58 |
| exp03_chunk_fixo | Fase 2 | chunking=fixo (vs hierárquico) | 0,467 | 0,306 | 0,287 | 0,312 | 2,57 |
| exp04_chunk_recursivo | Fase 2 | chunking=recursivo (vs hierárquico) | 0,600 | 0,372 | 0,418 | 0,426 | 2,90 |
| exp05_chunk_sentenca_janela | Fase 2 | chunking=sentença_janela (vs hierárquico) | 0,533 | 0,383 | 0,338 | 0,416 | 2,58 |
| exp06_chunk_semantico | Fase 2 | chunking=semântico (vs hierárquico) | 0,400 | 0,317 | 0,336 | 0,358 | 2,57 |
| exp07_embed_bge_m3 | Fase 3 | embedding=bge-m3 (vs nomic-embed-text) | 0,733 | 0,572 | 0,505 | 0,618 | 3,44 |
| exp08_embed_mxbai_embed_large | Fase 3 | embedding=mxbai-embed-large (vs nomic-embed-text) | 0,467 | 0,328 | 0,269 | 0,253 | 2,60 |
| exp09_embed_qwen3_embedding_0.6b | Fase 3 | embedding=qwen3-embedding:0.6b (vs nomic-embed-text) | 0,833 | 0,639 | 0,527 | 0,618 | 3,15 |
| exp10_embed_qwen3_embedding_4b | Fase 3 | embedding=qwen3-embedding:4b (vs nomic-embed-text) | 0,933 | 0,700 | 0,661 | 0,775 | 3,17 |
| exp11_embed_qwen3_embedding_8b | Fase 3 | embedding=qwen3-embedding:8b "latest" (vs nomic-embed-text) | 0,933 | 0,733 | 0,615 | 0,770 | 3,42 |
| exp12_densa_top5 | Fase 4 | tecnica=baseline (densa), top_k=5, base=Docling+hierárquico+qwen3-embedding:4b | 0,933 | 0,700 | 0,657 | 0,674 | 3,50 |
| exp13_densa_top20 | Fase 4 | tecnica=baseline (densa), top_k=20, base=Docling+hierárquico+qwen3-embedding:4b | 0,933 | 0,700 | 0,661 | 0,775 | 3,20 |
| exp14_hibrida_top5 | Fase 4 | tecnica=híbrida (BM25+densa/RRF), top_k=5, base=Docling+hierárquico+qwen3-embedding:4b | 0,800 | 0,633 | 0,548 | 0,599 | 3,25 |
| exp15_hibrida_top10 | Fase 4 | tecnica=híbrida (BM25+densa/RRF), top_k=10, base=Docling+hierárquico+qwen3-embedding:4b | 0,900 | 0,683 | 0,587 | 0,682 | 3,27 |
| exp16_hibrida_top20 | Fase 4 | tecnica=híbrida (BM25+densa/RRF), top_k=20, base=Docling+hierárquico+qwen3-embedding:4b | 0,800 | 0,600 | 0,569 | 0,660 | 3,27 |
| exp17_multi_query | Fase 5 | tecnica=multi_query (5 consultas, dedup), top_k=10, base=Docling+hierárquico+qwen3-embedding:4b | 0,933 | 0,722 | 0,616 | 0,705 | 7,49 |
| exp18_rag_fusion | Fase 5 | tecnica=rag_fusion (5 consultas, RRF), top_k=10, base=Docling+hierárquico+qwen3-embedding:4b | 0,933 | 0,706 | 0,579 | 0,723 | 7,40 |
| exp19_step_back | Fase 5 | tecnica=step_back (2 consultas, dedup), top_k=10, base=Docling+hierárquico+qwen3-embedding:4b | 0,900 | 0,706 | 0,624 | 0,756 | 4,81 |

*(atualizado automaticamente a partir de `avaliacao/resultados.csv` conforme novas fases rodam — gráficos entram aqui na consolidação final.)*

## 7. Melhor configuração final

`[PENDENTE]` — definido ao final da jornada (Fase 8), comparando todas as linhas de `resultados.csv`.

## 8. Análise crítica

`[PENDENTE]`

### 8.1 Decisões técnicas relevantes já tomadas

- **Destino de indexação forçado para `opensearch` na ingestão (Fase 0):** a
  heurística automática do projeto (`app/indexacao.py::decidir_destino`) roteou o
  PDF para o grafo (LightRAG) por ter ≥30 entidades nomeadas distintas — isso gerou
  muitas chamadas de LLM em paralelo para extração de entidades e estourou o limite
  de tokens do Groq. Como a Fase 0 do roteiro pede a baseline em OpenSearch (o grafo
  é uma técnica avançada opcional da Fase 7), forçamos `estrategia=opensearch`.
- **Avaliação de recuperação sem chamadas de LLM (técnica baseline):**
  `avaliar_recuperacao.py` foi escrito para não rodar a etapa de geração de resposta
  (desnecessária para medir apenas retrieval) — evita gasto de tokens do Groq e o
  risco de estourar o limite diário no meio de uma rodada de 30 perguntas.
- **A partir da Fase 4, a jornada passou de "isolada" para "progressiva"
  (decisão do autor, confirmada explicitamente):** as Fases 1-3 testaram cada
  variável isolada contra o baseline original (exp01, `nomic-embed-text`). A
  Fase 3 achou um ganho grande (`qwen3-embedding:4b`). Da Fase 4 em diante, a
  base fixa de cada fase passa a ser a melhor combinação confirmada até ali
  (Docling + `hierárquico` + `qwen3-embedding:4b`), em vez de continuar
  isolando contra `nomic-embed-text`. Trade-off consciente: ganhos das
  próximas fases podem não ser diretamente comparáveis aos ganhos das Fases
  1-3 (bases diferentes) — por isso cada tabela de fase deixa explícito
  contra qual base o `Δ` foi medido.

## 9. Conclusão

`[PENDENTE]`

## 10. Anexos

`[PENDENTE]` — traces do LangFuse, exemplos de perguntas certas/erradas antes e depois.
