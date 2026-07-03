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

### Fase 2 — Chunking

`[PENDENTE]`

### Fase 3 — Modelo de embedding

`[PENDENTE]`

### Fase 4 — Recuperação base (top_k e busca híbrida)

`[PENDENTE]`

### Fase 5 — Query enhancement

`[PENDENTE]`

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

## 9. Conclusão

`[PENDENTE]`

## 10. Anexos

`[PENDENTE]` — traces do LangFuse, exemplos de perguntas certas/erradas antes e depois.
