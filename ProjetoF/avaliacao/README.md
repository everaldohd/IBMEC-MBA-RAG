# avaliacao/ — dataset e scripts de medição

## `dataset.json`

Gabarito de avaliação para o Trabalho Final, construído sobre
`CadeiaCustódiaProvaPericial_lei13.964-2019.pdf` (GIACOMOLLI; AMARAL, 2020).

- **18 "documentos" (D01-D18)**: segmentos temáticos do artigo (não são os chunks
  reais gerados pela ingestão automática do projeto). Cada um tem `pagina_pdf`
  para localização manual no PDF.
- **30 perguntas (`queries_benchmark`)**, distribuídas em 3 tipos (conforme
  Seção 5 do `Roteiro_Final.md`):
  - `factual` (Q01-Q12): resposta está em 1 segmento.
  - `reformulavel` (Q13-Q22): mesmo conteúdo, mas perguntado em linguagem
    coloquial/leiga — bom para medir o ganho de multi-query / step-back.
  - `multihop` (Q23-Q30): a resposta exige combinar 2-3 segmentos — bom para
    medir o ganho de RAG-Fusion / grafo (LightRAG).
- `relevancia`: `{doc_id: nota}` — nota 2 = muito relevante, nota 1 = relevante
  (alimenta NDCG graduado).
- `resposta_referencia`: resposta curta de referência, usada pelo RAGAS
  (Context Recall / Answer Correctness).
- `marcadores`: lista de trechos distintivos (ex.: `"158-B"`, `"Dias Filho"`,
  `"Velásquez Paiz"`) usados por `avaliar_recuperacao.py` para identificar a
  qual `D0x` um chunk RECUPERADO pertence (ver seção seguinte).

## Como o D-id é atribuído a um chunk real (sem id_original)

Testamos mapear por página, mas **o pipeline do projeto não grava um
`id_original` estável por chunk** (`app/indexacao.py` não anota página de
origem, e `app/consulta.py` cai no fallback `meta.get("arquivo")`, igual para
todos os chunks do mesmo PDF). Por isso `avaliar_recuperacao.py` identifica
cada chunk **pelo conteúdo**: procura, no texto do chunk recuperado, os
`marcadores` de cada `D0x` (comparação sem acento/case-insensitive) e atribui
o chunk ao primeiro `D0x` cujo marcador aparece. Chunks sem marcador
reconhecido viram um id único `__nao_identificado_N` — contam como "não
relevante" mas ainda ocupam a posição no ranking, para @K não ficar
artificialmente melhor.

É uma aproximação (documentada aqui de propósito): se dois `D0x` tiverem
marcadores muito genéricos, um chunk pode ser atribuído ao primeiro da lista
por engano. Ao revisar os resultados, vale conferir `detalhe_<exp>.json`
(gerado a cada rodada, com o texto de cada chunk recuperado) se algum número
parecer estranho.

## Arquivos desta pasta

- `avaliar_recuperacao.py` — roda as 30 perguntas contra `app/busca_avancada`,
  calcula Hit@5, Recall@5, MRR, NDCG@10 e grava uma linha em `resultados.csv`
  (e um `detalhe_<exp>.json` com os chunks recuperados por pergunta). Uso:
  ```
  python avaliacao/avaliar_recuperacao.py --exp exp01_baseline --fase "Fase 0" \
      --mudanca "baseline: chunking=auto, embedding=nomic-embed-text, busca=baseline" \
      --tecnica baseline --top-k 10
  ```
  Rodar de dentro de `ProjetoF/`, com o venv do projeto ativado (precisa da
  API **não** precisar estar no ar — o script importa `app.busca_avancada`
  direto, sem passar pelo FastAPI).
- `rodar_fase1_extracao.py` — Fase 1 (Seção 7): compara extração Docling
  (markdown, baseline) vs fallback PyMuPDF (texto puro), com chunking FIXO
  (`hierarquico`, igual ao exp01) para isolar a extração como única variável.
  Limpa e reindexa o OpenSearch, roda a mesma avaliação de retrieval do
  exp01, grava `exp02_extracao_pymupdf` em `resultados.csv`, e por fim
  **restaura o índice para o texto do Docling** (estado da baseline) antes de
  terminar. Salva amostras de texto extraído em `avaliacao/amostras_extracao/`
  para inspeção manual. OCR não entra nessa fase: o PDF escolhido tem camada
  de texto nativa, não é escaneado. Uso:
  ```
  python avaliacao/rodar_fase1_extracao.py
  ```
- `rodar_fase2_chunking.py` — Fase 2 (Seção 7): compara as 5 técnicas de
  chunking (`app/indexacao.py::chunkar`) sobre o MESMO texto extraído
  (Docling, extração fica fixa). `hierarquico` já foi medido no exp01 e não é
  reindexado de novo; as outras 4 técnicas geram `exp03_chunk_fixo`,
  `exp04_chunk_recursivo`, `exp05_chunk_sentenca_janela`,
  `exp06_chunk_semantico`. Ao final restaura o índice para `hierarquico`
  (baseline) antes de terminar. Uso:
  ```
  python avaliacao/rodar_fase2_chunking.py
  ```
- `rodar_fase3_embedding.py` — Fase 3 (Seção 7): compara os modelos de
  embedding instalados no Ollama local (descobertos automaticamente via
  `/api/tags` — `bge-m3`, `mxbai-embed-large`, `qwen3-embedding` em qualquer
  tamanho instalado) contra `nomic-embed-text` (baseline, exp01), com
  extração e chunking fixos (Docling + `hierarquico`). Mede a dimensão real
  de cada embedding em runtime (em vez de supor pela tabela de
  `app/config.py::DIMENSAO_EMBEDDING`, que não cobre os modelos Qwen3) antes
  de indexar. Ao final restaura o índice para `nomic-embed-text`. Uso:
  ```
  python avaliacao/rodar_fase3_embedding.py
  ```
- `rodar_fase4_hibrida_topk.py` — Fase 4 (Seção 7): a partir daqui a jornada
  fica PROGRESSIVA — a base fixa passa a ser Docling + `hierarquico` +
  `qwen3-embedding:4b` (vencedora da Fase 3), em vez de continuar isolando
  contra `nomic-embed-text`. Testa a nova técnica `hibrida` (BM25 + densa via
  RRF, `OpenSearchHybridRetriever` — adicionada em
  `app/busca_avancada.py`/`avaliacao/avaliar_recuperacao.py`) contra a busca
  densa pura, em `top_k` 5/10/20. **Não restaura o índice ao final** — deixa
  na nova base para a Fase 5 em diante. Uso:
  ```
  python avaliacao/rodar_fase4_hibrida_topk.py
  ```
- `rodar_fase5_query_enhancement.py` — Fase 5 (Seção 7): testa as 3 técnicas de
  query enhancement já implementadas em `app/busca_avancada.py` (`multi_query`,
  `rag_fusion`, `step_back`) contra a busca densa pura (`exp10`, mesma base), no
  `top_k=10` (ponto de equilíbrio confirmado na Fase 4). **Não reindexa nada** —
  extração/chunking/embedding continuam os da Fase 4
  (Docling+`hierarquico`+`qwen3-embedding:4b`); a única variável é a técnica de
  busca. Diferença das fases anteriores: `multi_query`/`rag_fusion`/`step_back`
  fazem 1 chamada de LLM (Groq) por pergunta para reescrever a query (intrínseco
  à técnica, sem geração de resposta final). Uso:
  ```
  python avaliacao/rodar_fase5_query_enhancement.py
  ```
- `rodar_fase6_reranking.py` — Fase 6 (Seção 7): testa a nova técnica `rerank`
  (`app/busca_avancada.py`/`avaliar_recuperacao.py`) — busca densa recupera um
  pool maior de candidatos (`top_k_inicial`) e um cross-encoder
  (`TransformersSimilarityRanker`, modelo `BAAI/bge-reranker-v2-m3`, Aula 3)
  reordena e corta no `top_k=10` final — contra a busca densa pura (`exp10`,
  mesma base), em 2 tamanhos de pool (20 e 40). **Não reindexa nada** — mesma
  base da Fase 4/5. 100% local (sem chamada de LLM/Groq — o cross-encoder roda
  via `transformers`/`torch`), mas mais lento que a busca densa pura (reranqueia
  `top_k_inicial` pares pergunta+chunk por pergunta). Uso:
  ```
  python avaliacao/rodar_fase6_reranking.py
  ```
- `rodar_fase7_hyde.py` — Fase 7 (Seção 7, técnica avançada): testa a nova
  técnica `hyde` (`app/busca_avancada.py`/`avaliar_recuperacao.py`) — HyDE
  (Hypothetical Document Embeddings, Aula 6): o LLM gera um documento
  hipotético (um trecho que PARECE a resposta, no estilo do corpus) e a busca
  densa é feita pelo embedding desse trecho hipotético, em vez da pergunta
  original — contra a busca densa pura (`exp10`, mesma base), no `top_k=10`.
  **Não reindexa nada** — mesma base da Fase 4/5/6. Faz 1 chamada de LLM
  (Groq) por pergunta para gerar o documento hipotético (intrínseco à
  técnica, sem geração de resposta final). Uso:
  ```
  python avaliacao/rodar_fase7_hyde.py
  ```
- `avaliar_ragas.py` — a criar: Faithfulness, Answer Relevancy, Context
  Precision/Recall (padrão RAGAS + Groq das Aulas 5/8, `strictness=1`).
- `resultados.csv` — uma linha por experimento (gerado automaticamente pelos
  scripts `avaliar_recuperacao.py`/`rodar_fase1_extracao.py`/
  `rodar_fase2_chunking.py`/`rodar_fase3_embedding.py`/
  `rodar_fase4_hibrida_topk.py`/`rodar_fase5_query_enhancement.py`/
  `rodar_fase6_reranking.py`/`rodar_fase7_hyde.py`; colunas seguem o template
  da Seção 7 do `Roteiro_Final.md`).
