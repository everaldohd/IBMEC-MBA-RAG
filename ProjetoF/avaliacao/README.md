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
- `avaliar_ragas.py` — a criar: Faithfulness, Answer Relevancy, Context
  Precision/Recall (padrão RAGAS + Groq das Aulas 5/8, `strictness=1`).
- `resultados.csv` — uma linha por experimento (gerado automaticamente pelos
  scripts `avaliar_recuperacao.py`/`rodar_fase1_extracao.py`/
  `rodar_fase2_chunking.py`; colunas seguem o template da Seção 7 do
  `Roteiro_Final.md`).
