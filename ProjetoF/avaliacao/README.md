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

## Importante: mapeamento D-id → chunk real

Os IDs `D01`-`D18` **não** correspondem automaticamente aos IDs dos chunks que
o pipeline do projeto vai gerar ao ingerir o PDF de verdade (a ingestão do
Docling + o avaliador de chunking do `app/indexacao.py` vão cortar o documento
do jeito deles, não nesses 18 blocos). Antes de rodar `avaliar_recuperacao.py`
(Fase 0), será preciso:

1. Ingerir o PDF real via `/ingestao` (ou script equivalente).
2. Conferir os `id_original`/página de cada chunk retornado.
3. Fazer a correspondência: para cada pergunta, quais chunks reais cobrem a
   mesma página/conteúdo do `D0x` marcado como relevante — e usar essa
   correspondência (por página, já que `pagina_pdf` está anotado em cada
   documento) para calcular Hit@K/Recall@K/MRR/NDCG@K.

Uma forma simples: gravar em cada chunk indexado a página de origem do PDF
(o Docling normalmente preserva isso nos metadados) e comparar contra
`pagina_pdf` do documento relevante, em vez de comparar por ID exato.

## Próximos arquivos desta pasta (a criar nas próximas fases)

- `avaliar_recuperacao.py` — Hit@K, Recall@K, MRR, NDCG@K (reaproveita
  `bench_embeddings/app/metricas.py`).
- `avaliar_ragas.py` — Faithfulness, Answer Relevancy, Context Precision/Recall
  (padrão RAGAS + Groq das Aulas 5/8, `strictness=1`).
- `resultados.csv` — uma linha por experimento (template na Seção 7 do
  `Roteiro_Final.md`).
