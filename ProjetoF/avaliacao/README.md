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
  calcula Hit@5