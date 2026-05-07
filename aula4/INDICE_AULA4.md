# Índice — Aula 4: OpenSearch Completo — Dense, Hybrid Search, Neural Sparse e Contextual Retrieval
## Vetores + Texto, RRF, Search Pipelines e Contextual Retrieval
### MBA em RAG & CAG Aplicados a Direito e Segurança Pública

**Aula:** 4 de 12 | **Carga:** 5h | **Proporção:** 30% teoria / 70% prática  
**Pré-requisito:** Aula 3 concluída (Advanced RAG + Modular RAG + LangFuse) | **Stack:** OpenSearch 3.x · Neural Search Plugin · ML Commons Plugin · BGE-M3 · Neural Sparse · RRF · Search Pipelines · vLLM · RAGAS

---

## Estrutura de Arquivos

```
aula4/
│
├── INDICE_AULA4.md                                      ← Este arquivo
├── AVALIACAO_AULA4.md                                   ← Rubricas e critérios (professor)
│
├── teoria/
│   └── AULA4_TEORIA.md                                  ← Material teórico completo (7 seções)
│
├── labs/
│   ├── LAB1_OpenSearch_Hybrid_Index.ipynb               ← Criar índice kNN + BM25 no OpenSearch 3.x
│   ├── LAB2_Search_Pipeline_RRF.ipynb                   ← Configurar search pipeline com RRF
│   ├── LAB3_Hybrid_Search_Juridico.ipynb                ← Busca híbrida em corpus jurídico real
│   ├── LAB4_Neural_Sparse_Search.ipynb                  ← Neural Sparse Search com SPLADE
│   ├── LAB5_Contextual_Retrieval.ipynb                  ← #T09: pré-processar chunks com vLLM + medir RAGAS antes/depois
│   └── LAB6_LangFuse_Dashboard_Comparativo.ipynb        ← Dashboard comparativo BM25 vs Dense vs Hybrid
│
├── exemplos/
│   ├── EXEMPLO1_Hybrid_Query_Basico.ipynb               ← Consulta híbrida mínima (referência rápida)
│   ├── EXEMPLO2_RRF_vs_Normalization.ipynb              ← Comparação RRF × min-max normalization
│   └── EXEMPLO3_Conversational_RAG_Pipeline.ipynb       ← Conversational Search com RAG pipeline (referência extra)
│
└── datasets/
    └── corpus_juridico_aula4.json                       ← 20 docs + 15 queries com relevância anotada
```

---

## Roteiro da Aula (5 horas)

| Bloco | Duração | Tipo | Conteúdo | Arquivo |
|---|---|---|---|---|
| **1. Revisão + Motivação** | 15 min | Teoria | Limitações da busca puramente vetorial ou BM25 em textos jurídicos | `teoria/AULA4_TEORIA.md §1` |
| **2. Hybrid Search — Arquitetura** | 30 min | Teoria | kNN dense + BM25 sparse, score fusion, search pipelines | `teoria/AULA4_TEORIA.md §2–3` |
| **3. LAB 1 — Índice Híbrido** | 40 min | Prática | Criar índice com campo kNN (BGE-M3) + campo text (BM25) no OpenSearch | `labs/LAB1_OpenSearch_Hybrid_Index.ipynb` |
| **4. RRF e Score Fusion — Teoria** | 20 min | Teoria | Reciprocal Rank Fusion, min-max normalization, arithmetic combination | `teoria/AULA4_TEORIA.md §4` |
| **5. LAB 2 — Search Pipeline RRF** | 40 min | Prática | Criar e ativar search pipeline com normalization processor + RRF | `labs/LAB2_Search_Pipeline_RRF.ipynb` |
| **6. LAB 3 — Busca Híbrida Jurídica** | 40 min | Prática | Executar hybrid search em corpus jurídico, comparar MRR e Recall | `labs/LAB3_Hybrid_Search_Juridico.ipynb` |
| **7. Neural Sparse — Teoria** | 15 min | Teoria | SPLADE, sparse neural vectors, eficiência vs. dense | `teoria/AULA4_TEORIA.md §5` |
| **8. LAB 4 — Neural Sparse** | 30 min | Prática | Indexar com modelo SPLADE, executar consultas neural sparse | `labs/LAB4_Neural_Sparse_Search.ipynb` |
| **9. Contextual Retrieval — Teoria** | 15 min | Teoria | Pré-processamento de chunks com contexto situacional (vLLM), impacto no Context Precision | `teoria/AULA4_TEORIA.md §6` |
| **10. LAB 5 — Contextual Retrieval** | 45 min | Prática | Pré-processar 100 chunks com vLLM adicionando contexto situacional, re-indexar, medir Context Precision (RAGAS) antes/depois | `labs/LAB5_Contextual_Retrieval.ipynb` |
| **11. LAB 6 — Dashboard LangFuse** | 30 min | Prática | Dashboard comparativo de latência e qualidade entre BM25, dense e hybrid com LangFuse | `labs/LAB6_LangFuse_Dashboard_Comparativo.ipynb` |

---

## Objetivos de Aprendizagem (conforme ementa)

Ao final desta aula, o aluno será capaz de:

1. **Explicar** a diferença entre busca densa (kNN), esparsa (BM25) e híbrida em contexto jurídico
2. **Criar índices híbridos** no OpenSearch 3.x com Neural Search Plugin + ML Commons Plugin
3. **Configurar search pipelines** com normalization processor e RRF (Reciprocal Rank Fusion)
4. **Comparar estratégias de fusão de scores** (RRF vs. min-max normalization) com métricas objetivas
5. **Implementar Neural Sparse Search** com modelos SPLADE no OpenSearch 3.x
6. **Aplicar Contextual Retrieval (#T09)** — enriquecer chunks com contexto via vLLM e medir impacto via RAGAS
7. **Monitorar e comparar estratégias de busca** com dashboard LangFuse

---

## Stack Tecnológico

| Componente | Ferramenta | Papel no Pipeline |
|---|---|---|
| Motor de busca | **OpenSearch 3.x** | Índice híbrido (kNN + BM25) + search pipelines |
| Plugin Neural | **Neural Search Plugin** | Integração de modelos de embedding diretamente no OpenSearch |
| Plugin ML | **ML Commons Plugin** | Registro e deploy de modelos de embedding/sparse no cluster |
| Embeddings densos | **BGE-M3** (BAAI, dim=1024) | Vetorização multilíngue para kNN |
| Neural Sparse | **opensearch-neural-sparse-encoding-doc-v2** | Sparse vectors para busca esparsa neural |
| Score Fusion | **RRF + min-max normalization** | Combinação de scores vetorial + lexical |
| LLM | **Llama 3.1 8B Instruct** | Pré-processamento de chunks (Contextual Retrieval) + geração |
| Servidor LLM | **vLLM** (PagedAttention) | API OpenAI-compatible em `localhost:8000/v1` |
| Orquestração | **LangChain LCEL + OpenSearch Python SDK** | Pipeline RAG modular e composável |
| Avaliação | **RAGAS** | Context Precision antes/depois do Contextual Retrieval |
| Observabilidade | **LangFuse** | Rastreamento de queries, latência e dashboard comparativo |

---

## Fichas de Técnicas RAG — Esta Aula

### Ficha T04 — Hybrid Search

| Campo | Conteúdo |
|---|---|
| **ID** | #T04 |
| **Categoria** | Retrieval Avançado |
| **Subtítulo** | Fusão de busca vetorial e lexical |
| **Descrição** | Combina busca densa (embeddings + kNN) e busca esparsa (BM25/TF-IDF) numa única query. Os scores são normalizados e fundidos via RRF ou arithmetic combination. Captura tanto semântica quanto match exato de termos — essencial em textos jurídicos com terminologia técnica. |
| **Aplicabilidades** | Pesquisa em legislação, jurisprudência e doutrina; sistemas de compliance; investigação policial com linguagem técnica; portais de e-gov |
| **Vantagens** | Melhor Recall e MRR que abordagem única; robusto para variações de vocabulário jurídico |
| **Limitações** | Latência maior; requer tuning de pesos alpha; dependência de infraestrutura OpenSearch |
| **Lab** | LAB1 (índice) + LAB2 (pipeline RRF) + LAB3 (avaliação jurídica) |
| **Referência** | CHEN et al. (2022). arXiv:2210.11934; OpenSearch Docs, 2024. |

### Ficha — Neural Sparse Search (OpenSearch 3.x Extra)

| Campo | Conteúdo |
|---|---|
| **Categoria** | Retrieval Avançado |
| **Subtítulo** | Representação esparsa neural (SPLADE) |
| **Descrição** | Usa modelos neurais (SPLADE) para gerar vetores esparsos de alta dimensão que combinam a interpretabilidade do BM25 com a compreensão semântica dos embeddings densos. Permite busca eficiente em índices invertidos com compreensão de sinônimos e contexto. No OpenSearch 3.x é habilitado via ML Commons Plugin. |
| **Aplicabilidades** | Pesquisa em grandes corpora legislativos; busca de precedentes com variação terminológica; integração com sistemas legacy baseados em índices invertidos |
| **Vantagens** | Interpretabilidade (termos com pesos); eficiência computacional; qualidade próxima ao dense retrieval |
| **Limitações** | Requer modelo de sparse encoding dedicado; menos maduro que dense retrieval em português |
| **Lab** | LAB4 (Neural Sparse Search com SPLADE) |
| **Referência** | FORMAL et al. (2021). SPLADE. arXiv:2107.05720. |

### Ficha T09 — Contextual Retrieval

| Campo | Conteúdo |
|---|---|
| **ID** | #T09 |
| **Categoria** | Retrieval Avançado |
| **Subtítulo** | Enriquecimento contextual de chunks para indexação |
| **Descrição** | Antes da indexação, cada chunk é enriquecido com um trecho de contexto situacional gerado por um LLM (vLLM). O prompt descreve o papel do chunk no documento completo. O chunk enriquecido é então reindexado, melhorando Context Precision e Recall na recuperação. Desenvolvido pela Anthropic (2024). |
| **Aplicabilidades** | Acórdãos extensos onde chunks isolados perdem referência ao caso; laudos periciais com seções interdependentes; procedimentos policiais multi-fase |
| **Vantagens** | Melhora significativa em Context Precision (~35% segundo Anthropic); compatível com qualquer vector store; sem mudança na arquitetura de retrieval |
| **Limitações** | Custo de pré-processamento (1 chamada LLM por chunk); latência na ingestão; depende de qualidade do prompt de contextualização |
| **Lab** | LAB5 (pré-processar 100 chunks, re-indexar, medir Context Precision RAGAS antes/depois) |
| **Referência** | ANTHROPIC. *Introducing Contextual Retrieval*. Blog, 2024. Disponível em: <https://www.anthropic.com/news/contextual-retrieval>. |

---

## Avaliação

Ver `AVALIACAO_AULA4.md` para rubricas completas.

| Entregável | Peso | Lab |
|---|---|---|
| Índice híbrido funcional no OpenSearch 3.x (kNN + BM25) | 20% | LAB1 |
| Search pipeline RRF configurado e testado | 20% | LAB2 |
| Análise comparativa: hybrid vs. single-mode com MRR/Recall | 20% | LAB3 |
| Neural Sparse Search indexado e consultado | 10% | LAB4 |
| Contextual Retrieval implementado com melhoria RAGAS comprovada | 20% | LAB5 |
| Dashboard LangFuse comparativo BM25/dense/hybrid | 10% | LAB6 |

---

## Referências Bibliográficas (ABNT)

CHEN, Y. et al. **Out-of-Domain Semantics to the Rescue! Zero-Shot Hybrid Retrieval Models**. arXiv:2210.11934, 2022.

FORMAL, T. et al. **SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking**. arXiv:2107.05720, 2021.

LIN, J.; MA, X. **A Few Brief Notes on DeepImpact, COIL, and a Conceptual Framework for Information Retrieval Techniques**. arXiv:2106.14807, 2021.

OPENSEARCH PROJECT. **Hybrid Search**. Disponível em: <https://docs.opensearch.org/latest/vector-search/ai-search/hybrid-search/>. Acesso em: abr. 2026.

OPENSEARCH PROJECT. **Neural Sparse Search**. Disponível em: <https://docs.opensearch.org/latest/vector-search/ai-search/neural-sparse-search/>. Acesso em: abr. 2026.

ANTHROPIC. **Introducing Contextual Retrieval**. Blog Anthropic, 2024. Disponível em: <https://www.anthropic.com/news/contextual-retrieval>. Acesso em: abr. 2026.

ES, S. et al. **RAGAS: Automated Evaluation of Retrieval Augmented Generation**. arXiv:2309.15217, 2023.

ROBERTSON, S.; ZARAGOZA, H. **The Probabilistic Relevance Framework: BM25 and Beyond**. *Foundations and Trends in Information Retrieval*, v. 3, n. 4, p. 333-389, 2009.

CHEN, J. et al. **BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation**. arXiv:2309.07597, 2024.
