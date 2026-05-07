# MBA RAG & CAG — AULA 1: FUNDAMENTOS
## Índice de Materiais

> **Carga Horária:** 5 horas | **Proporção:** 35% Teoria / 65% Prática

---

## 📚 DOCUMENTOS DA AULA

### Teoria
| Arquivo | Conteúdo | Formato |
|---------|----------|---------|
| `teoria/AULA1_TEORIA.md` | 7 tópicos teóricos completos com analogias, insights, perguntas e resoluções, exercícios e referências ABNT | Markdown |

### Laboratórios (Google Colab)
| Arquivo | Conteúdo | Tempo |
|---------|----------|-------|
| `labs/LAB1_OpenSearch_Docker.ipynb` | Docker + OpenSearch 3.x + Dashboards — instalação e validação | 25 min |
| `labs/LAB2_Setup_Colab_Ambiente.ipynb` | GPU, dependências Python, variáveis de ambiente, smoke test | 25 min |
| `labs/LAB3_vLLM_LLM_Local.ipynb` | vLLM + Llama 3.1 8B, completions, temperature, API OpenAI-compat. | 45 min |
| `labs/LAB4_LangFuse_Observabilidade.ipynb` | LangFuse traces, spans, generations, @observe decorator | 20 min |
| `labs/LAB5_Embeddings_BGE_M3_UMAP.ipynb` | BGE-M3 vs sentence-transformers, similaridade coseno, UMAP 2D | 60 min |

### Exemplos
| Arquivo | Conteúdo | Tipo |
|---------|----------|------|
| `exemplos/EXEMPLO_Pipeline_RAG_Minimo.ipynb` | Pipeline RAG completo: corpus → embedding → FAISS → vLLM → LangFuse | Demo |

### Datasets
| Arquivo | Conteúdo | Formato |
|---------|----------|---------|
| `datasets/corpus_juridico_aula1.json` | 10 documentos jurídicos fictícios: acórdãos, BO, laudos, sentenças | JSON |

### Roteiro de Instalação
| Arquivo | Conteúdo |
|---------|----------|
| `ROTEIRO_INSTALACAO_FERRAMENTAS.md` | Guia passo a passo: Docker, OpenSearch, Python, vLLM, LangFuse, .env |

---

## 🔧 STACK TECNOLÓGICO DESTA AULA

```
OpenSearch 3.x       → Motor de busca vetorial + textual
OpenSearch Dashboards → Interface visual para exploração
Docker + Compose      → Infraestrutura de contêineres
vLLM                 → Servidor de LLMs locais (PagedAttention, API OpenAI-compatible)
Llama 3.1 8B Instruct → LLM principal do curso
LangFuse             → Observabilidade e rastreamento
sentence-transformers → Framework de embeddings
BGE-M3 (BAAI)        → Modelo de embedding estado da arte
FAISS                → Índice vetorial local
UMAP                 → Redução dimensional para visualização
Python 3.11          → Linguagem do curso
Google Colab (T4 GPU) → Ambiente de execução dos labs
```

---

## ✅ CRITÉRIOS DE AVALIAÇÃO

| Item | Descrição | Pontos |
|------|-----------|--------|
| Lab 1 | Checklist de 6 itens (OpenSearch funcionando) | 15 pts |
| Lab 2 | Checklist de 6 itens (ambiente Colab configurado) | 15 pts |
| Lab 3 | Checklist de 6 itens (vLLM + completion jurídico) | 20 pts |
| Lab 4 | Checklist de 7 itens (LangFuse trace visível) | 20 pts |
| Lab 5 | Checklist de 7 itens (UMAP com clusters visíveis) | 20 pts |
| Exercícios | 5 exercícios de fixação (teoria/AULA1_TEORIA.md) | 10 pts |
| **TOTAL** | | **100 pts** |

---

## 🗺️ SEQUÊNCIA RECOMENDADA

```
[1] Leia teoria/AULA1_TEORIA.md — Tópicos 1-4 (45 min)
    ↓
[2] Execute LAB1 — OpenSearch + Docker (25 min)
    ↓
[3] Execute LAB2 — Setup Colab (25 min)
    ↓
[4] Leia teoria/AULA1_TEORIA.md — Tópicos 5-7 (30 min)
    ↓
[5] Execute LAB3 — vLLM + LLM Local (45 min)
    ↓
[6] Execute LAB4 — LangFuse (20 min)
    ↓
[7] Execute LAB5 — Embeddings + UMAP (60 min)
    ↓
[8] Resolva os Exercícios de Fixação na teoria (30 min)
    ↓
[9] BÔNUS: Execute o EXEMPLO_Pipeline_RAG_Minimo.ipynb
```

---

*MBA RAG & CAG Aplicados a Direito e Segurança Pública — Aula 1*
