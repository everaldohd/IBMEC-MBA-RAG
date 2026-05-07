# Instruções de Execução — Labs 3 & 4

## Quick Start (5 minutos)

### 1. Preparar Ambiente
```bash
# No Google Colab, execute em uma célula:
!pip install -q langchain langchain-community langchain-text-splitters sentence-transformers faiss-cpu pandas numpy matplotlib seaborn umap-learn docling
```

### 2. LAB 3: Análise Qualitativa
```
1. Abrir LAB3_Analise_Qualitativa_Chunks.ipynb em Google Colab
2. Executar células sequencialmente (do topo para baixo)
3. Observar:
   - 4 critérios de qualidade sendo calculados
   - Gráficos de distribuição (histograma, boxplot, UMAP)
   - Score final de 0-10 para cada estratégia
   - Recomendação automática
```

### 3. LAB 4: Pipeline RAG Completo
```
1. Abrir LAB4_Naive_RAG_Pipeline_Completo.ipynb
2. Executar célula por célula
3. Na etapa 7 (vLLM), escolha:
   a) Se tem acesso a GPU cloud: iniciar vLLM em terminal separado
   b) Se não tem GPU: usar OpenAI API (export OPENAI_API_KEY=sk-...)
4. Testar RAG com queries jurídicas
```

---

## Detalhes de Cada Lab

### LAB 3: Análise Qualitativa de Chunks

#### O que você vai aprender
- ✅ Por que avaliar chunks qualitativamente ANTES de indexar
- ✅ 4 critérios de qualidade objetivos
- ✅ Como medir coerência semântica com embeddings
- ✅ Heurísticas para detectar chunks "órfãos"
- ✅ Análise estatística de distribuição de tamanhos
- ✅ Comparação de 3 estratégias de chunking

#### Fluxo Executável
```
[Célula 1] Instalar dependências (NLTK, sentence-transformers, sklearn)
    ↓
[Célula 2] Definir corpus jurídico (acórdão, lei, relatório)
    ↓
[Célula 3] Implementar 3 estratégias de chunking
    ↓
[Célula 4] Calcular coerência semântica (cosine similarity)
    ↓
[Célula 5] Detectar chunks órfãos (análise manual)
    ↓
[Célula 6] Análise de distribuição de tamanhos
    ↓
[Célula 7] Score agregado 0-10
    ↓
[Célula 8] Análise jurídica (artigos cortados)
    ↓
[Célula 9] Visualização UMAP dos embeddings
    ↓
[Célula 10] Relatório final
    ↓
[Célula 11-12] Exercício prático
```

#### Tempo de Execução
- Instalação: 3-5 minutos
- Carregamento BGE-M3: 2-3 minutos
- Chunking: < 1 segundo
- Coerência: ~20 segundos (100 chunks)
- Cálculos estatísticos: < 5 segundos
- UMAP: ~30 segundos
- **TOTAL: ~2 horas** (incluindo leitura do código)

#### Saídas Esperadas
```
✅ Coerência semântica
   FIXED: 0.612 (avg)
   RECURSIVE: 0.738 (melhor!)
   SEMANTIC: 0.654

✅ Chunks órfãos
   FIXED: 15% (ruim)
   RECURSIVE: 2% (ótimo!)
   SEMANTIC: 8%

✅ Score final (0-10)
   FIXED: 6.2
   RECURSIVE: 8.1 ← RECOMENDADO
   SEMANTIC: 7.3

✅ Gráficos salvos em /tmp/
   - analise_tamanhos_chunks.png
   - umap_chunks.png
```

---

### LAB 4: Pipeline Naive RAG Completo

#### O que você vai aprender
- ✅ Arquitetura completa de RAG (ingestão até resposta)
- ✅ Ingestão inteligente com Docling
- ✅ Chunking jurídico customizado
- ✅ Embeddings BGE-M3 (1024 dimensões)
- ✅ Indexação vetorial com FAISS
- ✅ Configuração de vLLM
- ✅ Prompt engineering para juristas
- ✅ RAG chain declarativa (LCEL)
- ✅ Debugging e rastreamento
- ✅ Persistência e extensibilidade

#### Fluxo Executável
```
[Célula 1] Instalar all dependencies
    ↓
[Célula 2] Criar 5 PDFs jurídicos com ReportLab
    ↓
[Célula 3] Ingerir PDFs com Docling → Markdown
    ↓
[Célula 4] Chunking jurídico customizado
    ↓
[Célula 5] Carregar BGE-M3 e gerar embeddings
    ↓
[Célula 6] Criar índice FAISS
    ↓
[Célula 7] Iniciar vLLM (ou usar OpenAI)
    ↓
[Célula 8] Definir prompt template jurídico
    ↓
[Célula 9] Montar RAG chain (LCEL)
    ↓
[Célula 10] Executar 3 queries de teste
    ↓
[Célula 11] Salvar índice FAISS
    ↓
[Célula 12] Exercício: adicionar novo documento
```

#### Tempo de Execução
```
Instalação:        3-5 min
Criar PDFs:        2 seg
Ingestão Docling:  5-10 min (primeiro download do modelo)
Chunking:          < 1 seg
BGE-M3:            2-3 min (primeiro download)
Embeddings:        20-30 seg
FAISS:             5 seg
vLLM setup:        5-10 min (se iniciar server) OU rápido (se OpenAI)
RAG queries:       5-15 seg por query
Persistência:      2 seg

TOTAL: ~2 horas
```

#### Saídas Esperadas
```
✅ 5 PDFs criados em /tmp/corpus_juridico/
   - acordao_hc.pdf (12 KB)
   - lei_11343.pdf (11 KB)
   - relatorio_inteligencia.pdf (13 KB)
   - parecer_mp.pdf (10 KB)
   - sumulas.pdf (9 KB)

✅ Chunks criados
   Total: 35-40 chunks
   Tamanho médio: 750 chars
   Distribuição por documento e seção

✅ Embeddings gerados
   Dimensão: 1024
   Total: 40 vetores
   RAM: ~160 MB

✅ FAISS Index
   Arquivos em /tmp/rag_index/faiss_index/
   - index.faiss
   - index.pkl

✅ RAG queries de teste
   Query 1: Pena para tráfico? → Resposta com [Fonte N]
   Query 2: Prisão preventiva? → Resposta com citações
   Query 3: Estatísticas 2025? → Resposta do relatório

✅ Índice persistido
   Load em nova sessão em < 1 seg
```

---

## Troubleshooting

### Problema: "ModuleNotFoundError: No module named 'langchain'"
**Solução**: 
```python
!pip install langchain langchain-community langchain-text-splitters
```

### Problema: "CUDA out of memory" ou "GPU not available"
**Solução**:
```python
# Use CPU no Colab
from langchain_community.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": "cpu"},  # Force CPU
)
```

### Problema: "vLLM not found at localhost:8000"
**Solução (Opção 1)**: Iniciar vLLM em terminal separado
```bash
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --port 8000 \
  --dtype float16
```

**Solução (Opção 2)**: Usar OpenAI API como fallback
```python
import os
os.environ['OPENAI_API_KEY'] = 'sk-your-key-here'
```

### Problema: "Docling not available"
**Solução**:
```bash
!pip install docling
# Pode levar alguns minutos no Colab
```

### Problema: "FAISS index is empty"
**Solução**: Certifique-se de que:
1. Chunks foram criados (len(chunks) > 0)
2. Embeddings foram gerados
3. FAISS foi criado com from_documents()

---

## Extensões Sugeridas

### Após LAB 3
- [ ] Adicionar novo critério de qualidade (ex: quantidade de termos jurídicos)
- [ ] Comparar 5 estratégias de chunking (não apenas 3)
- [ ] Testar embeddings diferentes (OpenAI vs Hugging Face)
- [ ] Aplicar análise em documento jurídico real (seu PDF)

### Após LAB 4
- [ ] Adicionar re-ranking (ex: ColBERT)
- [ ] Implementar query expansion
- [ ] Adicionar sumarização de contexto
- [ ] Testar múltiplos LLMs (GPT-4, Claude, etc.)
- [ ] Implementar logging estruturado
- [ ] Criar API FastAPI/Flask para RAG

---

## Configurações de Ambiente

### Google Colab (Recomendado)
```python
# No início do notebook
!pip install -q langchain langchain-community langchain-text-splitters \
    sentence-transformers faiss-cpu docling pandas numpy matplotlib seaborn

import warnings
warnings.filterwarnings('ignore')

# Memory optimization
import gc
gc.collect()
```

### Local (Python 3.11+)
```bash
# Criar venv
python3.11 -m venv rag_env
source rag_env/bin/activate

# Instalar dependências
pip install langchain langchain-community langchain-text-splitters \
    sentence-transformers faiss-cpu docling pandas numpy matplotlib seaborn umap-learn

# Executar Jupyter
jupyter notebook
```

### Hardware Recomendado
- **CPU**: Intel i5+ ou AMD Ryzen 5+
- **RAM**: 16 GB mínimo (32 GB ideal)
- **GPU**: Opcional (CPU funciona bem)
- **Disco**: 10 GB livre (para modelos)

---

## Próximas Aulas (Roadmap)

| Lab | Tópico | Horas |
|-----|--------|-------|
| Lab 3 | ✅ Análise Qualitativa de Chunks | 2 |
| Lab 4 | ✅ Pipeline Naive RAG | 2 |
| Lab 5 | Query Expansion & Rewriting | 2 |
| Lab 6 | Re-ranking (ColBERT, MonoT5) | 2 |
| Lab 7 | Agentic RAG (routing, tools) | 2 |
| Lab 8 | Evaluation (Hit@k, MRR, NDCG) | 2 |
| Lab 9 | Produção (FastAPI, monitoring) | 3 |
| | **TOTAL** | **15 horas** |

---

## Referências Rápidas

### Documentação
- LangChain: https://python.langchain.com/
- FAISS: https://github.com/facebookresearch/faiss
- BGE-M3: https://huggingface.co/BAAI/bge-m3
- vLLM: https://github.com/vllm-project/vllm

### Artigos Seminal
- Lewis et al. (2020): "RAG for Knowledge-Intensive NLP"
- Gao et al. (2023): "RAG Survey"
- Bai et al. (2024): "BGE-M3 Embeddings"

---

**Última atualização**: Abril 2026  
**Status**: Pronto para execução  
**Suporte**: Incluído nas células de debugging  
