# Índice — Aula 5: Avaliação e Observabilidade — RAGAS, DeepEval e LangFuse Avançado
## Medindo e Melhorando a Qualidade do Pipeline RAG Jurídico
### MBA em RAG & CAG Aplicados a Direito e Segurança Pública

**Aula:** 5 de 12 | **Carga:** 5h | **Proporção:** 30% teoria / 70% prática  
**Pré-requisito:** Aulas 1–4 concluídas (Naive RAG, Advanced RAG, Hybrid Search, Contextual Retrieval)  
**Stack:** RAGAS · DeepEval · LangFuse (Scores API) · Pandas · Matplotlib · vLLM · OpenSearch 3.x

---

## Estrutura de Arquivos

```
aula5/
│
├── INDICE_AULA5.md                                        ← Este arquivo
├── AVALIACAO_AULA5.md                                     ← Rubricas e critérios (professor)
│
├── teoria/
│   └── AULA5_TEORIA.md                                    ← Material teórico completo (8 seções)
│
├── labs/
│   ├── LAB1_Dataset_Avaliacao_GroundTruth.ipynb           ← Construir 50 pares query+resposta+contexto
│   ├── LAB2_RAGAS_Baseline_Naive_RAG.ipynb                ← 4 métricas RAGAS no Naive RAG (#T01)
│   ├── LAB3_RAGAS_LangFuse_Scores_API.ipynb               ← Integração RAGAS → LangFuse automático
│   ├── LAB4_DeepEval_Testes_Unitarios.ipynb               ← 5 testes unitários DeepEval no pipeline
│   ├── LAB5_Dashboard_Naive_vs_Advanced.ipynb             ← Comparação Naive RAG vs Advanced RAG
│   └── LAB6_Analise_Erros_Faithfulness.ipynb              ← Diagnóstico de queries com Faithfulness < 0.7
│
├── exemplos/
│   ├── EXEMPLO1_RAGAS_Minimo.ipynb                        ← Avaliação RAGAS em 5 linhas (referência rápida)
│   └── EXEMPLO2_DeepEval_Basico.ipynb                     ← Teste DeepEval mínimo (referência rápida)
│
└── datasets/
    └── corpus_avaliacao_aula5.json                        ← 50 pares QA com ground-truth jurídico
```

---

## Roteiro da Aula (5 horas)

| Bloco | Duração | Tipo | Conteúdo | Arquivo |
|---|---|---|---|---|
| **1. Por que avaliar RAG?** | 20 min | Teoria | O problema da "alucinação silenciosa", métricas clássicas vs. métricas LLM-as-judge | `teoria/AULA5_TEORIA.md §1–2` |
| **2. Framework RAGAS** | 25 min | Teoria | 4 métricas, pipeline de avaliação, metas mínimas do syllabus | `teoria/AULA5_TEORIA.md §3–4` |
| **3. LAB 1 — Ground-Truth** | 40 min | Prática | Construir dataset de 50 pares QA jurídicos; formato RAGAS-compatible | `labs/LAB1_Dataset_Avaliacao_GroundTruth.ipynb` |
| **4. LAB 2 — RAGAS Baseline** | 45 min | Prática | Calcular Faithfulness, Answer Relevancy, Context Recall, Context Precision no Naive RAG | `labs/LAB2_RAGAS_Baseline_Naive_RAG.ipynb` |
| **5. LangFuse Scores API** | 15 min | Teoria | Scores API, integração automática, dashboards de qualidade | `teoria/AULA5_TEORIA.md §5` |
| **6. LAB 3 — RAGAS + LangFuse** | 30 min | Prática | Pipeline que calcula RAGAS e envia scores ao LangFuse automaticamente | `labs/LAB3_RAGAS_LangFuse_Scores_API.ipynb` |
| **7. DeepEval** | 15 min | Teoria | Testes unitários de LLM, métricas disponíveis, integração com pytest | `teoria/AULA5_TEORIA.md §6` |
| **8. LAB 4 — DeepEval** | 30 min | Prática | 5 testes unitários: Faithfulness, AnswerRelevancy, Hallucination, Toxicity, Bias | `labs/LAB4_DeepEval_Testes_Unitarios.ipynb` |
| **9. LAB 5 — Dashboard** | 30 min | Prática | Comparação Naive RAG vs Advanced RAG nas 4 métricas RAGAS + gráficos | `labs/LAB5_Dashboard_Naive_vs_Advanced.ipynb` |
| **10. LAB 6 — Análise de Erros** | 25 min | Prática | Identificar queries com Faithfulness < 0.7, diagnosticar causas, propor correções | `labs/LAB6_Analise_Erros_Faithfulness.ipynb` |

---

## Objetivos de Aprendizagem (conforme ementa)

Ao final desta aula, o aluno será capaz de:

1. **Explicar** por que métricas tradicionais (BLEU, ROUGE) são insuficientes para avaliar sistemas RAG jurídicos
2. **Construir** um dataset de avaliação com ground-truth a partir de documentos jurídicos reais
3. **Calcular** as 4 métricas RAGAS (Faithfulness, Answer Relevancy, Context Recall, Context Precision) em qualquer pipeline RAG
4. **Integrar** avaliação RAGAS com LangFuse Scores API para monitoração contínua em produção
5. **Escrever testes unitários** com DeepEval para validar comportamento do pipeline RAG
6. **Comparar** pipelines RAG objetivamente com dashboards de qualidade
7. **Diagnosticar** falhas de Faithfulness e propor estratégias de mitigação

---

## Stack Tecnológico

| Componente | Ferramenta | Papel |
|---|---|---|
| Framework de avaliação | **RAGAS** (≥0.1) | Cálculo das 4 métricas de qualidade RAG |
| Testes unitários LLM | **DeepEval** (≥0.21) | Assertions de Faithfulness, Hallucination, Toxicity, Bias |
| Observabilidade | **LangFuse** (Scores API) | Armazenamento e visualização de métricas de qualidade |
| LLM judge | **Llama 3.1 8B Instruct via vLLM** | LLM-as-judge para avaliação Faithfulness e Relevância |
| Vector Store | **OpenSearch 3.x** | Recuperação de contexto para avaliação |
| Orquestração | **LangChain LCEL** | Pipeline RAG avaliado |
| Manipulação de dados | **Pandas** | Análise e exportação de resultados |
| Visualização | **Matplotlib / Seaborn** | Dashboards comparativos |

---

## Metas de Qualidade RAGAS (conforme syllabus)

| Métrica | Meta Mínima | Descrição |
|---|---|---|
| **Faithfulness** | ≥ 0.80 | Respostas fundamentadas nos contextos recuperados (sem alucinação) |
| **Answer Relevancy** | ≥ 0.75 | Resposta pertinente à pergunta feita |
| **Context Recall** | ≥ 0.70 | Ground-truth coberto pelos contextos recuperados |
| **Context Precision** | ≥ 0.70 | Contextos recuperados são realmente relevantes |

> **Por que essas metas?** No domínio jurídico, Faithfulness abaixo de 0.80 indica risco de alucinação sobre artigos de lei ou jurisprudência — o que pode causar dano real ao usuário.

---

## Fichas de Técnicas — Esta Aula

### Ficha — RAGAS (Framework de Avaliação)

| Campo | Conteúdo |
|---|---|
| **Categoria** | Avaliação de Sistemas RAG |
| **Subtítulo** | Automated Evaluation of Retrieval Augmented Generation |
| **Descrição** | Framework open-source para avaliação automática de pipelines RAG. Usa LLM-as-judge para calcular 4 métricas sem necessidade de anotação humana exaustiva: Faithfulness (fundamentação), Answer Relevancy (pertinência), Context Recall (cobertura) e Context Precision (precisão do retrieval). |
| **Aplicabilidades** | Avaliação de chatbots jurídicos; benchmarking de chunking strategies; monitoração de degradação em produção; comparação de modelos de embedding |
| **Vantagens** | Automático (não precisa de anotadores humanos); alinhado com percepção humana; extensível com métricas customizadas |
| **Limitações** | Depende de LLM judge (custo/latência); resultados podem variar com modelo judge; requer ground-truth para Context Recall |
| **Lab** | LAB2 (baseline) + LAB3 (integração LangFuse) + LAB5 (dashboard) |
| **Referência** | ES et al. arXiv:2309.15217, 2023. |

### Ficha — DeepEval (Testes Unitários)

| Campo | Conteúdo |
|---|---|
| **Categoria** | Testes de Qualidade LLM |
| **Subtítulo** | Pytest-like unit tests for LLM outputs |
| **Descrição** | Framework de testes unitários para pipelines LLM. Permite escrever assertions como `assert_test(case, [FaithfulnessMetric(threshold=0.7)])` integradas ao pytest. Suporta CI/CD para evitar regressões de qualidade em deploys. |
| **Aplicabilidades** | Pipeline RAG jurídico com requisito de Faithfulness; detecção de Hallucination em respostas sobre legislação; testes de regressão após atualização do corpus |
| **Vantagens** | Integração nativa com pytest; suporte a CI/CD; relatórios detalhados de falha com diagnóstico |
| **Limitações** | Também usa LLM judge; threshold fixo pode não capturar nuances; custo por teste em modelos pagos |
| **Lab** | LAB4 |
| **Referência** | CONFIDENT AI. *DeepEval Documentation*. 2024. |

---

## Avaliação

Ver `AVALIACAO_AULA5.md` para rubricas completas.

| Entregável | Peso | Lab |
|---|---|---|
| Dataset ground-truth com 50 pares QA exportado | 15% | LAB1 |
| 4 métricas RAGAS calculadas com pelo menos 1 acima da meta | 25% | LAB2 |
| Pipeline RAGAS → LangFuse automático funcionando | 20% | LAB3 |
| 5 testes DeepEval executados (≥3 passando) | 20% | LAB4 |
| Dashboard comparativo Naive vs Advanced RAG com gráficos | 15% | LAB5 |
| Análise de erros com ≥3 queries diagnosticadas | 5% | LAB6 |

---

## Referências Bibliográficas (ABNT)

ES, S. et al. **RAGAS: Automated Evaluation of Retrieval Augmented Generation**. arXiv:2309.15217, 2023. Disponível em: <https://arxiv.org/abs/2309.15217>. Acesso em: abr. 2026.

CONFIDENT AI. **DeepEval — The Open-Source LLM Evaluation Framework**. Documentação oficial, 2024. Disponível em: <https://docs.confident-ai.com>. Acesso em: abr. 2026.

SAAD-FALCON, J. et al. **ARES: An Automated Evaluation Framework for Retrieval-Augmented Generation Systems**. arXiv:2311.09476, 2023.

GENG, S. et al. **Faithful and Informative Question Answering over Knowledge Graphs**. arXiv:2302.09060, 2023.

CHEN, B. et al. **RAGAS v0.2: Towards Production-Ready RAG Evaluation**. arXiv:2404.14744, 2024.

KWON, W. et al. **Efficient Memory Management for LLM Serving with PagedAttention**. *ACM SOSP*, 2023.

LEWIS, P. et al. **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks**. *NeurIPS*, v. 33, 2020.

LANGFUSE. **Scores API Documentation**. Disponível em: <https://langfuse.com/docs/scores/custom>. Acesso em: abr. 2026.
