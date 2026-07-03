"""
avaliar_recuperacao.py - mede a RECUPERACAO (retrieval) do projeto contra o gabarito de
avaliacao/dataset.json: Hit@5, Recall@5, MRR e NDCG@10 (convencao da Secao 7 do
Roteiro_Final.md). Roda de dentro de ProjetoF/ (mesmo diretorio de app/), com o venv
do projeto ativado.

Uso (de dentro de ProjetoF/, com o venv ativado):

    python avaliacao/avaliar_recuperacao.py --exp exp01_baseline --fase "Fase 0" \
        --mudanca "baseline: chunking=auto(hierarquico), embedding=nomic-embed-text, busca=baseline" \
        --tecnica baseline --top-k 10

    python avaliacao/avaliar_recuperacao.py --exp exp05_multiquery --fase "Fase 5" \
        --mudanca "busca=multi_query" --tecnica multi_query --top-k 10

Cada execucao acrescenta UMA linha em avaliacao/resultados.csv (nao sobrescreve).

--- Por que "identificar" o chunk por CONTEUDO, e nao por ID -------------------------
O pipeline do projeto (app/indexacao.py) nao grava um id_original estavel por chunk:
todos os chunks do mesmo PDF compartilham o mesmo meta['arquivo'], entao
d.meta.get('id_original') e sempre None e cairia no fallback de app/consulta.py
(mesmo id p/ todo mundo). Por isso este script NAO casa por id: ele procura, no texto de
cada chunk recuperado, os 'marcadores' (trechos distintivos) de cada documento do
gabarito (D01..D18, em avaliacao/dataset.json) e atribui o chunk ao primeiro Dxx cujo
marcador aparece (comparacao sem acento, case-insensitive). Chunks sem nenhum marcador
reconhecido viram um id unico "__nao_identificado_N" (nao relevante p/ nenhuma pergunta,
mas ainda ocupam a posicao no ranking - importante para @K nao ficar artificialmente
melhor). E uma aproximacao, documentada em avaliacao/README.md.

--- Por que NAO usa busca_avancada.construir() direto -------------------------------
busca_avancada.construir() monta um pipeline que TERMINA numa geracao de resposta
(prompt+llm), mesmo quando so pedimos o output do retriever via include_outputs_from
(o Haystack roda o grafo inteiro, geracao inclusa). Como este script so mede
RECUPERACAO, essa geracao e desperdicio de tokens/tempo - e foi o que estourou o
limite diario do Groq (TPD) numa rodada anterior. construir_retrieval_only() replica a
logica de busca_avancada.py SEM os componentes finais de geracao: as tecnicas 'baseline',
'hibrida' e 'rerank' ficam 100% locais (Ollama + cross-encoder via transformers/torch,
sem chamada de LLM), 'multi_query'/'rag_fusion'/'step_back'/'hyde' usam 1 chamada de LLM
(a reescrita da query ou a geracao do documento hipotetico, intrinseca a cada tecnica).
"""

import argparse
import csv
import json
import math
import sys
import time
import traceback
import unicodedata
from pathlib import Path

PASTA_AVAL = Path(__file__).resolve().parent
PASTA_PROJETO = PASTA_AVAL.parent
sys.path.insert(0, str(PASTA_PROJETO))  # para "from app import ..."

DATASET = PASTA_AVAL / "dataset.json"
RESULTADOS = PASTA_AVAL / "resultados.csv"

CAMPOS_CSV = ["exp", "fase", "mudanca", "tecnica", "top_k", "hit@5", "recall@5", "mrr",
              "ndcg@10", "ragas_faith", "ragas_ans_rel", "ragas_ctx_recall",
              "latencia_media_s", "n_queries", "n_falhas", "observacao"]


# ---------------------------------------------------------------------------
# Pipeline de RECUPERACAO (sem geracao de resposta) - espelha busca_avancada.py
# ---------------------------------------------------------------------------
def construir_retrieval_only(tecnica, top_k, pergunta, top_k_inicial=None):
    from haystack import Pipeline
    from haystack.components.builders import PromptBuilder
    from haystack.components.generators import OpenAIGenerator
    from haystack.utils import Secret
    from haystack_integrations.components.embedders.ollama import OllamaTextEmbedder
    from haystack_integrations.components.retrievers.opensearch import (
        OpenSearchEmbeddingRetriever, OpenSearchHybridRetriever)

    from app import busca_avancada, config, indexacao, prompts

    if tecnica not in busca_avancada.TECNICAS:
        tecnica = "baseline"
    base_ollama, modelo_emb = config.config_ollama()
    store = indexacao._store_opensearch()
    pipe = Pipeline()

    if tecnica == "baseline":
        # 100% local: sem nenhuma chamada de LLM.
        pipe.add_component("embedder", OllamaTextEmbedder(model=modelo_emb, url=base_ollama))
        pipe.add_component("retriever", OpenSearchEmbeddingRetriever(document_store=store, top_k=top_k))
        pipe.connect("embedder.embedding", "retriever.query_embedding")
        return pipe, {"embedder": {"text": pergunta}}, "retriever"

    if tecnica == "hibrida":
        # BM25 (lexico) + denso, fundidos por RRF - tambem 100% local (sem LLM).
        embedder_hib = OllamaTextEmbedder(model=modelo_emb, url=base_ollama)
        pipe.add_component("retriever", OpenSearchHybridRetriever(
            document_store=store, embedder=embedder_hib,
            top_k_bm25=top_k, top_k_embedding=top_k, top_k=top_k,
            join_mode="reciprocal_rank_fusion"))
        return pipe, {"retriever": {"query": pergunta}}, "retriever"

    if tecnica == "rerank":
        # busca densa (pool maior) -> cross-encoder local (BGE-reranker-v2-m3) -> top_k.
        # 100% local: sem nenhuma chamada de LLM (o reranker roda via transformers/torch).
        from haystack.components.rankers import TransformersSimilarityRanker

        pool = top_k_inicial or max(2 * top_k, 20)
        pipe.add_component("embedder", OllamaTextEmbedder(model=modelo_emb, url=base_ollama))
        pipe.add_component("retriever", OpenSearchEmbeddingRetriever(document_store=store, top_k=pool))
        pipe.add_component("ranker", TransformersSimilarityRanker(
            model=busca_avancada.MODELO_RERANKER, top_k=top_k))
        pipe.connect("embedder.embedding", "retriever.query_embedding")
        pipe.connect("retriever.documents", "ranker.documents")
        return pipe, {"embedder": {"text": pergunta}, "ranker": {"query": pergunta}}, "ranker"

    if tecnica == "hyde":
        # LLM gera um documento hipotetico -> embeda ESSE texto (nao a pergunta) -> busca.
        # 1 chamada de LLM (intrinseca a tecnica), sem geracao final de resposta.
        p = prompts.get_prompts()
        api_key, gmodelo, llm_base = config.config_llm()
        hyde_llm = OpenAIGenerator(api_key=Secret.from_token(api_key), model=gmodelo,
                                    api_base_url=llm_base,
                                    generation_kwargs={"temperature": 0.3, "max_tokens": 300})
        pipe.add_component("hyde_prompt", PromptBuilder(template=p["hyde"], required_variables="*"))
        pipe.add_component("hyde_llm", hyde_llm)
        pipe.add_component("buscar", busca_avancada.BuscarPorHyde(store, top_k=top_k))
        pipe.connect("hyde_prompt.prompt", "hyde_llm.prompt")
        pipe.connect("hyde_llm.replies", "buscar.textos")
        return pipe, {"hyde_prompt": {"pergunta": pergunta}}, "buscar"

    # multi_query / rag_fusion / step_back: 1 chamada de LLM p/ reescrever a query
    # (intrinseca a tecnica) - mas SEM a geracao final de resposta.
    p = prompts.get_prompts()
    api_key, gmodelo, llm_base = config.config_llm()
    rw_llm = OpenAIGenerator(api_key=Secret.from_token(api_key), model=gmodelo,
                              api_base_url=llm_base,
                              generation_kwargs={"temperature": 0.3, "max_tokens": 300})

    if tecnica == "step_back":
        pipe.add_component("rw_prompt", PromptBuilder(template=p["stepback"], required_variables="*"))
        modo_montar, modo_fusao = "stepback", "dedup"
    else:
        pipe.add_component("rw_prompt", PromptBuilder(template=p["variacoes"], required_variables="*"))
        modo_montar = "variacoes"
        modo_fusao = "rrf" if tecnica == "rag_fusion" else "dedup"

    pipe.add_component("rw_llm", rw_llm)
    pipe.add_component("montar", busca_avancada.MontarConsultas(modo=modo_montar, n=4))
    pipe.add_component("buscar", busca_avancada.BuscarMultiplas(store, top_k=top_k, modo=modo_fusao))
    pipe.connect("rw_prompt.prompt", "rw_llm.prompt")
    pipe.connect("rw_llm.replies", "montar.textos")
    pipe.connect("montar.queries", "buscar.queries")
    inputs = {"rw_prompt": {"pergunta": pergunta}, "montar": {"question": pergunta}}
    return pipe, inputs, "buscar"


# ---------------------------------------------------------------------------
# Identificacao do chunk por conteudo + metricas
# ---------------------------------------------------------------------------
def _normalizar(txt):
    """minusculo + remove acentos, p/ comparacao robusta a diferencas do Docling."""
    txt = unicodedata.normalize("NFKD", txt.lower())
    return "".join(c for c in txt if not unicodedata.combining(c))


def carregar_dataset():
    dados = json.loads(DATASET.read_text(encoding="utf-8"))
    for doc in dados["documentos"]:
        doc["_marcadores_norm"] = [_normalizar(m) for m in doc.get("marcadores", [])]
    return dados


def identificar_doc(texto_chunk, documentos):
    """1o Dxx cujo marcador aparece no texto do chunk recuperado (ou None)."""
    alvo = _normalizar(texto_chunk)
    for doc in documentos:
        for marcador in doc["_marcadores_norm"]:
            if marcador in alvo:
                return doc["id"]
    return None


def dcg(ganhos):
    return sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(ganhos))


def metricas_query(ids_recuperados, gold_relevancia, k5=5, k10=10):
    relset = set(gold_relevancia)
    top5 = ids_recuperados[:k5]
    hit = 1.0 if (relset & set(top5)) else 0.0
    recall = len(relset & set(top5)) / len(relset) if relset else 0.0

    rr = 0.0
    for rank, doc_id in enumerate(ids_recuperados, 1):
        if doc_id in relset:
            rr = 1.0 / rank
            break

    top10 = ids_recuperados[:k10]
    ganhos = [gold_relevancia.get(d, 0) for d in top10]
    ideal = sorted(gold_relevancia.values(), reverse=True)[:k10]
    dcg_r, dcg_i = dcg(ganhos), dcg(ideal)
    ndcg = (dcg_r / dcg_i) if dcg_i > 0 else 0.0
    return hit, recall, rr, ndcg


# ---------------------------------------------------------------------------
# Execucao (resiliente: 1 falha nao derruba as outras 29; salva o que der)
# ---------------------------------------------------------------------------
def rodar(tecnica, top_k, top_k_inicial=None):
    dados = carregar_dataset()
    documentos, queries = dados["documentos"], dados["queries_benchmark"]

    hits = recalls = mrrs = ndcgs = 0.0
    latencias = []
    detalhes = []
    falhas = []

    for i, q in enumerate(queries, 1):
        print(f"[{i:02d}/{len(queries)}] ({q['tipo']}) {q['query'][:70]}...", flush=True)
        try:
            t0 = time.perf_counter()
            pipe, inputs, chave_docs = construir_retrieval_only(tecnica, top_k, q["query"], top_k_inicial)
            saida = pipe.run(inputs, include_outputs_from={chave_docs})
            docs = saida[chave_docs]["documents"]
            dt = time.perf_counter() - t0
        except Exception as e:
            print(f"    FALHOU: {e}", flush=True)
            falhas.append({"id": q["id"], "erro": str(e)})
            continue

        latencias.append(dt)
        ids_recuperados = [identificar_doc(d.content, documentos) or f"__nao_identificado_{j}"
                            for j, d in enumerate(docs)]

        hit, recall, rr, ndcg = metricas_query(ids_recuperados, q["relevancia"])
        hits += hit
        recalls += recall
        mrrs += rr
        ndcgs += ndcg
        detalhes.append({"id": q["id"], "tipo": q["tipo"], "hit@5": hit, "recall@5": recall,
                          "rr": round(rr, 4), "ndcg@10": round(ndcg, 4),
                          "ids_recuperados": ids_recuperados, "latencia_s": round(dt, 2)})

    n = len(detalhes)
    if n == 0:
        raise RuntimeError(f"Nenhuma query foi processada com sucesso ({len(falhas)} falha(s)). Nada foi salvo.")
    resumo = {"hit@5": round(hits / n, 4), "recall@5": round(recalls / n, 4),
              "mrr": round(mrrs / n, 4), "ndcg@10": round(ndcgs / n, 4),
              "latencia_media_s": round(sum(latencias) / n, 3), "n_queries": n, "n_falhas": len(falhas)}
    return resumo, detalhes, falhas


def salvar_csv(linha):
    novo = not RESULTADOS.exists()
    with open(RESULTADOS, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CAMPOS_CSV)
        if novo:
            w.writeheader()
        w.writerow({k: linha.get(k, "") for k in CAMPOS_CSV})


def salvar_detalhes(exp_nome, detalhes, falhas):
    caminho = PASTA_AVAL / f"detalhe_{exp_nome}.json"
    caminho.write_text(json.dumps({"detalhes": detalhes, "falhas": falhas}, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    return caminho


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", required=True, help="nome do experimento (ex.: exp01_baseline)")
    ap.add_argument("--fase", default="", help='ex.: "Fase 0"')
    ap.add_argument("--mudanca", default="", help="o que mudou nesta rodada (texto livre)")
    ap.add_argument("--tecnica", default="baseline",
                     choices=["baseline", "multi_query", "rag_fusion", "step_back", "hibrida",
                              "rerank", "hyde"])
    ap.add_argument("--top-k", type=int, default=10,
                     help="chunks recuperados por consulta (usa 10 p/ cobrir hit@5/recall@5 e ndcg@10)")
    ap.add_argument("--top-k-inicial", type=int, default=None,
                     help="so p/ --tecnica rerank: tamanho do pool antes do cross-encoder cortar em top-k")
    ap.add_argument("--observacao", default="")
    args = ap.parse_args()

    print(f"== avaliar_recuperacao: exp={args.exp} tecnica={args.tecnica} top_k={args.top_k} ==")
    try:
        resumo, detalhes, falhas = rodar(args.tecnica, args.top_k, args.top_k_inicial)
    except Exception:
        traceback.print_exc()
        print("\nNada foi salvo (falha antes de completar nenhuma query). Rode de novo.")
        sys.exit(1)

    linha = {"exp": args.exp, "fase": args.fase, "mudanca": args.mudanca,
             "tecnica": args.tecnica, "top_k": args.top_k, "observacao": args.observacao, **resumo}
    salvar_csv(linha)
    caminho_detalhe = salvar_detalhes(args.exp, detalhes, falhas)

    print("\n== resultado ==")
    for k in ("hit@5", "recall@5", "mrr", "ndcg@10", "latencia_media_s", "n_queries", "n_falhas"):
        print(f"  {k}: {resumo[k]}")
    if falhas:
        print(f"\n  ATENCAO: {len(falhas)} pergunta(s) falharam e NAO entraram na media:")
        for f in falhas:
            print(f"    - {f['id']}: {f['erro'][:120]}")
        print("  (rode de novo mais tarde p/ tentar cobrir 100% das perguntas)")
    print(f"\nLinha registrada em {RESULTADOS}")
    print(f"Detalhe por pergunta salvo em {caminho_detalhe}")


if __name__ == "__main__":
    main()
