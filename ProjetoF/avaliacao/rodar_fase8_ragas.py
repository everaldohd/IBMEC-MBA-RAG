"""
rodar_fase8_ragas.py - Fase 8 do Roteiro_Final.md (Secao 7): "melhor contexto -> resposta
mais fiel" - avaliacao da GERACAO (nao so recuperacao) com RAGAS.

Roda o RAG COMPLETO (busca + geracao, via app/busca_avancada.py::construir) em duas
configuracoes e mede 4 metricas RAGAS (padrao das Aulas 5/8, juiz Groq):
  - Faithfulness                    : a resposta se apoia nos trechos recuperados?
  - ResponseRelevancy(strictness=1) : a resposta responde a pergunta? (strictness=1
                                       porque a Groq so aceita n=1, nao o default=3 do RAGAS)
  - LLMContextPrecisionWithReference: os trechos relevantes vieram no topo do ranking?
  - LLMContextRecall                : os trechos cobrem o que a resposta_referencia exige?

As DUAS configuracoes avaliadas (Roteiro: "rode na baseline e na melhor configuracao"):
  1. BASELINE (exp01, Fase 0): Docling + hierarquico + nomic-embed-text + busca=baseline.
     O indice atual (herdado das Fases 4-7) esta em qwen3-embedding:4b - este script
     REINDEXA de volta para nomic-embed-text so para esta medicao.
  2. MELHOR CONFIGURACAO (exp10, vencedora de toda a jornada): Docling + hierarquico +
     qwen3-embedding:4b + busca=baseline (densa pura) - nenhuma tecnica alternativa
     (hibrida/query-enhancement/rerank/hyde) bateu a densa pura em nenhuma fase (4-7).

Ao final, o indice e RESTAURADO para qwen3-embedding:4b (a melhor config), que e o
estado final do projeto para o relatorio.

resultados.csv so tem colunas ragas_faith/ragas_ans_rel/ragas_ctx_recall (sem coluna
propria para Context Precision, conforme o template da Secao 7 do Roteiro) - o valor de
Context Precision fica registrado no campo 'observacao' de cada linha.

RESUMIVEL (pula o que ja esta em resultados.csv) e com try/except por pergunta, como
as fases anteriores - mas aqui cada pergunta faz 2 chamadas de LLM (geracao da resposta
+ o proprio RAGAS chama o juiz varias vezes por amostra), entao e mais lento e mais
sensivel a limite de taxa/cota do provedor. Rode com --limite N para testar num
subconjunto antes de rodar as 30 perguntas completas.

CACHE DE GERACAO (avaliacao/cache_geracao_<exp>.json, so quando --limite=0): a geracao
(busca+LLM) e a parte mais cara em tokens desta fase (prompt com ate 10 chunks de
contexto por pergunta). Se a cota do provedor estourar no meio das 30 perguntas (ja
aconteceu com o Groq), rodar de novo reaproveita as respostas ja geradas com sucesso em
vez de gastar tokens de novo nelas - o cache e apagado automaticamente depois que a
configuracao inteira e registrada em resultados.csv.

NOTA (3 bugs corrigidos, todos no juiz do RAGAS - avaliar_ragas()):
  1. A 1a rodada (com Groq) passava base_url=llm_base ao ChatGroq, mas o ChatGroq
     (langchain_groq) ja usa "https://api.groq.com/openai/v1" internamente por padrao -
     passar de novo duplicava o path (POST /openai/v1/openai/v1/chat/completions -> 404)
     e derrubava TODAS as chamadas de juiz.
  2. Depois de trocar o provedor para OpenRouter no .env (Groq TPD=100k/dia esgotou 3
     chaves seguidas - cada config desta fase sozinha ja consome quase toda a cota
     diaria), o ChatGroq CONTINUAVA fixo na API do Groq (ignora LLM_BASE_URL) - a chave
     do OpenRouter foi enviada pro endpoint do Groq -> 401 Invalid API Key em 100% dos
     jobs, mesmo com as 30 respostas geradas certinho (a geracao usa o OpenAIGenerator
     do Haystack, que ja e agnostico e usava o endpoint certo). Corrigido trocando para
     ChatOpenAI (langchain_openai) generico, com base_url explicito de config_llm().
  3. Com o ChatOpenAI sem max_tokens explicito, o juiz cortava a resposta no meio
     (LLMDidNotFinishException) - Faithfulness decompoe a resposta em varias afirmacoes
     e verifica cada uma, o que gera uma saida longa (JSON estruturado). Corrigido com
     max_tokens=2048 no ChatOpenAI.

Uso (de dentro de ProjetoF/, venv ativado, precisa de OpenSearch + Ollama + LLM no ar):

    pip install -r requirements.txt   # instala ragas/langchain-openai/langchain-ollama
    python avaliacao/rodar_fase8_ragas.py
    python avaliacao/rodar_fase8_ragas.py --limite 5     # teste rapido, nao salva no csv
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

PASTA_AVAL = Path(__file__).resolve().parent
PASTA_PROJETO = PASTA_AVAL.parent
sys.path.insert(0, str(PASTA_PROJETO))

PDF = PASTA_PROJETO / "CadeiaCustodia_ProvaPericial_lei13.964-2019.pdf"
CHUNKING_FIXO = "hierarquico"
MODELO_BASELINE = "nomic-embed-text"
MODELO_MELHOR = "qwen3-embedding:4b"
TOP_K = 10

CONFIGURACOES = [
    # (exp_nome, modelo_embedding, descricao)
    ("exp23_ragas_baseline", MODELO_BASELINE,
     "Fase 0 exata: Docling+hierarquico+nomic-embed-text+busca=baseline"),
    ("exp24_ragas_melhor", MODELO_MELHOR,
     "melhor config da jornada: Docling+hierarquico+qwen3-embedding:4b+busca=baseline "
     "(densa pura venceu todas as tecnicas alternativas nas Fases 4-7)"),
]


# ---------------------------------------------------------------------------
# Reindexacao (mesmo padrao de rodar_fase3_embedding.py)
# ---------------------------------------------------------------------------
def medir_dimensao(tag):
    from app import config
    from haystack_integrations.components.embedders.ollama import OllamaTextEmbedder
    base_url, _ = config.config_ollama()
    emb = OllamaTextEmbedder(model=tag, url=base_url)
    vetor = emb.run(text="teste de dimensao do embedding")["embedding"]
    return len(vetor)


def preparar_ambiente_embedding(tag):
    from app import config
    dim = medir_dimensao(tag)
    config.DIMENSAO_EMBEDDING[tag.split(":")[0].lower()] = dim
    os.environ["EMBEDDING_MODEL"] = tag
    print(f"  Ambiente preparado: EMBEDDING_MODEL={tag} (dimensao={dim})")
    return dim


def limpar_indice():
    import requests
    from app import config
    cfg = config.config_opensearch()
    auth = (cfg["usuario"], cfg["senha"]) if cfg["usuario"] else None
    r = requests.delete(f"{cfg['url']}/{cfg['indice']}", auth=auth, verify=False)
    print(f"  limpar_indice: DELETE {cfg['url']}/{cfg['indice']} -> {r.status_code}")


def reindexar(conteudo):
    from app import indexacao
    docs = indexacao.chunkar(conteudo, CHUNKING_FIXO)
    n = indexacao.indexar_opensearch(docs, meta={"arquivo": PDF.name})
    print(f"  reindexar: {n} chunk(s) ({CHUNKING_FIXO})")
    return n


def extrair_docling():
    from app.extracao import _impl_texto
    conteudo, _ = _impl_texto(str(PDF))
    return conteudo


# ---------------------------------------------------------------------------
# RAG completo (busca + geracao) usando o pipeline de PRODUCAO
# ---------------------------------------------------------------------------
def gerar_resposta(pergunta, top_k=TOP_K):
    from app import busca_avancada
    pipe, inputs, chave_docs = busca_avancada.construir("baseline", top_k, pergunta)
    saida = pipe.run(inputs, include_outputs_from={chave_docs})
    docs = saida[chave_docs]["documents"]
    resposta = saida["llm"]["replies"][0]
    return resposta, [d.content for d in docs]


# ---------------------------------------------------------------------------
# RAGAS
# ---------------------------------------------------------------------------
def avaliar_ragas(amostras):
    """amostras: [{"pergunta", "resposta", "contextos", "referencia"}, ...] -> DataFrame."""
    from langchain_openai import ChatOpenAI
    from ragas import EvaluationDataset, evaluate
    from ragas.dataset_schema import SingleTurnSample
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import (Faithfulness, LLMContextPrecisionWithReference,
                                LLMContextRecall, ResponseRelevancy)

    from app import config

    api_key, gmodelo, llm_base = config.config_llm()
    # O projeto e agnostico de provedor (LLM_BASE_URL/LLM_MODEL/LLM_API_KEY no .env) -
    # o juiz do RAGAS precisa respeitar isso, entao usa ChatOpenAI generico (aceita
    # qualquer endpoint OpenAI-compativel via base_url) em vez de ChatGroq (que so fala
    # com a API do Groq, ignorando LLM_BASE_URL). Ver NOTA no topo do arquivo.
    # max_tokens=2048: sem isso, o ChatOpenAI (via OpenRouter) cortou respostas do juiz
    # no meio (LLMDidNotFinishException) - Faithfulness decompoe a resposta em varias
    # afirmacoes e verifica cada uma, o que pode gerar uma saida longa (JSON estruturado).
    juiz = LangchainLLMWrapper(ChatOpenAI(model=gmodelo, api_key=api_key,
                                           base_url=llm_base, temperature=0,
                                           max_tokens=2048))

    base_url, modelo_emb = config.config_ollama()
    try:
        from langchain_ollama import OllamaEmbeddings
    except ImportError:
        from langchain_community.embeddings import OllamaEmbeddings
    emb = LangchainEmbeddingsWrapper(OllamaEmbeddings(model=modelo_emb, base_url=base_url))

    samples = [SingleTurnSample(user_input=a["pergunta"], response=a["resposta"],
                                 retrieved_contexts=a["contextos"] or ["(sem contexto)"],
                                 reference=a["referencia"])
               for a in amostras]
    dataset = EvaluationDataset(samples=samples)

    # strictness=1: ResponseRelevancy gera N perguntas por amostra p/ comparar com a
    # original (default N=3 no RAGAS); a Groq rejeita 'n>1' na chamada de completion.
    metricas = [Faithfulness(), ResponseRelevancy(strictness=1),
                LLMContextPrecisionWithReference(), LLMContextRecall()]
    resultado = evaluate(dataset=dataset, metrics=metricas, llm=juiz, embeddings=emb)
    return resultado.to_pandas()


# ---------------------------------------------------------------------------
# Execucao por configuracao
# ---------------------------------------------------------------------------
def exps_ja_registrados(caminho_csv):
    if not Path(caminho_csv).exists():
        return set()
    with open(caminho_csv, newline="", encoding="utf-8") as f:
        return {row["exp"] for row in csv.DictReader(f)}


def _cache_path(exp_nome):
    return PASTA_AVAL / f"cache_geracao_{exp_nome}.json"


def _carregar_cache(exp_nome):
    p = _cache_path(exp_nome)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {}


def _salvar_cache(exp_nome, cache):
    _cache_path(exp_nome).write_text(json.dumps(cache, ensure_ascii=False, indent=2),
                                      encoding="utf-8")


def rodar_configuracao(exp_nome, queries, limite):
    """Gera a resposta completa (RAG) para cada pergunta e roda RAGAS em cima.

    CACHE por pergunta (avaliacao/cache_geracao_<exp>.json, so quando --limite=0):
    a geracao (busca+LLM) e a parte mais cara em tokens desta fase (cada resposta
    usa um prompt com ate 10 chunks de contexto). Se a cota estourar no meio das 30
    perguntas, rodar de novo SEM o cache perderia e re-gastaria tokens nas perguntas
    que ja tinham gerado resposta com sucesso. Com o cache, so as perguntas que ainda
    nao tem resposta salva fazem uma nova chamada de LLM - o restante e reaproveitado
    do arquivo.
    """
    perguntas = queries[:limite] if limite else queries
    cache = {} if limite else _carregar_cache(exp_nome)
    amostras = []
    falhas = []
    latencias = []
    for i, q in enumerate(perguntas, 1):
        print(f"  [{i:02d}/{len(perguntas)}] ({q['tipo']}) {q['query'][:70]}...", flush=True)
        if q["id"] in cache:
            c = cache[q["id"]]
            amostras.append({"id": q["id"], "pergunta": q["query"], "resposta": c["resposta"],
                              "contextos": c["contextos"], "referencia": q["resposta_referencia"]})
            latencias.append(c.get("latencia_s", 0.0))
            print("      (cache: resposta ja gerada numa rodada anterior, sem nova chamada de LLM)")
            continue
        try:
            t0 = time.perf_counter()
            resposta, contextos = gerar_resposta(q["query"])
            dt = time.perf_counter() - t0
        except Exception as e:
            print(f"      FALHOU (geracao): {e}", flush=True)
            falhas.append({"id": q["id"], "erro": str(e)})
            continue
        latencias.append(dt)
        amostras.append({"id": q["id"], "pergunta": q["query"], "resposta": resposta,
                          "contextos": contextos, "referencia": q["resposta_referencia"]})
        if not limite:
            cache[q["id"]] = {"resposta": resposta, "contextos": contextos,
                               "latencia_s": round(dt, 3)}
            _salvar_cache(exp_nome, cache)   # incremental - sobrevive a TPD/crash no meio

    if not amostras:
        raise RuntimeError(f"Nenhuma resposta gerada para {exp_nome} ({len(falhas)} falha(s)).")

    print(f"  Rodando RAGAS em {len(amostras)} amostra(s) (varias chamadas de LLM por "
          f"amostra - pode demorar)...", flush=True)
    df = avaliar_ragas(amostras)

    def media(col):
        return float(df[col].mean()) if col in df.columns else float("nan")

    medias = {
        "faithfulness": round(media("faithfulness"), 4),
        "answer_relevancy": round(media("answer_relevancy"), 4),
        "context_precision": round(media("llm_context_precision_with_reference"), 4),
        "context_recall": round(media("context_recall"), 4),
    }
    if all(v != v for v in medias.values()):   # v != v <=> math.isnan(v), sem precisar importar math
        raise RuntimeError(
            f"RAGAS devolveu NaN em TODAS as 4 metricas para {exp_nome} - as "
            f"{len(amostras)} resposta(s) foram geradas, mas nenhuma chamada de juiz do "
            f"RAGAS deu certo (confira o log acima por 404/429/401/timeout). NADA foi "
            f"salvo em resultados.csv - as respostas geradas continuam no cache "
            f"({_cache_path(exp_nome).name}) para a proxima tentativa nao gastar tokens "
            f"de novo.")
    detalhes = []
    for a, (_, row) in zip(amostras, df.iterrows()):
        detalhes.append({
            "id": a["id"], "pergunta": a["pergunta"], "resposta": a["resposta"],
            "faithfulness": row.get("faithfulness"),
            "answer_relevancy": row.get("answer_relevancy"),
            "context_precision": row.get("llm_context_precision_with_reference"),
            "context_recall": row.get("context_recall"),
        })
    resumo = {**medias, "latencia_media_s": round(sum(latencias) / len(latencias), 3),
              "n_queries": len(amostras), "n_falhas": len(falhas)}
    return resumo, detalhes, falhas


def salvar_csv(linha):
    import avaliar_recuperacao as av
    av.salvar_csv(linha)


def salvar_detalhes(exp_nome, detalhes, falhas):
    caminho = PASTA_AVAL / f"detalhe_{exp_nome}.json"
    caminho.write_text(json.dumps({"detalhes": detalhes, "falhas": falhas},
                                   ensure_ascii=False, indent=2, default=float),
                        encoding="utf-8")
    return caminho


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limite", type=int, default=0,
                     help="avalia so as N primeiras perguntas (0 = todas as 30). "
                          "Com --limite, NAO reindexa nem salva em resultados.csv - so testa.")
    args = ap.parse_args()

    print("== Fase 8: RAGAS (avaliacao da geracao) ==\n")

    with open(PASTA_AVAL / "dataset.json", encoding="utf-8") as f:
        dataset = json.load(f)
    queries = dataset["queries_benchmark"]

    import avaliar_recuperacao as av
    ja_feitos = exps_ja_registrados(av.RESULTADOS)

    print("Extraindo com Docling (fixo, igual as fases anteriores)...")
    t0 = time.time()
    conteudo = extrair_docling()
    print(f"  {len(conteudo)} caracteres em {time.time() - t0:.1f}s\n")

    resumos = {}
    falhas_config = []

    for exp_nome, modelo_emb, descricao in CONFIGURACOES:
        if not args.limite and exp_nome in ja_feitos:
            print(f"--- {exp_nome} --- JA REGISTRADO, pulando.\n")
            continue
        print(f"--- {exp_nome} ({modelo_emb}) ---")
        print(f"  {descricao}")
        try:
            preparar_ambiente_embedding(modelo_emb)
            if not args.limite:
                print("  Limpando indice...")
                limpar_indice()
                print("  Reindexando...")
                reindexar(conteudo)
            else:
                print("  --limite ativo: usando o indice JA carregado no OpenSearch, sem reindexar "
                      "(garanta que o EMBEDDING_MODEL do .env bate com o indice atual antes de "
                      "confiar nos resultados de um teste rapido).")

            resumo, detalhes, falhas = rodar_configuracao(exp_nome, queries, args.limite)
        except Exception as e:
            print(f"  FALHOU: {e}")
            falhas_config.append((exp_nome, str(e)))
            continue

        print(f"  faithfulness={resumo['faithfulness']} answer_relevancy={resumo['answer_relevancy']} "
              f"context_precision={resumo['context_precision']} context_recall={resumo['context_recall']} "
              f"({resumo['n_queries']} ok, {resumo['n_falhas']} falha(s))")

        if args.limite:
            print("  (--limite ativo: NAO salvo em resultados.csv - rode sem --limite para registrar)\n")
            resumos[exp_nome] = resumo
            continue

        linha = {"exp": exp_nome, "fase": "Fase 8",
                 "mudanca": f"RAGAS (geracao completa) em cima da tecnica=baseline, top_k={TOP_K}, "
                            f"embedding={modelo_emb}",
                 "tecnica": "baseline", "top_k": TOP_K,
                 "ragas_faith": resumo["faithfulness"],
                 "ragas_ans_rel": resumo["answer_relevancy"],
                 "ragas_ctx_recall": resumo["context_recall"],
                 "latencia_media_s": resumo["latencia_media_s"],
                 "n_queries": resumo["n_queries"], "n_falhas": resumo["n_falhas"],
                 "observacao": f"context_precision(RAGAS)={resumo['context_precision']} "
                               f"(sem coluna propria no csv - ver Roteiro Secao 7)"}
        salvar_csv(linha)
        caminho_detalhe = salvar_detalhes(exp_nome, detalhes, falhas)
        resumos[exp_nome] = resumo
        print(f"  detalhe: {caminho_detalhe}\n")
        cache_path = _cache_path(exp_nome)
        if cache_path.exists():
            cache_path.unlink()   # ja registrado em resultados.csv/detalhe - cache nao e mais util

    if not args.limite:
        print("== restaurando o indice para qwen3-embedding:4b (melhor config, estado final "
              "do projeto) ==")
        try:
            preparar_ambiente_embedding(MODELO_MELHOR)
            limpar_indice()
            reindexar(conteudo)
            print("OK: OpenSearch de volta a qwen3-embedding:4b/hierarquico (melhor config).")
        except Exception as e:
            print(f"  FALHOU a restauracao final: {e}")
            print("  ATENCAO: o indice pode nao estar na melhor config agora - rode de novo.")

    print("\n== resumo Fase 8 (RAGAS) ==")
    print(f"  {'config':<22} {'faith':>7} {'ans_rel':>8} {'ctx_prec':>9} {'ctx_rec':>8}")
    for exp_nome, _, _ in CONFIGURACOES:
        if exp_nome in resumos:
            r = resumos[exp_nome]
            print(f"  {exp_nome:<22} {r['faithfulness']:>7} {r['answer_relevancy']:>8} "
                  f"{r['context_precision']:>9} {r['context_recall']:>8}")
        elif exp_nome in ja_feitos:
            print(f"  {exp_nome:<22} (ja estava em resultados.csv de uma rodada anterior)")
        else:
            print(f"  {exp_nome:<22} FALHOU nesta rodada - nao registrado")

    if falhas_config:
        print(f"\nATENCAO: {len(falhas_config)} configuracao(oes) falharam e NAO foram registradas:")
        for exp_nome, erro in falhas_config:
            print(f"  - {exp_nome}: {erro[:200]}")
        print("Corrija a causa e rode o script de novo - o que ja passou sera pulado.")
        sys.exit(1)


if __name__ == "__main__":
    main()
