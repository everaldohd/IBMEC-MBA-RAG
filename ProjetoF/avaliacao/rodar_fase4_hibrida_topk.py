"""
rodar_fase4_hibrida_topk.py - Fase 4 do Roteiro_Final.md (Secao 7): "combinar lexico
(BM25) + denso melhora cobertura" + variar top_k.

MUDANCA DE METODOLOGIA A PARTIR DESTA FASE (decisao registrada em Relatorio.md, Secao
8.1): as Fases 1-3 testaram cada variavel ISOLADA contra o baseline original (exp01:
nomic-embed-text). A Fase 3 achou um ganho grande com qwen3-embedding:4b. A partir
daqui a jornada e PROGRESSIVA: a base fixa passa a ser
  extracao=Docling + chunking=hierarquico + embedding=qwen3-embedding:4b
(a melhor combinacao confirmada ate a Fase 3) e as fases seguintes testam suas
variaveis EM CIMA dela, em vez de continuar isolando contra nomic-embed-text. Por isso
este script NAO restaura o indice para nomic-embed-text ao final - ele deixa o indice
na nova base (qwen3-embedding:4b), que passa a valer para a Fase 5 em diante.

O QUE E TESTADO: a tecnica 'hibrida' (BM25 lexico + denso, fundidos por Reciprocal
Rank Fusion via OpenSearchHybridRetriever - novidade adicionada em
app/busca_avancada.py::construir() e avaliacao/avaliar_recuperacao.py::
construir_retrieval_only() nesta fase) contra a busca densa pura ('baseline'), em 3
valores de top_k (5/10/20). A combinacao baseline(densa)+top_k=10 NAO e reindexada de
novo aqui: e exatamente o exp10 (Fase 3), que ja usa essa mesma base.

Experimentos gerados (fase = "Fase 4"):
  exp12_densa_top5, exp13_densa_top20,
  exp14_hibrida_top5, exp15_hibrida_top10, exp16_hibrida_top20
  (exp10_embed_qwen3_embedding_4b == densa/top_k=10 nesta mesma base, ja medido)

RESUMIVEL (pula o que ja esta em resultados.csv) e com try/except por combinacao,
como as fases anteriores.

Uso (de dentro de ProjetoF/, venv ativado, precisa de OpenSearch + Ollama no ar, com
qwen3-embedding:4b ja baixado):

    python avaliacao/rodar_fase4_hibrida_topk.py
"""

import csv
import os
import sys
import time
from pathlib import Path

import requests

PASTA_AVAL = Path(__file__).resolve().parent
PASTA_PROJETO = PASTA_AVAL.parent
sys.path.insert(0, str(PASTA_PROJETO))

PDF = PASTA_PROJETO / "CadeiaCustodia_ProvaPericial_lei13.964-2019.pdf"
CHUNKING_FIXO = "hierarquico"
MODELO_BASE = "qwen3-embedding:4b"   # nova base fixa (vencedora da Fase 3)

# (tecnica, top_k) -> nome do experimento. baseline/top_k=10 fica de fora (== exp10).
COMBINACOES = [
    ("baseline", 5, "exp12_densa_top5"),
    ("baseline", 20, "exp13_densa_top20"),
    ("hibrida", 5, "exp14_hibrida_top5"),
    ("hibrida", 10, "exp15_hibrida_top10"),
    ("hibrida", 20, "exp16_hibrida_top20"),
]

REFERENCIA_EXP10 = {"hit@5": 0.9333, "recall@5": 0.7, "mrr": 0.6608, "ndcg@10": 0.7754}


def medir_dimensao(tag):
    from app import config
    from haystack_integrations.components.embedders.ollama import OllamaTextEmbedder
    base_url, _ = config.config_ollama()
    emb = OllamaTextEmbedder(model=tag, url=base_url)
    vetor = emb.run(text="teste de dimensao do embedding")["embedding"]
    return len(vetor)


def limpar_indice():
    from app import config
    cfg = config.config_opensearch()
    auth = (cfg["usuario"], cfg["senha"]) if cfg["usuario"] else None
    r = requests.delete(f"{cfg['url']}/{cfg['indice']}", auth=auth, verify=False)
    print(f"  limpar_indice: DELETE {cfg['url']}/{cfg['indice']} -> {r.status_code}")


def reindexar(conteudo):
    from app import indexacao
    docs = indexacao.chunkar(conteudo, CHUNKING_FIXO)
    n = indexacao.indexar_opensearch(docs, meta={"arquivo": PDF.name})
    print(f"  reindexar: {n} chunk(s) ({CHUNKING_FIXO}, embedding={MODELO_BASE})")
    return n


def extrair_docling():
    from app.extracao import _impl_texto
    conteudo, _ = _impl_texto(str(PDF))
    return conteudo


def exps_ja_registrados(caminho_csv):
    if not Path(caminho_csv).exists():
        return set()
    with open(caminho_csv, newline="", encoding="utf-8") as f:
        return {row["exp"] for row in csv.DictReader(f)}


def main():
    from app import config

    print("== Fase 4: busca hibrida (BM25+densa/RRF) e top_k ==")
    print(f"   Nova base fixa a partir daqui: Docling + {CHUNKING_FIXO} + {MODELO_BASE}\n")

    print("Medindo dimensao real do embedding e preparando o indice na nova base...")
    dim = medir_dimensao(MODELO_BASE)
    config.DIMENSAO_EMBEDDING[MODELO_BASE.split(":")[0].lower()] = dim
    os.environ["EMBEDDING_MODEL"] = MODELO_BASE
    print(f"  dimensao={dim}")

    print("Extraindo com Docling...")
    t0 = time.time()
    conteudo = extrair_docling()
    print(f"  {len(conteudo)} caracteres em {time.time() - t0:.1f}s")

    print("Reindexando (uma vez - top_k so afeta a BUSCA, nao a indexacao)...")
    limpar_indice()
    reindexar(conteudo)
    print()

    import avaliar_recuperacao as av

    ja_feitos = exps_ja_registrados(av.RESULTADOS)
    resumos = {}
    falhas_combo = []

    for tecnica, top_k, exp_nome in COMBINACOES:
        if exp_nome in ja_feitos:
            print(f"--- {tecnica} top_k={top_k} ({exp_nome}) --- JA REGISTRADO, pulando.\n")
            continue
        print(f"--- {tecnica} top_k={top_k} ({exp_nome}) ---")
        try:
            resumo, detalhes, falhas = av.rodar(tecnica, top_k)
        except Exception as e:
            print(f"  FALHOU: {e}")
            print("  (rode de novo depois de corrigir - as combinacoes ja OK nao serao repetidas)\n")
            falhas_combo.append((exp_nome, str(e)))
            continue

        linha = {"exp": exp_nome, "fase": "Fase 4",
                 "mudanca": f"tecnica={tecnica}, top_k={top_k}, base=Docling+{CHUNKING_FIXO}+{MODELO_BASE} "
                            f"(nova base progressiva pos-Fase 3)",
                 "tecnica": tecnica, "top_k": top_k,
                 "observacao": f"dimensao={dim}",
                 **resumo}
        av.salvar_csv(linha)
        caminho_detalhe = av.salvar_detalhes(exp_nome, detalhes, falhas)
        resumos[exp_nome] = resumo
        print(f"  hit@5={resumo['hit@5']} recall@5={resumo['recall@5']} "
              f"mrr={resumo['mrr']} ndcg@10={resumo['ndcg@10']} "
              f"({resumo['n_queries']} ok, {resumo['n_falhas']} falha(s))")
        print(f"  detalhe: {caminho_detalhe}\n")

    print("\n== resumo Fase 4 (base: Docling + hierarquico + qwen3-embedding:4b) ==")
    print(f"  {'combinacao':<28} {'hit@5':>7} {'recall@5':>9} {'mrr':>7} {'ndcg@10':>8}")
    print(f"  {'densa top_k=10(exp10)':<28} {REFERENCIA_EXP10['hit@5']:>7} "
          f"{REFERENCIA_EXP10['recall@5']:>9} {REFERENCIA_EXP10['mrr']:>7} "
          f"{REFERENCIA_EXP10['ndcg@10']:>8}  (referencia, ja medido)")
    for tecnica, top_k, exp_nome in COMBINACOES:
        if exp_nome in resumos:
            r = resumos[exp_nome]
            print(f"  {exp_nome:<28} {r['hit@5']:>7} {r['recall@5']:>9} {r['mrr']:>7} {r['ndcg@10']:>8}")
        elif exp_nome in ja_feitos:
            print(f"  {exp_nome:<28} (ja estava em resultados.csv de uma rodada anterior)")
        else:
            print(f"  {exp_nome:<28} FALHOU nesta rodada - nao registrado")
    print(f"\nLinhas registradas em {av.RESULTADOS}")
    print("\nNOTA: o indice NAO foi restaurado para nomic-embed-text - fica na nova base "
          f"(Docling + {CHUNKING_FIXO} + {MODELO_BASE}) para a Fase 5 em diante.")

    if falhas_combo:
        print(f"\nATENCAO: {len(falhas_combo)} combinacao(oes) falharam e NAO foram registradas:")
        for exp_nome, erro in falhas_combo:
            print(f"  - {exp_nome}: {erro[:200]}")
        print("Corrija a causa e rode o script de novo - o que ja passou sera pulado.")
        sys.exit(1)


if __name__ == "__main__":
    main()
