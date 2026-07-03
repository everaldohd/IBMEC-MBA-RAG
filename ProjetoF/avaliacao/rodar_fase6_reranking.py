"""
rodar_fase6_reranking.py - Fase 6 do Roteiro_Final.md (Secao 7): "reordenar os top-N com
um cross-encoder sobe o relevante".

Testa a tecnica 'rerank' (busca DENSA recupera um pool maior de candidatos ->
TransformersSimilarityRanker com o cross-encoder BAAI/bge-reranker-v2-m3, Aula 3 ->
reordena -> corta no top_k final) contra a busca densa pura (exp10, mesma base), em 2
tamanhos de pool de candidatos (para medir sensibilidade ao tamanho do pool antes do
corte) - o Roteiro sugere "recupere top-20 -> reranqueie -> top-5"; aqui usamos
top_k final=10 (em vez de 5) para manter comparabilidade direta com a referencia exp10 e
com as Fases 4/5 (todas medidas em top_k=10, cobrindo Hit@5/Recall@5 e NDCG@10 na mesma
rodada).

NAO reindexa nada: a extracao/chunking/embedding continuam os da Fase 4
(Docling + hierarquico + qwen3-embedding:4b) - a UNICA variavel aqui e a tecnica de busca
(reranking) e o tamanho do pool de candidatos antes do corte.

IMPORTANTE (mesma licao da Fase 5): o indice no OpenSearch continua na dimensao da
Fase 4 (2560, qwen3-embedding:4b), mas a variavel de ambiente EMBEDDING_MODEL so vale
DENTRO do processo Python que a setou - cada novo script (como este) precisa medir a
dimensao e resetar EMBEDDING_MODEL de novo no INICIO, ou o OpenSearch rejeita a query
com "Query vector has invalid dimension". Ver preparar_ambiente_embedding() abaixo.

Diferenca em relacao a Fase 5 (query enhancement): reranking e 100% LOCAL (embedding via
Ollama + cross-encoder via transformers/torch, rodando na CPU/GPU da maquina) - nao
chama o Groq, entao nao ha risco de estourar a cota diaria de tokens (TPD) aqui. Em
compensacao, o cross-encoder e mais pesado que os embedders: a 1a chamada de cada
combinacao baixa o modelo BAAI/bge-reranker-v2-m3 (~560MB, cacheado pelo
huggingface_hub depois disso) e cada pergunta reranqueia `top_k_inicial` pares
(pergunta, chunk) na CPU - espere latencia bem maior que a busca densa pura.

Experimentos gerados (fase = "Fase 6"):
  exp20_rerank_pool20 (top_k_inicial=20, top_k final=10)
  exp21_rerank_pool40 (top_k_inicial=40, top_k final=10)

RESUMIVEL (pula o que ja esta em resultados.csv) e com try/except por combinacao, como
as fases anteriores.

Uso (de dentro de ProjetoF/, venv ativado, precisa de OpenSearch + Ollama no ar - Groq
NAO e necessario para esta fase):

    python avaliacao/rodar_fase6_reranking.py
"""

import csv
import os
import sys
from pathlib import Path

PASTA_AVAL = Path(__file__).resolve().parent
PASTA_PROJETO = PASTA_AVAL.parent
sys.path.insert(0, str(PASTA_PROJETO))

TOP_K = 10
MODELO_BASE = "qwen3-embedding:4b"   # herdado da Fase 4 - precisa ser re-setado no env
BASE_DESCRICAO = "Docling+hierarquico+qwen3-embedding:4b (mesma base da Fase 4)"


def medir_dimensao(tag):
    from app import config
    from haystack_integrations.components.embedders.ollama import OllamaTextEmbedder
    base_url, _ = config.config_ollama()
    emb = OllamaTextEmbedder(model=tag, url=base_url)
    vetor = emb.run(text="teste de dimensao do embedding")["embedding"]
    return len(vetor)


def preparar_ambiente_embedding():
    """Reseta EMBEDDING_MODEL/DIMENSAO_EMBEDDING neste processo (ver docstring do
    modulo) - o indice no OpenSearch ja esta na base certa desde a Fase 4, so o
    processo Python precisa ser lembrado de qual modelo embedar a query."""
    from app import config
    dim = medir_dimensao(MODELO_BASE)
    config.DIMENSAO_EMBEDDING[MODELO_BASE.split(":")[0].lower()] = dim
    os.environ["EMBEDDING_MODEL"] = MODELO_BASE
    print(f"Ambiente preparado: EMBEDDING_MODEL={MODELO_BASE} (dimensao={dim})\n")


# (top_k_inicial, nome do experimento)
COMBINACOES = [
    (20, "exp20_rerank_pool20"),
    (40, "exp21_rerank_pool40"),
]

REFERENCIA_EXP10 = {"hit@5": 0.9333, "recall@5": 0.7, "mrr": 0.6608, "ndcg@10": 0.7754}


def exps_ja_registrados(caminho_csv):
    if not Path(caminho_csv).exists():
        return set()
    with open(caminho_csv, newline="", encoding="utf-8") as f:
        return {row["exp"] for row in csv.DictReader(f)}


def main():
    print("== Fase 6: reranking (cross-encoder BAAI/bge-reranker-v2-m3) ==")
    print(f"   Base (sem reindexar - herdada da Fase 4): {BASE_DESCRICAO}\n")

    preparar_ambiente_embedding()

    import avaliar_recuperacao as av

    ja_feitos = exps_ja_registrados(av.RESULTADOS)
    resumos = {}
    falhas_combo = []

    for top_k_inicial, exp_nome in COMBINACOES:
        if exp_nome in ja_feitos:
            print(f"--- rerank top_k_inicial={top_k_inicial} top_k={TOP_K} ({exp_nome}) --- "
                  f"JA REGISTRADO, pulando.\n")
            continue
        print(f"--- rerank top_k_inicial={top_k_inicial} top_k={TOP_K} ({exp_nome}) ---")
        try:
            resumo, detalhes, falhas = av.rodar("rerank", TOP_K, top_k_inicial)
        except Exception as e:
            print(f"  FALHOU: {e}")
            print("  (rode de novo depois de corrigir - as combinacoes ja OK nao serao repetidas)\n")
            falhas_combo.append((exp_nome, str(e)))
            continue

        linha = {"exp": exp_nome, "fase": "Fase 6",
                 "mudanca": f"tecnica=rerank, top_k_inicial={top_k_inicial}, top_k={TOP_K}, "
                            f"base={BASE_DESCRICAO}",
                 "tecnica": "rerank", "top_k": TOP_K,
                 "observacao": f"cross-encoder BAAI/bge-reranker-v2-m3 (local, sem LLM), "
                               f"pool={top_k_inicial}",
                 **resumo}
        av.salvar_csv(linha)
        caminho_detalhe = av.salvar_detalhes(exp_nome, detalhes, falhas)
        resumos[exp_nome] = resumo
        print(f"  hit@5={resumo['hit@5']} recall@5={resumo['recall@5']} "
              f"mrr={resumo['mrr']} ndcg@10={resumo['ndcg@10']} "
              f"({resumo['n_queries']} ok, {resumo['n_falhas']} falha(s))")
        print(f"  detalhe: {caminho_detalhe}\n")

    print("\n== resumo Fase 6 (base: Docling + hierarquico + qwen3-embedding:4b) ==")
    print(f"  {'combinacao':<24} {'hit@5':>7} {'recall@5':>9} {'mrr':>7} {'ndcg@10':>8}")
    print(f"  {'densa top_k=10(exp10)':<24} {REFERENCIA_EXP10['hit@5']:>7} "
          f"{REFERENCIA_EXP10['recall@5']:>9} {REFERENCIA_EXP10['mrr']:>7} "
          f"{REFERENCIA_EXP10['ndcg@10']:>8}  (referencia, ja medido)")
    for top_k_inicial, exp_nome in COMBINACOES:
        if exp_nome in resumos:
            r = resumos[exp_nome]
            print(f"  {exp_nome:<24} {r['hit@5']:>7} {r['recall@5']:>9} {r['mrr']:>7} {r['ndcg@10']:>8}")
        elif exp_nome in ja_feitos:
            print(f"  {exp_nome:<24} (ja estava em resultados.csv de uma rodada anterior)")
        else:
            print(f"  {exp_nome:<24} FALHOU nesta rodada - nao registrado")
    print(f"\nLinhas registradas em {av.RESULTADOS}")
    print("\nNOTA: nenhuma reindexacao foi feita - o indice segue na mesma base da Fase 4 "
          "para a Fase 7 em diante (lembrar de repetir preparar_ambiente_embedding() la).")

    if falhas_combo:
        print(f"\nATENCAO: {len(falhas_combo)} combinacao(oes) falharam e NAO foram registradas:")
        for exp_nome, erro in falhas_combo:
            print(f"  - {exp_nome}: {erro[:200]}")
        print("Corrija a causa e rode o script de novo - o que ja passou sera pulado.")
        sys.exit(1)


if __name__ == "__main__":
    main()
