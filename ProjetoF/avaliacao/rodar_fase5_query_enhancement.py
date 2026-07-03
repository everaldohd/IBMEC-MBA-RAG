"""
rodar_fase5_query_enhancement.py - Fase 5 do Roteiro_Final.md (Secao 7): "reescrever a
pergunta ajuda a recuperar o que a pergunta original, sozinha, nao acha".

Testa as 3 tecnicas de query enhancement ja implementadas em app/busca_avancada.py
(multi_query, rag_fusion, step_back) contra a busca densa pura (exp10/exp13, mesma
base), no top_k=10 (ponto de equilibrio confirmado na Fase 4).

NAO reindexa nada: a extracao/chunking/embedding continuam os mesmos da Fase 4
(Docling + hierarquico + qwen3-embedding:4b) - a UNICA variavel aqui e a tecnica de
busca. Por isso este script nao tem etapa de limpar/reindexar OpenSearch, so roda a
avaliacao de retrieval 3 vezes.

IMPORTANTE (licao da 1a tentativa desta fase): o indice no OpenSearch continua na
dimensao da Fase 4 (2560, qwen3-embedding:4b) porque nunca foi apagado - mas a variavel
de ambiente EMBEDDING_MODEL que o app usa para saber COM QUAL MODELO EMBEDAR A QUERY
(app/config.py::config_ollama) so foi setada dentro do processo Python do script da
Fase 4 (rodar_fase4_hibrida_topk.py) e nao persiste em .env nem no SO - cada novo
processo (como este script) volta a usar o default 'nomic-embed-text' (768d) se
ninguem setar de novo. Isso causa erro do OpenSearch ("Query vector has invalid
dimension: 768. Dimension should be: 2560"), porque a query e embedada com 768d contra
um indice de 2560d. Por isso este script mede a dimensao e seta EMBEDDING_MODEL de
novo no INICIO, antes de qualquer chamada a avaliar_recuperacao - e as Fases 6-8 (que
tambem herdam esta mesma base) precisam repetir esse mesmo cuidado.

Diferenca importante em relacao as Fases 1-4: multi_query/rag_fusion/step_back fazem
1 chamada de LLM (Groq) por pergunta para reescrever/gerar variacoes da query - isso e
intrinseco a tecnica (ver avaliar_recuperacao.py::construir_retrieval_only). Sao 30
perguntas x 3 tecnicas = ate 90 chamadas curtas de reescrita (max_tokens=300, sem
geracao de resposta final) - bem abaixo do que estourou o limite diario do Groq numa
rodada anterior (que usava o pipeline completo com geracao).

Experimentos gerados (fase = "Fase 5"):
  exp17_multi_query, exp18_rag_fusion, exp19_step_back

RESUMIVEL (pula o que ja esta em resultados.csv) e com try/except por tecnica, como as
fases anteriores.

Uso (de dentro de ProjetoF/, venv ativado, precisa de OpenSearch + Ollama + Groq no ar):

    python avaliacao/rodar_fase5_query_enhancement.py
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

# (tecnica, top_k) -> nome do experimento
COMBINACOES = [
    ("multi_query", TOP_K, "exp17_multi_query"),
    ("rag_fusion", TOP_K, "exp18_rag_fusion"),
    ("step_back", TOP_K, "exp19_step_back"),
]

REFERENCIA_EXP10 = {"hit@5": 0.9333, "recall@5": 0.7, "mrr": 0.6608, "ndcg@10": 0.7754}


def exps_ja_registrados(caminho_csv):
    if not Path(caminho_csv).exists():
        return set()
    with open(caminho_csv, newline="", encoding="utf-8") as f:
        return {row["exp"] for row in csv.DictReader(f)}


def main():
    print("== Fase 5: query enhancement (multi_query / rag_fusion / step_back) ==")
    print(f"   Base (sem reindexar - herdada da Fase 4): {BASE_DESCRICAO}\n")

    preparar_ambiente_embedding()

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

        linha = {"exp": exp_nome, "fase": "Fase 5",
                 "mudanca": f"tecnica={tecnica}, top_k={top_k}, base={BASE_DESCRICAO}",
                 "tecnica": tecnica, "top_k": top_k,
                 "observacao": "reescrita de query via Groq (sem geracao de resposta final)",
                 **resumo}
        av.salvar_csv(linha)
        caminho_detalhe = av.salvar_detalhes(exp_nome, detalhes, falhas)
        resumos[exp_nome] = resumo
        print(f"  hit@5={resumo['hit@5']} recall@5={resumo['recall@5']} "
              f"mrr={resumo['mrr']} ndcg@10={resumo['ndcg@10']} "
              f"({resumo['n_queries']} ok, {resumo['n_falhas']} falha(s))")
        print(f"  detalhe: {caminho_detalhe}\n")

    print("\n== resumo Fase 5 (base: Docling + hierarquico + qwen3-embedding:4b) ==")
    print(f"  {'combinacao':<24} {'hit@5':>7} {'recall@5':>9} {'mrr':>7} {'ndcg@10':>8}")
    print(f"  {'densa top_k=10(exp10)':<24} {REFERENCIA_EXP10['hit@5']:>7} "
          f"{REFERENCIA_EXP10['recall@5']:>9} {REFERENCIA_EXP10['mrr']:>7} "
          f"{REFERENCIA_EXP10['ndcg@10']:>8}  (referencia, ja medido)")
    for tecnica, top_k, exp_nome in COMBINACOES:
        if exp_nome in resumos:
            r = resumos[exp_nome]
            print(f"  {exp_nome:<24} {r['hit@5']:>7} {r['recall@5']:>9} {r['mrr']:>7} {r['ndcg@10']:>8}")
        elif exp_nome in ja_feitos:
            print(f"  {exp_nome:<24} (ja estava em resultados.csv de uma rodada anterior)")
        else:
            print(f"  {exp_nome:<24} FALHOU nesta rodada - nao registrado")
    print(f"\nLinhas registradas em {av.RESULTADOS}")
    print("\nNOTA: nenhum reindexacao foi feita - o indice segue na mesma base da Fase 4 "
          "para a Fase 6 em diante.")

    if falhas_combo:
        print(f"\nATENCAO: {len(falhas_combo)} combinacao(oes) falharam e NAO foram registradas:")
        for exp_nome, erro in falhas_combo:
            print(f"  - {exp_nome}: {erro[:200]}")
        print("Corrija a causa e rode o script de novo - o que ja passou sera pulado.")
        sys.exit(1)


if __name__ == "__main__":
    main()
