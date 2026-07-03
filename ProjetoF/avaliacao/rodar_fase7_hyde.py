"""
rodar_fase7_hyde.py - Fase 7 do Roteiro_Final.md (Secao 7): "casos dificeis pedem
estrategias especificas" - tecnica avancada escolhida: HyDE (Hypothetical Document
Embeddings, Aula 6).

Testa a tecnica 'hyde' (LLM gera um documento hipotetico - um trecho que PARECE a
resposta, no estilo do corpus - e a busca densa e feita pelo embedding desse trecho
hipotetico, em vez da pergunta original) contra a busca densa pura (exp10, mesma
base), no top_k=10 (ponto de equilibrio confirmado na Fase 4).

NAO reindexa nada: a extracao/chunking/embedding continuam os da Fase 4
(Docling + hierarquico + qwen3-embedding:4b) - a UNICA variavel aqui e a tecnica de
busca. IMPORTANTE (mesma licao das Fases 5/6): o indice no OpenSearch continua na
dimensao da Fase 4 (2560, qwen3-embedding:4b), mas a variavel de ambiente
EMBEDDING_MODEL so vale DENTRO do processo Python que a setou - por isso este script
mede a dimensao e reseta EMBEDDING_MODEL de novo no INICIO (preparar_ambiente_embedding),
e a Fase 8 (que tambem herda esta base) precisa repetir esse mesmo cuidado.

Diferenca em relacao a Fase 6 (reranking): HyDE faz 1 chamada de LLM (Groq) por
pergunta para gerar o documento hipotetico - intrinseco a tecnica (ver
avaliar_recuperacao.py::construir_retrieval_only). Sao so 30 chamadas curtas
(max_tokens=300, sem geracao de resposta final), bem abaixo do limite diario do Groq.

Experimentos gerados (fase = "Fase 7"):
  exp22_hyde

RESUMIVEL (pula o que ja esta em resultados.csv) e com try/except, como as fases
anteriores.

Uso (de dentro de ProjetoF/, venv ativado, precisa de OpenSearch + Ollama + Groq no ar):

    python avaliacao/rodar_fase7_hyde.py
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


COMBINACOES = [
    ("hyde", TOP_K, "exp22_hyde"),
]

REFERENCIA_EXP10 = {"hit@5": 0.9333, "recall@5": 0.7, "mrr": 0.6608, "ndcg@10": 0.7754}


def exps_ja_registrados(caminho_csv):
    if not Path(caminho_csv).exists():
        return set()
    with open(caminho_csv, newline="", encoding="utf-8") as f:
        return {row["exp"] for row in csv.DictReader(f)}


def main():
    print("== Fase 7: tecnica avancada - HyDE (Hypothetical Document Embeddings) ==")
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

        linha = {"exp": exp_nome, "fase": "Fase 7",
                 "mudanca": f"tecnica=hyde, top_k={top_k}, base={BASE_DESCRICAO}",
                 "tecnica": tecnica, "top_k": top_k,
                 "observacao": "documento hipotetico via Groq, embedado com qwen3-embedding:4b "
                               "(sem geracao de resposta final)",
                 **resumo}
        av.salvar_csv(linha)
        caminho_detalhe = av.salvar_detalhes(exp_nome, detalhes, falhas)
        resumos[exp_nome] = resumo
        print(f"  hit@5={resumo['hit@5']} recall@5={resumo['recall@5']} "
              f"mrr={resumo['mrr']} ndcg@10={resumo['ndcg@10']} "
              f"({resumo['n_queries']} ok, {resumo['n_falhas']} falha(s))")
        print(f"  detalhe: {caminho_detalhe}\n")

    print("\n== resumo Fase 7 (base: Docling + hierarquico + qwen3-embedding:4b) ==")
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
    print("\nNOTA: nenhuma reindexacao foi feita - o indice segue na mesma base da Fase 4 "
          "para a Fase 8 em diante.")

    if falhas_combo:
        print(f"\nATENCAO: {len(falhas_combo)} combinacao(oes) falharam e NAO foram registradas:")
        for exp_nome, erro in falhas_combo:
            print(f"  - {exp_nome}: {erro[:200]}")
        print("Corrija a causa e rode o script de novo - o que ja passou sera pulado.")
        sys.exit(1)


if __name__ == "__main__":
    main()
