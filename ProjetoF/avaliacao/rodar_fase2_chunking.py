"""
rodar_fase2_chunking.py - Fase 2 do Roteiro_Final.md (Secao 7): "a granularidade do
chunk muda o que e recuperavel". Compara as 5 tecnicas de chunking do projeto
(app/indexacao.py::chunkar) sobre o MESMO texto extraido (Docling, igual ao baseline
exp01) - a extracao fica FIXA, chunking e a UNICA variavel.

Tecnica 'hierarquico' NAO e reindexada aqui: e exatamente a configuracao do exp01
(baseline), entao seus numeros ja existem em resultados.csv - o script so referencia
exp01 na tabela final em vez de duplicar a rodada.

Cada tecnica testada vira uma linha em resultados.csv (fase = "Fase 2"):
  exp03_chunk_fixo, exp04_chunk_recursivo, exp05_chunk_sentenca_janela,
  exp06_chunk_semantico

Ao final, o indice e restaurado para 'hierarquico' (o estado do baseline/exp01),
para a Fase 3 nao herdar um chunking diferente do combinado.

RESUMIVEL: antes de rodar cada tecnica, o script confere se o exp ja esta em
resultados.csv e PULA se ja estiver (assim, se um chunking falhar no meio do lote -
ex.: dependencia do Haystack faltando, como o 'nltk' que 'recursivo'/'sentenca_janela'
usam por baixo dos panos - da pra rodar de novo sem duplicar o que ja passou).

Uso (de dentro de ProjetoF/, venv ativado, precisa de OpenSearch + Ollama no ar):

    python avaliacao/rodar_fase2_chunking.py
"""

import csv
import sys
import time
from pathlib import Path

import requests

PASTA_AVAL = Path(__file__).resolve().parent
PASTA_PROJETO = PASTA_AVAL.parent
sys.path.insert(0, str(PASTA_PROJETO))

PDF = PASTA_PROJETO / "CadeiaCustodia_ProvaPericial_lei13.964-2019.pdf"

# tecnica -> nome do experimento (fixo, na ordem em que a Secao 7 do roteiro lista)
TECNICAS = {
    "fixo": "exp03_chunk_fixo",
    "recursivo": "exp04_chunk_recursivo",
    "sentenca_janela": "exp05_chunk_sentenca_janela",
    "semantico": "exp06_chunk_semantico",
}
TECNICA_BASELINE = "hierarquico"   # ja medida no exp01 - restaurada ao final


def limpar_indice():
    from app import config
    cfg = config.config_opensearch()
    auth = (cfg["usuario"], cfg["senha"]) if cfg["usuario"] else None
    r = requests.delete(f"{cfg['url']}/{cfg['indice']}", auth=auth, verify=False)
    print(f"  limpar_indice: DELETE {cfg['url']}/{cfg['indice']} -> {r.status_code}")


def reindexar(conteudo, tecnica):
    from app import indexacao
    docs = indexacao.chunkar(conteudo, tecnica)
    n = indexacao.indexar_opensearch(docs, meta={"arquivo": PDF.name, "chunking": tecnica})
    print(f"  reindexar[{tecnica}]: {n} chunk(s)")
    return n


def extrair_docling():
    from app.extracao import _impl_texto
    conteudo, _ = _impl_texto(str(PDF))
    return conteudo


def exps_ja_registrados(caminho_csv):
    """Le resultados.csv (se existir) e devolve o conjunto de 'exp' ja gravados."""
    if not Path(caminho_csv).exists():
        return set()
    with open(caminho_csv, newline="", encoding="utf-8") as f:
        return {row["exp"] for row in csv.DictReader(f)}


def main():
    print("== Fase 2: chunking (fixo / recursivo / sentenca_janela / semantico) ==")
    print("   (hierarquico ja medido no exp01 - nao reindexado aqui)\n")

    print("Extraindo com Docling (igual ao baseline, extracao fica FIXA nesta fase)...")
    t0 = time.time()
    conteudo = extrair_docling()
    print(f"  {len(conteudo)} caracteres em {time.time() - t0:.1f}s\n")

    import avaliar_recuperacao as av

    ja_feitos = exps_ja_registrados(av.RESULTADOS)
    resumos = {}
    falhas_tecnica = []
    for tecnica, exp_nome in TECNICAS.items():
        if exp_nome in ja_feitos:
            print(f"--- {tecnica} ({exp_nome}) --- JA REGISTRADO, pulando.\n")
            continue
        print(f"--- {tecnica} ({exp_nome}) ---")
        try:
            print("  Limpando indice...")
            limpar_indice()
            print("  Reindexando...")
            n_chunks = reindexar(conteudo, tecnica)

            print("  Rodando avaliacao de retrieval (tecnica=baseline, top_k=10)...")
            resumo, detalhes, falhas = av.rodar("baseline", 10)
        except Exception as e:
            print(f"  FALHOU: {e}")
            print("  (rode de novo depois de corrigir - as tecnicas ja OK nao serao repetidas)\n")
            falhas_tecnica.append((tecnica, str(e)))
            continue

        linha = {"exp": exp_nome, "fase": "Fase 2",
                 "mudanca": f"chunking={tecnica} (vs hierarquico no exp01), extracao=Docling fixa",
                 "tecnica": "baseline", "top_k": 10,
                 "observacao": f"{n_chunks} chunks gerados",
                 **resumo}
        av.salvar_csv(linha)
        caminho_detalhe = av.salvar_detalhes(exp_nome, detalhes, falhas)
        resumos[exp_nome] = resumo
        print(f"  hit@5={resumo['hit@5']} recall@5={resumo['recall@5']} "
              f"mrr={resumo['mrr']} ndcg@10={resumo['ndcg@10']} "
              f"({resumo['n_queries']} ok, {resumo['n_falhas']} falha(s))")
        print(f"  detalhe: {caminho_detalhe}\n")

    restauracao_ok = True
    print("== restaurando o indice para hierarquico (estado do baseline/exp01) ==")
    try:
        limpar_indice()
        reindexar(conteudo, TECNICA_BASELINE)
    except Exception as e:
        restauracao_ok = False
        print(f"  FALHOU: {e}")
        print("  O indice NAO foi restaurado para hierarquico - provavelmente o "
              "OpenSearch esta fora do ar (confira 'docker compose ps'/'http://localhost:9200/'). "
              "Rode o script de novo depois de subir a infra: ele so vai repetir o que "
              "ainda nao passou (incluindo esta restauracao).")

    print("\n== resumo Fase 2 (linhas novas desta rodada + o que ja estava em resultados.csv) ==")
    print(f"  {'tecnica':<18} {'hit@5':>7} {'recall@5':>9} {'mrr':>7} {'ndcg@10':>8}")
    print(f"  {'hierarquico(exp01)':<18} {0.6333:>7} {0.4222:>9} {0.3512:>7} {0.3381:>8}  (referencia, ja medido)")
    for tecnica, exp_nome in TECNICAS.items():
        if exp_nome in resumos:
            r = resumos[exp_nome]
            print(f"  {tecnica:<18} {r['hit@5']:>7} {r['recall@5']:>9} {r['mrr']:>7} {r['ndcg@10']:>8}")
        elif exp_nome in ja_feitos:
            print(f"  {tecnica:<18} (ja estava em resultados.csv de uma rodada anterior)")
        else:
            print(f"  {tecnica:<18} FALHOU nesta rodada - nao registrado")
    print(f"\nLinhas registradas em {av.RESULTADOS}")
    if restauracao_ok:
        print("OK: OpenSearch de volta ao chunking hierarquico (baseline) para as proximas fases.")

    if falhas_tecnica or not restauracao_ok:
        if falhas_tecnica:
            print(f"\nATENCAO: {len(falhas_tecnica)} tecnica(s) falharam e NAO foram registradas:")
            for tecnica, erro in falhas_tecnica:
                print(f"  - {tecnica}: {erro[:200]}")
        if not restauracao_ok:
            print("\nATENCAO: a restauracao final para hierarquico tambem falhou (ver acima) - "
                  "o indice NAO esta garantidamente no estado da baseline agora.")
        print("Corrija a causa (provavelmente subir/checar OpenSearch e Ollama) e rode o "
              "script de novo - o que ja passou sera pulado.")
        sys.exit(1)


if __name__ == "__main__":
    main()
