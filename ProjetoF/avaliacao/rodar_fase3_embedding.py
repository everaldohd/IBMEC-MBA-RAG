"""
rodar_fase3_embedding.py - Fase 3 do Roteiro_Final.md (Secao 7): "o modelo de embedding
define o teto da busca densa". Compara os modelos de embedding disponiveis no Ollama
LOCAL do usuario contra o baseline (nomic-embed-text, ja medido no exp01), mantendo
extracao (Docling) e chunking (hierarquico) FIXOS - embedding e a UNICA variavel.

DESCOBERTA AUTOMATICA dos modelos: em vez de supor os nomes exatos das tags no Ollama
(ex.: 'qwen3-embedding:0.6b' vs 'qwen3-embedding-0.6b' etc.), o script consulta
GET {OLLAMA_BASE_URL}/api/tags e busca por FAMILIAS conhecidas (substring, case-
insensitive): 'bge-m3', 'mxbai-embed-large', 'qwen3-embedding'. Cada tag encontrada
(inclusive as 3 variantes de tamanho do Qwen3, se estiverem instaladas) vira um
experimento. 'nomic-embed-text' NAO entra aqui: e o baseline (exp01), ja medido.

DIMENSAO DO EMBEDDING: app/config.py::DIMENSAO_EMBEDDING so tem 3 modelos mapeados
(nomic-embed-text/mxbai-embed-large/bge-m3) e cai em 768 por padrao pra qualquer outro
- errado para os Qwen3 (que tem dimensao MAIOR e variavel conforme o tamanho do
modelo/Matryoshka). Por isso este script MEDE a dimensao real (embeda uma string de
teste e olha o tamanho do vetor) e registra em config.DIMENSAO_EMBEDDING ANTES de
indexar - senao o OpenSearch cria o indice com a dimensao errada e trava.

Cada modelo testado vira uma linha em resultados.csv (fase = "Fase 3"):
  exp07_embed_<modelo> em diante (numeracao sequencial, nomes sanitizados)

Ao final, reindexado de volta com nomic-embed-text (estado do baseline/exp01) para a
Fase 4 nao herdar um embedding diferente do combinado.

RESUMIVEL, como o script da Fase 2: pula experimentos ja em resultados.csv; erro numa
tag nao derruba as outras (ex.: OpenSearch fora do ar, modelo nao respondeu).

Uso (de dentro de ProjetoF/, venv ativado, precisa de OpenSearch + Ollama no ar, com
os modelos ja baixados via 'ollama pull'):

    python avaliacao/rodar_fase3_embedding.py
"""

import csv
import os
import re
import sys
import time
from pathlib import Path

import requests

PASTA_AVAL = Path(__file__).resolve().parent
PASTA_PROJETO = PASTA_AVAL.parent
sys.path.insert(0, str(PASTA_PROJETO))

PDF = PASTA_PROJETO / "CadeiaCustodia_ProvaPericial_lei13.964-2019.pdf"
CHUNKING_FIXO = "hierarquico"          # igual ao baseline - so o embedding varia
MODELO_BASELINE = "nomic-embed-text"   # ja medido no exp01 - restaurado ao final
FAMILIAS_A_TESTAR = ["bge-m3", "mxbai-embed-large", "qwen3-embedding"]


def _slug(tag):
    return re.sub(r"[^a-z0-9]+", "_", tag.lower()).strip("_")


def descobrir_modelos():
    """Consulta o Ollama local e devolve as tags instaladas que batem com as familias."""
    from app import config
    base_url, _ = config.config_ollama()
    r = requests.get(f"{base_url}/api/tags", timeout=10)
    r.raise_for_status()
    instalados = [m["name"] for m in r.json().get("models", [])]
    achados = []
    for familia in FAMILIAS_A_TESTAR:
        for tag in instalados:
            if familia.lower() in tag.lower():
                achados.append(tag)
    return sorted(set(achados))


def medir_dimensao(tag):
    """Embeda uma string curta e devolve o tamanho real do vetor (nao supor pela tabela)."""
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


def reindexar(conteudo, tecnica_chunk):
    from app import indexacao
    docs = indexacao.chunkar(conteudo, tecnica_chunk)
    n = indexacao.indexar_opensearch(docs, meta={"arquivo": PDF.name})
    print(f"  reindexar: {n} chunk(s) ({tecnica_chunk})")
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

    print("== Fase 3: modelo de embedding ==")
    print("   (nomic-embed-text ja medido no exp01 - nao reindexado aqui)\n")

    print("Descobrindo modelos de embedding instalados no Ollama...")
    modelos = descobrir_modelos()
    if not modelos:
        print("  Nenhum modelo das familias", FAMILIAS_A_TESTAR, "encontrado via "
              "'ollama list'/'/api/tags'. Confira 'ollama pull' dos modelos e rode de novo.")
        sys.exit(1)
    print(f"  Encontrados: {modelos}\n")

    print("Extraindo com Docling (fixo, igual ao baseline)...")
    t0 = time.time()
    conteudo = extrair_docling()
    print(f"  {len(conteudo)} caracteres em {time.time() - t0:.1f}s\n")

    import avaliar_recuperacao as av

    ja_feitos = exps_ja_registrados(av.RESULTADOS)
    n_exp = 7  # exp01..exp06 ja usados nas Fases 0-2
    resumos = {}
    falhas_modelo = []

    for tag in modelos:
        exp_nome = f"exp{n_exp:02d}_embed_{_slug(tag)}"
        n_exp += 1
        if exp_nome in ja_feitos:
            print(f"--- {tag} ({exp_nome}) --- JA REGISTRADO, pulando.\n")
            continue
        print(f"--- {tag} ({exp_nome}) ---")
        try:
            print("  Medindo dimensao real do embedding...")
            dim = medir_dimensao(tag)
            familia_key = tag.split(":")[0].lower()
            config.DIMENSAO_EMBEDDING[familia_key] = dim
            print(f"  dimensao={dim} (registrada em config.DIMENSAO_EMBEDDING['{familia_key}'])")

            os.environ["EMBEDDING_MODEL"] = tag
            print("  Limpando indice...")
            limpar_indice()
            print("  Reindexando...")
            n_chunks = reindexar(conteudo, CHUNKING_FIXO)

            print("  Rodando avaliacao de retrieval (tecnica=baseline, top_k=10)...")
            resumo, detalhes, falhas = av.rodar("baseline", 10)
        except Exception as e:
            print(f"  FALHOU: {e}")
            print("  (rode de novo depois de corrigir - os modelos ja OK nao serao repetidos)\n")
            falhas_modelo.append((tag, str(e)))
            continue

        linha = {"exp": exp_nome, "fase": "Fase 3",
                 "mudanca": f"embedding={tag} (vs nomic-embed-text no exp01, dim={dim}), "
                            f"chunking={CHUNKING_FIXO} e extracao=Docling fixos",
                 "tecnica": "baseline", "top_k": 10,
                 "observacao": f"{n_chunks} chunks, dimensao={dim}",
                 **resumo}
        av.salvar_csv(linha)
        caminho_detalhe = av.salvar_detalhes(exp_nome, detalhes, falhas)
        resumos[exp_nome] = resumo
        print(f"  hit@5={resumo['hit@5']} recall@5={resumo['recall@5']} "
              f"mrr={resumo['mrr']} ndcg@10={resumo['ndcg@10']} "
              f"({resumo['n_queries']} ok, {resumo['n_falhas']} falha(s))")
        print(f"  detalhe: {caminho_detalhe}\n")

    restauracao_ok = True
    print("== restaurando o indice para nomic-embed-text (estado do baseline/exp01) ==")
    try:
        os.environ["EMBEDDING_MODEL"] = MODELO_BASELINE
        limpar_indice()
        reindexar(conteudo, CHUNKING_FIXO)
    except Exception as e:
        restauracao_ok = False
        print(f"  FALHOU: {e}")
        print("  O indice NAO foi restaurado para nomic-embed-text - confira OpenSearch/Ollama "
              "e rode o script de novo (so repete o que faltou).")

    print("\n== resumo Fase 3 (linhas novas desta rodada + o que ja estava em resultados.csv) ==")
    print(f"  {'modelo':<28} {'hit@5':>7} {'recall@5':>9} {'mrr':>7} {'ndcg@10':>8}")
    print(f"  {'nomic-embed-text(exp01)':<28} {0.6333:>7} {0.4222:>9} {0.3512:>7} {0.3381:>8}  (referencia, ja medido)")
    for exp_nome, r in resumos.items():
        print(f"  {exp_nome:<28} {r['hit@5']:>7} {r['recall@5']:>9} {r['mrr']:>7} {r['ndcg@10']:>8}")
    print(f"\nLinhas registradas em {av.RESULTADOS}")
    if restauracao_ok:
        print("OK: OpenSearch de volta ao nomic-embed-text/hierarquico (baseline) para as proximas fases.")

    if falhas_modelo or not restauracao_ok:
        if falhas_modelo:
            print(f"\nATENCAO: {len(falhas_modelo)} modelo(s) falharam e NAO foram registrados:")
            for tag, erro in falhas_modelo:
                print(f"  - {tag}: {erro[:200]}")
        if not restauracao_ok:
            print("\nATENCAO: a restauracao final tambem falhou - o indice pode nao estar no "
                  "estado da baseline agora.")
        print("Corrija a causa e rode o script de novo - o que ja passou sera pulado.")
        sys.exit(1)


if __name__ == "__main__":
    main()
