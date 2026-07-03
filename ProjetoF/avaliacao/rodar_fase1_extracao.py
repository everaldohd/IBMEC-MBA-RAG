"""
rodar_fase1_extracao.py - Fase 1 do Roteiro_Final.md (Secao 7): "texto mal extraido
limita tudo a jusante". Compara a extracao atual (Docling markdown, sem OCR - a que
ja esta indexada como baseline/exp01) contra o fallback PyMuPDF (texto puro, sem
estrutura), isolando a extracao como UNICA variavel: mesmo chunking (hierarquico,
igual ao baseline), mesmo embedding, mesma tecnica de busca (baseline), mesmo top_k.

OCR nao entra nessa fase: o PDF escolhido (CadeiaCustodia_ProvaPericial) tem camada
de texto nativa (nao e escaneado), entao 'extrair_com_ocr' nunca seria escolhido pela
extracao.probe()/decidir_e_extrair() - comparar OCR aqui nao faria sentido p/ essa fonte.

O que o script faz, em ordem:
  1) extrai o PDF das 2 formas (Docling markdown e PyMuPDF) e salva amostras em
     avaliacao/amostras_extracao/ p/ inspecao manual (pedido pelo roteiro).
  2) LIMPA o indice do OpenSearch (index e recriado no proximo write).
  3) reindexa com o texto do PyMuPDF, chunking FORCADO = hierarquico (isola a variavel).
  4) roda a mesma avaliacao de retrieval do exp01 (reaproveita avaliar_recuperacao.rodar)
     e grava exp02_extracao_pymupdf em resultados.csv (fase = "Fase 1").
  5) REINDEXA DE VOLTA com Docling (o texto original) para deixar o OpenSearch no
     estado da baseline outra vez - importante pras proximas fases nao herdarem o
     indice "errado".

Uso (de dentro de ProjetoF/, venv ativado, precisa de OpenSearch + Ollama no ar):

    python avaliacao/rodar_fase1_extracao.py
"""

import sys
import time
from pathlib import Path

import requests

PASTA_AVAL = Path(__file__).resolve().parent
PASTA_PROJETO = PASTA_AVAL.parent
sys.path.insert(0, str(PASTA_PROJETO))

PDF = PASTA_PROJETO / "CadeiaCustodia_ProvaPericial_lei13.964-2019.pdf"
PASTA_AMOSTRAS = PASTA_AVAL / "amostras_extracao"
CHUNKING_FIXO = "hierarquico"   # igual ao baseline (exp01) - unica variavel = extracao


def limpar_indice():
    """Apaga o indice do OpenSearch (recriado automaticamente no proximo write)."""
    from app import config
    cfg = config.config_opensearch()
    auth = (cfg["usuario"], cfg["senha"]) if cfg["usuario"] else None
    r = requests.delete(f"{cfg['url']}/{cfg['indice']}", auth=auth, verify=False)
    print(f"  limpar_indice: DELETE {cfg['url']}/{cfg['indice']} -> {r.status_code}")


def reindexar(conteudo, tag):
    """Chunkar (tecnica fixa) + indexar no OpenSearch. Retorna n_chunks."""
    from app import indexacao
    docs = indexacao.chunkar(conteudo, CHUNKING_FIXO)
    n = indexacao.indexar_opensearch(docs, meta={"arquivo": PDF.name, "extracao": tag})
    print(f"  reindexar[{tag}]: {n} chunk(s) ({CHUNKING_FIXO})")
    return n


def extrair_docling():
    from app.extracao import _impl_texto
    conteudo, _ = _impl_texto(str(PDF))
    return conteudo


def extrair_pymupdf():
    from app.extracao import _pymupdf_texto
    return _pymupdf_texto(str(PDF))


def salvar_amostra(nome, texto):
    PASTA_AMOSTRAS.mkdir(exist_ok=True)
    caminho = PASTA_AMOSTRAS / nome
    caminho.write_text(texto, encoding="utf-8")
    return caminho


def main():
    print("== Fase 1: extracao (Docling vs PyMuPDF) ==\n")

    print("[1/5] Extraindo com Docling (markdown, sem OCR - igual ao baseline)...")
    t0 = time.time()
    conteudo_docling = extrair_docling()
    print(f"  {len(conteudo_docling)} caracteres em {time.time() - t0:.1f}s")

    print("[2/5] Extraindo com PyMuPDF (texto puro)...")
    t0 = time.time()
    conteudo_pymupdf = extrair_pymupdf()
    print(f"  {len(conteudo_pymupdf)} caracteres em {time.time() - t0:.1f}s")

    c_docling = salvar_amostra("docling.txt", conteudo_docling)
    c_pymupdf = salvar_amostra("pymupdf.txt", conteudo_pymupdf)
    print(f"  amostras salvas em {c_docling} e {c_pymupdf} (inspecao manual)")

    n_titulos_docling = len([l for l in conteudo_docling.splitlines() if l.lstrip().startswith("#")])
    n_titulos_pymupdf = len([l for l in conteudo_pymupdf.splitlines() if l.lstrip().startswith("#")])
    print(f"  sinal de estrutura: {n_titulos_docling} titulos markdown (Docling) "
          f"vs {n_titulos_pymupdf} (PyMuPDF - nao gera markdown, deve ser 0)")

    print("\n[3/5] Limpando indice do OpenSearch...")
    limpar_indice()

    print("\n[4/5] Reindexando com o texto do PyMuPDF (chunking forcado = "
          f"{CHUNKING_FIXO}, igual ao baseline)...")
    reindexar(conteudo_pymupdf, tag="pymupdf")

    print("\n[5/5] Rodando avaliacao de retrieval (tecnica=baseline, top_k=10)...")
    import avaliar_recuperacao as av
    resumo, detalhes, falhas = av.rodar("baseline", 10)
    linha = {"exp": "exp02_extracao_pymupdf", "fase": "Fase 1",
             "mudanca": f"extracao=PyMuPDF (vs Docling no exp01), chunking={CHUNKING_FIXO} fixo p/ isolar a variavel",
             "tecnica": "baseline", "top_k": 10,
             "observacao": f"{n_titulos_docling} titulos(Docling) vs {n_titulos_pymupdf}(PyMuPDF); "
                            f"{len(conteudo_docling)} vs {len(conteudo_pymupdf)} caracteres",
             **resumo}
    av.salvar_csv(linha)
    caminho_detalhe = av.salvar_detalhes("exp02_extracao_pymupdf", detalhes, falhas)
    print("\n== resultado (exp02_extracao_pymupdf) ==")
    for k in ("hit@5", "recall@5", "mrr", "ndcg@10", "latencia_media_s", "n_queries", "n_falhas"):
        print(f"  {k}: {resumo[k]}")
    print(f"Linha registrada em {av.RESULTADOS}")
    print(f"Detalhe salvo em {caminho_detalhe}")

    print("\n== restaurando o indice para o estado baseline (Docling) ==")
    limpar_indice()
    reindexar(conteudo_docling, tag="docling")
    print("\nOK: OpenSearch de volta ao estado da baseline (exp01) para as proximas fases.")


if __name__ == "__main__":
    main()
