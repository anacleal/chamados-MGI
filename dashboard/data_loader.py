"""
Carregamento e preparação de dados para o Dashboard de Análise de Chamados MGI
================================================================================
Centraliza a leitura de todos os artefatos gerados pelo pipeline:
  - model_bertopic.py     -> modelos BERTopic salvos, Topicos_Dominantes.csv
  - summarization.py      -> summary_topic_{N}.txt (Padrão Dominante / Impacto Operacional)
  - title.py              -> titulo_topic_{N}.txt

Estrutura de pastas esperada (raiz do projeto):
  data/                                  chamados_{sistema}.csv
  topic_modeling/bertopic_resultados/    {sistema}/Topicos_Dominantes.csv, topicos.json
  topic_modeling/bertopic_graphs/        {sistema}/intertopic_map.html (gerado nativamente pelo BERTopic)
  summarization/outLLM/detailed_summarization/{sistema}/  summary_topic_N.txt, titulo_topic_N.txt
  dashboard/                             <- este módulo roda daqui
"""

import os
import re
import json
import functools

import numpy as np
import pandas as pd

# ============================================================
# CAMINHOS BASE
# ============================================================
# dashboard/ é irmã de data/, topic_modeling/ e summarization/ na raiz do projeto
BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR        = os.path.join(BASE_DIR, "data")
TOPIC_RESULT_DIR = os.path.join(BASE_DIR, "topic_modeling", "bertopic_resultados")
TOPIC_GRAPH_DIR  = os.path.join(BASE_DIR, "topic_modeling", "bertopic_graphs")
SUMMARY_DIR      = os.path.join(BASE_DIR, "summarization", "outLLM", "detailed_summarization")

K_POR_SISTEMA = {
    "SIASS":  6,
    "SIAPE":  8,
    "SIGEPE": 5,
    "SOUGOV": 5,
    "TOTAIS": 10,
}

SISTEMAS = list(K_POR_SISTEMA.keys())

COL_DATA = "Data de abertura"   # coluna de data nos CSVs de chamados


# ============================================================
# RESUMOS E TÍTULOS
# ============================================================
def _read_txt(path: str) -> str | None:
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return f.read().strip()


def parse_summary(texto: str | None) -> dict:
    """Extrai os campos Padrão Dominante e Impacto Operacional do texto bruto."""
    if not texto:
        return {"padrao_dominante": "", "impacto_operacional": ""}

    padrao, impacto = "", ""
    m = re.search(r'\*\*Padrão Dominante\*\*:\s*(.*?)(?=\*\*Impacto Operacional\*\*|$)', texto, re.S)
    if m:
        padrao = m.group(1).strip()
    m = re.search(r'\*\*Impacto Operacional\*\*:\s*(.*)', texto, re.S)
    if m:
        impacto = m.group(1).strip()

    return {"padrao_dominante": padrao, "impacto_operacional": impacto}


def load_keywords(sistema: str, topic: int, top_n: int = 8) -> list[str]:
    json_path = os.path.join(TOPIC_RESULT_DIR, sistema, "topicos.json")
    if not os.path.exists(json_path):
        return []
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    entry = data.get(str(topic)) or data.get(topic)
    if not entry:
        return []
    return [word for word, _ in entry[:top_n] if word]


# ============================================================
# CONTAGEM DE DOCUMENTOS / PREVALÊNCIA
# ============================================================
@functools.lru_cache(maxsize=None)
def load_dominant_topics(sistema: str) -> pd.DataFrame | None:
    """
    Lê Resumo_Topicos_Dominantes.csv (id, dominant_topic).
    Usado para contagem de documentos por tópico e para cruzar com datas.
    """
    path = os.path.join(TOPIC_RESULT_DIR, sistema, "Resumo_Topicos_Dominantes.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    id_col = "id" if "id" in df.columns else "Id"
    df[id_col] = df[id_col].astype(str)
    df = df.rename(columns={id_col: "id"})
    return df


def count_docs_por_topico(sistema: str) -> dict[int, int]:
    df = load_dominant_topics(sistema)
    if df is None:
        return {}
    return df["dominant_topic"].value_counts().to_dict()


# ============================================================
# TABELA CONSOLIDADA DE TÓPICOS (para mapa, ranking, tabela)
# ============================================================
@functools.lru_cache(maxsize=None)
def build_topic_table(sistema: str) -> pd.DataFrame:
    """
    Monta uma linha por tópico do sistema, com:
      sistema, topico, titulo, padrao_dominante, impacto_operacional,
      keywords, n_documentos
    """
    k = K_POR_SISTEMA.get(sistema, 0)
    contagem = count_docs_por_topico(sistema)

    rows = []
    for t in range(k):
        titulo = _read_txt(os.path.join(SUMMARY_DIR, sistema, f"titulo_topic_{t}.txt")) or f"Tópico {t}"
        resumo_raw = _read_txt(os.path.join(SUMMARY_DIR, sistema, f"summary_topic_{t}.txt"))
        parsed = parse_summary(resumo_raw)
        keywords = load_keywords(sistema, t)

        rows.append({
            "sistema": sistema,
            "topico": t,
            "titulo": titulo,
            "padrao_dominante": parsed["padrao_dominante"],
            "impacto_operacional": parsed["impacto_operacional"],
            "keywords": ", ".join(keywords),
            "n_documentos": int(contagem.get(t, 0)),
        })

    return pd.DataFrame(rows)


def build_all_systems_table() -> pd.DataFrame:
    """Concatena a tabela de tópicos de todos os sistemas."""
    frames = [build_topic_table(s) for s in SISTEMAS]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame(columns=[
            "sistema", "topico", "titulo", "padrao_dominante",
            "impacto_operacional", "keywords", "n_documentos"
        ])
    return pd.concat(frames, ignore_index=True)


# ============================================================
# COORDENADAS 2D DO MAPA INTERTÓPICOS
# (lidas de topic_coordinates_2d.csv, salvo por model_bertopic.py a partir
#  dos embeddings SEMÂNTICOS reais dos tópicos — topic_embeddings_ — e não
#  do c-TF-IDF léxico usado pelo visualize_topics() nativo do BERTopic.
#  Isso mantém a mesma noção de similaridade usada para formar os clusters
#  via KMeans, em vez de uma projeção léxica que tende a ficar "genérica"
#  com vocabulário de domínio restrito.)
# ============================================================
def get_intertopic_map_path(sistema: str) -> str | None:
    """
    Retorna o caminho absoluto do intertopic_map.html nativo do BERTopic
    para o sistema, se existir. Mantido apenas como referência/fallback
    visual secundário — o mapa principal do dashboard usa
    topic_coordinates_2d.csv (ver load_topic_coordinates).
    """
    path = os.path.join(TOPIC_GRAPH_DIR, sistema, "intertopic_map.html")
    return path if os.path.exists(path) else None


@functools.lru_cache(maxsize=None)
def load_topic_coordinates(sistema: str) -> pd.DataFrame | None:
    """
    Lê as coordenadas 2D dos tópicos salvas por
    BERTopic.save_topic_coordinates_2d() em model_bertopic.py.

    Essas coordenadas vêm de um UMAP 2D dedicado aplicado sobre
    topic_embeddings_ (embeddings semânticos reais dos tópicos), e não
    do c-TF-IDF léxico que o visualize_topics() nativo usa — por isso a
    distância entre os pontos aqui reflete a mesma similaridade usada
    para formar os clusters via KMeans no pipeline de modelagem.

    Retorna DataFrame: topico, x, y, n_documentos (prevalência)
    """
    path = os.path.join(TOPIC_RESULT_DIR, sistema, "topic_coordinates_2d.csv")
    if not os.path.exists(path):
        return None

    df = pd.read_csv(path)
    required = {"topico", "x", "y"}
    if not required.issubset(df.columns):
        print(f"[AVISO] topic_coordinates_2d.csv de {sistema} sem colunas esperadas {required}.")
        return None

    df = df[df["topico"] >= 0].reset_index(drop=True)

    # Recalcula a contagem de documentos a partir da fonte mais atual
    # (Resumo_Topicos_Dominantes.csv), em vez de confiar apenas no que foi
    # salvo no momento do treino — garante consistência com o resto do dashboard.
    contagem = count_docs_por_topico(sistema)
    if contagem:
        df["n_documentos"] = df["topico"].map(contagem).fillna(df.get("n_documentos", 0)).astype(int)
    elif "n_documentos" not in df.columns:
        df["n_documentos"] = 0

    return df


def get_topic_info_path(sistema: str) -> str | None:
    """Retorna o caminho do topic_info.csv salvo junto com os gráficos do BERTopic."""
    path = os.path.join(TOPIC_GRAPH_DIR, sistema, "topic_info.csv")
    return path if os.path.exists(path) else None


# ============================================================
# EVOLUÇÃO TEMPORAL
# ============================================================
@functools.lru_cache(maxsize=None)
def load_chamados_com_topico(sistema: str) -> pd.DataFrame | None:
    """
    Cruza o CSV de chamados (com a coluna de data) com o tópico dominante
    de cada chamado. Retorna DataFrame: id, data, topico.
    """
    csv_path = os.path.join(DATA_DIR, f"chamados_{sistema.lower()}.csv")
    if not os.path.exists(csv_path):
        return None

    df = pd.read_csv(csv_path, low_memory=False)
    id_col = "Id" if "Id" in df.columns else "id"
    df[id_col] = df[id_col].astype(str)

    if COL_DATA not in df.columns:
        return None

    df["data"] = pd.to_datetime(df[COL_DATA], errors="coerce")
    df = df.dropna(subset=["data"])

    dominante = load_dominant_topics(sistema)
    if dominante is None:
        return None

    merged = df[[id_col, "data"]].rename(columns={id_col: "id"}).merge(
        dominante[["id", "dominant_topic"]], on="id", how="inner"
    )
    merged = merged.rename(columns={"dominant_topic": "topico"})
    return merged


def build_monthly_evolution(sistema: str, topicos: list[int] | None = None) -> pd.DataFrame:
    """
    Agrega chamados por mês/ano e tópico.
    Se topicos for None, agrega todos os tópicos.
    Retorna DataFrame: mes_ano (Timestamp), topico, titulo, n_chamados
    """
    df = load_chamados_com_topico(sistema)
    if df is None or df.empty:
        return pd.DataFrame(columns=["mes_ano", "topico", "titulo", "n_chamados"])

    if topicos is not None:
        df = df[df["topico"].isin(topicos)]

    df["mes_ano"] = df["data"].dt.to_period("M").dt.to_timestamp()
    agg = df.groupby(["mes_ano", "topico"]).size().reset_index(name="n_chamados")

    tabela_topicos = build_topic_table(sistema)[["topico", "titulo"]]
    agg = agg.merge(tabela_topicos, on="topico", how="left")
    agg["titulo"] = agg["titulo"].fillna(agg["topico"].apply(lambda t: f"Tópico {t}"))

    return agg.sort_values("mes_ano")