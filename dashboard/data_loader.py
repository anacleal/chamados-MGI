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
  topic_modeling/models/                 {sistema}/modelo  (modelo BERTopic salvo)
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
BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR        = os.path.join(BASE_DIR, "data")
TOPIC_RESULT_DIR = os.path.join(BASE_DIR, "topic_modeling", "bertopic_resultados")
TOPIC_MODEL_DIR  = os.path.join(BASE_DIR, "topic_modeling", "models")
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
# COORDENADAS 2D DO MAPA INTERTÓPICOS (via modelo BERTopic salvo)
# ============================================================
@functools.lru_cache(maxsize=None)
def load_topic_coordinates(sistema: str) -> pd.DataFrame | None:
    """
    Carrega o modelo BERTopic salvo e calcula as coordenadas 2D dos tópicos
    a partir do espaço reduzido pelo UMAP do próprio modelo (mesma lógica
    usada pelo visualize_topics() nativo do BERTopic).

    Retorna DataFrame: topico, x, y, n_documentos (prevalência)
    """
    model_path = os.path.join(TOPIC_MODEL_DIR, sistema, "modelo")
    if not os.path.exists(model_path):
        return None

    try:
        # Import local para não exigir bertopic/torch quando não necessário
        from bertopic import BERTopic
        topic_model = BERTopic.load(model_path)

        # topic_embeddings_ já é a representação centróide de cada tópico
        # no espaço de embeddings original (antes do UMAP). Reduzimos para 2D
        # com o próprio umap_model do modelo treinado, se disponível.
        topic_embeddings = getattr(topic_model, "topic_embeddings_", None)
        if topic_embeddings is None:
            return None

        umap_model = topic_model.umap_model
        # Reduz para 2 componentes reaproveitando o umap_model treinado
        # (transform funciona pois o modelo já foi ajustado no fit original)
        try:
            coords = umap_model.transform(topic_embeddings)
        except Exception:
            # Fallback: re-treina um UMAP 2D rápido só para visualização,
            # caso o umap_model salvo não seja transformável (ex: n_components != 2)
            from umap import UMAP
            coords = UMAP(n_neighbors=15, n_components=2, metric="cosine",
                           min_dist=0.0, random_state=42).fit_transform(topic_embeddings)

        topic_info = topic_model.get_topic_info()
        # Remove o tópico -1 (outliers), se existir, e mantém ordem de get_topic_info
        topic_ids = topic_info["Topic"].tolist()

        df = pd.DataFrame({
            "topico": topic_ids,
            "x": coords[:len(topic_ids), 0],
            "y": coords[:len(topic_ids), 1],
        })
        df = df[df["topico"] >= 0].reset_index(drop=True)

        contagem = count_docs_por_topico(sistema)
        df["n_documentos"] = df["topico"].map(contagem).fillna(0).astype(int)

        return df

    except Exception as e:
        print(f"[AVISO] Não foi possível calcular coordenadas 2D para {sistema}: {e}")
        return None


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