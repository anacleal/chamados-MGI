from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import os
from dashboard import data_loader as dl

BASE_DIR = Path(__file__).resolve().parents[1]
RECOMENDADOR_DATA_DIR = BASE_DIR / "data" / "recomendador"
CHAMADOS_CSV = RECOMENDADOR_DATA_DIR / "df_chamados.csv"
TOPICOS_CSV = RECOMENDADOR_DATA_DIR / "df_topicos.csv"
DATA_DIR = Path(dl.DATA_DIR)
K_POR_SISTEMA = dl.K_POR_SISTEMA


COL_ID = "Id"
COL_TITULO = "Titulo"
COL_TITULO_ORIGINAL = "Título"
COL_DESCRICAO = "Descrição do chamado"
COL_SOLUCAO = "Última ação de acompanhamento"
COL_TIME = "Time"
COL_TEMPO_H = "Tempo_Total_Atendimento_H"
COL_TEMPO_ALT = "Tempo_Total_Atendimento"


def _normalizar_sistema(sistema: str) -> str:
    return sistema.upper().replace(" ", "")


def _texto_limpo(valor: object) -> str:
    if pd.isna(valor):
        return ""
    return " ".join(str(valor).split())


def _criar_texto_completo(row: pd.Series) -> str:
    titulo = _texto_limpo(row.get(COL_TITULO_ORIGINAL, ""))
    descricao = _texto_limpo(row.get(COL_DESCRICAO, ""))
    partes = []
    if titulo:
        partes.append(f"Título: {titulo}")
    if descricao:
        partes.append(f"Descrição: {descricao}")
    return " | ".join(partes)


def build_topicos() -> pd.DataFrame:
    frames = []
    for sistema in K_POR_SISTEMA:
        tabela = dl.build_topic_table(sistema).copy()
        if tabela.empty:
            continue

        tabela = tabela.rename(
            columns={
                "sistema": "sistema_origem",
                "topico": "topico_id_original",
                "titulo": "label_curto",
                "n_documentos": "n_documentos",
                "padrao_dominante": "padrao_dominante",
                "impacto_operacional": "impacto_operacional",
            }
        )
        tabela["topico_id_unico"] = (
            tabela["sistema_origem"].map(_normalizar_sistema)
            + "_"
            + tabela["topico_id_original"].astype(str)
        )
        tabela["texto_para_embedding"] = (
            tabela["label_curto"].fillna("")
            + ". "
            + tabela["padrao_dominante"].fillna("")
            + " "
            + tabela["impacto_operacional"].fillna("")
            + " "
            + tabela.get("keywords", "").fillna("")
        ).map(_texto_limpo)
        frames.append(tabela)

    if not frames:
        return pd.DataFrame(
            columns=[
                "topico_id_unico",
                "sistema_origem",
                "topico_id_original",
                "label_curto",
                "resumo_sumarizado",
                "n_documentos",
                "keywords",
                "padrao_dominante",
                "impacto_operacional",
                "texto_para_embedding",
            ]
        )

    df = pd.concat(frames, ignore_index=True)
    df["resumo_sumarizado"] = (
        "Padrão Dominante: "
        + df["padrao_dominante"].fillna("")
        + " Impacto Operacional: "
        + df["impacto_operacional"].fillna("")
    ).map(_texto_limpo)

    ordered = [
        "topico_id_unico",
        "sistema_origem",
        "topico_id_original",
        "label_curto",
        "resumo_sumarizado",
        "n_documentos",
        "keywords",
        "padrao_dominante",
        "impacto_operacional",
        "texto_para_embedding",
    ]
    return df[ordered]


def build_chamados() -> pd.DataFrame:
    map_frames = []
    for sistema in K_POR_SISTEMA:
        dom = dl.load_dominant_topics(sistema)
        if dom is not None and not dom.empty:
            df_map = dom[["id", "dominant_topic"]].copy()
            # BLINDAGEM DO ID: Força string, tira espaço, arranca o .0 decimal
            df_map["id"] = df_map["id"].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
            
            prefixo = _normalizar_sistema(sistema)
            df_map["topico_id_unico"] = df_map["dominant_topic"].apply(lambda t: f"{prefixo}_{t}")
            df_map["sistema_origem"] = sistema
            map_frames.append(df_map)
            
    if map_frames:
        df_mappings = pd.concat(map_frames, ignore_index=True)
        df_topicos_agrupados = df_mappings.groupby("id", as_index=False).agg({
            "topico_id_unico": lambda x: ",".join(x.dropna().astype(str).unique()),
            "sistema_origem": lambda x: ",".join(x.dropna().astype(str).unique())
        })
    else:
        df_topicos_agrupados = pd.DataFrame(columns=["id", "topico_id_unico", "sistema_origem"])

    NOME_ARQUIVO_RAW = "chatbot_table.csv"
    csv_raw_path = DATA_DIR / NOME_ARQUIVO_RAW
    
    if not csv_raw_path.exists():
        print(f"[ERRO] Base original não encontrada em: {csv_raw_path}")
        return pd.DataFrame()

    df_raw = pd.read_csv(csv_raw_path, low_memory=False)
    
    if COL_ID in df_raw.columns:
        # BLINDAGEM DO ID NO CSV: Garante que "123.0" bata exatamente com "123"
        df_raw[COL_ID] = df_raw[COL_ID].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
    
    df_raw = df_raw.drop_duplicates(subset=[COL_ID]).copy()

    df_final = df_raw.merge(df_topicos_agrupados, left_on=COL_ID, right_on="id", how="left")
    df_final = df_final.drop(columns=["id"], errors="ignore")
    
    df_final["sistema_origem"] = df_final["sistema_origem"].fillna("NAO_CLASSIFICADO")

    if COL_SOLUCAO not in df_final.columns:
        df_final[COL_SOLUCAO] = ""
    if COL_TEMPO_H not in df_final.columns and COL_TEMPO_ALT in df_final.columns:
        df_final[COL_TEMPO_H] = df_final[COL_TEMPO_ALT]

    df_final["texto_completo"] = df_final.apply(_criar_texto_completo, axis=1)
    df_final["link_solucao"] = df_final[COL_SOLUCAO].astype(str).str.extract(
        r"(https?://[^\s<>\"]+|www\.[^\s<>\"]+)", expand=False
    )

    preferidas = [
        COL_ID,
        "sistema_origem",      # <-- Agora essa coluna foi injetada pelo nosso mapa!
        "topico_id_unico",
        COL_TITULO_ORIGINAL,
        COL_DESCRICAO,
        COL_SOLUCAO,
        COL_TIME,
        "Categoria",
        "Status",
        "Prioridade",
        "Data de abertura",
        "Data de Solução",
        COL_TEMPO_H,
        "texto_completo",
        "link_solucao",
    ]
    cols = [c for c in preferidas if c in df_final.columns]
    extras = [c for c in df_final.columns if c not in cols]
    
    return df_final[cols + extras]


def build_csvs() -> tuple[Path, Path]:
    RECOMENDADOR_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df_topicos = build_topicos()
    df_chamados = build_chamados()

    df_topicos.to_csv(TOPICOS_CSV, index=False, encoding="utf-8")
    df_chamados.to_csv(CHAMADOS_CSV, index=False, encoding="utf-8")

    topicos_validos = set(df_topicos["topico_id_unico"])
    
    # Validador Inteligente: Lê cada linha, separa por vírgula e checa se ALGUM tópico está na lista
    def checar_validade(tags):
        if pd.isna(tags) or tags == "NAO_CLASSIFICADO":
            return False
        # Separa os tópicos (ex: "SIAPE_1,SIAPE_4") e vê se algum deles é válido
        lista_tags = str(tags).split(",")
        return any(t in topicos_validos for t in lista_tags)

    chamados_mapeados = df_chamados["topico_id_unico"].apply(checar_validade).sum()
    
    print(f"Tópicos salvos: {TOPICOS_CSV} ({len(df_topicos)} linhas)")
    print(f"Chamados salvos: {CHAMADOS_CSV} ({len(df_chamados)} linhas totais)")
    print(f"Chamados Mapeados/Úteis (com tópico válido): {chamados_mapeados}/{len(df_chamados)}")
    
    return TOPICOS_CSV, CHAMADOS_CSV


def main() -> None:
    parser = argparse.ArgumentParser(description="Gera os CSVs do recomendador.")
    parser.parse_args()
    build_csvs()


if __name__ == "__main__":
    main()