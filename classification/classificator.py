import os
import re
import pandas as pd
import ollama
from keybert import KeyBERT
from concurrent.futures import ThreadPoolExecutor, as_completed

# =============================================================================
# CLASSIFICADOR v5 — KeyBERT extrai palavras-chave; LLM só monta 4-5 palavras
# =============================================================================

MODEL = "llama3.1:8b-instruct-q4_K_S"

# --- CAMINHOS ---
BASE_DADOS_CSV    = "../data/base_de_dados.csv"
BERTOPIC_DIR      = "../topic_modeling/bertopic_resultados"
SUMMARIZATION_DIR = "../summarization/summarization/outLLM/postprocessed_summarization"
OUTPUT_CSV        = "../summarization/summarization/topic_labels.csv"

# --- PARÂMETROS ---
TOP_N_CATEGORIAS  = 5
MAX_WORKERS       = 4
SISTEMAS          = ["SIAPE", "SIASS", "SIGEPE", "SOUGOV", "TOTAIS"]
CONFIGS_TOPICOS   = [5, 10, 15]

# KeyBERT carregado uma única vez (modelo leve, roda em CPU)
print("[→] Carregando KeyBERT...")
kw_model = KeyBERT(model="paraphrase-multilingual-MiniLM-L12-v2")

# -----------------------------------------------------------------------------
# PROMPT — agora o LLM recebe só as keywords + módulo dominante
# A tarefa é trivial: compor um título fluente com o que já foi extraído
# -----------------------------------------------------------------------------
SYSTEM_PROMPT = """\
Você recebe palavras-chave de um grupo de chamados e o módulo do sistema.
Sua única tarefa: escrever UM título em português, 4 a 6 palavras, Title Case.
Responda SOMENTE o título. Sem explicação, sem ponto final, sem aspas.
"""

# -----------------------------------------------------------------------------
# 1. HELPERS
# -----------------------------------------------------------------------------

def get_col(df, name):
    for col in df.columns:
        if col.strip().lower() == name.lower():
            return col
    raise ValueError(f"Coluna '{name}' não encontrada. Disponíveis: {list(df.columns)}")


def carregar_base(csv_path):
    df = pd.read_csv(csv_path, low_memory=False, encoding="utf-8")
    col_id  = get_col(df, "id")
    col_cat = get_col(df, "categoria")
    df = df[[col_id, col_cat]].rename(columns={col_id: "Id", col_cat: "Categoria"})
    df["Id"] = df["Id"].astype(str)
    return df


def modulo_dominante(categorias_str: str) -> str:
    """Extrai o submódulo mais específico da primeira categoria."""
    primeira = categorias_str.split("|")[0].strip()
    # Pega o que está após o último ">"
    partes = [p.strip() for p in primeira.split(">")]
    if len(partes) >= 2:
        return partes[-1]          # ex: "Auxílios e Indenizações"
    elif len(partes) == 1:
        # Tenta pegar o sistema — remove prefixos como "SIAPE - "
        return re.sub(r"^[A-Z\s\-]+>\s*", "", partes[0]).strip()
    return primeira


def extrair_keywords(texto: str, top_n: int = 8) -> list[str]:
    """Retorna lista de palavras-chave em ordem de relevância."""
    # Remove ruído antes de passar ao KeyBERT
    texto_limpo = re.sub(r"\*\*|\[DADO REMOVIDO\]|[•\*\-]", " ", texto)
    texto_limpo = re.sub(r"\s+", " ", texto_limpo).strip()

    if len(texto_limpo) < 30:
        return []

    resultados = kw_model.extract_keywords(
        texto_limpo,
        keyphrase_ngram_range=(1, 2),   # unigramas e bigramas
        stop_words=None,                # sem stop words — modelo multilingual cuida disso
        top_n=top_n,
        use_mmr=True,                   # Maximal Marginal Relevance → diversidade
        diversity=0.5,
    )
    return [kw for kw, _ in resultados]


# -----------------------------------------------------------------------------
# 2. MAPA DE CATEGORIAS
# -----------------------------------------------------------------------------

def montar_mapa_categorias(df_base):
    mapa = {}
    for sis in SISTEMAS:
        for total_t in CONFIGS_TOPICOS:
            path = os.path.join(BERTOPIC_DIR, sis, f"{total_t}_topicos", "Resumo_Topicos_Dominantes.csv")
            if not os.path.exists(path):
                print(f"  [AVISO] Não encontrado: {path}")
                continue

            df_dom = pd.read_csv(path, low_memory=False, encoding="utf-8")
            try:
                col_id    = get_col(df_dom, "id")
                col_topic = get_col(df_dom, "dominant_topic")
            except ValueError as e:
                print(f"  [AVISO] {path}: {e}")
                continue

            df_dom = df_dom[[col_id, col_topic]].rename(
                columns={col_id: "Id", col_topic: "dominant_topic"}
            )
            df_dom["Id"] = df_dom["Id"].astype(str)
            df_merged = df_dom.merge(df_base, on="Id", how="left")

            if df_merged["Categoria"].isna().all():
                print(f"  [AVISO] Nenhum id cruzou em {path}.")
                continue

            for topico_id, grupo in df_merged.groupby("dominant_topic"):
                top_cats = (
                    grupo["Categoria"]
                    .value_counts()
                    .head(TOP_N_CATEGORIAS)
                    .index
                    .tolist()
                )
                mapa[(sis, total_t, int(topico_id))] = " | ".join(top_cats)

    print(f"[✓] Mapa montado: {len(mapa)} combinações")
    return mapa


# -----------------------------------------------------------------------------
# 3. CHAMADA AO LLM — input enxuto, tarefa mínima
# -----------------------------------------------------------------------------

def get_topic_label(keywords: list[str], modulo: str) -> str:
    kw_str = ", ".join(keywords)
    user_msg = f"MÓDULO: {modulo}\nPALAVRAS-CHAVE: {kw_str}\nTÍTULO:"

    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        options={"temperature": 0.05, "top_p": 0.9, "num_gpu": 99},
    )

    raw = response["message"]["content"].strip()
    # Pega só a primeira linha não vazia, remove marcadores e pontuação final
    titulo = next(
        (
            l.strip().lstrip("*•-123456789. ").replace('"', "").rstrip(".:")
            for l in raw.splitlines()
            if l.strip()
        ),
        raw,
    )
    return titulo


# -----------------------------------------------------------------------------
# 4. PROCESSA UM TÓPICO
# -----------------------------------------------------------------------------

def process_single_label(sis, total_t, t, mapa_categorias):
    summary_path = os.path.join(
        SUMMARIZATION_DIR, sis, f"{total_t}_topicos", f"summary_topic_{t}.txt"
    )
    if not os.path.exists(summary_path):
        return None

    with open(summary_path, "r", encoding="utf-8") as f:
        resumo = f.read()

    categorias_str = mapa_categorias.get((sis, total_t, t), "")
    modulo         = modulo_dominante(categorias_str) if categorias_str else sis
    keywords       = extrair_keywords(resumo)

    if not keywords:
        print(f"  [AVISO] Sem keywords: {sis} {total_t}t T{t}")
        return None

    print(f"  > {sis} | {total_t}t | T{t:02d} | módulo: {modulo}")
    print(f"    kw: {', '.join(keywords[:5])}")

    label = get_topic_label(keywords, modulo)
    print(f"    → {label}\n")

    return {
        "Sistema":       sis,
        "Total_Topicos": total_t,
        "Topico_ID":     t,
        "Rotulo":        label,
        "Keywords":      ", ".join(keywords),
        "Modulo":        modulo,
    }


# -----------------------------------------------------------------------------
# 5. ORQUESTRADOR
# -----------------------------------------------------------------------------

def run_labeling():
    print("=" * 60)
    print("CLASSIFICADOR v5 — KeyBERT + LLM como compositor")
    print("=" * 60)

    print(f"\n[→] Carregando {BASE_DADOS_CSV}...")
    df_base = carregar_base(BASE_DADOS_CSV)
    print(f"    {len(df_base):,} registros.")

    print("\n[→] Montando mapa de categorias...")
    mapa = montar_mapa_categorias(df_base)

    resultados = []
    print(f"\n[→] Gerando rótulos ({MAX_WORKERS} threads)...\n")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futuros = [
            executor.submit(process_single_label, sis, total_t, t, mapa)
            for sis in SISTEMAS
            for total_t in CONFIGS_TOPICOS
            for t in range(total_t)
        ]
        for futuro in as_completed(futuros):
            try:
                r = futuro.result()
                if r:
                    resultados.append(r)
            except Exception as e:
                print(f"  [ERRO]: {e}")

    if resultados:
        df_out = (
            pd.DataFrame(resultados)
            .sort_values(["Sistema", "Total_Topicos", "Topico_ID"])
            .reset_index(drop=True)
        )
        os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
        df_out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
        print(f"\n[✓] {len(df_out)} rótulos salvos em: {OUTPUT_CSV}")
        print("\nAmostra:")
        print(df_out[["Sistema", "Total_Topicos", "Topico_ID", "Rotulo", "Keywords"]]
              .head(15).to_string(index=False))
    else:
        print("\n[!] Nenhum resultado.")


if __name__ == "__main__":
    run_labeling()