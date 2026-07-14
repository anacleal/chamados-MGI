import os
import json
import random

import pandas as pd
import ollama

MODEL = "llama3.1:8b-instruct-q4_K_S"

MAX_OUTPUT_TOKENS = 1000
NUM_CTX           = 8192

OLLAMA_OPTIONS = {
    "temperature": 0.05,
    "top_p":       0.9,
    "num_gpu":     99,
    "num_ctx":     NUM_CTX,
    "num_predict": MAX_OUTPUT_TOKENS,
}

K_POR_SISTEMA = {
    "SIASS":  6,
    "SIAPE":  8,
    "SIGEPE": 5,
    "SOUGOV": 5,
    "TOTAIS": 10,
}

# Seleção de documentos por tópico
N_NUCLEO    = 30   # documentos com maior probabilidade (centro do tópico)
N_PERIFERIA = 15   # documentos aleatórios dentre os restantes do mesmo tópico

SEED = 42

def load_keywords(sistema: str, topic: int) -> list[str]:
    json_path = f'../topic_modeling/bertopic_resultados/{sistema}/topicos.json'
    if not os.path.exists(json_path):
        return []
    with open(json_path, encoding='utf-8') as f:
        data = json.load(f)
    entry = data.get(str(topic)) or data.get(topic)
    if not entry:
        return []
    return [word for word, _ in entry if word]


def load_data(sistema: str, topic: int) -> tuple[list[str], list[str]]:
    df_path  = f'../data/chamados_{sistema.lower()}.csv'
    td_path  = f'../topic_modeling/bertopic_resultados/{sistema}/Topicos_Dominantes.csv'
    rdt_path = f'../topic_modeling/bertopic_resultados/{sistema}/Resumo_Topicos_Dominantes.csv'

    if not os.path.exists(df_path):
        print(f"  [AVISO] CSV de chamados não encontrado: {df_path}")
        return [], []

    df = pd.read_csv(df_path, low_memory=False)
    id_col = 'Id' if 'Id' in df.columns else 'id'
    df[id_col] = df[id_col].astype(str)

    col_texto = None
    for candidato in ['Descrição do chamado', 'Descrição do chamado_y', 'Descrição do chamado_x']:
        if candidato in df.columns:
            col_texto = candidato
            break
    if not col_texto:
        for col in df.columns:
            if 'descri' in col.lower() or 'chamado' in col.lower():
                col_texto = col
                break
    if not col_texto:
        print(f"  [AVISO] Coluna de descrição não encontrada para {sistema}.")
        return [], []

    #Topicos_Dominantes.csv (com probabilidades)
    col_prob = f'Topico {topic}'

    if os.path.exists(td_path):
        td = pd.read_csv(td_path, sep='|')
        td_id_col = 'id' if 'id' in td.columns else 'Id'
        td[td_id_col] = td[td_id_col].astype(str)

        if col_prob in td.columns and 'dominant_topic' in td.columns:
            td_topico = td[td['dominant_topic'] == topic].copy()

            if not td_topico.empty:
                td_topico = td_topico.sort_values(col_prob, ascending=False)

                ids_nucleo    = td_topico.head(N_NUCLEO)[td_id_col].tolist()
                ids_restantes = td_topico.iloc[N_NUCLEO:][td_id_col].tolist()

                rng = random.Random(SEED)
                ids_periferia = rng.sample(
                    ids_restantes,
                    min(N_PERIFERIA, len(ids_restantes))
                )

                def get_textos(ids):
                    return (
                        df[df[id_col].isin(ids)][col_texto]
                        .dropna()
                        .astype(str)
                        .tolist()
                    )

                nucleo    = get_textos(ids_nucleo)
                periferia = get_textos(ids_periferia)

                print(f"    Seleção via probabilidade: {len(nucleo)} núcleo + {len(periferia)} periferia")
                return nucleo, periferia

    #(sem probabilidade)
    if os.path.exists(rdt_path):
        print(f"  [AVISO] Topicos_Dominantes.csv sem coluna '{col_prob}' — usando fallback sem ranking.")
        rdt = pd.read_csv(rdt_path)
        rdt_id_col = 'id' if 'id' in rdt.columns else 'Id'
        rdt[rdt_id_col] = rdt[rdt_id_col].astype(str)

        ids_topico = rdt[rdt['dominant_topic'] == topic][rdt_id_col].tolist()
        df_topico  = df[df[id_col].isin(ids_topico)][col_texto].dropna().astype(str).tolist()

        rng = random.Random(SEED)
        rng.shuffle(df_topico)
        nucleo    = df_topico[:N_NUCLEO]
        periferia = df_topico[N_NUCLEO: N_NUCLEO + N_PERIFERIA]

        print(f"    Seleção via fallback (sem probabilidade): {len(nucleo)} núcleo + {len(periferia)} periferia")
        return nucleo, periferia

    print(f"  [AVISO] Nenhum arquivo de tópicos dominantes encontrado para {sistema}.")
    return [], []


# PROMPT

SYSTEM_PROMPT = """\
You are a data analysis pipeline specialized in identifying operational bottlenecks in Brazilian \
government IT support systems. You will receive a set of support tickets (chamados) from a \
specific topic identified by a topic modeling algorithm. The tickets are written in \
Brazilian Portuguese (pt-BR).

[CONTEXT]
The tickets were selected from a BERTopic cluster. The first group (labeled NÚCLEO) contains \
the most representative tickets of the topic. The second group (labeled PERIFERIA) contains \
more peripheral tickets that still belong to the same topic.
The keywords listed below were extracted by BERTopic and represent the dominant terms of \
this topic — use them as a secondary signal to interpret the tickets.

[AUDIENCE]
The output will be read by non-technical civil servants at MGI (Ministério da Gestão e da \
Inovação em Serviços Públicos) who have no background in data science or machine learning. \
Write as if describing a recurring operational problem directly, not as if describing the \
output of an algorithm.

[ANONYMIZATION RULE]
Replace ALL personal data (names, CPFs, registration numbers, e-mails, phone numbers) with \
[DADO REMOVIDO].

[TASK]
Read all tickets and produce a structured report with exactly two fields:
1. What is the dominant pattern in these tickets? (the core issue or behavior these tickets share)
2. What is the operational impact? (what this causes in practice for users and operators)

[OUTPUT FORMAT — FOLLOW EXACTLY]
- Exactly two fields, no more, no less.
- Each field starts with its bold label on its own line.
- Written in Brazilian Portuguese (pt-BR).
- No bullet points, no numbered lists, no sub-items.
- No extra sections, notes, conclusions, or meta-commentary of any kind.
- Do not mention the number of tickets or refer to NÚCLEO/PERIFERIA in the output.
- Stop writing immediately after the Impacto Operacional field.

**Padrão Dominante**: [Describe the core pattern shared by the tickets. Cover the main issue \
and its relevant variations observed across the tickets. Length should match the complexity \
of what was found — be concise when the pattern is simple, more detailed when there are \
meaningful nuances worth capturing. Do not include bullet points or lists. \
Do not describe consequences here — save those for Impacto Operacional.]

**Impacto Operacional**: [One or two sentences on the practical consequences for users and operators.]

"""


# SUMARIZAÇÃO
def chamar_ollama(prompt: list[dict]) -> str:
    response = ollama.chat(
        model=MODEL,
        messages=prompt,
        options=OLLAMA_OPTIONS,
    )
    return response["message"]["content"].strip()


def validar_resposta(text: str) -> bool:
    campos = ["**Padrão Dominante**", "**Impacto Operacional**"]
    return all(campo in text for campo in campos)


def formatar_chamados(nucleo: list[str], periferia: list[str]) -> str:
    #Monta o bloco de chamados para o prompt, separando núcleo e periferia.
    linhas = ["=== NÚCLEO (mais representativos do tópico) ==="]
    for i, c in enumerate(nucleo, 1):
        linhas.append(f"Chamado N{i}: {c}")

    if periferia:
        linhas.append("\n=== PERIFERIA (variações do mesmo tópico) ===")
        for i, c in enumerate(periferia, 1):
            linhas.append(f"Chamado P{i}: {c}")

    return "\n".join(linhas)


def get_summary(nucleo: list[str], periferia: list[str], keywords: list[str], sis: str, t: int) -> str:
    kw_str     = ", ".join(f"'{w}'" for w in keywords) if keywords else "(não disponível)"
    chamados   = formatar_chamados(nucleo, periferia)

    user_content = (
        f"Keywords do tópico: {kw_str}\n\n"
        f"{chamados}"
    )

    prompt = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]

    resposta = chamar_ollama(prompt)

    if not validar_resposta(resposta):
        print(f"  [{sis} | Tópico {t}] [AVISO] Resposta fora do formato esperado.")
        print(f"  [{sis} | Tópico {t}] [DEBUG] Início: {resposta[:200]!r}")

    return resposta




def save_summary(text: str, sistema: str, topic: int) -> None:
    dir_path = f'.//outLLM/detailed_summarization/{sistema}'
    os.makedirs(dir_path, exist_ok=True)
    with open(f'{dir_path}/summary_topic_{topic}.txt', 'w', encoding='utf-8') as f:
        f.write(text)


def load_saved_summary(sistema: str, topic: int) -> str | None:
    filepath = (f'./'
                f'/outLLM/detailed_summarization/{sistema}/summary_topic_{topic}.txt')
    if not os.path.exists(filepath):
        return None
    with open(filepath, encoding='utf-8') as f:
        return f.read()


def already_done(sistema: str, topic: int) -> bool:
    return load_saved_summary(sistema, topic) is not None



def run() -> None:
    print("=" * 60)
    print("  SUMARIZAÇÃO DETALHADA — LLAMA 3.1 (OLLAMA LOCAL)")
    print("=" * 60)

    for sis, k in K_POR_SISTEMA.items():
        print(f"\n{'=' * 60}")
        print(f"  SISTEMA: {sis} ({k} tópicos)")
        print(f"{'=' * 60}")

        for t in range(k):
            prefixo = f"[{sis} | Tópico {t}/{k - 1}]"

            if already_done(sis, t):
                print(f"  {prefixo} → já concluído, pulando.")
                continue

            nucleo, periferia = load_data(sis, t)
            if not nucleo:
                print(f"  {prefixo} → sem chamados, ignorando.")
                continue

            keywords = load_keywords(sis, t)

            print(f"  {prefixo} → {len(nucleo)} núcleo + {len(periferia)} periferia. Gerando resumo...")
            try:
                resumo = get_summary(nucleo, periferia, keywords, sis, t)
                save_summary(resumo, sis, t)
                print(f"  {prefixo} → resumo salvo.")
            except Exception as e:
                print(f"  {prefixo} → ERRO: {e}")

    print("\n" + "=" * 60)
    print("  Concluído.")
    print("=" * 60)


if __name__ == "__main__":
    run()