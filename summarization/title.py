"""
Titulação Iterativa de Tópicos — Llama 3.1 (Ollama local)
==========================================================
Lê os resumos gerados pelo detailed_summarization.py e atribui
um título único a cada tópico, por sistema.

Estratégia (Opção B):
  - Tópicos titulados um por vez
  - Cada chamada recebe os títulos já atribuídos como contexto proibido
  - Garante distinção mesmo entre tópicos com conteúdo próximo

Saída: titulo_topic_{N}.txt por tópico, na mesma pasta dos resumos.
"""

import os
import ollama

# ============================================================
# CONFIGURAÇÕES
# ============================================================
MODEL = "llama3.1:8b-instruct-q4_K_S"

OLLAMA_OPTIONS = {
    "temperature": 0.3,   # ligeiramente mais criativo para variar títulos
    "top_p":       0.9,
    "num_gpu":     99,
    "num_ctx":     4096,
    "num_predict": 50,    # título curto — não precisa de mais
}

K_POR_SISTEMA = {
    "SIASS":  6,
    "SIAPE":  8,
    "SIGEPE": 5,
    "SOUGOV": 5,
    "TOTAIS": 10,
}

RESUMOS_DIR = "outLLM/detailed_summarization"


# ============================================================
# PROMPT
# ============================================================
SYSTEM_PROMPT = """\
You are a topic labeling specialist for Brazilian government IT support systems.
You will receive the summary of one topic cluster and must generate a single short title for it.
Do not reuse suffix patterns or structural templates from previously used titles, even if the wording itself differs.

[RULES]
- The title must be in Brazilian Portuguese (pt-BR).
- Maximum 8 words.
- Must be specific enough to distinguish this topic from all others already titled.
- Must NOT be a generic phrase like "Problemas de Integração" or "Erros no Sistema" alone \
— always include the specific context (which system, which operation, which user group).
- Must NOT repeat or closely paraphrase any of the already used titles listed below.

[OUTPUT]
Output ONLY the title. No quotes, no numbering, no explanation.
"""


# ============================================================
# CHAMADA AO MODELO
# ============================================================
def chamar_ollama(prompt: list[dict]) -> str:
    response = ollama.chat(
        model=MODEL,
        messages=prompt,
        options=OLLAMA_OPTIONS,
    )
    return response["message"]["content"].strip()


def get_titulo(resumo: str, titulos_usados: list[str], sis: str, t: int) -> str:
    proibidos = ""
    if titulos_usados:
        lista = "\n".join(f"- {titulo}" for titulo in titulos_usados)
        proibidos = f"[ALREADY USED TITLES — DO NOT REPEAT OR PARAPHRASE]\n{lista}\n\n"

    user_content = f"{proibidos}[TOPIC SUMMARY]\n{resumo}"

    prompt = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]

    titulo = chamar_ollama(prompt).strip().strip('"').strip("'")

    # Remove numeração acidental (ex: "1. Título")
    if titulo and titulo[0].isdigit():
        titulo = titulo.lstrip("0123456789. ").strip()

    if not titulo:
        titulo = f"Tópico {t}"
        print(f"  [{sis} | Tópico {t}] [AVISO] Título vazio — usando placeholder.")

    return titulo


# ============================================================
# PERSISTÊNCIA
# ============================================================
def load_resumo(sistema: str, topic: int) -> str | None:
    path = f"{RESUMOS_DIR}/{sistema}/summary_topic_{topic}.txt"
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return f.read().strip()


def save_titulo(titulo: str, sistema: str, topic: int) -> None:
    dir_path = f"{RESUMOS_DIR}/{sistema}"
    os.makedirs(dir_path, exist_ok=True)
    with open(f"{dir_path}/titulo_topic_{topic}.txt", "w", encoding="utf-8") as f:
        f.write(titulo)


def load_titulo(sistema: str, topic: int) -> str | None:
    path = f"{RESUMOS_DIR}/{sistema}/titulo_topic_{topic}.txt"
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return f.read().strip()


def titulo_done(sistema: str, topic: int) -> bool:
    return load_titulo(sistema, topic) is not None


# ============================================================
# LOOP PRINCIPAL
# ============================================================
def run() -> None:
    print("=" * 60)
    print("  TITULAÇÃO ITERATIVA — LLAMA 3.1 (OLLAMA LOCAL)")
    print("=" * 60)

    for sis, k in K_POR_SISTEMA.items():
        print(f"\n{'=' * 60}")
        print(f"  SISTEMA: {sis} ({k} tópicos)")
        print(f"{'=' * 60}")

        # Pré-popula com títulos já gerados (permite retomar sem quebrar contexto)
        titulos_usados: list[str] = []
        for t in range(k):
            titulo_existente = load_titulo(sis, t)
            if titulo_existente:
                titulos_usados.append(titulo_existente)

        for t in range(k):
            prefixo = f"[{sis} | Tópico {t}/{k - 1}]"

            if titulo_done(sis, t):
                print(f"  {prefixo} → já concluído ({load_titulo(sis, t)!r}), pulando.")
                continue

            resumo = load_resumo(sis, t)
            if not resumo:
                print(f"  {prefixo} → resumo não encontrado, pulando.")
                continue

            print(f"  {prefixo} → gerando título ({len(titulos_usados)} títulos já usados)...")
            try:
                titulo = get_titulo(resumo, titulos_usados, sis, t)
                save_titulo(titulo, sis, t)
                titulos_usados.append(titulo)
                print(f"  {prefixo} → '{titulo}'")
            except Exception as e:
                print(f"  {prefixo} → ERRO: {e}")

    print("\n" + "=" * 60)
    print("  Concluído.")
    print("=" * 60)


if __name__ == "__main__":
    run()