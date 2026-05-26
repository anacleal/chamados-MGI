import os
import json
import time
import pandas as pd
import ollama

TOPICOS_PATH   = "../topic_modeling/bertopic_resultados/TOTAIS/5_topicos/Resumo_Topicos_Dominantes.csv"
CHATBOT_PATH   = "../data/chatbot_table.csv"   # descrições sem pré-proc

OUTPUT_PATH    = "../topic_modeling/bertopic_resultados/TOTAIS/5_topicos/Topicos_Nomeados.csv"

# Coluna de ID nas bases (ajuste se o nome for diferente)
ID_COL_TOPICOS  = "Id"
ID_COL_CHATBOT  = "Id"          # maiúsculo conforme mencionado
DESC_COL        = "Descrição do chamado"

AMOSTRAS_POR_TOPICO = 10        # quantos chamados mandar ao LLM por tópico
MODEL           = "qwen2.5"  # Certifique-se de que este modelo existe no seu Ollama
SLEEP_ENTRE_CHAMADAS = 0.5      # segundos entre chamadas


def montar_prompt(topico_id: int, amostras: list[str]) -> str:
    exemplos = "\n".join(f"- {s}" for s in amostras)
    return f"""Você é um especialista em análise de chamados de suporte técnico.
Abaixo estão {len(amostras)} exemplos de chamados que pertencem ao Tópico {topico_id}
de uma modelagem de tópicos (BERTopic + KMeans).

CHAMADOS:
{exemplos}

Com base nesses exemplos, gere:
1. Um NOME CURTO para o tópico (2 a 5 palavras, em português).
2. Uma DESCRIÇÃO de uma frase explicando o padrão central do tópico.

Responda SOMENTE em JSON válido, sem markdown, no formato:
{{"nome": "...", "descricao": "..."}}"""


def nomear_topico(topico_id: int, amostras: list[str]) -> dict:
    prompt = montar_prompt(topico_id, amostras)
    try:
        response = ollama.chat(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.2},
        )
        texto = response["message"]["content"].strip()
        
        # Limpeza de markdown
        texto = texto.replace("```json", "").replace("```", "").strip()
        
        # Extração de JSON
        inicio = texto.find("{")
        fim = texto.rfind("}") + 1
        if inicio != -1 and fim > inicio:
            texto = texto[inicio:fim]
            
        resultado = json.loads(texto)
        return {
            "topico_id": topico_id,
            "nome": resultado.get("nome", "SEM_NOME").strip(),
            "descricao": resultado.get("descricao", "").strip(),
        }
    except json.JSONDecodeError as e:
        print(f"  [AVISO] JSON inválido para tópico {topico_id}: {e}")
        return {"topico_id": topico_id, "nome": "ERRO_PARSE", "descricao": texto[:300]}
    except Exception as e:
        print(f"  [ERRO] Tópico {topico_id}: {e}")
        return {"topico_id": topico_id, "nome": "ERRO", "descricao": str(e)}




def main():
    print("=" * 60)
    print("NOMEAÇÃO DE TÓPICOS COM QWEN 2.5 (OLLAMA LOCAL)")
    print("=" * 60)

    # 0. Verifica Ollama
    print("\n[0] Verificando conexão com Ollama...")

    # 1. Carrega tópicos
    print("\n[1] Carregando tópicos dominantes...")
    if not os.path.exists(TOPICOS_PATH):
        print(f"Erro: Arquivo {TOPICOS_PATH} não encontrado.")
        return

    df_topicos = pd.read_csv(TOPICOS_PATH)
    df_topicos.columns = df_topicos.columns.str.strip()
    
    id_col_top = next(
        (c for c in df_topicos.columns if c.lower() == ID_COL_TOPICOS.lower()),
        df_topicos.columns[0]
    )
    topic_col = next(
        (c for c in df_topicos.columns 
         if "topic" in c.lower() or "topico" in c.lower() or "tópico" in c.lower()),
        df_topicos.columns[1]
    )

    # 2. Carrega base de descrições
    print("\n[2] Carregando base de chamados...")
    if not os.path.exists(CHATBOT_PATH):
        print(f"Erro: Arquivo {CHATBOT_PATH} não encontrado.")
        return

    df_chat = pd.read_csv(CHATBOT_PATH, low_memory=False)
    df_chat.columns = df_chat.columns.str.strip()

    id_col_chat = next(
        (c for c in df_chat.columns if c.upper() == ID_COL_CHATBOT.upper()),
        df_chat.columns[0]
    )

    # 3. Join
    print("\n[3] Fazendo join entre tópicos e descrições...")
    df_topicos[id_col_top] = df_topicos[id_col_top].astype(str).str.strip()
    df_chat[id_col_chat] = df_chat[id_col_chat].astype(str).str.strip()

    df_merged = df_topicos.merge(
        df_chat[[id_col_chat, DESC_COL]],
        left_on=id_col_top,
        right_on=id_col_chat,
        how="left"
    )

    # 4. Nomeação por tópico
    topicos_unicos = sorted(df_merged[topic_col].dropna().unique())
    print(f"\n[4] Tópicos encontrados: {topicos_unicos}")
    print(f"    Usando {AMOSTRAS_POR_TOPICO} amostras por tópico no Ollama ({MODEL})...\n")

    resultados = []
    for t in topicos_unicos:
        subset = df_merged[df_merged[topic_col] == t][DESC_COL].dropna()
        if len(subset) == 0: continue
        
        amostras = subset.sample(min(AMOSTRAS_POR_TOPICO, len(subset)), random_state=42).tolist()
        amostras = [str(s)[:500] for s in amostras]

        print(f"  Tópico {int(t)} — {len(subset)} chamados...")
        resultado = nomear_topico(int(t), amostras)
        print(f"    → Nome: {resultado['nome']}")
        resultados.append(resultado)
        time.sleep(SLEEP_ENTRE_CHAMADAS)

    # 5. Salva resultado
    if resultados:
        df_nomes = pd.DataFrame(resultados)
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        df_nomes.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
        
        # 7. Base completa
        df_final = df_merged.merge(df_nomes, left_on=topic_col, right_on="topico_id", how="left")
        path_completo = OUTPUT_PATH.replace(".csv", "_completo.csv")
        df_final.to_csv(path_completo, index=False, encoding="utf-8-sig")
        print(f"\n[5] Resultados salvos em: {OUTPUT_PATH}")

    print("\nConcluído!")


if __name__ == "__main__":
    main()
