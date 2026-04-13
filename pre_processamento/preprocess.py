import re
import pandas as pd
from unidecode import unidecode
from nltk.corpus import stopwords
#import nltk

#nltk.download('stopwords', quiet=True)

def normalize(text):
    #lower e tira acentos
    return unidecode(str(text).lower())

def get_stopwords_formatadas():
    #lista de stopwords normalizadas
    stopwords_pt = {normalize(w) for w in stopwords.words('portuguese')}
    custom_stops = ["ola", "oi", "saudacoes", "senhores", "senhoras", "cordialmente", "att", "atenciosamente", "id"]
    stopwords_pt.update(custom_stops)
    return stopwords_pt

def load_nomes_formatados(file_path):
    #carrega os nomes normalizados da base de dados do ibge
    with open(file_path, "r", encoding="utf-8") as f:
        nomes = {normalize(nome.strip()) for nome in f if nome.strip()}
    return nomes


def preprocess_text(text, stopwords_pt, nomes):
    #limpa e tokeniza o texto, e limpa os nomes e numeros
    if pd.isna(text): return ""

    text = normalize(text)
    # remove URLs
    text = re.sub(r'http\S+|www\S+', ' ', text)
    # remove pontuação
    text = re.sub(r'[^\w\s]', ' ', text)

    tokens = text.split()
    resultado = []

    for token in tokens:
        if token == "nao":
            resultado.append("nao")
            continue

        # remove stopwords
        if token in stopwords_pt or token.startswith("prezad") or token.startswith("servidor"):
            continue

        # tokeniza os nomes
        if token in nomes:
            resultado.append("[nome]")
            continue

        # substitui os numeros por X
        if any(char.isdigit() for char in token):
            continue

        else:
            resultado.append(token)

    return " ".join(resultado).strip()

def run_preprocessing(input_path, names_path, output_clean_path):
    # execucao pra main
    df = pd.read_csv(input_path, encoding="utf-8", low_memory=False)

    # normaliza o titulo das colunas
    cols_originais = df.columns
    normalized_cols_map = {normalize(col): col for col in cols_originais}

    labels_to_process = [
        "titulo",
        "descricao do chamado",
        "ultima acao de acompanhamento",
        "titulo da ultima acao padrao"
    ]

    nomes = load_nomes_formatados(names_path)
    stopwords_pt = get_stopwords_formatadas()

    processed_cols = []
    for label in labels_to_process:
        if label in normalized_cols_map:
            original_col = normalized_cols_map[label]
            col_clean = f"{original_col}"
            print(f"Processando coluna: {original_col}")
            df[col_clean] = df[original_col].apply(lambda x: preprocess_text(x, stopwords_pt, nomes))
            processed_cols.append(col_clean)
        else:
            print(f"Aviso: Coluna correspondente a '{label}' não encontrada no CSV.")

    # Save full CSV
    # df.to_csv(output_full_path, index=False, encoding="utf-8")

    # Save only processed columns
    if processed_cols:
        df[processed_cols].to_csv(output_clean_path, index=False, encoding="utf-8")
        print(f"Arquivos salvos:\n - {output_clean_path}")
    else:
        print("Nenhuma coluna foi processada.")

if __name__ == "__main__":
    # Defina aqui os caminhos dos seus arquivos
    INPUT_CSV = "../data/DESIN2025.csv"
    NAMES_FILE = "../data/nomes_formatados.txt"
    OUTPUT_CLEAN = "../data/data_pre_processed.csv"

    run_preprocessing(
        input_path=INPUT_CSV,
        names_path=NAMES_FILE,
        output_clean_path=OUTPUT_CLEAN
    )