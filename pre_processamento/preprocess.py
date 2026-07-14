import os
import re
import pandas as pd
from unidecode import unidecode
from nltk.corpus import stopwords

# import nltk
# nltk.download('stopwords', quiet=True)

def normalize(text):
    # lower e tira acentos
    return unidecode(str(text).lower())

def get_stopwords_formatadas():
    # lista de stopwords normalizadas
    stopwords_pt = {normalize(w) for w in stopwords.words('portuguese')}
    custom_stops = ['ola', 'oi', 'saudacoes', 'senhores', 'senhoras', 'cordialmente', 'att', 'atenciosamente', 'id', 'nome',
                    'nao']
    stopwords_pt.update(custom_stops)
    return stopwords_pt

def load_nomes_formatados(file_path):
    # carrega os nomes normalizados da base de dados do ibge
    with open(file_path, "r", encoding="utf-8") as f:
        nomes = {normalize(nome.strip()) for nome in f if nome.strip()}
    return nomes
def preprocess_text(text, stopwords_pt, nomes):
    # limpa e tokeniza o texto, e limpa os nomes e numeros
    if pd.isna(text): return ""

    text = normalize(text)

    expressoes_remover = ['bom dia', 'boa tarde', 'boa noite',]
    for expr in expressoes_remover:
        text = text.replace(expr, ' ')

    # remove emails
    text = re.sub(r'\S+@\S+', ' ', text)
    # remove URLs
    text = re.sub(r'http\S+|www\S+', ' ', text)
    # remove pontuação
    text = re.sub(r'[^\w\s]', ' ', text)

    tokens = text.split()
    resultado = []

    for token in tokens:

        #remove stopwords
        if token in stopwords_pt or token in nomes or token.startswith("prezad") or token.startswith("servidor"):
            continue

        if token in nomes:
            continue

        if all(char == 'x' for char in token):
            continue

        if any(char.isdigit() for char in token):
            continue

        if len(token) <= 2:
            continue

        else:
            resultado.append(token)

    return " ".join(resultado).strip()


def process_original_file(input_path, output_path, names_path):
    print(f"Iniciando pré-processamento na base completa: {input_path}")

    # carrega a base
    df = pd.read_csv(input_path, encoding="utf-8", low_memory=False)

    nomes = load_nomes_formatados(names_path)
    stopwords_pt = get_stopwords_formatadas()

    labels_to_process = [
        "titulo", "descricao do chamado", "ultima acao de acompanhamento",
        "titulo da ultima acao padrao", "chamados relacionados"
    ]

    cols_originais = df.columns
    normalized_cols_map = {normalize(col): col for col in cols_originais}

    for label in labels_to_process:
        if label in normalized_cols_map:
            original_col = normalized_cols_map[label]
            print(f" -> Limpando coluna: {original_col}...")
            df[original_col] = df[original_col].apply(lambda x: preprocess_text(x, stopwords_pt, nomes))

    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Concluído! Base mestre salva em: {output_path}")


if __name__ == "__main__":
    INPUT_CSV = "../data/chatbot_table.csv"
    OUTPUT_CSV = "../data/base_de_dados.csv"
    NAMES_FILE = "../data/nomes_formatados.txt"

    process_original_file(INPUT_CSV, OUTPUT_CSV, NAMES_FILE)