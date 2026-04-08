import unicodedata
import pandas as pd
import re

entrada_csv = "tabela_teste.csv"
entrada_nomes_formatados = "nomes_formatados.txt"

with open(entrada_nomes_formatados, "r", encoding="utf-8") as f:
    nomes = {nome.strip().lower() for nome in f if nome.strip()}

df = pd.read_csv(entrada_csv, encoding="utf-8")

labels = [
    "Título",
    "Descrição do chamado",
    "Última ação de acompanhamento",
    "Título da ultima ação padrão"
]

def substituir(match):
    palavra = match.group()
    if palavra.lower() in nomes:
        return "[ANONIMIZADO]"
    return palavra

def anonimizar(texto):
    if not isinstance(texto, str):
        return texto
    return re.sub(r'\b[A-Za-zÀ-ÿ]+\b', substituir, texto)

for coluna in labels:
    if coluna in df.columns:
        df[coluna] = df[coluna].apply(anonimizar)


df.to_csv("tabela_anonimizada.csv", index=False, encoding="utf-8")

# def preprocess(entrada):
#   entrada['textclear'] = entrada['textclear'].str.replace(r"\d+", "<NUM>", regex = True)
