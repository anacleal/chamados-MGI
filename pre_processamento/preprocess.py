import re
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from unidecode import unidecode

# ntlk.download('stopwords')

entrada_csv = "tabela_teste.csv"
entrada_nomes_formatados = "nomes_formatados.txt"

# lower e tirar acento
def normalize(text):
  return unidecode(str(text).lower())

#lista de stopwords normalizadas
def stopwords_formatadas():
  stopwords_pt = {normalize(w) for w in stopwords.words('portuguese')}
  stopwords_pt.discard("nao")
  stopwords_pt.discard("sem")
  stopwords_pt.add("prezad")
  return stopwords_pt

#pega todos os nomes da base do ibge
def nomes_formatados():
  with open(entrada_nomes_formatados, "r", encoding="utf-8") as f:
    nomes = {normalize(nome.strip()) for nome in f if nome.strip()}
  return nomes

#mascara da base de dados
df = pd.read_csv(entrada_csv, encoding="utf-8")

#colunas da base
labels = [
  "Titulo",
  "Descrição do chamado",
  "Última ação de acompanhamento",
  "Título da ultima ação padrão"
]

#inicialização variaveis
nomes = nomes_formatados()
stopwords_pt = stopwords_formatadas()


def preprocess(text):
  # normaliza o texto
  text = normalize(text)
  # remove URLs
  text = re.sub(r'http\S+|www\S+', '', text)
  # remove pontuação
  text = re.sub(r'[^\w\s]', '', text)

  
  tokens = text.split()
  resultado = []
  for token in tokens:
        # remove stopwords
        if token in stopwords_pt:
            continue
        #tokeniza os nomes
        elif token in nomes:
           resultado.append("[nome]")
        #se não for token só da append
        else: 
           resultado.append(token)
  return " ".join(resultado).strip()


for coluna in labels:
    coluna_clean = f"{coluna}_clean"
    if coluna in df.columns:
        df[coluna_clean] = df[coluna].apply(lambda x: preprocess(x))

df.to_csv("DESIN2025_clean.csv", index=False, encoding="utf-8")