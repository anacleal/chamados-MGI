import re
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from unidecode import unidecode

# ntlk.download('stopwords')

entrada_csv = "../data/DESIN2025.csv"
entrada_nomes_formatados = "../data/nomes_formatados.txt"

# lower e tirar acento
def normalize(text):
  return unidecode(str(text).lower())

#lista de stopwords normalizadas
def stopwords_formatadas():
  stopwords_pt = {normalize(w) for w in stopwords.words('portuguese')}
  stopwords_pt.add("ola")
  stopwords_pt.add("oi")
  stopwords_pt.add("saudacoes")
  stopwords_pt.add("senhores")
  stopwords_pt.add("senhoras")
  stopwords_pt.add("cordialmente")
  stopwords_pt.add("att")
  stopwords_pt.add("atenciosamente")
  return stopwords_pt

#pega todos os nomes da base do ibge
def nomes_formatados():
  with open(entrada_nomes_formatados, "r", encoding="utf-8") as f:
    nomes = {normalize(nome.strip()) for nome in f if nome.strip()}
  return nomes

#mascara da base de dados
df = pd.read_csv(entrada_csv, encoding="utf-8", low_memory=False)

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
  #remover valores nulos
  if pd.isna(text): return ""

  # normaliza o texto
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
        if token in stopwords_pt or token.startswith("prezad"):
          continue

        #tokeniza os nomes
        if token in nomes:
           resultado.append("[nome]")
           continue
        
        #se não for token só da append
        if any(char.isdigit() for char in token):
           token_num = re.sub(r'\d', 'X', token)
           resultado.append(token_num)
        else:
           resultado.append(token)

  return " ".join(resultado).strip()


for coluna in labels:
    coluna_clean = f"{coluna}_clean"
    if coluna in df.columns:
        df[coluna_clean] = df[coluna].apply(lambda x: preprocess(x))

df.to_csv("../data/DESIN2025_clean.csv", index=False, encoding="utf-8")
