import pandas as pd
import re

print("Carregando base...")
df = pd.read_csv('../data/base_de_dados.csv', low_memory=False)

# As palavras que você quer apagar
palavras_remover = ['siape', 'sigepe', 'siass', 'sougov', 'sou gov', 'bom dia', 'boa tarde', 'boa noite']

# Cria uma regra Regex para achar essas palavras
pattern = r'\b(?:' + '|'.join(palavras_remover) + r')\b'

colunas = ['Título', 'Descrição do chamado']

for col in colunas:
    print(f"Limpando coluna: {col}...")

    # BLINDAGEM: Preenche os vazios (NaN) com texto vazio ('') para não dar erro de float
    df[col] = df[col].fillna('')

    # 1. Substitui as palavras, forçando o 'x' a ser string dentro da função
    df[col] = df[col].apply(lambda x: re.sub(pattern, '', str(x), flags=re.IGNORECASE))

    # 2. Arruma os espaços duplos que ficam quando a gente arranca uma palavra
    df[col] = df[col].apply(lambda x: re.sub(r'\s+', ' ', str(x)).strip())

# Salva a base limpa
df.to_csv('../data/base_de_dados_0.csv', index=False)
print("✅ Concluído! Sistemas removidos dos textos.")