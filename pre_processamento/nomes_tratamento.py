import pandas as pd

arquivo_fem = "ibge-fem-10000.csv"
arquivo_masc = "ibge-mas-10000.csv"
arquivo_saida = "nomes_formatados.txt"

df_fem = pd.read_csv(arquivo_fem, header=None, encoding="utf-8")
df_masc = pd.read_csv(arquivo_masc, header=None, encoding="utf-8")

df_total = pd.concat([df_fem, df_masc], ignore_index=True)

nomes = df_total[0]

nomes = nomes.drop_duplicates()

nomes.to_csv(arquivo_saida, index=False, header=False, encoding="utf-8")

print(f"Pronto! {len(nomes)} nomes foram salvos em {arquivo_saida}")