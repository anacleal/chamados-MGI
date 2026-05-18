import pandas as pd
from unidecode import unidecode

def format_names(file_fem, file_masc, output_file):
    try: # tratamento de erro
        df_fem = pd.read_csv(file_fem, header=None, encoding="utf-8")
        df_masc = pd.read_csv(file_masc, header=None, encoding="utf-8")
    except FileNotFoundError as e:
        print(f"Erro: Arquivo IBGE não encontrado: {e.filename}")
        return

    df_total = pd.concat([df_fem, df_masc], ignore_index=True)
    nomes = df_total[0].drop_duplicates()

    nomes = nomes.apply(lambda x: unidecode(str(x)).lower().strip())

    # remove "NAO" e nomes com menos de 2 letras
    nomes = nomes[~nomes.isin(["nao"])]
    nomes = nomes[nomes.str.len() > 2]

    nomes.to_csv(output_file, index=False, header=False, encoding="utf-8")
    print(f"Pronto! {len(nomes)} nomes foram salvos em {output_file}")

if __name__ == "__main__":
    format_names("ibge-data/ibge-fem-10000.csv", "ibge-data/ibge-mas-10000.csv", "../data/nomes_formatados.txt")
