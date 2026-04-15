import pandas as pd
import numpy as np


def aplicar_transformacoes_adjacentes(input_path, output_path):
    print(f"Carregando dados de: {input_path}")
    df = pd.read_csv(input_path, low_memory=False)

    # ---------------------------------------------------------
    # valores ausentes
    # ---------------------------------------------------------
    print(" -> Tratando valores ausentes e datas...")
    df['Unidade'] = df['Unidade'].fillna('Não Informado')
    df['Estado'] = df['Estado'].fillna('Não Informado')
    df['Time'] = df['Time'].fillna('Não Atribuído')

    # Converter datas de string para datetime
    df['Data de abertura'] = pd.to_datetime(df['Data de abertura'], errors='coerce')
    df['Data de Solução'] = pd.to_datetime(df['Data de Solução'], errors='coerce')

    # ---------------------------------------------------------
    # 2. CONSTRUÇÃO DE DESCRITORES (Feature Engineering)
    # ---------------------------------------------------------

    # # ---------------------------------------------------------
    # # 3. BINNING / DISCRETIZAÇÃO (Exigência do TP1)
    # # ---------------------------------------------------------
    # print(" -> Aplicando Binning (Discretização) no tempo de solução...")
    # # Transforma uma variável contínua (Horas) em categorias (Rápido, Normal, Demorado)
    # bins = [-1, 24, 72, np.inf]  # Até 24h, 24h a 72h, mais de 72h
    # labels = ['Rapido (Ate 1 dia)', 'Normal (1 a 3 dias)', 'Demorado (+3 dias)']
    # df['Categoria_Tempo'] = pd.cut(df['Total_Horas'], bins=bins, labels=labels)
    #
    # # ---------------------------------------------------------
    # # 4. NORMALIZAÇÃO / PREPARAÇÃO PARA CLASSIFICAÇÃO
    # # ---------------------------------------------------------
    # print(" -> Simplificando a coluna Prioridade...")
    # # Como vimos que 'Baixa' domina 99% da base, podemos criar uma flag binária
    # df['Prioridade_Alta'] = df['Prioridade'].apply(lambda x: 1 if x in ['Alta', 'Crítica', 'Urgente'] else 0)
    #
    # # Salva o arquivo finalizado
    df.to_csv(output_path, index=False)
    print(f"Pré-processamento adjacente concluído! Salvo em: {output_path}")

def comparar_tempos_solucao(input_path, output_path):
    print(f"Carregando dados de: {input_path}")
    df = pd.read_csv(input_path, low_memory=False)

    # 1. Garantir que as datas são objetos datetime
    df['Data de abertura'] = pd.to_datetime(df['Data de abertura'], errors='coerce')
    df['Data de Solução'] = pd.to_datetime(df['Data de Solução'], errors='coerce')

    # 2. Calcular o Delta Real (Diferença entre Solução e Abertura)
    # pd.Timedelta(hours=1) converte o resultado automaticamente para horas decimais
    delta_real_horas = ((df['Data de Solução'] - df['Data de abertura']) / pd.Timedelta(hours=1)).round(1)

    # 3. Localizar 'Total_Horas' e inserir as novas colunas ao lado
    if 'Total_Horas' in df.columns:
        idx = df.columns.get_loc('Total_Horas')

        # Inserir o Delta Real logo à direita (+1)
        df.insert(idx + 1, 'Delta_Real_Horas', delta_real_horas)

        # Calcular a diferença matemática (Delta Real - Total Sistema)
        diferenca = (df['Delta_Real_Horas'] - df['Total_Horas']).round(1)

        # Inserir a Diferença logo após o Delta (+2)
        df.insert(idx + 2, 'Diferenca_Calculo', diferenca)

        print(f" -> Sucesso! Colunas de validação inseridas nas posições {idx + 1} e {idx + 2}.")

        # 4. Análise Estatística Rápida (Print para você colocar no relatório)
        print("\n" + "=" * 50)
        print("📊 COMPARAÇÃO: TOTAL DO SISTEMA vs DELTA REAL")
        print("=" * 50)
        print(df[['Total_Horas', 'Delta_Real_Horas', 'Diferenca_Calculo']].describe())

        # Contar chamados com discrepância absurda (diferença maior que 24 horas)
        erros_graves = df[df['Diferenca_Calculo'].abs() > 24.0]
        print("\n🚨 ALERTA DE QUALIDADE DE DADOS:")
        print(f" -> Total de chamados analisados: {len(df)}")
        print(f" -> Chamados com diferença maior que 24h entre os cálculos: {len(erros_graves)}")
        print(f" -> Isso representa {len(erros_graves) / len(df) * 100:.1f}% da base com tempos corrompidos.")
        print("=" * 50 + "\n")

    else:
        print(" -> Aviso: Coluna 'Total_Horas' não encontrada. Verifique os passos anteriores.")

    # Salva o arquivo com as comparações
    df.to_csv(output_path, index=False)
    print(f"Concluído! Base atualizada salva em: {output_path}")

def remover_colunas_inuteis(input_path, output_path):
    print(f"Limpando colunas de: {input_path}")
    df = pd.read_csv(input_path, low_memory=False)
    colunas_para_remover = ['Total_Horas',
                            'Diferenca_Calculo',
                            'Delta_Real_Horas',
                            'Tempo de atendimento',
                            'Tempo total',
                            'Tempo de Interação',
                            'Tempo de interação dentro do SLA',
                            'Tempo de Início do Atendimento',
                            'Última ação de acompanhamento',
                            'Título da ultima ação padrão',
                            'Chamados Relacionados ',
                            'Contém Anexo']
    # Remove as colunas
    df_final = df.drop(columns=colunas_para_remover, errors='ignore')

    # SALVAR o resultado no arquivo de saída
    df_final.to_csv(output_path, index=False, encoding="utf-8")
    print(f"✅ Sucesso! Arquivo limpo salvo em: {output_path}")
    print(f"Colunas restantes: {len(df_final.columns)}")


import pandas as pd


def adicionar_tempo_atendimento(input_path, output_path):
    print(f"Lendo base: {input_path}")
    df = pd.read_csv(input_path, low_memory=False)

    # 1. Garantir que as colunas de data sejam reconhecidas como tempo
    df['Data de abertura'] = pd.to_datetime(df['Data de abertura'], errors='coerce')
    df['Data de Solução'] = pd.to_datetime(df['Data de Solução'], errors='coerce')

    # 2. Calcular o tempo total (Delta) em horas, arredondando para 1 casa decimal
    # Se quiser em dias, troque 'hours' por 'days'
    tempo_total = ((df['Data de Solução'] - df['Data de abertura']) / pd.Timedelta(hours=1)).round(1)

    # 3. Identificar a posição da 'Data de Solução' para inserir ao lado
    if 'Data de Solução' in df.columns:
        idx = df.columns.get_loc('Data de Solução')

        # Inserir na posição logo após a Data de Solução (idx + 1)
        df.insert(idx + 1, 'Tempo_Total_Atendimento', tempo_total)
        print(f" -> Coluna 'Tempo_Total_Atendimento' inserida na posição {idx + 1}.")
    else:
        # Se não achar a coluna, joga no final por segurança
        df['Tempo_Total_Atendimento'] = tempo_total
        print(" -> Aviso: 'Data de Solução' não encontrada. Coluna adicionada ao final.")

    # 4. Salvar o arquivo
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"✅ Arquivo atualizado salvo em: {output_path}")


if __name__ == "__main__":
    # Escolha qual tabela você quer "tunar" para a análise final
    INPUT_CSV = "../data/base_de_dados.csv"
    OUTPUT_CSV = "../data/base_de_dados_0.csv"

    #aplicar_transformacoes_adjacentes(OUTPUT_CSV, OUTPUT_CSV)
    #comparar_tempos_solucao(INPUT_CSV, OUTPUT_CSV)
    #remover_colunas_inuteis(INPUT_CSV, OUTPUT_CSV)
    adicionar_tempo_atendimento(INPUT_CSV, OUTPUT_CSV)