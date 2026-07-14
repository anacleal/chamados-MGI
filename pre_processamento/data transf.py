import pandas as pd

class ProcessadorBase:
    def __init__(self, input_path):
        self.input_path = input_path
        self.df = None

    def carregar_dados(self):
        print(f"--- Carregando dados de: {self.input_path} ---")
        self.df = pd.read_csv(self.input_path, low_memory=False)

        # Converter datas logo no início para evitar repetição
        self.df['Data de abertura'] = pd.to_datetime(self.df['Data de abertura'], errors='coerce')
        self.df['Data de Solução'] = pd.to_datetime(self.df['Data de Solução'], errors='coerce')
        return self

    def tratar_nulos_e_limpeza(self):
        print("-> Tratando nulos e preenchendo categorias...")
        self.df['Unidade'] = self.df['Unidade'].fillna('Não Informado')
        self.df['Estado'] = self.df['Estado'].fillna('Não Informado')
        self.df['Time'] = self.df['Time'].fillna('Não Atribuído')
        return self

    def calcular_metricas_tempo(self):
        print("-> Calculando métricas de tempo (Delta Real)...")
        # Cálculo do tempo total em horas
        delta_horas = ((self.df['Data de Solução'] - self.df['Data de abertura']) / pd.Timedelta(hours=1)).round(1)

        if 'Data de Solução' in self.df.columns:
            idx = self.df.columns.get_loc('Data de Solução')
            self.df.insert(idx + 1, 'Tempo em horas do atendimento', delta_horas)
        else:
            self.df['Tempo_Total_Atendimento_H'] = delta_horas

        return self

    def filtrar_colunas_finais(self):

        print("-> Removendo colunas inúteis...")
        colunas_lixo = [
            'Total_Horas', 'Tempo de atendimento', 'Tempo total',
            'Tempo de Interação', 'Tempo de interação dentro do SLA',
            'Tempo de Início do Atendimento', 'Contém Anexo',
            'Origem da requisição', 'Prioridade', 'Total em Minutos'
        ]

        self.df.drop(columns=[c for c in colunas_lixo if c in self.df.columns], inplace=True)
        return self

    def salvar(self, output_path):
        self.df.to_csv(output_path, index=False, encoding="utf-8")
        print(f"\nProcessamento concluído!")
        print(f"Salvo em: {output_path}")
        print(f"Colunas finais: {list(self.df.columns)}")


if __name__ == "__main__":
    ARQUIVO_ENTRADA = "../data/base_de_dados.csv"
    ARQUIVO_SAIDA = "../data/base_clean.csv"

    processador = ProcessadorBase(ARQUIVO_ENTRADA)

    (processador
     .carregar_dados()
     .tratar_nulos_e_limpeza()
     .calcular_metricas_tempo()
     .filtrar_colunas_finais()
     .salvar(ARQUIVO_SAIDA))