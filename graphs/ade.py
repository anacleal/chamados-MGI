import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Função para encurtar nomes muito longos e evitar quebras no layout
def encurtar_nome(nome, limite=30):
    nome = str(nome)
    return (nome[:limite-10] + "...") if len(nome) > limite else nome

# ==========================================
# 1. CARREGAR E PREPARAR DADOS
# ==========================================
arquivo = "../data/base_de_dados.csv"
df = pd.read_csv(arquivo)

# Aplicar encurtamento nos nomes dos times
df['Time'] = df['Time'].astype(str).apply(encurtar_nome)
df['Categoria'] = df['Categoria'].astype(str)
df = df.dropna(subset=['Ano', 'Mês'])

# Normalização de Categorias
def normalizar_categoria(cat):
    cat = str(cat).upper()
    if cat.startswith("SIGEPE"): return "SIGEPE"
    elif cat.startswith("SIAPE"): return "SIAPE"
    elif cat.startswith("SOU GOV.BR"): return "SOUGOV"
    elif cat.startswith("SIASS"): return "SIASS"

df['Categoria_Resumo'] = df['Categoria'].apply(normalizar_categoria)

# Tratamento de Tempo
df['Tempo_Total_Atendimento'] = pd.to_numeric(df['Tempo_Total_Atendimento'], errors='coerce')
df.loc[df['Tempo_Total_Atendimento'] < 0, 'Tempo_Total_Atendimento'] = np.nan
df['Tempo_dias'] = df['Tempo_Total_Atendimento'] / (60 * 24)

# Configurações globais
sns.set_theme(style="ticks")

# ==========================================
# 2. EVOLUÇÃO TEMPORAL
# ==========================================
temporal = df.groupby(['Ano', 'Mês']).size().reset_index(name='Quantidade')
temporal['Data'] = temporal.apply(lambda x: f"{int(x['Ano'])}-{int(x['Mês']):02d}", axis=1)

plt.figure(figsize=(10, 5))
sns.lineplot(data=temporal.sort_values(['Ano', 'Mês']), x='Data', y='Quantidade', marker='o', color='#2c3e50')
plt.title("Abertura Mensal de Chamados")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('../graficos/graphs_ade/01_evolucao_temporal.png')

# ==========================================
# 3. TOP 10 TIMES (VOLUME)
# ==========================================
top_times = df['Time'].value_counts().head(10)

plt.figure(figsize=(10, 8))
sns.barplot(x=top_times.values, y=top_times.index, palette='viridis')
plt.title("Top 10 times por volume de demanda")
plt.xlabel("Total de chamados")
# print(f"{df['Time'].value_counts().values, df['Time'].value_counts().index}")
plt.tight_layout()
plt.savefig('../graficos/graphs_ade/02_top_10_times.png')

# ==========================================
# 4. DISTRIBUIÇÃO POR SISTEMA
# ==========================================
resumo = df['Categoria_Resumo'].value_counts()

plt.figure(figsize=(6, 6))
plt.pie(resumo, labels=resumo.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'), textprops={'fontsize': 14})
plt.title("Representatividade por sistema")
plt.tight_layout()
plt.savefig('../graficos/graphs_ade/03_distribuicao_sistemas.png')

# ==========================================
# 5. TIME X CATEGORIA (TOP 10)
# ==========================================
top10_times_lista = df['Time'].value_counts().head(10).index
df_top = df[df['Time'].isin(top10_times_lista)]
pivot = pd.crosstab(df_top['Time'], df_top['Categoria_Resumo'])

ax = pivot.plot(kind='bar', stacked=True, figsize=(10, 5), colormap='tab10')
plt.title("Perfil de atendimento por time")
plt.xticks(rotation=45, ha='right')
plt.legend(title="Sistemas", bbox_to_anchor=(1, 1), loc='upper left')
plt.tight_layout()
plt.savefig('../graficos/graphs_ade/04_time_vs_categoria.png')

# ==========================================
# 6. PERFORMANCE (Z-SCORE BAYESIANO)
# ==========================================
agrupado = df.groupby('Time').agg({
    'Tempo_dias': 'mean',
    'Time': 'count'
}).rename(columns={'Tempo_dias': 'Tempo_Medio', 'Time': 'Quantidade'})

media_global = df['Tempo_dias'].mean()
v = agrupado['Quantidade'].quantile(0.25)
agrupado['Tempo_Bayesiano'] = ((v * media_global) + (agrupado['Quantidade'] * agrupado['Tempo_Medio'])) / (v + agrupado['Quantidade'])

m_bayes = agrupado['Tempo_Bayesiano'].mean()
s_bayes = agrupado['Tempo_Bayesiano'].std()
agrupado['Z_Score_Ajustado'] = (agrupado['Tempo_Bayesiano'] - m_bayes) / s_bayes

z_plot = agrupado.sort_values('Z_Score_Ajustado')

# Deixei a figura um pouco mais larga (14 de largura, 8 de altura) para caber as barras
plt.figure(figsize=(14, 8))

colors = ['#27ae60' if x < -1 else '#e74c3c' if x > 1 else '#95a5a6' for x in z_plot['Z_Score_Ajustado']]

# Plotagem vertical
sns.barplot(x=z_plot.index, y=z_plot['Z_Score_Ajustado'], palette=colors)

# Linhas de referência HORIZONTAIS (axhline)
plt.axhline(0, color='black', lw=1)
plt.axhline(1, color='red', ls='--', alpha=0.5)
plt.axhline(-1, color='green', ls='--', alpha=0.5)

plt.title("Ranking de Performance por Time (Z-Score Bayesiano)", fontsize=20)
plt.ylabel("Desvios Padrão (Z-Score)", fontsize=20)

# O segredo para os nomes não ficarem bagunçados: 
# Rotacionar em 45 graus (ou 90) e alinhar à direita (ha='right')
plt.xticks(rotation=60, ha='right', fontsize=14)
plt.tight_layout()
plt.savefig('../graficos/graphs_ade/05_zscore_performance.png')