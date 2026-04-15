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
arquivo = "graphs_ade/Planilha Chamados MGI.xlsx"
df = pd.read_excel(arquivo)

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
    elif "E-SOCIAL" in cat or "ESOCIAL" in cat: return "E-SOCIAL"
    elif cat.startswith("SIASS"): return "SIASS"
    else: return "OUTROS"

df['Categoria_Resumo'] = df['Categoria'].apply(normalizar_categoria)

# Tratamento de Tempo
df['Total em Minutos'] = pd.to_numeric(df['Total em Minutos'], errors='coerce')
df.loc[df['Total em Minutos'] < 0, 'Total em Minutos'] = np.nan
df['Tempo_dias'] = df['Total em Minutos'] / (60 * 24)

# Configurações globais
sns.set_theme(style="whitegrid")

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
plt.savefig('graphs_ade/01_evolucao_temporal.png')

# ==========================================
# 3. TOP 10 TIMES (VOLUME)
# ==========================================
top_times = df['Time'].value_counts().head(10)

plt.figure(figsize=(10, 5))
sns.barplot(x=top_times.values, y=top_times.index, palette='viridis')
plt.title("Top 10 times por volume de demanda")
plt.xlabel("Total de chamados")
plt.tight_layout()
plt.savefig('graphs_ade/02_top_10_times.png')

# ==========================================
# 4. DISTRIBUIÇÃO POR SISTEMA
# ==========================================
resumo = df['Categoria_Resumo'].value_counts()

plt.figure(figsize=(7, 7))
plt.pie(resumo, labels=resumo.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
plt.title("Representatividade por sistema")
plt.tight_layout()
plt.savefig('graphs_ade/03_distribuicao_sistemas.png')

# ==========================================
# 5. TIME X CATEGORIA (TOP 10)
# ==========================================
top10_times_lista = df['Time'].value_counts().head(10).index
df_top = df[df['Time'].isin(top10_times_lista)]
pivot = pd.crosstab(df_top['Time'], df_top['Categoria_Resumo'])

ax = pivot.plot(kind='bar', stacked=True, figsize=(10, 5), colormap='tab10')
plt.title("Perfil de Atendimento por Time (Top 10)")
plt.xticks(rotation=45, ha='right')
plt.legend(title="Sistemas", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('graphs_ade/04_time_vs_categoria.png')

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

# Altura reduzida para 8 (era 12) para melhor visualização
plt.figure(figsize=(10, 8))
colors = ['#27ae60' if x < -1 else '#e74c3c' if x > 1 else '#95a5a6' for x in z_plot['Z_Score_Ajustado']]
sns.barplot(x=z_plot['Z_Score_Ajustado'], y=z_plot.index, palette=colors)
plt.axvline(0, color='black', lw=1)
plt.axvline(1, color='red', ls='--', alpha=0.5)
plt.axvline(-1, color='green', ls='--', alpha=0.5)
plt.title("Ranking de Performance por Time (Z-Score Bayesiano)")
plt.xlabel("Desvios Padrão (Z-Score)")
plt.tight_layout()
plt.savefig('graphs_ade/05_zscore_performance.png')