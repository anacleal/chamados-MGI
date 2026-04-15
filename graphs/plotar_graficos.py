import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from wordcloud import WordCloud
import os

# =========================================================
# CONFIGURAÇÕES GERAIS E CRIAÇÃO DE PASTAS
# =========================================================
INPUT_FILE = '../data/base_de_dados.csv'
OUTPUT_DIR = '../graficos/' # Nova pasta exclusiva para as imagens!

# Cria a pasta 'graficos' automaticamente se ela não existir
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Estilo premium para slides/documentos (fundo branco, sem grades pesadas)
sns.set_theme(style="white", font_scale=1.1)

print(f"Carregando base de dados: {INPUT_FILE}...")
df = pd.read_csv(INPUT_FILE, low_memory=False)

# Garantir que as datas estão no formato correto
df['Data de abertura'] = pd.to_datetime(df['Data de abertura'], errors='coerce')
df['Data de início de atendimento'] = pd.to_datetime(df['Data de início de atendimento'], errors='coerce')

# =========================================================
# GRÁFICO 1: TOP 10 UNIDADES SOLICITANTES
# =========================================================
print("1/5 - Gerando Gráfico: Top Unidades...")
plt.figure(figsize=(10, 6))

top_unidades = df['Unidade'].fillna('Não Informada').value_counts().head(10)
ax = sns.barplot(x=top_unidades.values, y=top_unidades.index, hue=top_unidades.index, palette="crest", legend=False)
sns.despine()

plt.title('Top 10 Instituições Solicitantes de Suporte', fontweight='bold', pad=20)
plt.xlabel('Volume de Chamados')
plt.ylabel('')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}01_Top_Unidades.png', dpi=300, transparent=True)
plt.close()

# =========================================================
# GRÁFICO 2: CURVA DE TEMPO (KDE)
# =========================================================
print("2/5 - Gerando Gráfico: Curva de Densidade de Tempo...")
plt.figure(figsize=(10, 6))

horas = pd.to_numeric(df['Tempo_Total_Atendimento'], errors='coerce').dropna()
limite_visual = horas.quantile(0.95)

sns.kdeplot(horas[horas <= limite_visual], fill=True, color="#2ecc71", alpha=0.6, linewidth=2)
sns.despine()

plt.title(f'Distribuição do Tempo de Solução (Cortado no P95: Até {limite_visual:.1f}h)', fontweight='bold', pad=20)
plt.xlabel('Horas para Solução (Tempo Total)')
plt.ylabel('Densidade de Chamados')

mediana = horas.median()
plt.axvline(mediana, color='#e74c3c', linestyle='--', label=f'Mediana ({mediana:.1f}h)')
plt.legend()

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}02_Curva_Tempo.png', dpi=300, transparent=True)
plt.close()

# =========================================================
# GRÁFICO 3: NUVEM DE PALAVRAS COM PADRONIZAÇÃO
# =========================================================
print("3/5 - Gerando Gráfico: Wordcloud...")
textos_limpos = " ".join(df['Título'].dropna().astype(str).tolist())

substituicoes = {
    "sigep ": "sigepe ", "sigepweb": "sigepe", "siap ": "siape ",
    "gov br": "sougov", "sou gov": "sougov", "app": "aplicativo"
}
for errado, correto in substituicoes.items():
    textos_limpos = textos_limpos.replace(errado, correto)

wordcloud = WordCloud(
    width=1600, height=800, background_color='white', colormap='inferno', max_words=100, collocations=False
).generate(textos_limpos)

plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Principais Termos Relatados nos Chamados', fontweight='bold', pad=20, fontsize=16)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}03_Wordcloud.png', dpi=300, transparent=True)
plt.close()

# =========================================================
# GRÁFICO 4: TEMPO DE RESPOSTA vs TEMPO DE SOLUÇÃO
# =========================================================
print("4/5 - Gerando Gráfico: SLA por Prioridade...")
df['Tempo_Resposta_Horas'] = ((df['Data de início de atendimento'] - df['Data de abertura']) / pd.Timedelta(hours=1)).round(1)
df.loc[df['Tempo_Resposta_Horas'] < 0, 'Tempo_Resposta_Horas'] = 0

plt.figure(figsize=(10, 6))
df_valid_resp = df.dropna(subset=['Tempo_Resposta_Horas', 'Tempo_Total_Atendimento', 'Prioridade'])
metricas_prioridade = df_valid_resp.groupby('Prioridade')[['Tempo_Resposta_Horas', 'Tempo_Total_Atendimento']].median().reset_index()

# CORREÇÃO: Pega todas as prioridades que existem na base e organiza as principais primeiro
ordem_desejada = ['Crítica', 'Alta', 'Média', 'Baixa', 'Sem Prioridade', 'Não Informado']
categorias_existentes = metricas_prioridade['Prioridade'].unique().tolist()
ordem_final = [p for p in ordem_desejada if p in categorias_existentes] + [p for p in categorias_existentes if p not in ordem_desejada]

metricas_prioridade['Prioridade'] = pd.Categorical(metricas_prioridade['Prioridade'], categories=ordem_final, ordered=True)
metricas_prioridade = metricas_prioridade.sort_values('Prioridade')

df_melted = metricas_prioridade.melt(id_vars='Prioridade', var_name='Métrica', value_name='Horas (Mediana)')
df_melted['Métrica'] = df_melted['Métrica'].replace({
    'Tempo_Resposta_Horas': 'Tempo até o 1º Contato',
    'Tempo_Total_Atendimento': 'Tempo até a Solução'
})

sns.set_theme(style="whitegrid", font_scale=1.1)
ax = sns.barplot(data=df_melted, x='Prioridade', y='Horas (Mediana)', hue='Métrica', palette=['#f39c12', '#2ecc71'])
sns.despine()

for container in ax.containers:
    ax.bar_label(container, fmt='%.1fh', padding=3, fontsize=10, fontweight='bold')

plt.title('Efetividade do SLA por Nível de Prioridade', fontweight='bold', pad=20)
plt.ylabel('Horas (Mediana)')
plt.xlabel('Prioridade')
plt.legend(title='Etapa', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}04_Prioridade_Tempo.png', dpi=300, transparent=True)
plt.close()

# =========================================================
# GRÁFICO 5: FUNIL DE SLA (Faixas de Tempo)
# =========================================================
print("5/5 - Gerando Gráfico: Funil de SLA...")
plt.figure(figsize=(10, 6))

condicoes = [
    (df['Tempo_Total_Atendimento'] <= 24),
    (df['Tempo_Total_Atendimento'] > 24) & (df['Tempo_Total_Atendimento'] <= 72),
    (df['Tempo_Total_Atendimento'] > 72) & (df['Tempo_Total_Atendimento'] <= 168),
    (df['Tempo_Total_Atendimento'] > 168)
]
escolhas = ['1. Até 24h', '2. 1 a 3 dias', '3. 3 a 7 dias', '4. Mais de 7 dias']
df['Faixa_SLA'] = np.select(condicoes, escolhas, default='Desconhecido')

contagem_sla = df['Faixa_SLA'].value_counts().sort_index()

# CORREÇÃO: Agora criamos um dicionário de cores fixas. Se o 'Desconhecido' aparecer, ele fica cinza.
paleta_cores = {
    '1. Até 24h': '#27ae60',
    '2. 1 a 3 dias': '#f1c40f',
    '3. 3 a 7 dias': '#e67e22',
    '4. Mais de 7 dias': '#c0392b',
    'Desconhecido': '#95a5a6' # Cinza
}

ax2 = sns.barplot(x=contagem_sla.values, y=contagem_sla.index, palette=paleta_cores, legend=False, hue=contagem_sla.index)
sns.despine()

total_chamados = len(df)
for i, v in enumerate(contagem_sla.values):
    porc = (v / total_chamados) * 100
    ax2.text(v + (total_chamados*0.01), i, f"{v} ({porc:.1f}%)", color='black', va='center', fontweight='bold')

plt.title('Distribuição de Chamados por Faixa de Solução', fontweight='bold', pad=20)
plt.xlabel('Quantidade de Chamados')
plt.ylabel('Faixa de Tempo')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}05_Faixas_SLA.png', dpi=300, transparent=True)
plt.close()

print(f"\n✅ Concluído! Vá na sua pasta do projeto e procure a nova pasta '{OUTPUT_DIR}' para ver as imagens.")