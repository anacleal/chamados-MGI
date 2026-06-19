import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import plotly.express as px
import ollama

# =====================================================================
# 1. INICIALIZAÇÃO (A Cozinha) - Roda apenas uma vez quando você liga o app
# =====================================================================

# Carrega a base de chamados que já passou pela sua modelagem de tópicos
df = pd.read_csv("data/Dataset_Rotulado_Final.csv")

# Calcula os maiores ofensores para o gráfico da tela inicial
df_grafico = df['nome'].value_counts().reset_index().head(10)
df_grafico.columns = ['Categoria', 'Volume']

app = dash.Dash(__name__)

# =====================================================================
# 2. LAYOUT (O Salão) - O que o usuário vê na tela
# =====================================================================
app.layout = html.Div(style={'fontFamily': 'Arial, sans-serif', 'padding': '20px'}, children=[
    html.H1("Painel de Inteligência Estratégica - MGI", style={'color': '#003366'}),

    # Seção 1: O Gráfico Estratégico
    html.Div([
        html.H3("Maiores Ofensores de Chamados"),
        dcc.Graph(
            id='grafico-ofensores',
            figure=px.bar(df_grafico, x='Volume', y='Categoria', orientation='h', color='Volume',
                          color_continuous_scale='Blues')
        )
    ], style={'border': '1px solid #ccc', 'padding': '20px', 'marginBottom': '30px'}),

    # Seção 2: O Chatbot Interno
    html.Div([
        html.H3("Assistente IA (Llama 3.1)"),
        dcc.Textarea(
            id='input-chamado',
            placeholder='Cole aqui a descrição do chamado para a IA analisar...',
            style={'width': '100%', 'height': '100px'}
        ),
        html.Button('Consultar IA', id='btn-consultar', n_clicks=0,
                    style={'marginTop': '10px', 'padding': '10px 20px', 'backgroundColor': '#003366', 'color': 'white',
                           'border': 'none'}),

        # Onde a resposta vai aparecer com um símbolo de carregamento (Loading)
        dcc.Loading(
            id="loading-ia",
            type="circle",
            children=html.Div(id='resposta-ia',
                              style={'marginTop': '20px', 'whiteSpace': 'pre-wrap', 'backgroundColor': '#f9f9f9',
                                     'padding': '15px'})
        )
    ], style={'border': '1px solid #ccc', 'padding': '20px'})
])


# =====================================================================
# 3. CALLBACKS (Os Garçons) - A inteligência interativa
# =====================================================================
@app.callback(
    Output('resposta-ia', 'children'),
    Input('btn-consultar', 'n_clicks'),
    State('input-chamado', 'value'),
    prevent_initial_call=True
)
def conversar_com_ia(n_clicks, texto_do_chamado):
    if not texto_do_chamado:
        return "Por favor, insira a descrição de um chamado."

    # Aqui o Dash envia o pedido para o Ollama rodando no Windows
    prompt_sistema = "Você é um assistente do MGI. Responda em Português de forma executiva."

    try:
        resposta = ollama.chat(
            model='llama3.1:8b-instruct-q4_K_S',
            messages=[
                {'role': 'system', 'content': prompt_sistema},
                {'role': 'user', 'content': texto_do_chamado}
            ],
            options={'temperature': 0.0}
        )
        return resposta['message']['content']
    except Exception as e:
        return f"Erro ao conectar com o Ollama: {str(e)}\n(Verifique se o Ollama está rodando no terminal)"


# =====================================================================
# 4. EXECUÇÃO
# =====================================================================
if __name__ == '__main__':
    app.run_server(debug=True)