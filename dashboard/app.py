"""
Dashboard de Análise de Chamados — MGI
========================================
Visualização para identificação de gargalos operacionais a partir da
modelagem de tópicos (BERTopic) e sumarização (Llama 3.1 via Ollama).

Rodar com:  python app.py
Acesse em:  http://127.0.0.1:8050
"""

import dash
from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd
import os
import threading
import sys
from pathlib import Path

import data_loader as dl

chatbot_path = Path(__file__).resolve().parent.parent / "chatbot"
if str(chatbot_path) not in sys.path:
    sys.path.append(str(chatbot_path))

recommender = None
chatbot_status = "Carregando"

def load_chatbot_background():
    import time
    # Pequena pausa para garantir que o Dash suba o servidor e renderize a página imediatamente
    time.sleep(2)
    
    global recommender, chatbot_status
    try:
        import os
        from pathlib import Path
        base_dir = Path(__file__).resolve().parent.parent
        path_topicos = base_dir / 'data' / 'chatbot' / 'df_topicos.csv'
        path_chamados = base_dir / 'data' / 'chatbot' / 'df_chamados.csv'
        
        if not (os.path.exists(path_topicos) and os.path.exists(path_chamados)):
            chatbot_status = "Construindo base de dados (Vetorizando chamados)..."
            import build_datasets  # type: ignore
            build_datasets.build_csvs()
            chatbot_status = "Carregando motor de busca..."
            
        from rag_data_loader import carregar_dados as carregar_dados_chatbot  # type: ignore
        from embeddings_manager import carregar_modelo_embedding, get_embeddings  # type: ignore
        from search_engine import SearchEngine  # type: ignore
        from recommender import RecommenderSystem  # type: ignore
        
        _df_topicos, _df_chamados = carregar_dados_chatbot()
        _emb_model = carregar_modelo_embedding()
        _emb_t, _emb_c = get_embeddings(_df_topicos, _df_chamados, _emb_model)
        _search_engine = SearchEngine(_df_topicos, _df_chamados, _emb_t, _emb_c, _emb_model)
        recommender = RecommenderSystem(_search_engine, _df_chamados)
        chatbot_status = "Pronto"
    except Exception as e:
        chatbot_status = f"Erro: {e}"

threading.Thread(target=load_chatbot_background, daemon=True).start()

# ============================================================
# CORES POR SISTEMA
# ============================================================
COR_SISTEMA = {
    "SIASS":  "#2B6CB0",
    "SIAPE":  "#2F7D6B",
    "SIGEPE": "#8B5CF6",
    "SOUGOV": "#B8860B",
    "TOTAIS": "#475569",
}

FONT_FAMILY = "IBM Plex Sans, sans-serif"
MONO_FAMILY = "IBM Plex Mono, monospace"

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    title="Chamados MGI — Painel de Gargalos",
    suppress_callback_exceptions=True,
)
server = app.server


@server.route("/bertopic-graph/<sistema>")
def serve_bertopic_graph(sistema: str):
    """
    [Não utilizado pelo mapa principal — mantido apenas como referência.]
    Serviria o intertopic_map.html nativo do BERTopic via <iframe>, caso
    seja necessário no futuro. O mapa de tópicos do dashboard usa as
    coordenadas extraídas desse mesmo HTML (ver data_loader.load_topic_coordinates),
    renderizadas com o estilo visual do dashboard via Plotly/dcc.Graph.
    """
    from flask import send_file, abort

    if sistema not in dl.SISTEMAS:
        abort(404)

    html_path = dl.get_intertopic_map_path(sistema)
    if html_path is None:
        abort(404)

    return send_file(html_path)


# ============================================================
# COMPONENTES — SIDEBAR
# ============================================================
def render_sidebar():
    items = []
    for sis in dl.SISTEMAS:
        items.append(
            html.Div(
                [
                    html.Span(className="system-dot", style={"backgroundColor": COR_SISTEMA[sis]}),
                    html.Span(sis),
                ],
                id={"type": "system-item", "index": sis},
                className="system-item" + (" active" if sis == dl.SISTEMAS[0] else ""),
                n_clicks=0,
            )
        )

    return html.Div(
        [
            html.Div(
                [
                    html.Div("PAINEL DE GARGALOS", className="brand-eyebrow"),
                    html.Div("Chamados de Suporte MGI", className="brand-title"),
                ],
                className="brand",
            ),
            html.Div(
                [
                    html.Span("Sistema", className="sidebar-label"),
                    html.Div(items, className="system-list", id="system-list"),
                ]
            ),
        ],
        className="sidebar",
    )


# ============================================================
# COMPONENTES — KPIs
# ============================================================
def render_kpis(sistema: str):
    tabela = dl.build_topic_table(sistema)
    n_topicos = len(tabela)
    n_docs = int(tabela["n_documentos"].sum())
    top_topico = tabela.sort_values("n_documentos", ascending=False).iloc[0] if not tabela.empty else None
    media_docs = round(n_docs / n_topicos, 1) if n_topicos else 0

    cards = [
        ("TÓPICOS IDENTIFICADOS", f"{n_topicos}", "clusters via BERTopic"),
        ("CHAMADOS ANALISADOS", f"{n_docs:,}".replace(",", "."), "documentos classificados"),
        ("MÉDIA POR TÓPICO", f"{media_docs}".replace(".", ","), "chamados / tópico"),
        ("MAIOR GARGALO",
         top_topico["titulo"] if top_topico is not None else "—",
         f"{int(top_topico['n_documentos'])} chamados" if top_topico is not None else ""),
    ]

    return html.Div(
        [
            html.Div(
                [
                    html.Div(label, className="kpi-label"),
                    html.Div(
                        value,
                        className="kpi-value-title" if i == 3 else "kpi-value",
                    ),
                    html.Div(delta, className="kpi-delta"),
                ],
                className="kpi-card",
            )
            for i, (label, value, delta) in enumerate(cards)
        ],
        className="kpi-row",
    )


# ============================================================
# COMPONENTES — MAPA INTERTÓPICOS
# (coordenadas reais extraídas do intertopic_map.html, desenhadas com o
#  estilo visual do dashboard — ver data_loader.load_topic_coordinates)
# ============================================================
def render_topic_map(sistema: str, topico_selecionado: int | None):
    coords = dl.load_topic_coordinates(sistema)
    tabela = dl.build_topic_table(sistema)
    cor = COR_SISTEMA.get(sistema, "#2B6CB0")

    if coords is None or coords.empty:
        # Fallback: layout circular simples baseado apenas em volume,
        # caso o intertopic_map.html ainda não tenha sido gerado para este sistema.
        import numpy as np
        n = len(tabela)
        if n == 0:
            fig = go.Figure()
            fig.update_layout(
                annotations=[dict(text="Sem dados de tópicos para este sistema.",
                                   showarrow=False, font=dict(size=13, color="#5B6776"))],
                paper_bgcolor="white", plot_bgcolor="white",
                height=420,
            )
            return fig
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        coords = pd.DataFrame({
            "topico": tabela["topico"],
            "x": np.cos(angles),
            "y": np.sin(angles),
            "n_documentos": tabela["n_documentos"],
        })

    merged = coords.merge(tabela[["topico", "titulo"]], on="topico", how="left")
    merged["titulo"] = merged["titulo"].fillna(merged["topico"].apply(lambda t: f"Tópico {t}"))

    import numpy as _np
    raw = merged["n_documentos"].clip(lower=1).values.astype(float)
    # Normaliza para [1, 10] com raiz quadrada para comprimir outliers,
    # depois usa sizemode="area" com sizeref pequeno para que as diferenças
    # de volume sejam claramente visíveis sem que os círculos se sobreponham.
    sqrt_vals = _np.sqrt(raw)
    vmin, vmax = sqrt_vals.min(), sqrt_vals.max()
    if vmax > vmin:
        norm = 1 + (sqrt_vals - vmin) / (vmax - vmin) * 9   # range [1, 10]
    else:
        norm = _np.full_like(sqrt_vals, 5.0)
    sizes = norm
    size_ref = 2.0 * norm.max() / (52 ** 2)

    is_selected = merged["topico"] == topico_selecionado
    line_widths = [3 if sel else 1 for sel in is_selected]
    line_colors = ["#1C2530" if sel else "rgba(255,255,255,0.7)" for sel in is_selected]
    opacities = [1.0 if (topico_selecionado is None or sel) else 0.35 for sel in is_selected]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=merged["x"], y=merged["y"],
        mode="markers+text",
        text=merged["topico"].astype(str),
        textposition="middle center",
        textfont=dict(size=11, color="white", family=FONT_FAMILY),
        marker=dict(
            size=sizes, sizemode="area", sizeref=size_ref, sizemin=14,
            color=cor, opacity=opacities,
            line=dict(width=line_widths, color=line_colors),
        ),
        customdata=merged[["titulo", "n_documentos", "topico"]],
        hovertemplate=(
            "<b>%{customdata[2]} · %{customdata[0]}</b><br>"
            "%{customdata[1]} chamados<extra></extra>"
        ),
    ))

    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        height=420,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(visible=False, zeroline=False),
        yaxis=dict(visible=False, zeroline=False),
        font=dict(family=FONT_FAMILY),
        showlegend=False,
        clickmode="event+select",
        uirevision=sistema,
    )
    return fig


# ============================================================
# COMPONENTES — CARTÃO DE DIAGNÓSTICO DO TÓPICO
# ============================================================
def render_topic_card(sistema: str, topico: int | None):
    if topico is None:
        return html.Div(
            "Selecione um tópico no ranking ou na tabela abaixo "
            "para ver o diagnóstico completo gerado pela sumarização.",
            className="topic-card-empty",
        )

    tabela = dl.build_topic_table(sistema)
    row = tabela[tabela["topico"] == topico]
    if row.empty:
        return html.Div("Tópico não encontrado.", className="topic-card-empty")

    row = row.iloc[0]
    keywords = [k.strip() for k in row["keywords"].split(",") if k.strip()]

    return html.Div(
        [
            html.Div(f"SISTEMA {sistema} · TÓPICO {topico}", className="topic-card-badge"),
            html.Div(row["titulo"], className="topic-card-title"),

            html.Div("Padrão Dominante", className="topic-card-section-label"),
            html.Div(
                row["padrao_dominante"] or "Resumo não disponível.",
                className="topic-card-section-text",
            ),

            html.Div("Impacto Operacional", className="topic-card-section-label"),
            html.Div(
                row["impacto_operacional"] or "Resumo não disponível.",
                className="topic-card-section-text topic-card-impact",
            ),

            html.Div(
                [html.Span(kw, className="keyword-chip") for kw in keywords] or
                [html.Span("sem palavras-chave", className="keyword-chip")],
                className="topic-card-keywords",
            ),
        ],
        className="topic-card",
    )


# ============================================================
# COMPONENTES — RANKING DE GARGALOS
# ============================================================
def render_ranking_chart(sistema: str, topico_selecionado: int | None):
    tabela = dl.build_topic_table(sistema).sort_values("n_documentos", ascending=True)
    cor = COR_SISTEMA.get(sistema, "#2B6CB0")

    if tabela.empty:
        fig = go.Figure()
        fig.update_layout(height=420, paper_bgcolor="white", plot_bgcolor="white")
        return fig

    colors = [
        "#1C2530" if t == topico_selecionado else cor
        for t in tabela["topico"]
    ]
    def _wrap_label(t, tit, max_chars=32):
        prefix = f"T{t}"
        if len(tit) <= max_chars:
            return f"{prefix} · {tit}"
        break_at = tit.rfind(" ", 0, max_chars)
        if break_at == -1:
            break_at = max_chars
        linha1 = tit[:break_at].rstrip()
        linha2 = tit[break_at:].strip()
        if len(linha2) > max_chars:
            linha2 = linha2[:max_chars - 1] + "…"
        return f"{prefix} · {linha1}<br>       {linha2}"

    labels = [_wrap_label(t, tit) for t, tit in zip(tabela["topico"], tabela["titulo"])]

    fig = go.Figure(go.Bar(
        x=tabela["n_documentos"],
        y=labels,
        orientation="h",
        marker=dict(color=colors),
        customdata=tabela[["topico"]],
        hovertemplate="<b>%{y}</b><br>%{x} chamados<extra></extra>",
    ))
    fig.update_layout(
        height=max(320, 56 * len(tabela)),
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=20, r=30, t=10, b=40),
        font=dict(family=FONT_FAMILY, size=13, color="#1C2530"),
        xaxis=dict(title="Nº de chamados", gridcolor="#E2E5EA", tickfont=dict(size=12)),
        yaxis=dict(title="", automargin=True, tickfont=dict(size=13), ticksuffix="      "),
        clickmode="event+select",
        uirevision=sistema,
    )
    return fig


# ============================================================
# COMPONENTES — EVOLUÇÃO TEMPORAL
# ============================================================
def render_timeline(sistema: str, topico_selecionado: int | None):
    tabela = dl.build_topic_table(sistema)
    cor_base = COR_SISTEMA.get(sistema, "#2B6CB0")

    if topico_selecionado is not None:
        agg = dl.build_monthly_evolution(sistema, topicos=[topico_selecionado])
    else:
        agg = dl.build_monthly_evolution(sistema)

    fig = go.Figure()

    if agg.empty:
        fig.update_layout(
            height=300, paper_bgcolor="white", plot_bgcolor="white",
            annotations=[dict(
                text="Sem coluna de data disponível para este sistema, ou nenhum chamado encontrado.",
                showarrow=False, font=dict(size=12, color="#5B6776"),
            )],
        )
        return fig

    if topico_selecionado is not None:
        fig.add_trace(go.Scatter(
            x=agg["mes_ano"], y=agg["n_chamados"],
            mode="lines+markers",
            line=dict(color=cor_base, width=2.5),
            marker=dict(size=6, color=cor_base),
            fill="tozeroy",
            fillcolor=cor_base.replace(")", ", 0.12)").replace("rgb", "rgba") if cor_base.startswith("rgb") else None,
            name=agg["titulo"].iloc[0] if not agg.empty else "",
            hovertemplate="%{x|%b/%Y}<br><b>%{y} chamados</b><extra></extra>",
        ))
    else:
        # Total agregado por mês, somando todos os tópicos
        total = agg.groupby("mes_ano")["n_chamados"].sum().reset_index()
        fig.add_trace(go.Scatter(
            x=total["mes_ano"], y=total["n_chamados"],
            mode="lines+markers",
            line=dict(color=cor_base, width=2.5),
            marker=dict(size=6, color=cor_base),
            fill="tozeroy",
            name="Total de chamados",
            hovertemplate="%{x|%b/%Y}<br><b>%{y} chamados</b><extra></extra>",
        ))

    fig.update_layout(
        height=300,
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=10, r=10, t=10, b=10),
        font=dict(family=FONT_FAMILY, size=12, color="#1C2530"),
        xaxis=dict(title="", gridcolor="#E2E5EA"),
        yaxis=dict(title="Chamados / mês", gridcolor="#E2E5EA"),
        showlegend=False,
    )
    return fig


# ============================================================
# COMPONENTES — TABELA COMPLETA
# ============================================================
def render_full_table(sistema: str):
    tabela = dl.build_topic_table(sistema).sort_values("n_documentos", ascending=False)
    display = tabela[["topico", "titulo", "n_documentos", "padrao_dominante", "impacto_operacional"]].copy()
    display.columns = ["Tópico", "Título", "Nº Chamados", "Padrão Dominante", "Impacto Operacional"]

    return dash_table.DataTable(
        id={"type": "full-table-datatable", "index": sistema},
        data=display.to_dict("records"),
        columns=[{"name": c, "id": c} for c in display.columns],
        cell_selectable=True,
        style_table={"overflowX": "auto"},
        style_cell={
            "fontFamily": "IBM Plex Sans, sans-serif",
            "fontSize": "12.5px",
            "padding": "10px 12px",
            "textAlign": "left",
            "whiteSpace": "normal",
            "height": "auto",
            "border": "none",
            "borderBottom": "1px solid #E2E5EA",
        },
        style_header={
            "fontFamily": "IBM Plex Mono, monospace",
            "fontSize": "10.5px",
            "letterSpacing": "0.05em",
            "textTransform": "uppercase",
            "color": "#5B6776",
            "backgroundColor": "#F7F8FA",
            "border": "none",
            "borderBottom": "1px solid #E2E5EA",
        },
        style_cell_conditional=[
            {"if": {"column_id": "Tópico"}, "width": "60px", "fontFamily": "IBM Plex Mono, monospace"},
            {"if": {"column_id": "Título"}, "width": "190px", "fontWeight": "600"},
            {"if": {"column_id": "Nº Chamados"}, "width": "100px", "fontFamily": "IBM Plex Mono, monospace"},
        ],
        style_data={"cursor": "pointer"},
        page_size=10,
        sort_action="native",
        filter_action="none",
    )


# ============================================================
# COMPONENTES DO CHATBOT (UI)
# ============================================================
def get_initial_message():
    global chatbot_status
    msgs = [
        html.Div(
            [
                html.B("SISTEMA DE BUSCA E RECOMENDAÇÃO DE SUPORTE"),
                html.Br(), html.Br(),
                "Descreva o problema ou use um dos comandos abaixo:", html.Br(),
                html.Code("/id <numero>"), " recomenda baseado em um ID existente", html.Br(),
                html.Code("/time <nome>"), " filtra recomendação para um time especifico", html.Br(),
                html.Code("/top <numero>"), " define quantos chamados exibir (ex: /top 5)", html.Br(),
                html.Code("/times"), " lista todos os times disponíveis", html.Br(),
                html.Br(),
                html.Em("A busca respeita a aba de sistema selecionada no painel.")
            ], 
            style={"marginBottom": "20px", "backgroundColor": "#EBF8FF", "padding": "10px", "borderRadius": "8px", "fontSize": "14px"}
        )
    ]
    
    # A mensagem de carregamento sempre faz parte do histórico inicial
    status_str = str(chatbot_status) if chatbot_status else ""
    msgs.append(
        html.Div(
            [html.B("Sistema: "), "Aguarde, inicializando o motor de busca... (Vetorizando chamados)" if "Construindo" in status_str else "Aguarde, inicializando o motor de busca... (Carregando)"], 
            style={"marginBottom": "20px", "color": "#C53030", "fontSize": "14px"}
        )
    )
    
    if status_str == "Pronto" or "Erro" in status_str:
        msgs.append(
            html.Div(
                [html.B("Sistema: "), "Motor de busca carregado e pronto para uso!" if status_str == "Pronto" else f"Falha: {status_str}"], 
                style={"marginBottom": "20px", "color": "#2F855A" if status_str == "Pronto" else "#C53030", "fontSize": "14px"}
            )
        )
        
    return msgs

chatbot_button = dbc.Button(
    "Recomendador Inteligente",
    id="open-chatbot",
    color="primary",
    style={
        "position": "fixed", 
        "bottom": "20px", 
        "right": "100px", 
        "borderRadius": "50px", 
        "zIndex": 9999, 
        "boxShadow": "0px 4px 10px rgba(0,0,0,0.2)"
    }
)

chatbot_offcanvas = dbc.Offcanvas(
    html.Div([
        html.Div(
            id="chatbot-messages", 
            children=get_initial_message(),
            style={
                "height": "75vh", 
                "overflowY": "auto", 
                "padding": "10px", 
                "backgroundColor": "#F7F8FA", 
                "borderRadius": "8px", 
                "marginBottom": "10px"
            }
        ),
        dbc.InputGroup([
            dbc.Input(id="chatbot-input", placeholder="Descreva o problema...", n_submit=0),
            dbc.Button("Enviar", id="chatbot-submit", color="primary")
        ])
    ]),
    id="chatbot-offcanvas",
    title="Recomendador Inteligente",
    is_open=False,
    placement="end",
    style={"width": "450px"}
)


# ============================================================
# LAYOUT PRINCIPAL
# ============================================================
app.layout = html.Div(
    [
        dcc.Store(id="store-sistema", data=dl.SISTEMAS[0]),
        dcc.Store(id="store-topico", data=None),
        dcc.Store(id="store-chat-top", data=3),
        dcc.Store(id="store-chat-time", data=None),
        dcc.Store(id="chatbot-announced", data=False),
        dcc.Interval(id="chatbot-load-interval", interval=2000, n_intervals=0),
        dcc.Store(id="chat-size", data=0),
        dbc.Modal(
            [
                dbc.ModalHeader(dbc.ModalTitle("Aviso de Lentidão")),
                dbc.ModalBody("O histórico do recomendador possui muitos chamados carregados e abri-lo agora pode causar travamentos no seu navegador. Deseja limpar o histórico antes de abrir?"),
                dbc.ModalFooter([
                    dbc.Button("Limpar e Abrir", id="btn-clear-chat", color="danger", className="ms-auto"),
                    dbc.Button("Abrir Mesmo Assim", id="btn-open-heavy-chat", color="secondary"),
                ]),
            ],
            id="heavy-chat-modal",
            is_open=False,
            centered=True,
        ),
        chatbot_button,
        chatbot_offcanvas,
        html.Div(
            [
                render_sidebar(),
                html.Div(
                    [
                        html.Div(
                            [
                                html.H1("Visão geral do sistema", className="page-title", id="page-title"),
                            ],
                            className="page-header",
                        ),
                        html.Div(id="kpi-row"),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.H3("Mapa de tópicos", className="panel-title"),
                                                html.Span(
                                                    "tamanho = volume de chamados | distância = similaridade · clique para detalhar",
                                                    className="panel-note",
                                                ),
                                            ],
                                            className="panel-header",
                                        ),
                                        dcc.Graph(id="topic-map", config={"displayModeBar": False}),
                                    ],
                                    className="panel",
                                ),
                                html.Div(
                                    [
                                        html.Div(id="topic-card"),
                                    ],
                                    className="panel",
                                ),
                            ],
                            className="two-col",
                        ),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.H3("Ranking de gargalos por volume", className="panel-title"),
                                            ],
                                            className="panel-header",
                                        ),
                                        dcc.Graph(id="ranking-chart", config={"displayModeBar": False}),
                                    ],
                                    className="panel",
                                ),
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.H3("Evolução mensal", className="panel-title"),
                                                html.Span(id="timeline-note", className="panel-note"),
                                            ],
                                            className="panel-header",
                                        ),
                                        dcc.Graph(id="timeline-chart", config={"displayModeBar": False}),
                                    ],
                                    className="panel",
                                ),
                            ],
                            style={"display": "grid", "gridTemplateColumns": "1fr", "gap": "0"},
                        ),
                        html.Div(
                            [
                                html.Div(id="full-table"),
                            ],
                            className="panel",
                        ),
                    ],
                    className="main-content",
                ),
            ],
            className="app-shell",
        ),
    ]
)


# ============================================================
# CALLBACKS DO CHATBOT E CLIENTSIDE
# ============================================================
app.clientside_callback(
    """
    function(children) {
        setTimeout(function(){
            var objDiv = document.getElementById('chatbot-messages');
            if (objDiv) {
                objDiv.scrollTop = objDiv.scrollHeight;
            }
        }, 100);
        return window.dash_clientside.no_update;
    }
    """,
    Output("chatbot-messages", "id"),
    Input("chatbot-messages", "children")
)

@app.callback(
    Output("open-chatbot", "style"),
    Output("chatbot-submit", "style"),
    Input("store-sistema", "data")
)
def atualizar_cores_chatbot(sistema):
    cor = COR_SISTEMA.get(sistema, "#2B6CB0")
    btn_style = {
        "position": "fixed", 
        "bottom": "20px", 
        "left": "20px", 
        "borderRadius": "50px", 
        "zIndex": 9999, 
        "boxShadow": "0px 4px 10px rgba(0,0,0,0.2)",
        "backgroundColor": cor,
        "borderColor": cor,
        "color": "white"
    }
    submit_style = {
        "backgroundColor": cor,
        "borderColor": cor,
        "color": "white"
    }
    return btn_style, submit_style

@app.callback(
    Output("chatbot-offcanvas", "is_open"),
    Output("heavy-chat-modal", "is_open"),
    Input("open-chatbot", "n_clicks"),
    Input("btn-open-heavy-chat", "n_clicks"),
    Input("btn-clear-chat", "n_clicks"),
    State("chatbot-offcanvas", "is_open"),
    State("chat-size", "data"),
    prevent_initial_call=True
)
def handle_open_chat(n_open, n_proceed, n_clear, is_open, chat_size):
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update, dash.no_update
        
    trigger_id = ctx.triggered[0]["prop_id"]
    
    if "open-chatbot" in trigger_id:
        if not is_open and chat_size and chat_size > 50000:
            # Ao invés de abrir o chat direto, mostra o modal
            return False, True
        else:
            return not is_open, False
            
    if "btn-open-heavy-chat" in trigger_id:
        return True, False
        
    if "btn-clear-chat" in trigger_id:
        return True, False
        
    return dash.no_update, dash.no_update



@app.callback(
    Output("chatbot-messages", "children", allow_duplicate=True),
    Output("chatbot-load-interval", "disabled"),
    Output("chatbot-announced", "data"),
    Input("chatbot-load-interval", "n_intervals"),
    State("chatbot-messages", "children"),
    State("chatbot-announced", "data"),
    prevent_initial_call=True
)
def update_chatbot_status(n, chat_history, announced):
    global chatbot_status
    if announced:
        return dash.no_update, True, dash.no_update
        
    if not chat_history:
        chat_history = get_initial_message()
        
    if "Construindo" in chatbot_status or "Carregando" in chatbot_status:
        # get_initial_message já inclui a mensagem de carregamento, não precisamos fazer nada
        return dash.no_update, False, False
        
    if chatbot_status == "Pronto" or "Erro" in chatbot_status:
        
        # Para evitar duplicação, verificamos se a última string diz "pronto para uso"
        ultimo_texto = str(chat_history[-1]) if chat_history else ""
        if "pronto para uso" not in ultimo_texto and "Falha" not in ultimo_texto:
            msg = html.Div(
                [html.B("Sistema: "), "Motor de busca carregado e pronto para uso!" if chatbot_status == "Pronto" else f"Falha: {chatbot_status}"], 
                style={"marginBottom": "20px", "color": "#2F855A" if chatbot_status == "Pronto" else "#C53030", "fontSize": "14px"}
            )
            chat_history.append(msg)
            return chat_history, True, True
            
        return dash.no_update, True, True
        
    return dash.no_update, False, False

@app.callback(
    Output("chatbot-messages", "children"),
    Output("chatbot-input", "value"),
    Output("store-chat-top", "data"),
    Output("store-chat-time", "data"),
    Output("chat-size", "data"),
    Input("chatbot-submit", "n_clicks"),
    Input("chatbot-input", "n_submit"),
    Input("store-sistema", "data"),
    State("chatbot-input", "value"),
    State("chatbot-messages", "children"),
    State("store-chat-top", "data"),
    State("store-chat-time", "data"),
    prevent_initial_call=True
)
def chat_interaction(n_clicks, n_submit, sistema_input, user_text, chat_history, chat_top, chat_time):
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
        
    trigger_id = ctx.triggered[0]["prop_id"]
    
    # Se o sistema mudou (mudança de aba), reseta tudo!
    if "store-sistema" in trigger_id:
        return get_initial_message(), "", 3, None, 0
        
    if not user_text:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
    
    if chat_history is None or len(chat_history) == 0:
        chat_history = get_initial_message()
        
    chat_history.append(
        html.Div(
            [html.B("Você: "), user_text], 
            style={"marginBottom": "10px", "textAlign": "right", "color": "#2B6CB0", "fontSize": "14px"}
        )
    )
    
    global recommender, chatbot_status
    if recommender is None:
        chat_history.append(
            html.Div(
                [
                    html.B("Recomendador: "), 
                    html.Span("Aguarde! O motor (RAG) ainda está sendo carregado no servidor.", style={"color": "#C53030"}),
                    html.Br(),
                    f"Status: {chatbot_status}. Tente novamente em alguns segundos."
                ], 
                style={"marginBottom": "20px", "backgroundColor": "#FED7D7", "padding": "10px", "borderRadius": "8px", "fontSize": "14px"}
            )
        )
        return chat_history, "", chat_top, chat_time
        
    user_text_lower = user_text.lower()
    bot_response = None
    chamados = []
    
    # Processa comandos
    if user_text_lower == '/times':
        times_disponiveis = recommender.df_chamados['Time'].value_counts()
        lista_times = [html.Li(f"{time} ({count} chamados)") for time, count in times_disponiveis.items() if str(time).strip()]
        bot_response = html.Div(
            [html.B("Recomendador: "), "Times disponíveis na base:", html.Ul(lista_times)], 
            style={"marginBottom": "20px", "fontSize": "14px"}
        )
        
    elif user_text_lower.startswith('/top '):
        novo_top = user_text_lower[5:].strip()
        if novo_top.isdigit() and int(novo_top) > 0:
            chat_top = int(novo_top)
            bot_response = html.Div(
                [html.B("Recomendador: "), f"Configuração atualizada! O sistema agora listará as {chat_top} melhores recomendações."], 
                style={"marginBottom": "20px", "fontSize": "14px", "color": "#2F855A"}
            )
        else:
            bot_response = html.Div(
                [html.B("Recomendador: "), "Erro: Por favor, informe um número válido maior que zero. (Ex: /top 5)"], 
                style={"marginBottom": "20px", "fontSize": "14px", "color": "#C53030"}
            )
            
    elif user_text_lower.startswith('/time '):
        novo_time = user_text_lower[6:].strip()
        chat_time = novo_time if novo_time else None
        if chat_time:
            bot_response = html.Div(
                [html.B("Recomendador: "), f"Filtro de time aplicado: '{chat_time}'. As buscas agora priorizarão esse time."], 
                style={"marginBottom": "20px", "fontSize": "14px", "color": "#2F855A"}
            )
        else:
            bot_response = html.Div(
                [html.B("Recomendador: "), "Filtro de time removido."], 
                style={"marginBottom": "20px", "fontSize": "14px"}
            )

    elif user_text_lower.startswith('/id '):
        id_chamado = user_text[4:].strip()
        chamados_similares = recommender.buscar_por_id(id_chamado, top_k=chat_top)
        chamados = chamados_similares
        if not chamados:
            bot_response = html.Div(
                [html.B("Recomendador: "), f"ID {id_chamado} não encontrado."], 
                style={"marginBottom": "20px", "fontSize": "14px"}
            )
            
    else:
        resultado = recommender.recomendar_solucao(user_text, top_k=chat_top, filtro_sistema=sistema_input, filtro_time=chat_time)
        chamados = resultado.get("chamados_recomendados", [])

    if bot_response is None:
        if not chamados:
            msg_filtro = f" no sistema '{sistema_input}'"
            if chat_time: msg_filtro += f" e time '{chat_time}'"
            bot_response = html.Div(
                [html.B("Recomendador: "), f"Não encontrei chamados similares{msg_filtro}."], 
                style={"marginBottom": "20px", "fontSize": "14px"}
            )
        else:
            solucoes = []
            for i, c in enumerate(chamados, 1):
                titulo = str(c.get('Título', 'Sem título'))
                acao = str(c.get('Última ação de acompanhamento', 'Sem ação'))
                if acao == "nan":
                    acao = "Sem última ação de acompanhamento documentada."
                desc = str(c.get('Descrição do chamado', 'Sem descrição'))
                id_ = str(c.get('Id', 'Sem ID'))
                time_ = str(c.get('Time', 'Sem time'))
                score = c.get('score_similaridade', 0)
                
                resumo_acao = acao[:100] + "..." if len(acao) > 100 else acao
                
                card = html.Div([
                    html.Details([
                        html.Summary(
                            html.Strong(f"[{i}] {titulo} (Score: {score:.0%})", style={"cursor": "pointer", "color": "#2B6CB0"}),
                            style={"outline": "none"}
                        ),
                        html.Div([
                            html.P([html.B("Time: "), time_], style={"margin": "2px 0"}),
                            html.Hr(style={"margin": "8px 0"}),
                            html.P([html.B("ID: "), id_], style={"margin": "2px 0"}),
                            html.Hr(style={"margin": "8px 0"}),
                            html.P([html.B("Descrição do Problema:"), html.Br(), desc], style={"margin": "2px 0"}),
                            html.Hr(style={"margin": "8px 0"}),
                            html.P([html.B("Solução (Ação de Acompanhamento):"), html.Br(), acao], style={"margin": "2px 0"})
                        ], style={"padding": "10px", "backgroundColor": "#F9FAFB", "border": "1px solid #E2E8F0", "borderRadius": "5px", "marginTop": "8px", "fontSize": "13px"})
                    ]),
                    html.Div(html.Em(resumo_acao), style={"marginTop": "5px", "color": "#718096", "fontSize": "13px"})
                ], style={"padding": "12px", "backgroundColor": "#fff", "border": "1px solid #E2E8F0", "borderRadius": "8px", "marginBottom": "10px", "boxShadow": "0 1px 3px rgba(0,0,0,0.05)"})
                
                solucoes.append(card)
                
            bot_response = html.Div([
                html.B("Recomendador: Encontrei as seguintes soluções históricas:"),
                html.Div(solucoes, style={"marginTop": "10px"})
            ], style={"marginBottom": "20px", "fontSize": "14px"})
            
    chat_history.append(bot_response)
    
    # Calculo matemático do peso do histórico
    peso_historico = len(str(chat_history))
        
    return chat_history, "", chat_top, chat_time, peso_historico

@app.callback(
    Output("chatbot-messages", "children", allow_duplicate=True),
    Output("chat-size", "data", allow_duplicate=True),
    Input("btn-clear-chat", "n_clicks"),
    prevent_initial_call=True
)
def clear_chat_history(submit_n_clicks):
    if submit_n_clicks:
        return get_initial_message(), 0
    return dash.no_update, dash.no_update

def _extrair_topico_customdata(point: dict, indice: int | None = None):
    """
    Extrai o tópico de um ponto clicado no Plotly, de forma defensiva.
    customdata pode chegar como escalar, lista, ou estar ausente
    dependendo da versão do plotly.js e de qual elemento foi clicado.
    """
    cd = point.get("customdata")
    if cd is None:
        return None
    if isinstance(cd, (list, tuple)):
        if indice is not None and indice < len(cd):
            return cd[indice]
        return cd[0] if cd else None
    # customdata veio como escalar direto
    return cd


@app.callback(
    Output("store-sistema", "data"),
    Output("store-topico", "data"),
    Input({"type": "system-item", "index": dash.ALL}, "n_clicks"),
    Input("topic-map", "clickData"),
    Input("ranking-chart", "clickData"),
    Input({"type": "full-table-datatable", "index": dash.ALL}, "active_cell"),
    State({"type": "full-table-datatable", "index": dash.ALL}, "data"),
    State("store-sistema", "data"),
    prevent_initial_call=True,
)
def atualizar_selecao(n_clicks_list, map_click, ranking_click, active_cells, table_data_list, sistema_atual):
    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update, dash.no_update

    trigger_id = ctx.triggered[0]["prop_id"]

    # Troca de sistema -> reseta o tópico selecionado
    if "system-item" in trigger_id and any(n_clicks_list):
        import json as _json
        sistema = _json.loads(trigger_id.split(".")[0])["index"]
        return sistema, None

    # Clique no mapa de tópicos
    if "topic-map" in trigger_id and map_click:
        points = map_click.get("points") or []
        if points:
            topico = _extrair_topico_customdata(points[0], indice=2)
            if topico is not None:
                return dash.no_update, topico
        return dash.no_update, dash.no_update

    # Clique no ranking
    if "ranking-chart" in trigger_id and ranking_click:
        points = ranking_click.get("points") or []
        if points:
            topico = _extrair_topico_customdata(points[0], indice=0)
            if topico is not None:
                return dash.no_update, topico
        return dash.no_update, dash.no_update

    # Clique em uma linha da tabela completa
    if "full-table-datatable" in trigger_id and active_cells:
        for cell, data in zip(active_cells, table_data_list):
            if cell:
                row = data[cell["row"]]
                return dash.no_update, row["Tópico"]

    return dash.no_update, dash.no_update


@app.callback(
    Output("system-list", "children"),
    Input("store-sistema", "data"),
)
def atualizar_destacado(sistema_ativo):
    items = []
    for sis in dl.SISTEMAS:
        items.append(
            html.Div(
                [
                    html.Span(className="system-dot", style={"backgroundColor": COR_SISTEMA[sis]}),
                    html.Span(sis),
                ],
                id={"type": "system-item", "index": sis},
                className="system-item" + (" active" if sis == sistema_ativo else ""),
                n_clicks=0,
            )
        )
    return items





@app.callback(
    Output("page-title", "children"),
    Output("kpi-row", "children"),
    Output("topic-map", "figure"),
    Output("topic-card", "children"),
    Output("ranking-chart", "figure"),
    Output("timeline-chart", "figure"),
    Output("timeline-note", "children"),
    Output("full-table", "children"),
    Input("store-sistema", "data"),
    Input("store-topico", "data"),
)
def atualizar_pagina(sistema, topico):
    titulo_pagina = f"Sistema {sistema}" + (f" · Tópico {topico}" if topico is not None else " · Visão geral")
    timeline_note = (
        f"filtrado pelo tópico {topico}" if topico is not None else "todos os tópicos somados"
    )

    return (
        titulo_pagina,
        render_kpis(sistema),
        render_topic_map(sistema, topico),
        render_topic_card(sistema, topico),
        render_ranking_chart(sistema, topico),
        render_timeline(sistema, topico),
        timeline_note,
        render_full_table(sistema),
    )


if __name__ == "__main__":
    app.run(debug=True, port=8050)