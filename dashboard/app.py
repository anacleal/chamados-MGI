import dash
from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd

import data_loader as dl

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
# LAYOUT PRINCIPAL
# ============================================================
app.layout = html.Div(
    [
        dcc.Store(id="store-sistema", data=dl.SISTEMAS[0]),
        dcc.Store(id="store-topico", data=None),
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
# CALLBACKS
# ============================================================
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