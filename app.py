"""
app.py — Escada do Saneamento · Dashboard interativo (Plotly Dash)

Produto de visualização aplicado à gestão municipal de saneamento (PFVD).
Dados: SNIS 2010-2021 via dankkom/snis-rawdata (pré-processados por data_prep.py).

Uso:
    python data_prep.py   # apenas na primeira vez
    python app.py         # abre em http://127.0.0.1:8050
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, dash_table

# ============================================================
# DADOS
# ============================================================
DADOS = pd.read_parquet("snis_slim.parquet")

UF_NOMES = {
    "AC": "Acre", "AL": "Alagoas", "AP": "Amapá", "AM": "Amazonas",
    "BA": "Bahia", "CE": "Ceará", "DF": "Distrito Federal",
    "ES": "Espírito Santo", "GO": "Goiás", "MA": "Maranhão",
    "MT": "Mato Grosso", "MS": "Mato Grosso do Sul", "MG": "Minas Gerais",
    "PA": "Pará", "PB": "Paraíba", "PR": "Paraná", "PE": "Pernambuco",
    "PI": "Piauí", "RJ": "Rio de Janeiro", "RN": "Rio Grande do Norte",
    "RS": "Rio Grande do Sul", "RO": "Rondônia", "RR": "Roraima",
    "SC": "Santa Catarina", "SP": "São Paulo", "SE": "Sergipe",
    "TO": "Tocantins",
}
REGIOES = {
    "Norte": ["AC", "AP", "AM", "PA", "RO", "RR", "TO"],
    "Nordeste": ["AL", "BA", "CE", "MA", "PB", "PE", "PI", "RN", "SE"],
    "Centro-Oeste": ["DF", "GO", "MT", "MS"],
    "Sudeste": ["ES", "MG", "RJ", "SP"],
    "Sul": ["PR", "RS", "SC"],
}
UF_REGIAO = {uf: reg for reg, ufs in REGIOES.items() for uf in ufs}

# Ponderação: indicadores urbanos pesam pela pop. urbana; totais, pela total
PESO = {
    "IN023_AE": "POP_URB", "IN047_AE": "POP_URB",
    "IN055_AE": "POP_TOT", "IN015_AE": "POP_TOT", "IN016_AE": "POP_TOT",
}

# Critérios de ranking oferecidos ao usuário
CRITERIOS = {
    "ciclo": "Ciclo completo (coleta × tratamento)",
    "IN023_AE": "Atendimento urbano de água",
    "IN047_AE": "Coleta de esgoto (urbano)",
    "IN016_AE": "Tratamento do esgoto coletado",
}

POP_MINIMA = 20_000  # elegibilidade do ranking (ver justificativa no rodapé)

# ---- Marco Legal do Saneamento (Lei 14.026/2020) ----
ANO_META_LEGAL = 2033        # prazo oficial da lei
ANO_HORIZONTE_PROJ = 2030    # até onde projetamos
JANELA_PROJ = 5              # regressão sobre os últimos N anos (tendência atual)
METAS_MARCO_LEGAL = {"Água": 99.0, "Ciclo completo": 90.0}

# ---- Paleta (mesma justificativa do notebook do PFVD) ----
# Sequencial de azuis: as 3 etapas da cascata são ORDINAIS.
AZ = ["#9CC9E8", "#3D8FD1", "#0B3C6B"]
DESTAQUE = "#E0712C"      
VERDE = "#1D6E4F"
VERMELHO = "#B23A2F"
CINZA = "#C7CBD1"
CINZA_TXT = "#5A6068"
FUNDO = "#FAFBFC"

FONTE = dict(family="Segoe UI, Helvetica, Arial, sans-serif",
             color="#22262A", size=13)


# ============================================================
# AGREGAÇÃO
# ============================================================
def media_ponderada(g: pd.DataFrame, ind: str) -> float:
    """Média do indicador ponderada pela população correspondente."""
    p = PESO[ind]
    s = g[[ind, p]].dropna()
    if s.empty or s[p].sum() == 0:
        return np.nan
    return float(np.average(s[ind], weights=s[p]))


def cascata(g: pd.DataFrame) -> dict:
    """As 3 etapas da Escada do Saneamento para um recorte qualquer."""
    agua = media_ponderada(g, "IN023_AE")
    coleta = media_ponderada(g, "IN047_AE")
    trat = media_ponderada(g, "IN016_AE")  # % do coletado que é tratado
    ciclo = coleta * trat / 100 if pd.notna(coleta) and pd.notna(trat) else np.nan
    return {"Água": agua, "Coleta": coleta, "Ciclo completo": ciclo}


def cascata_municipio(row: pd.Series) -> dict:
    """Cascata de um único município (sem ponderação — é 1 unidade)."""
    ciclo = (row.IN047_AE * row.IN016_AE / 100
             if pd.notna(row.IN047_AE) and pd.notna(row.IN016_AE) else np.nan)
    return {"Água": row.IN023_AE, "Coleta": row.IN047_AE,
            "Ciclo completo": ciclo}

def projetar_serie(anos: list, valores: list, anos_futuros: list,
                    janela: int = JANELA_PROJ) -> tuple:
    """
    Regressão linear (mínimos quadrados) sobre os últimos `janela` anos com
    dado válido. Janela curta = sensível à tendência recente, não à média
    histórica inteira (o ritmo de investimento de 2010 não representa o atual).
    Retorna (valores_projetados, taxa_anual_em_p.p.).
    """
    pares = [(a, v) for a, v in zip(anos, valores) if pd.notna(v)]
    if len(pares) < 2:
        return [np.nan] * len(anos_futuros), np.nan
    pares = pares[-janela:]
    xs = np.array([p[0] for p in pares])
    ys = np.array([p[1] for p in pares])
    taxa, intercepto = np.polyfit(xs, ys, 1)
    proj = np.clip(np.polyval([taxa, intercepto], anos_futuros), 0, 100)
    return proj.tolist(), float(taxa)


def avaliar_marco_legal(uf: str) -> dict:
    """
    Projeta Água e Ciclo completo até ANO_HORIZONTE_PROJ e estima em que ano
    (se algum) a meta do Marco Legal seria atingida no ritmo atual.
    """
    anos = sorted(DADOS.ano_referencia.unique())
    serie = {"Água": [], "Ciclo completo": []}
    for a in anos:
        g = DADOS[(DADOS.ano_referencia == a) & (DADOS.sigla_uf == uf)]
        c = cascata(g)
        serie["Água"].append(c["Água"])
        serie["Ciclo completo"].append(c["Ciclo completo"])

    anos_proj = list(range(anos[-1], ANO_HORIZONTE_PROJ + 1))
    resultado = {"anos_hist": anos, "anos_proj": anos_proj}
    for ind, meta in METAS_MARCO_LEGAL.items():
        proj, taxa = projetar_serie(anos, serie[ind], anos_proj)
        ultimo_valor = next((v for v in reversed(serie[ind]) if pd.notna(v)), np.nan)
        ano_meta = None
        if pd.notna(taxa) and pd.notna(ultimo_valor):
            if ultimo_valor >= meta:
                ano_meta = anos[-1]
            elif taxa > 0:
                ano_meta = int(round(anos[-1] + (meta - ultimo_valor) / taxa))
        resultado[ind] = {
            "serie": serie[ind], "proj": proj, "taxa_anual": taxa,
            "valor_2030": proj[-1] if proj else np.nan, "meta": meta,
            "ano_estimado_meta": ano_meta,
            "atinge_2033": ano_meta is not None and ano_meta <= ANO_META_LEGAL,
        }
    return resultado

def elegiveis(uf: str, ano: int) -> pd.DataFrame:
    """Municípios elegíveis ao ranking: pop. mínima + 3 indicadores presentes."""
    g = DADOS[(DADOS.sigla_uf == uf) & (DADOS.ano_referencia == ano)].copy()
    g = g[(g.POP_URB >= POP_MINIMA)
          & g.IN023_AE.notna() & g.IN047_AE.notna() & g.IN016_AE.notna()]
    g["ciclo"] = g.IN047_AE * g.IN016_AE / 100
    return g


# ============================================================
# FIGURAS
# ============================================================
def _layout_base(fig: go.Figure, altura: int) -> go.Figure:
    fig.update_layout(
        height=altura, font=FONTE,
        paper_bgcolor="white", plot_bgcolor="white",
        margin=dict(l=10, r=10, t=42, b=10),
        showlegend=False,
    )
    return fig


def fig_escada(vals: dict, titulo: str, altura: int = 260,
               compacta: bool = False) -> go.Figure:
    """
    Escada do Saneamento.
    Marca: barra | Canal: comprimento+posição (máx. precisão, Cleveland&McGill)
    Eixo x SEMPRE 0-100: em barras o comprimento É o dado (integridade visual).
    Cor sequencial: etapas ordinais. Laranja: só nas perdas (pré-atenção).
    Ordem fixa água->coleta->ciclo: a ordem é semântica, não estatística.
    """
    etapas = ["Água", "Coleta", "Ciclo completo"]
    v = [vals[e] for e in etapas]

    fig = go.Figure(go.Bar(
        x=v, y=etapas, orientation="h",
        marker_color=AZ, width=0.6,
        text=[f"<b>{x:.1f}%</b>" if pd.notna(x) else "s/ dado" for x in v],
        textposition="outside", cliponaxis=False,
        hovertemplate="%{y}: %{x:.1f}% da população urbana<extra></extra>",
    ))

    # Anotação das perdas entre etapas (o insight escrito, não deduzido — Knaflic)
    if not compacta:
        for i in range(2):
            if pd.notna(v[i]) and pd.notna(v[i + 1]):
                perda = v[i] - v[i + 1]
                fig.add_annotation(
                    x=(v[i] + v[i + 1]) / 2, y=i + 0.5,
                    text=f"<b>−{perda:.1f} p.p.</b>",
                    font=dict(color=DESTAQUE, size=12),
                    showarrow=False, bgcolor="white",
                )

    fig.update_yaxes(autorange="reversed", ticksuffix="  ")
    fig.update_xaxes(range=[0, 112], ticksuffix="%",
                     gridcolor="#EDEFF2", zeroline=False)
    fig.update_layout(title=dict(text=titulo, x=0, font=dict(size=15)))
    return _layout_base(fig, altura)


def fig_bullet(valor: float, media_uf: float, rotulo: str) -> go.Figure:
    """
    Bullet de contexto: onde o município está vs. média estadual.
    Marca: ponto | Referência: linha vertical, ajuda a analisar se é outlier ou é o padrão?
    """
    fig = go.Figure()
    fig.add_shape(type="line", x0=media_uf, x1=media_uf, y0=-0.5, y1=0.5,
                  line=dict(color=CINZA_TXT, width=2, dash="dot"))
    fig.add_trace(go.Scatter(
        x=[valor], y=[0], mode="markers",
        marker=dict(size=16, color=DESTAQUE, line=dict(color="white", width=2)),
        hovertemplate=f"{rotulo}: %{{x:.1f}}%<extra></extra>",
    ))
    fig.add_annotation(x=media_uf, y=0.85, text=f"média UF: {media_uf:.0f}%",
                       showarrow=False, font=dict(size=11, color=CINZA_TXT))
    fig.update_xaxes(range=[0, 105], ticksuffix="%", gridcolor="#EDEFF2",
                     zeroline=False)
    fig.update_yaxes(visible=False, range=[-1, 1.3])
    return _layout_base(fig, 110)


def fig_ranking(g: pd.DataFrame, criterio: str, uf: str) -> go.Figure:
    """
    Top e bottom 8 municípios pelo critério escolhido.
    Ordenação por valor: cria gradiente visual que dispensa ler rótulos.
    Barras cinza; extremos coloridos com parcimônia (verde/vermelho semânticos).
    """
    col = "ciclo" if criterio == "ciclo" else criterio
    g = g.dropna(subset=[col]).sort_values(col)
    n = len(g)
    if n > 16:
        g = pd.concat([g.head(8), g.tail(8)])

    cores = [CINZA] * len(g)
    if len(g) >= 2:
        cores[0] = VERMELHO      # pior
        cores[-1] = VERDE        # melhor

    fig = go.Figure(go.Bar(
        x=g[col], y=g.nome_municipio, orientation="h",
        marker_color=cores,
        hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
    ))
    sufixo = f" (top/bottom 8 de {n})" if n > 16 else f" ({n} elegíveis)"
    fig.update_layout(title=dict(
        text=f"Ranking — {CRITERIOS[criterio]}{sufixo}",
        x=0, font=dict(size=15)))
    fig.update_xaxes(range=[0, 105], ticksuffix="%", gridcolor="#EDEFF2",
                     zeroline=False)
    fig.update_yaxes(tickfont=dict(size=11))
    return _layout_base(fig, 420)


def fig_evolucao(uf: str, ano_sel: int) -> go.Figure:
    """
    Ciclo completo 2010-2021: UF (laranja=figura) vs. região (cinza=fundo).
    Rótulos diretos na ponta (sem legenda: menos carga cognitiva — Knaflic).
    Eixo y a partir de 0: o valor absoluto importa (é % de população).
    """
    regiao = UF_REGIAO[uf]
    ufs_reg = REGIOES[regiao]
    anos = sorted(DADOS.ano_referencia.unique())

    serie_uf, serie_reg = [], []
    for a in anos:
        g_uf = DADOS[(DADOS.ano_referencia == a) & (DADOS.sigla_uf == uf)]
        g_rg = DADOS[(DADOS.ano_referencia == a) & (DADOS.sigla_uf.isin(ufs_reg))]
        serie_uf.append(cascata(g_uf)["Ciclo completo"])
        serie_reg.append(cascata(g_rg)["Ciclo completo"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=anos, y=serie_reg, mode="lines", line=dict(color=CINZA, width=3),
        hovertemplate=f"{regiao} %{{x}}: %{{y:.1f}}%<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=anos, y=serie_uf, mode="lines+markers",
        line=dict(color=DESTAQUE, width=3), marker=dict(size=6),
        hovertemplate=f"{uf} %{{x}}: %{{y:.1f}}%<extra></extra>"))

    # Rótulo direto na ponta das linhas
    fig.add_annotation(x=anos[-1], y=serie_uf[-1], text=f"<b>{uf}</b>",
                       font=dict(color=DESTAQUE, size=13),
                       showarrow=False, xshift=24)
    fig.add_annotation(x=anos[-1], y=serie_reg[-1], text=regiao,
                       font=dict(color=CINZA_TXT, size=12),
                       showarrow=False, xshift=38)

    # Marca discreta do ano selecionado
    fig.add_vline(x=ano_sel, line=dict(color="#D9DDE2", width=1, dash="dot"))

    fig.update_layout(title=dict(
        text=f"Ciclo completo do esgoto: {uf} vs. {regiao} (2010–2021)",
        x=0, font=dict(size=15)))
    fig.update_xaxes(gridcolor="#EDEFF2", dtick=2)
    fig.update_yaxes(range=[0, 85], ticksuffix="%", gridcolor="#EDEFF2",
                     zeroline=False)
    return _layout_base(fig, 320)

def fig_comparacao_uf(casc_a: dict, uf_a: str, casc_b: dict, uf_b: str) -> go.Figure:
    """
    Compara a Escada de dois estados lado a lado, mesma escala 0-100.
    Reaproveita a marca (barra) da fig_escada — leitura já é familiar,
    só adiciona uma segunda série por cor (evita reinventar o padrão visual).
    """
    etapas = ["Água", "Coleta", "Ciclo completo"]
    v_a = [casc_a[e] for e in etapas]
    v_b = [casc_b[e] for e in etapas]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=etapas, x=v_a, orientation="h", name=uf_a, marker_color=DESTAQUE,
        text=[f"{x:.1f}%" if pd.notna(x) else "s/ dado" for x in v_a],
        textposition="outside", cliponaxis=False,
        hovertemplate=f"{uf_a} — %{{y}}: %{{x:.1f}}%<extra></extra>"))
    fig.add_trace(go.Bar(
        y=etapas, x=v_b, orientation="h", name=uf_b, marker_color=AZ[2],
        text=[f"{x:.1f}%" if pd.notna(x) else "s/ dado" for x in v_b],
        textposition="outside", cliponaxis=False,
        hovertemplate=f"{uf_b} — %{{y}}: %{{x:.1f}}%<extra></extra>"))

    fig.update_yaxes(autorange="reversed", ticksuffix="  ")
    fig.update_xaxes(range=[0, 112], ticksuffix="%", gridcolor="#EDEFF2", zeroline=False)
    fig.update_layout(title=dict(text=f"{uf_a} vs. {uf_b}", x=0, font=dict(size=15)),
                      barmode="group")
    fig = _layout_base(fig, 280)
    fig.update_layout(showlegend=True, legend=dict(orientation="h", y=1.15, x=0))
    return fig

def fig_marco_legal(marco: dict, uf: str) -> go.Figure:
    """
    Histórico (linha cheia) + projeção até 2030 (pontilhada = incerteza
    explícita) vs. metas do Marco Legal (prazo 2033).
    """
    fig = go.Figure()
    cores = {"Água": AZ[1], "Ciclo completo": DESTAQUE}
    anos_h, anos_p = marco["anos_hist"], marco["anos_proj"]

    for ind, cor in cores.items():
        d = marco[ind]
        fig.add_trace(go.Scatter(
            x=anos_h, y=d["serie"], mode="lines+markers", name=ind,
            line=dict(color=cor, width=3), marker=dict(size=5),
            hovertemplate=f"{ind} %{{x}}: %{{y:.1f}}%<extra></extra>"))
        fig.add_trace(go.Scatter(
            x=anos_p, y=d["proj"], mode="lines", showlegend=False,
            line=dict(color=cor, width=2, dash="dot"),
            hovertemplate=f"{ind} (projeção) %{{x}}: %{{y:.1f}}%<extra></extra>"))
        fig.add_hline(y=d["meta"], line=dict(color=cor, width=1, dash="dash"),
                      opacity=0.5, annotation_text=f"meta {ind.lower()}: {d['meta']:.0f}%",
                      annotation_font=dict(size=10, color=cor))

    fig.add_vline(x=ANO_META_LEGAL, line=dict(color=CINZA_TXT, width=1, dash="dot"),
                 annotation_text=f"prazo legal {ANO_META_LEGAL}",
                 annotation_font=dict(size=10, color=CINZA_TXT))
    fig.update_layout(title=dict(
        text=f"Marco Legal do Saneamento — projeção {uf} até {ANO_HORIZONTE_PROJ} "
             f"(tendência dos últimos {JANELA_PROJ} anos)",
        x=0, font=dict(size=15)),
        legend=dict(orientation="h", y=1.15, x=0))
    fig.update_xaxes(gridcolor="#EDEFF2", dtick=2,
                     range=[anos_h[0] - 0.5, ANO_META_LEGAL + 0.5])
    fig.update_yaxes(range=[0, 105], ticksuffix="%", gridcolor="#EDEFF2", zeroline=False)
    return _layout_base(fig, 320)


def fig_distribuicao(ele: pd.DataFrame, col: str, valor_uf: float,
                     val_melhor: float, val_pior: float) -> go.Figure:
    """
    Boxplot dos elegíveis: responde 'o pior é um outlier isolado, ou metade
    do estado está mal?' — pergunta que top/bottom 8 sozinho não responde.
    """
    fig = go.Figure()
    fig.add_trace(go.Box(x=ele[col], boxpoints="all", jitter=0.4,
                         marker=dict(color=CINZA, size=4, opacity=0.5),
                         line=dict(color=CINZA_TXT), fillcolor="white",
                         hovertemplate="%{x:.1f}%<extra></extra>", name=""))
    for val, cor, rotulo in [(val_melhor, VERDE, "melhor"), (val_pior, VERMELHO, "pior"),
                              (valor_uf, DESTAQUE, "média UF")]:
        if pd.notna(val):
            fig.add_vline(x=val, line=dict(color=cor, width=2, dash="dot"),
                         annotation_text=rotulo, annotation_font=dict(size=10, color=cor))
    fig.update_xaxes(range=[0, 105], ticksuffix="%", gridcolor="#EDEFF2", zeroline=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(title=dict(text="Distribuição dos municípios elegíveis",
                                  x=0, font=dict(size=13)))
    return _layout_base(fig, 160)

# ============================================================
# COMPONENTES DE LAYOUT
# ============================================================
def kpi_card(titulo, valor, delta):
    seta, cor = "", CINZA_TXT
    if pd.notna(delta):
        seta = f"{'▲' if delta >= 0 else '▼'} {abs(delta):.1f} p.p. vs. ano anterior"
        cor = VERDE if delta >= 0 else VERMELHO
    return html.Div(className="kpi", children=[
        html.Div(titulo, className="kpi-titulo"),
        html.Div(f"{valor:.1f}%" if pd.notna(valor) else "—", className="kpi-valor"),
        html.Div(seta, className="kpi-delta", style={"color": cor}),
    ])

def selo_marco_legal(marco: dict) -> html.Div:
    itens = []
    for ind in ("Água", "Ciclo completo"):
        d = marco[ind]
        ano_meta = d["ano_estimado_meta"]
        if ano_meta is None:
            texto, cor = "sem tendência suficiente para projetar", CINZA_TXT
        elif d["atinge_2033"]:
            texto, cor = f"no ritmo atual, atinge a meta em {ano_meta}", VERDE
        else:
            texto, cor = f"no ritmo atual, só atinge em {ano_meta} (após o prazo)", VERMELHO
        v2030 = f"{d['valor_2030']:.1f}%" if pd.notna(d["valor_2030"]) else "s/ dado"
        itens.append(html.Div(className="marco-item", children=[
            html.Span(f"{ind}: ", style={"fontWeight": 700}),
            html.Span(f"projeção {ANO_HORIZONTE_PROJ} = {v2030}"),
            html.Span(f" · {texto}", style={"color": cor}),
        ]))
    return html.Div(className="marco-selos", children=itens)

def justificativa(texto_md):
    """Painel recolhível com a fundamentação teórica do gráfico."""
    return html.Details(className="just", children=[
        html.Summary("ⓘ  Por que este gráfico é assim?"),
        dcc.Markdown(texto_md, className="just-corpo"),
    ])


JUST_ESCADA = """
| Decisão | Escolha | Fundamentação |
|---|---|---|
| Marca | Barra horizontal | Magnitudes comparáveis entre etapas; rótulos textuais leem-se sem rotação |
| Canal | Comprimento + posição | Topo da hierarquia de precisão de **Cleveland & McGill (1984)** — o objetivo é *quantificar* a perda |
| Escala | 0–100 fixo, início no zero | Em barras o comprimento **é** o dado; truncar distorceria a proporção (integridade visual) |
| Cor | Sequencial de azuis | As etapas são **ordinais** (água→coleta→tratamento); paleta qualitativa sugeriria independência — a leitura errada que o gráfico combate |
| Laranja | Exclusivo das perdas | Cor de uso único = propriedade **pré-atentiva**; o olho acha a perda antes de ler |
| Ordem | Fixa (semântica) | Reordenar por valor destruiria o significado de cascata |
| Anotação | Perda escrita em p.p. | **Knaflic**: o insight não deve exigir cálculo mental do leitor |
"""

JUST_CARDS = """
| Decisão | Escolha | Fundamentação |
|---|---|---|
| Mesma codificação nos 3 contextos | Escada idêntica p/ estado, melhor e pior | Lei da **similaridade** (Gestalt): aprende-se a ler uma vez, lê-se três |
| Eixos idênticos (0–100) | Compartilhados entre os cards | Sem eixo comum, o pior município *parece* igual ao melhor; com ele, a diferença vira **comprimento visível** |
| Verde/vermelho só no cabeçalho | Barras continuam azuis | A etapa da cadeia é a informação principal, não o "status" do município; semântica cultural (verde=bom) fica no rótulo |
| Bullet de contexto | Ponto vs. linha da média UF | Responde o que o número isolado não responde: *é outlier ou é o padrão do estado?* |
| Elegibilidade: pop ≥ 20 mil + 3 indicadores | Filtro antes do ranking | Municípios minúsculos com 100% declarado são ruído de reporte, não desempenho |
"""

JUST_RANKING = """
| Decisão | Escolha | Fundamentação |
|---|---|---|
| Ordenação por valor | Crescente | Cria gradiente visual que revela o ranking sem ler rótulos |
| Cinza como cor-base | Extremos em verde/vermelho | Parcimônia de destaque: se tudo é colorido, nada se destaca (**pré-atenção**) |
| Top/bottom 8 | Corte quando há muitos municípios | Barra ilegível não informa; o meio da distribuição está resumido na média do estado |
"""

JUST_EVOLUCAO = """
| Decisão | Escolha | Fundamentação |
|---|---|---|
| Marca | Linha | Dado temporal contínuo; lei da **continuidade** guia a leitura da trajetória |
| Figura-fundo | UF laranja, região cinza | Hierarquia visual: a série em foco é figura, o contexto é fundo (Gestalt) |
| Rótulo direto na ponta | Sem legenda separada | Elimina o vaivém olho–legenda (redução de carga cognitiva, **Knaflic**) |
| Eixo y desde 0 | Apesar de ser linha | O valor absoluto importa (% de população atendida), não só a tendência |
| ⚠ Limitação declarada | Nota abaixo do gráfico | O nº de municípios que reportam ao SNIS varia por ano; oscilações bruscas são artefato de reporte |
"""
JUST_MARCO_LEGAL = """
| Decisão | Escolha | Fundamentação |
|---|---|---|
| Janela de regressão | Últimos 5 anos, não a série toda | O ritmo de investimento de 2010 não representa a tendência atual; janela longa suavizaria demais uma eventual aceleração/desaceleração recente |
| Linha pontilhada na projeção | Estilo visual distinto do histórico | Projeção é incerteza explícita — o leitor não deve confundir com dado observado |
| Meta em linha horizontal | Não em barra separada | Comparação direta trajetória × meta no mesmo plano cartesiano |
| Prazo 2033 marcado, projeção até 2030 | Não estendemos a reta até 2033 | Regressão linear a 12 anos de distância tem erro grande demais para ser exibida como se fosse dado; o "ano estimado" é extrapolado à parte, só como texto |
"""

# ============================================================
# APP
# ============================================================
app = Dash(__name__, title="Escada do Saneamento")

app.index_string = """
<!DOCTYPE html>
<html><head>{%metas%}<title>{%title%}</title>{%favicon%}{%css%}
<style>
  body { background:#FAFBFC; margin:0; font-family:'Segoe UI',Helvetica,Arial,sans-serif; }
  .container { max-width:1180px; margin:0 auto; padding:22px 26px 60px; }
  h1 { color:#0B3C6B; font-size:26px; margin:6px 0 2px; }
  .subtitulo { color:#5A6068; font-size:14px; margin-bottom:18px; }
  .controles { display:flex; gap:18px; flex-wrap:wrap; background:white;
               border:1px solid #E4E7EB; border-radius:10px; padding:14px 18px;
               margin-bottom:18px; align-items:flex-end; }
  .controle { min-width:230px; flex:1; }
  .controle label { font-size:12px; font-weight:600; color:#5A6068;
                    text-transform:uppercase; letter-spacing:.04em; }
  .linha-kpi { display:flex; gap:14px; margin-bottom:18px; flex-wrap:wrap; }
  .kpi { flex:1; min-width:180px; background:white; border:1px solid #E4E7EB;
         border-radius:10px; padding:14px 18px; }
  .kpi-titulo { font-size:12px; font-weight:600; color:#5A6068;
                text-transform:uppercase; letter-spacing:.04em; }
  .kpi-valor { font-size:30px; font-weight:700; color:#0B3C6B; margin:2px 0; }
  .kpi-delta { font-size:12px; }
  .painel { background:white; border:1px solid #E4E7EB; border-radius:10px;
            padding:14px 16px 8px; margin-bottom:18px; }
  .dupla { display:flex; gap:18px; flex-wrap:wrap; }
  .dupla > .painel { flex:1; min-width:420px; }
  .card-mun-cab { display:flex; justify-content:space-between; align-items:baseline;
                  border-bottom:1px solid #EDEFF2; padding-bottom:8px; margin-bottom:4px; }
  .card-mun-nome { font-size:17px; font-weight:700; }
  .card-mun-info { font-size:12px; color:#5A6068; }
  .selo { font-size:11px; font-weight:700; padding:3px 10px; border-radius:20px; }
  .selo-melhor { background:#E8F4EA; color:#1D6E4F; }
  .selo-pior   { background:#FDECEA; color:#B23A2F; }
  .just { margin:2px 4px 10px; }
  .just summary { font-size:12px; color:#3D8FD1; cursor:pointer; }
  .just-corpo { font-size:12px; }
  .just-corpo table { border-collapse:collapse; }
  .just-corpo th, .just-corpo td { border:1px solid #E4E7EB; padding:4px 8px; text-align:left; }
  .nota { font-size:12px; color:#5A6068; margin:0 4px 8px; }
  .aviso { background:#FFF7ED; border:1px solid #F5D6B0; color:#8A5A1E;
           border-radius:8px; padding:10px 14px; font-size:13px; margin-bottom:18px; }
  .marco-selos { margin: 4px 0 10px; }
  .marco-item { font-size:13px; margin-bottom:4px; }
  .tabela-cab { display:flex; justify-content:space-between; align-items:center; margin-bottom:8px; }
  .tabela-titulo { font-size:13px; font-weight:700; color:#0B3C6B; }
  .tabela-busca { width:100%; padding:6px 10px; margin-bottom:10px; border:1px solid #E4E7EB;
                  border-radius:6px; font-size:13px; box-sizing:border-box; }
  .btn-export { background:#0B3C6B; color:white; border:none; border-radius:6px;
                padding:6px 14px; font-size:12px; cursor:pointer; }
  .btn-export:hover { background:#134A82; }
</style>
</head>
<body>{%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}</footer></body></html>
"""

app.layout = html.Div(className="container", children=[
    html.H1("Escada do Saneamento"),
    html.Div("Diagnóstico interativo de água e esgoto por estado — SNIS 2010–2021, "
             "médias ponderadas por população", className="subtitulo"),

    # ---- Controles ----
    html.Div(className="controles", children=[
        html.Div(className="controle", children=[
            html.Label("Estado"),
            dcc.Dropdown(
                id="uf",
                options=[{"label": f"{UF_NOMES[u]} ({u})", "value": u}
                         for u in sorted(UF_NOMES)],
                value="PB", clearable=False),
        ]),
        html.Div(className="controle", children=[
            html.Label("Critério do ranking (melhor/pior município)"),
            dcc.Dropdown(
                id="criterio",
                options=[{"label": v, "value": k} for k, v in CRITERIOS.items()],
                value="ciclo", clearable=False),
        ]),
        html.Div(className="controle", children=[
            html.Label("Comparar com outro estado (opcional)"),
            dcc.Dropdown(
                id="uf-comparacao",
                options=[{"label": "— Nenhuma —", "value": ""}] +
                        [{"label": f"{UF_NOMES[u]} ({u})", "value": u}
                         for u in sorted(UF_NOMES)],
                value="", clearable=False),
        ]),
        html.Div(className="controle", style={"flex": "2"}, children=[
            html.Label("Ano"),
            dcc.Slider(
                id="ano", min=2010, max=2021, step=1, value=2021,
                marks={a: str(a) for a in range(2010, 2022)}),
        ]),
    ]),

    html.Div(id="aviso-dados"),

    # ---- KPIs ----
    html.Div(id="linha-kpi", className="linha-kpi"),

    # ---- Escada do estado ----
    html.Div(className="painel", children=[
        dcc.Graph(id="g-escada-uf", config={"displayModeBar": False}),
        justificativa(JUST_ESCADA),
    ]),
    html.Div(id="painel-comparacao", className="painel", style={"display": "none"}, children=[
        dcc.Graph(id="g-comparacao-uf", config={"displayModeBar": False}),
    ]),
    # ---- Melhor x Pior ----
    html.Div(className="dupla", children=[
        html.Div(className="painel", children=[
            html.Div(id="cab-melhor"),
            dcc.Graph(id="g-escada-melhor", config={"displayModeBar": False}),
            dcc.Graph(id="g-bullet-melhor", config={"displayModeBar": False}),
        ]),
        html.Div(className="painel", children=[
            html.Div(id="cab-pior"),
            dcc.Graph(id="g-escada-pior", config={"displayModeBar": False}),
            dcc.Graph(id="g-bullet-pior", config={"displayModeBar": False}),
        ]),
    ]),
    html.Div(className="painel", children=[
        html.Div(id="nota-elegiveis", className="nota"),
        justificativa(JUST_CARDS),
    ]),

    # ---- Ranking + evolução ----
    html.Div(className="dupla", children=[
        html.Div(className="painel", children=[
            dcc.Graph(id="g-ranking", config={"displayModeBar": False}),
            dcc.Graph(id="g-distribuicao", config={"displayModeBar": False}),
            justificativa(JUST_RANKING),
        ]),
        html.Div(className="painel", children=[
            dcc.Graph(id="g-evolucao", config={"displayModeBar": False}),
            html.Div("⚠ O conjunto de municípios que reporta ao SNIS varia por ano; "
                     "oscilações bruscas podem ser artefato de reporte, não mudança real.",
                     className="nota"),
            justificativa(JUST_EVOLUCAO),
        ]),
    ]),
    html.Div(className="painel", children=[
        html.Div(id="marco-legal-status"),
        dcc.Graph(id="g-marco-legal", config={"displayModeBar": False}),
        justificativa(JUST_MARCO_LEGAL),
    ]),
    html.Div(className="painel", children=[
        html.Div(className="tabela-cab", children=[
            html.Span(id="tabela-titulo", className="tabela-titulo"),
            html.Button("⬇ Exportar CSV", id="btn-export-csv", n_clicks=0,
                       className="btn-export"),
        ]),
        dcc.Input(id="tabela-busca", type="text", placeholder="Buscar município…",
                 className="tabela-busca", debounce=True),
        dash_table.DataTable(
            id="tabela-dados",
            columns=[
                {"name": "Município", "id": "nome_municipio"},
                {"name": "Água (%)", "id": "IN023_AE", "type": "numeric", "format": {"specifier": ".1f"}},
                {"name": "Coleta (%)", "id": "IN047_AE", "type": "numeric", "format": {"specifier": ".1f"}},
                {"name": "Tratamento (%)", "id": "IN016_AE", "type": "numeric", "format": {"specifier": ".1f"}},
                {"name": "Ciclo (%)", "id": "ciclo", "type": "numeric", "format": {"specifier": ".1f"}},
                {"name": "Pop. urbana", "id": "POP_URB", "type": "numeric", "format": {"specifier": ",.0f"}},
            ],
            sort_action="native", page_size=10,
            style_table={"overflowX": "auto"},
            style_cell={"fontFamily": "Segoe UI, Helvetica, Arial, sans-serif",
                       "fontSize": "12px", "padding": "6px 10px"},
            style_header={"fontWeight": "700", "backgroundColor": "#F4F6F8",
                         "textTransform": "uppercase", "fontSize": "11px", "color": CINZA_TXT},
        ),
        dcc.Download(id="download-csv"),
    ]),
    html.Div(className="nota", children=[
        "Fonte: SNIS — Série Histórica municipal, via dankkom/snis-rawdata. ",
        f"Elegibilidade do ranking: população urbana ≥ {POP_MINIMA:,} hab. "
        "e os três indicadores da cascata preenchidos no ano.".replace(",", "."),
    ]),
])


# ============================================================
# CALLBACK
# ============================================================
def _fig_vazia(msg):
    fig = go.Figure()
    fig.add_annotation(text=msg, showarrow=False, font=dict(color=CINZA_TXT))
    fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
    return _layout_base(fig, 240)


def _cabecalho(row, tipo, criterio, pos, total):
    selo = ("🏆 MELHOR", "selo selo-melhor") if tipo == "melhor" \
        else ("⚠ PIOR", "selo selo-pior")
    return html.Div(className="card-mun-cab", children=[
        html.Div([
            html.Span(selo[0], className=selo[1]),
            html.Span(f"  ·  {CRITERIOS[criterio]}",
                      style={"fontSize": "12px", "color": CINZA_TXT}),
            html.Div(row.nome_municipio, className="card-mun-nome"),
        ]),
        html.Div(className="card-mun-info", children=[
            html.Div(f"{pos}º de {total} elegíveis"),
            html.Div(f"{row.POP_URB:,.0f} hab. urbanos".replace(",", ".")),
        ]),
    ])


@app.callback(
    Output("linha-kpi", "children"),
    Output("g-escada-uf", "figure"),
    Output("cab-melhor", "children"),
    Output("g-escada-melhor", "figure"),
    Output("g-bullet-melhor", "figure"),
    Output("cab-pior", "children"),
    Output("g-escada-pior", "figure"),
    Output("g-bullet-pior", "figure"),
    Output("g-ranking", "figure"),
    Output("g-evolucao", "figure"),
    Output("nota-elegiveis", "children"),
    Output("aviso-dados", "children"),
    Output("marco-legal-status", "children"),
    Output("g-marco-legal", "figure"),
    Output("g-distribuicao", "figure"),
    Output("painel-comparacao", "style"),
    Output("g-comparacao-uf", "figure"),
    Output("tabela-dados", "data"),
    Output("tabela-titulo", "children"),
    Input("uf", "value"),
    Input("ano", "value"),
    Input("criterio", "value"),
    Input("uf-comparacao", "value"),
    Input("tabela-busca", "value")
)
def atualizar(uf, ano, criterio,uf_comp, busca):
    g_uf = DADOS[(DADOS.sigla_uf == uf) & (DADOS.ano_referencia == ano)]
    g_ant = DADOS[(DADOS.sigla_uf == uf) & (DADOS.ano_referencia == ano - 1)]

    casc = cascata(g_uf)
    casc_ant = cascata(g_ant) if len(g_ant) else {k: np.nan for k in casc}

    # ---- KPIs ----
    kpis = [
        kpi_card("Água (urbano)", casc["Água"],
                 casc["Água"] - casc_ant["Água"]),
        kpi_card("Coleta de esgoto", casc["Coleta"],
                 casc["Coleta"] - casc_ant["Coleta"]),
        kpi_card("Ciclo completo", casc["Ciclo completo"],
                 casc["Ciclo completo"] - casc_ant["Ciclo completo"]),
    ]

    # ---- Escada do estado ----
    n_mun = len(g_uf)
    fig_uf = fig_escada(
        casc, f"Escada do Saneamento — {UF_NOMES[uf]} ({ano}) · "
              f"{n_mun} municípios no SNIS", altura=280)

    # ---- Melhor x Pior ----
    ele = elegiveis(uf, ano)
    col = "ciclo" if criterio == "ciclo" else criterio
    ele = ele.dropna(subset=[col])

    aviso = None
    if len(ele) == 0:
        vazia = _fig_vazia("Sem municípios elegíveis neste ano")
        bullet_vazio = _fig_vazia("")
        cab_m = cab_p = html.Div()
        fig_m = fig_p = vazia
        bul_m = bul_p = bullet_vazio
        nota = (f"Nenhum município de {UF_NOMES[uf]} atende aos "
                f"critérios de elegibilidade em {ano}.")
        aviso = html.Div(className="aviso", children=[
            f"⚠ {UF_NOMES[uf]} não tem municípios com dados completos em "
            f"{ano}. Experimente outro ano no slider."])
    elif len(ele) == 1:
        # Caso DF: só um município elegível — mostra o card único, sem "pior"
        unico = ele.iloc[0]
        media_uf = media_ponderada(g_uf, criterio) if criterio != "ciclo" \
            else casc["Ciclo completo"]
        cab_m = _cabecalho(unico, "melhor", criterio, 1, 1)
        fig_m = fig_escada(cascata_municipio(unico), "", altura=200,
                           compacta=True)
        bul_m = fig_bullet(unico["ciclo" if criterio == "ciclo" else criterio],
                           media_uf, CRITERIOS[criterio])
        cab_p = html.Div()
        fig_p = _fig_vazia("Único município elegível — não há comparação")
        bul_p = _fig_vazia("")
        nota = (f"Apenas 1 município elegível em {UF_NOMES[uf]} ({ano}); "
                f"o comparativo melhor × pior não se aplica.")
    else:
        ele_ord = ele.sort_values(col, ascending=False).reset_index(drop=True)
        melhor, pior = ele_ord.iloc[0], ele_ord.iloc[-1]
        media_uf = media_ponderada(g_uf, criterio) if criterio != "ciclo" \
            else casc["Ciclo completo"]

        cab_m = _cabecalho(melhor, "melhor", criterio, 1, len(ele_ord))
        cab_p = _cabecalho(pior, "pior", criterio, len(ele_ord), len(ele_ord))
        fig_m = fig_escada(cascata_municipio(melhor), "", altura=200,
                           compacta=True)
        fig_p = fig_escada(cascata_municipio(pior), "", altura=200,
                           compacta=True)
        val_m = melhor[col]
        val_p = pior[col]
        bul_m = fig_bullet(val_m, media_uf, CRITERIOS[criterio])
        bul_p = fig_bullet(val_p, media_uf, CRITERIOS[criterio])
        nota = (f"{len(ele_ord)} municípios elegíveis em {UF_NOMES[uf]} ({ano}) "
                f"— critério: {CRITERIOS[criterio]}.")

    # ---- Ranking + evolução ----
    fig_rank = fig_ranking(ele, criterio, uf) if len(ele) >= 2 \
        else _fig_vazia("Sem dados suficientes para o ranking")
    fig_evo = fig_evolucao(uf, ano)

    #---Marco Legal---
    marco = avaliar_marco_legal(uf)
    fig_marco = fig_marco_legal(marco, uf)
    status_marco = selo_marco_legal(marco)

    # ---- Distribuição ----
    if len(ele) >= 2:
        val_uf = casc.get("Ciclo completo") if criterio == "ciclo" else media_ponderada(g_uf, criterio)
        fig_dist = fig_distribuicao(ele, col, val_uf, ele_ord.iloc[0][col], ele_ord.iloc[-1][col])
    else:
        fig_dist = _fig_vazia("Elegíveis insuficientes para distribuição")

    if uf_comp:
        g_comp = DADOS[(DADOS.sigla_uf == uf_comp) & (DADOS.ano_referencia == ano)]
        fig_comp = fig_comparacao_uf(casc, uf, cascata(g_comp), uf_comp)
        style_comp = {"display": "block"}
    else:
        fig_comp = _fig_vazia("")
        style_comp = {"display": "none"}

    # ---- Tabela de dados brutos ----
    tabela_df = ele.copy()
    if busca:
        tabela_df = tabela_df[tabela_df.nome_municipio.str.contains(busca, case=False, na=False)]
    tabela_dados = tabela_df[["nome_municipio", "IN023_AE", "IN047_AE",
                              "IN016_AE", "ciclo", "POP_URB"]].to_dict("records")
    tabela_titulo = f"Dados brutos — {len(tabela_dados)} de {len(ele)} elegível(eis)"

    return (kpis, fig_uf, cab_m, fig_m, bul_m, cab_p, fig_p, bul_p,
            fig_rank, fig_evo, nota, aviso,
            status_marco, fig_marco, fig_dist,          
            style_comp, fig_comp, tabela_dados, tabela_titulo)

@app.callback(
    Output("download-csv", "data"),
    Input("btn-export-csv", "n_clicks"),
    State("uf", "value"),
    State("ano", "value"),
    State("tabela-busca", "value"),
    prevent_initial_call=True,
)
def exportar_csv(n_clicks, uf, ano, busca):
    ele = elegiveis(uf, ano)
    if busca:
        ele = ele[ele.nome_municipio.str.contains(busca, case=False, na=False)]
    cols = ["nome_municipio", "sigla_uf", "ano_referencia",
            "IN023_AE", "IN047_AE", "IN016_AE", "ciclo", "POP_URB"]
    nome_arquivo = f"saneamento_{uf}_{ano}.csv"
    # sep=";" e decimal="," porque é o padrão que o Excel em pt-BR espera
    return dcc.send_data_frame(ele[cols].to_csv, nome_arquivo, index=False,
                               sep=";", decimal=",")


if __name__ == "__main__":
    app.run(debug=True)  # use debug=False se a porta reclamar
