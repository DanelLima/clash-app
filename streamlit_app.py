"""
app.py
------
Aplicação Streamlit para análise de partidas de Clash Royale.

Execução:
    streamlit run app.py

Dependências:
    streamlit, pandas, matplotlib, scikit-learn
    + módulos locais: preprocessing, feature_engineering, analysis, visualization, model
"""

from __future__ import annotations

import io
import time

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Configuração da página — deve ser a PRIMEIRA chamada Streamlit
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Clash Royale Analytics",
    page_icon="👑",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS global — tema Clash Royale
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* ── Importação de fonte ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* ── Reset e base ── */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Fundo principal ── */
.stApp { background: #0B1623; }
section[data-testid="stSidebar"] { background: #0D1F35 !important; border-right: 1px solid #1A3050; }

/* ── Sidebar: título ── */
.sidebar-brand {
    text-align: center;
    padding: 1.2rem 0 1rem;
    border-bottom: 1px solid #1A3050;
    margin-bottom: 1rem;
}
.sidebar-brand h1 { font-size: 1.4rem; font-weight: 800; color: #FFD740; margin: 0; letter-spacing: .5px; }
.sidebar-brand p  { font-size: .75rem; color: #5A7A99; margin: .2rem 0 0; }

/* ── Cards de seção ── */
.cr-card {
    background: #111F33;
    border: 1px solid #1A3050;
    border-radius: 14px;
    padding: 1.5rem 1.75rem;
    margin-bottom: 1.25rem;
}
.cr-card-accent { border-left: 4px solid #FFD740; }
.cr-card-blue   { border-left: 4px solid #4FC3F7; }
.cr-card-purple { border-left: 4px solid #AB47BC; }
.cr-card-red    { border-left: 4px solid #EF5350; }

/* ── Hero da Home ── */
.hero-wrap {
    background: linear-gradient(135deg, #0D1F35 0%, #102840 50%, #0B1623 100%);
    border: 1px solid #1E3A55;
    border-radius: 18px;
    padding: 3rem 2.5rem;
    text-align: center;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-wrap::before {
    content: '';
    position: absolute;
    top: -60px; left: -60px;
    width: 240px; height: 240px;
    background: radial-gradient(circle, rgba(255,215,64,.08) 0%, transparent 70%);
    pointer-events: none;
}
.hero-crown  { font-size: 4rem; line-height: 1; margin-bottom: .5rem; }
.hero-title  { font-size: 2.6rem; font-weight: 800; color: #FFD740; margin: 0; letter-spacing: -1px; }
.hero-sub    { font-size: 1.05rem; color: #7A9CBF; margin: .6rem 0 0; }

/* ── Divisor com label ── */
.section-label {
    display: flex; align-items: center; gap: .75rem;
    margin: 1.8rem 0 1rem;
}
.section-label span.bar {
    flex: 1; height: 1px; background: #1A3050;
}
.section-label span.txt {
    font-size: .7rem; font-weight: 700; letter-spacing: 1.5px;
    color: #4A6A88; text-transform: uppercase;
}

/* ── Métricas customizadas ── */
[data-testid="metric-container"] {
    background: #111F33;
    border: 1px solid #1A3050;
    border-radius: 12px;
    padding: .9rem 1.1rem .7rem;
}
[data-testid="metric-container"] label { color: #5A7A99 !important; font-size: .78rem !important; font-weight: 600; letter-spacing: .5px; text-transform: uppercase; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color: #FFD740 !important; font-size: 1.8rem !important; font-weight: 800; }
[data-testid="metric-container"] [data-testid="stMetricDelta"] { font-size: .8rem !important; }

/* ── Tabelas ── */
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

/* ── Botões ── */
.stButton button {
    background: linear-gradient(135deg, #1E6FA8, #1A5A8A);
    color: #E8EDF2;
    border: 1px solid #2A7FC0;
    border-radius: 8px;
    font-weight: 600;
    font-size: .85rem;
    padding: .5rem 1.2rem;
    transition: all .2s;
}
.stButton button:hover { background: linear-gradient(135deg, #2A80C0, #1E6FA8); border-color: #4FC3F7; }

/* ── Upload area ── */
[data-testid="stFileUploader"] {
    background: #111F33;
    border: 2px dashed #1E3A55;
    border-radius: 12px;
    padding: 1rem;
}

/* ── Selectbox / radio lateral ── */
.stRadio label { color: #A0BDD4 !important; font-size: .88rem !important; }
.stRadio [data-testid="stMarkdownContainer"] p { color: #A0BDD4; }

/* ── Expanders ── */
.streamlit-expanderHeader { background: #111F33 !important; color: #A0BDD4 !important; border-radius: 8px; }

/* ── Info / warning boxes ── */
.stInfo    { background: #0D2A44 !important; border-left-color: #4FC3F7 !important; color: #A0BDD4 !important; }
.stWarning { background: #2A1F05 !important; border-left-color: #FFD740 !important; color: #C8A840 !important; }
.stSuccess { background: #0A2A1A !important; border-left-color: #66BB6A !important; color: #80C880 !important; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Importações dos módulos do projeto (com feedback amigável)
# ---------------------------------------------------------------------------
try:
    from src.preprocessing import preprocess
    from src.feature_engineering import create_features
    import src.analysis as an
    import src.visualization as viz
    from src.model import train_model, FEATURES as MODEL_FEATURES
    MODULES_OK = True
except ImportError as e:
    MODULES_OK = False
    MODULE_ERROR = str(e)


# ---------------------------------------------------------------------------
# Helpers de UI
# ---------------------------------------------------------------------------

def section_divider(label: str) -> None:
    st.markdown(
        f'<div class="section-label">'
        f'<span class="bar"></span><span class="txt">{label}</span><span class="bar"></span>'
        f'</div>',
        unsafe_allow_html=True,
    )


def card(content_fn, accent: str = "accent"):
    cls = {"accent": "cr-card-accent", "blue": "cr-card-blue",
           "purple": "cr-card-purple", "red": "cr-card-red"}.get(accent, "")
    st.markdown(f'<div class="cr-card {cls}">', unsafe_allow_html=True)
    content_fn()
    st.markdown('</div>', unsafe_allow_html=True)


def render_fig(fig: plt.Figure, key: str | None = None) -> None:
    """Renderiza Figure Matplotlib com fundo transparente."""
    fig.patch.set_alpha(0)
    for ax in fig.get_axes():
        ax.patch.set_alpha(0.0)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor="none", transparent=True)
    buf.seek(0)
    st.image(buf, use_container_width=True)
    plt.close(fig)


def kpi_row(items: list[tuple]) -> None:
    """Renderiza uma linha de KPIs. items = [(label, value, delta, delta_color), ...]"""
    cols = st.columns(len(items))
    for col, (label, value, delta, dcolor) in zip(cols, items):
        with col:
            st.metric(label=label, value=value,
                      delta=delta if delta else None,
                      delta_color=dcolor if dcolor else "normal")


# ---------------------------------------------------------------------------
# Cache de dados
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_and_process(file_bytes: bytes, filename: str) -> pd.DataFrame:
    import tempfile, os
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        df = preprocess(tmp_path)
        df = create_features(df)
    finally:
        os.unlink(tmp_path)
    return df


@st.cache_data(show_spinner=False)
def get_analysis(_df: pd.DataFrame) -> dict:
    return an.run_all(_df)


@st.cache_resource(show_spinner=False)
def get_model(_df: pd.DataFrame):
    return train_model(_df)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar() -> tuple[str, pd.DataFrame | None]:
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-brand">
            <h1>👑 CR Analytics</h1>
            <p>Clash Royale · Dashboard</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**📂 Dataset**")
        uploaded = st.file_uploader(
            "Carregar CSV de partidas",
            type=["csv"],
            help="CSV exportado da API do Clash Royale com as colunas padrão do projeto.",
            label_visibility="collapsed",
        )

        df = None
        if uploaded:
            with st.spinner("Processando dataset…"):
                try:
                    df = load_and_process(uploaded.read(), uploaded.name)
                    st.success(f"✅ {len(df):,} partidas carregadas")
                except Exception as e:
                    st.error(f"Erro ao processar: {e}")

        st.markdown("---")
        st.markdown("**🧭 Navegação**")
        page = st.radio(
            "Página",
            ["🏠 Home", "📊 Análise Geral", "🃏 Análise de Cartas",
             "🏆 Análise de Troféus", "🤖 Modelo Preditivo", "📝 Conclusões"],
            label_visibility="collapsed",
        )

        if df is not None:
            st.markdown("---")
            st.markdown("**ℹ️ Dataset Info**")
            st.caption(f"Partidas: **{len(df):,}**")
            st.caption(f"Colunas: **{len(df.columns)}**")
            if "battle_time" in df.columns:
                try:
                    mn = df["battle_time"].min().strftime("%d/%m/%Y")
                    mx = df["battle_time"].max().strftime("%d/%m/%Y")
                    st.caption(f"Período: **{mn}** → **{mx}**")
                except Exception:
                    pass

        st.markdown("---")
        st.caption("Desenvolvido com Python · Streamlit · Scikit-Learn")

    return page, df


# ---------------------------------------------------------------------------
# Páginas
# ---------------------------------------------------------------------------

def page_home(df: pd.DataFrame | None) -> None:
    st.markdown("""
    <div class="hero-wrap">
        <div class="hero-crown">👑</div>
        <h1 class="hero-title">Clash Royale Analytics</h1>
        <p class="hero-sub">Dashboard de análise estatística e preditiva de partidas</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""<div class="cr-card cr-card-accent">
            <h4 style="color:#FFD740;margin:0 0 .4rem">📊 Análise Geral</h4>
            <p style="color:#7A9CBF;font-size:.88rem;margin:0">Taxa de vitória, distribuição de resultados e padrões por hora do dia.</p>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="cr-card cr-card-blue">
            <h4 style="color:#4FC3F7;margin:0 0 .4rem">🃏 Análise de Cartas</h4>
            <p style="color:#7A9CBF;font-size:.88rem;margin:0">Cartas mais usadas no meta e quais entregam o maior win rate.</p>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class="cr-card cr-card-purple">
            <h4 style="color:#AB47BC;margin:0 0 .4rem">🤖 Modelo Preditivo</h4>
            <p style="color:#7A9CBF;font-size:.88rem;margin:0">Random Forest treinado para prever vitórias com base em features de batalha.</p>
        </div>""", unsafe_allow_html=True)

    section_divider("como usar")
    steps = [
        ("1", "Carregue seu CSV", "Use o painel lateral para importar o dataset de partidas no formato padrão da API do Clash Royale."),
        ("2", "Explore as páginas", "Navegue pelas seções do menu para análises de resultados, cartas, troféus e o modelo de ML."),
        ("3", "Interprete os insights", "Cada seção traz KPIs, gráficos interativos e tabelas detalhadas para orientar sua estratégia."),
    ]
    cols = st.columns(3)
    for col, (num, title, desc) in zip(cols, steps):
        with col:
            st.markdown(f"""<div class="cr-card" style="text-align:center">
                <div style="font-size:2rem;font-weight:800;color:#1E3A55;margin-bottom:.3rem">{num}</div>
                <h4 style="color:#E8EDF2;margin:0 0 .4rem;font-size:.95rem">{title}</h4>
                <p style="color:#5A7A99;font-size:.83rem;margin:0">{desc}</p>
            </div>""", unsafe_allow_html=True)

    if df is None:
        st.info("👈 Carregue um arquivo CSV no painel lateral para começar a análise.")

    section_divider("colunas esperadas")
    expected_cols = [
        "battle_time", "battle_type", "game_mode", "player_tag", "player_name",
        "player_crowns", "player_starting_trophies", "opponent_tag", "opponent_name",
        "opponent_crowns", "opponent_starting_trophies", "result",
        "player_deck", "opponent_deck", "player_deck_hash", "opponent_deck_hash",
        "player_king_tower_hp", "opponent_king_tower_hp",
    ]
    col_df = pd.DataFrame({"Coluna": expected_cols,
                           "Tipo esperado": ["datetime", "str", "str", "str", "str",
                                             "int", "int", "str", "str", "int", "int",
                                             "str", "str", "str", "str", "str", "int", "int"]})
    st.dataframe(col_df, use_container_width=True, hide_index=True)


def page_analise_geral(df: pd.DataFrame | None) -> None:
    st.markdown("## 📊 Análise Geral")
    if df is None:
        st.warning("Carregue um dataset para visualizar esta seção.")
        return

    results = get_analysis(df)
    wr = results["overall_win_rate"].iloc[0]
    rd = results["result_distribution"]
    cr = results["avg_crowns_by_result"]

    section_divider("indicadores principais")
    win_rate_val = f"{wr['win_rate_pct']:.1f}%"
    total_val    = f"{int(wr['total_battles']):,}"
    wins_val     = f"{int(wr['total_wins']):,}"
    losses_val   = f"{int(wr['total_losses']):,}"

    kpi_row([
        ("Taxa de Vitória", win_rate_val, None, None),
        ("Total de Partidas", total_val, None, None),
        ("Vitórias", wins_val, None, None),
        ("Derrotas / Empates", losses_val, None, None),
    ])

    section_divider("distribuição de resultados")
    col1, col2 = st.columns([3, 2])
    with col1:
        fig = viz.plot_result_distribution(rd)
        render_fig(fig)
    with col2:
        st.markdown("##### Tabela de Resultados")
        display_rd = rd.copy()
        if "result" not in display_rd.columns:
            display_rd = display_rd.reset_index()
        display_rd.columns = [c.replace("_", " ").title() for c in display_rd.columns]
        st.dataframe(display_rd, use_container_width=True, hide_index=True)

        st.markdown("##### Coroas Médias por Resultado")
        display_cr = cr[["result", "avg_player_crowns", "avg_opponent_crowns"]].copy()
        display_cr.columns = ["Resultado", "Coroas Jogador", "Coroas Oponente"]
        st.dataframe(display_cr, use_container_width=True, hide_index=True)

    section_divider("win rate por hora do dia")
    wh = results["win_rate_by_hour"]
    if not wh.empty:
        peak = wh.loc[wh["win_rate_pct"].idxmax()]
        low  = wh.loc[wh["win_rate_pct"].idxmin()]
        kpi_row([
            ("Melhor Horário", f"{int(peak['battle_hour'])}h", f"{peak['win_rate_pct']:.1f}% win rate", "normal"),
            ("Pior Horário",   f"{int(low['battle_hour'])}h",  f"{low['win_rate_pct']:.1f}% win rate",  "inverse"),
            ("Amplitude",      f"{(peak['win_rate_pct'] - low['win_rate_pct']):.1f}pp", "pico − vale", None),
        ])
        fig2 = viz.plot_win_rate_by_hour(wh)
        render_fig(fig2)
    else:
        st.info("Coluna 'battle_hour' não encontrada — execute feature_engineering.")


def page_analise_cartas(df: pd.DataFrame | None) -> None:
    st.markdown("## 🃏 Análise de Cartas")
    if df is None:
        st.warning("Carregue um dataset para visualizar esta seção.")
        return

    results = get_analysis(df)
    top_n = st.slider("Número de cartas exibidas", min_value=5, max_value=30, value=15, step=5)

    section_divider("cartas mais utilizadas")
    mu = results["most_used_cards"].head(top_n)
    col1, col2 = st.columns([3, 2])
    with col1:
        fig = viz.plot_most_used_cards(results["most_used_cards"], top_n=top_n)
        render_fig(fig)
    with col2:
        st.markdown("##### Top Cartas — Uso")
        display_mu = mu[["card", "usage_count", "usage_rate_pct"]].copy()
        display_mu.columns = ["Carta", "Usos", "Taxa (%)"]
        st.dataframe(display_mu, use_container_width=True, hide_index=True)

    section_divider("cartas com maior win rate")
    cwr = results["cards_win_rate"].head(top_n)
    col3, col4 = st.columns([3, 2])
    with col3:
        fig2 = viz.plot_cards_win_rate(results["cards_win_rate"], top_n=top_n)
        render_fig(fig2)
    with col4:
        st.markdown("##### Top Cartas — Win Rate")
        display_cwr = cwr[["card", "win_rate_pct", "appearances", "wins"]].copy()
        display_cwr.columns = ["Carta", "Win Rate (%)", "Aparições", "Vitórias"]
        st.dataframe(display_cwr, use_container_width=True, hide_index=True)

        if not cwr.empty:
            best = cwr.iloc[0]
            st.success(f"🏅 Melhor carta: **{best['card']}** com **{best['win_rate_pct']:.1f}%** de win rate")

    section_divider("comparativo: uso vs win rate")
    if not mu.empty and not cwr.empty:
        merged = pd.merge(
            mu[["card", "usage_rate_pct"]],
            cwr[["card", "win_rate_pct"]],
            on="card", how="inner"
        ).sort_values("win_rate_pct", ascending=False).head(20)
        merged.columns = ["Carta", "Taxa de Uso (%)", "Win Rate (%)"]
        st.dataframe(merged, use_container_width=True, hide_index=True)


def page_analise_trofeus(df: pd.DataFrame | None) -> None:
    st.markdown("## 🏆 Análise de Troféus")
    if df is None:
        st.warning("Carregue um dataset para visualizar esta seção.")
        return

    results = get_analysis(df)

    section_divider("distribuição de troféus")
    col_hist, col_stats = st.columns([3, 1])
    with col_hist:
        fig = viz.plot_trophy_histogram(df)
        render_fig(fig)
    with col_stats:
        st.markdown("##### Estatísticas")
        trophy_col = "player_starting_trophies"
        if trophy_col in df.columns:
            data = df[trophy_col].dropna()
            stats = {
                "Média":    f"{data.mean():,.0f}",
                "Mediana":  f"{data.median():,.0f}",
                "Mínimo":   f"{data.min():,.0f}",
                "Máximo":   f"{data.max():,.0f}",
                "Desvio P.":f"{data.std():,.0f}",
            }
            for label, val in stats.items():
                st.metric(label=label, value=val)

    section_divider("win rate por faixa de troféus")
    tr = results["win_rate_by_trophy_range"]
    if not tr.empty:
        best_range  = tr.loc[tr["win_rate_pct"].idxmax()]
        worst_range = tr.loc[tr["win_rate_pct"].idxmin()]
        kpi_row([
            ("Melhor Faixa",  best_range["trophy_range"],  f"{best_range['win_rate_pct']:.1f}% win rate", "normal"),
            ("Pior Faixa",    worst_range["trophy_range"], f"{worst_range['win_rate_pct']:.1f}% win rate", "inverse"),
            ("Faixas com dados", f"{len(tr[tr['total_battles']>0])}", None, None),
        ])

        col_table, col_bar = st.columns([2, 3])
        with col_table:
            display_tr = tr[["trophy_range","total_battles","wins","win_rate_pct"]].copy()
            display_tr.columns = ["Faixa", "Partidas", "Vitórias", "Win Rate (%)"]
            st.dataframe(display_tr, use_container_width=True, hide_index=True)
        with col_bar:
            fig2, ax = plt.subplots(figsize=(8, 4))
            fig2.patch.set_facecolor("#111F33")
            ax.set_facecolor("#111F33")
            valid = tr[tr["total_battles"] > 0]
            colors = ["#FFD740" if v >= 50 else "#4FC3F7" for v in valid["win_rate_pct"]]
            ax.bar(valid["trophy_range"].astype(str), valid["win_rate_pct"],
                   color=colors, edgecolor="#0B1623", linewidth=.8, zorder=3)
            ax.axhline(50, color="#EF5350", linewidth=1.2, linestyle="--", alpha=.7, label="50%")
            ax.set_title("Win Rate por Faixa de Troféus", color="#E8EDF2", fontsize=12, fontweight="bold", pad=10)
            ax.set_xlabel("Faixa de Troféus", color="#7A9CBF", fontsize=9)
            ax.set_ylabel("Win Rate (%)", color="#7A9CBF", fontsize=9)
            ax.tick_params(colors="#7A9CBF", labelsize=8)
            ax.tick_params(axis="x", rotation=35)
            for spine in ax.spines.values():
                spine.set_edgecolor("#1A3050")
            ax.grid(axis="y", color="#1A3050", linewidth=.6, alpha=.8)
            ax.set_ylim(0, max(valid["win_rate_pct"].max() * 1.15, 60))
            ax.legend(fontsize=8, facecolor="#111F33", edgecolor="#1A3050", labelcolor="#E8EDF2")
            render_fig(fig2)

    section_divider("diferença de troféus")
    if "trophy_difference" in df.columns:
        diff = df["trophy_difference"]
        kpi_row([
            ("Diferença Média",   f"{diff.mean():+.0f}", None, None),
            ("Diferença Mediana", f"{diff.median():+.0f}", None, None),
            ("Máx. Vantagem",     f"{diff.max():+,.0f}", None, None),
            ("Máx. Desvantagem",  f"{diff.min():+,.0f}", None, None),
        ])


def page_modelo(df: pd.DataFrame | None) -> None:
    st.markdown("## 🤖 Modelo Preditivo")
    if df is None:
        st.warning("Carregue um dataset para treinar o modelo.")
        return

    st.markdown("""<div class="cr-card cr-card-purple">
        <p style="color:#A0BDD4;font-size:.9rem;margin:0">
        O modelo <strong style="color:#AB47BC">Random Forest Classifier</strong> é treinado para prever
        vitórias com base em features de batalha. O treinamento usa 80% dos dados e avalia
        no 20% restante, com validação cruzada 5-fold para detectar overfitting.
        </p>
    </div>""", unsafe_allow_html=True)

    section_divider("features do modelo")
    feat_df = pd.DataFrame({
        "Feature": MODEL_FEATURES,
        "Descrição": [
            "Troféus iniciais do jogador",
            "Troféus iniciais do oponente",
            "Diferença de troféus (player − opponent)",
            "Hora do dia da partida (0–23)",
            "Diferença de coroas (player − opponent)",
        ]
    })
    st.dataframe(feat_df, use_container_width=True, hide_index=True)

    col_btn, _ = st.columns([1, 3])
    with col_btn:
        train_btn = st.button("🚀 Treinar Modelo", use_container_width=True)

    if train_btn or "model_result" in st.session_state:
        if train_btn:
            with st.spinner("Treinando Random Forest…"):
                try:
                    result = get_model(df)
                    st.session_state["model_result"] = result
                except Exception as e:
                    st.error(f"Erro no treinamento: {e}")
                    return

        result = st.session_state.get("model_result")
        if result is None:
            return

        m = result.metrics

        section_divider("métricas de avaliação")
        kpi_row([
            ("Acurácia",  f"{m.accuracy*100:.2f}%",  None, None),
            ("Precisão",  f"{m.precision*100:.2f}%", None, None),
            ("Recall",    f"{m.recall*100:.2f}%",    None, None),
            ("F1-Score",  f"{m.f1*100:.2f}%",        None, None),
        ])

        col_cv, col_cm = st.columns(2)
        with col_cv:
            st.markdown("##### Validação Cruzada (5-fold F1)")
            st.metric("CV F1 Médio",  f"{m.cv_mean*100:.2f}%")
            st.metric("CV F1 Desvio", f"± {m.cv_std*100:.2f}pp")
            st.caption("Medido no conjunto de treino para detectar overfitting.")

        with col_cm:
            st.markdown("##### Matriz de Confusão")
            cm = m.confusion_matrix
            cm_df = pd.DataFrame(
                cm,
                index=["Real: Derrota (0)", "Real: Vitória (1)"],
                columns=["Pred: Derrota (0)", "Pred: Vitória (1)"],
            )
            st.dataframe(cm_df, use_container_width=True)

        section_divider("importância das features")
        imp = m.feature_importances
        imp_df = pd.DataFrame(
            sorted(imp.items(), key=lambda x: x[1], reverse=True),
            columns=["Feature", "Importância"],
        )

        col_imp_chart, col_imp_table = st.columns([3, 2])
        with col_imp_chart:
            fig, ax = plt.subplots(figsize=(7, 3.5))
            fig.patch.set_facecolor("#111F33")
            ax.set_facecolor("#111F33")
            sorted_feats = imp_df.sort_values("Importância")
            bar_colors = ["#FFD740" if i == len(sorted_feats) - 1 else "#4FC3F7"
                          for i in range(len(sorted_feats))]
            ax.barh(sorted_feats["Feature"], sorted_feats["Importância"],
                    color=bar_colors, edgecolor="#0B1623", linewidth=.8)
            ax.set_title("Importância das Features", color="#E8EDF2", fontsize=11,
                         fontweight="bold", pad=10)
            ax.tick_params(colors="#7A9CBF", labelsize=9)
            for spine in ax.spines.values():
                spine.set_edgecolor("#1A3050")
            ax.grid(axis="x", color="#1A3050", linewidth=.6)
            render_fig(fig)
        with col_imp_table:
            imp_df["Importância (%)"] = (imp_df["Importância"] * 100).round(2)
            st.dataframe(imp_df[["Feature", "Importância (%)"]], use_container_width=True, hide_index=True)

        section_divider("relatório de classificação")
        with st.expander("Ver relatório completo", expanded=False):
            st.code(m.classification_report, language=None)

        section_divider("informações do treinamento")
        kpi_row([
            ("Amostras de Treino", f"{m.train_size:,}", None, None),
            ("Amostras de Teste",  f"{m.test_size:,}",  None, None),
            ("Split",              "80% / 20%",          None, None),
        ])


def page_conclusoes(df: pd.DataFrame | None) -> None:
    st.markdown("## 📝 Conclusões")

    st.markdown("""<div class="cr-card cr-card-accent">
        <h4 style="color:#FFD740;margin:0 0 .75rem">Sobre este Dashboard</h4>
        <p style="color:#A0BDD4;font-size:.9rem;line-height:1.7;margin:0">
        Este painel foi construído para transformar dados brutos de partidas de Clash Royale
        em insights estratégicos acionáveis. Cada módulo do projeto tem responsabilidade única,
        seguindo princípios de engenharia de software como SRP, imutabilidade de dados e
        separação entre lógica, análise e visualização.
        </p>
    </div>""", unsafe_allow_html=True)

    section_divider("arquitetura do projeto")
    modules = [
        ("preprocessing.py",       "📥", "Carrega, limpa e padroniza o dataset bruto.", "accent"),
        ("feature_engineering.py", "⚙️", "Cria features derivadas: trophy_difference, battle_hour, win_binary…", "blue"),
        ("analysis.py",            "📊", "Calcula métricas agregadas e retorna DataFrames prontos para visualização.", "purple"),
        ("visualization.py",       "🎨", "Gera gráficos Matplotlib com tema Clash Royale e retorna objetos Figure.", "accent"),
        ("model.py",               "🤖", "Treina RandomForest, avalia com CV 5-fold e retorna modelo + métricas.", "blue"),
        ("app.py",                 "🖥️", "Orquestra tudo via Streamlit com cache, KPIs e navegação por seções.", "purple"),
    ]
    for name, icon, desc, accent in modules:
        cls = {"accent": "cr-card-accent", "blue": "cr-card-blue", "purple": "cr-card-purple"}[accent]
        st.markdown(f"""<div class="cr-card {cls}" style="padding:1rem 1.25rem;margin-bottom:.75rem">
            <span style="font-size:1.1rem">{icon}</span>
            <strong style="color:#E8EDF2;margin-left:.5rem">{name}</strong>
            <span style="color:#5A7A99;font-size:.85rem;margin-left:.75rem">{desc}</span>
        </div>""", unsafe_allow_html=True)

    section_divider("boas práticas aplicadas")
    practices = [
        ("Type Hints",        "Todos os módulos usam anotações de tipo completas (PEP 484)."),
        ("Docstrings",        "Documentação no estilo Google em todas as funções públicas."),
        ("Cache Streamlit",   "@st.cache_data e @st.cache_resource evitam reprocessamento."),
        ("Imutabilidade",     "Funções sempre operam em df.copy() — o DataFrame original nunca é mutado."),
        ("Pipeline sklearn",  "StandardScaler + RandomForest em pipeline evita data leakage."),
        ("Validação antecipada", "_require_columns() / _validate_dataframe() falham cedo com mensagens claras."),
        ("Logging estruturado", "Todos os módulos usam logging com timestamp em vez de print()."),
        ("SRP",               "Cada módulo tem uma única responsabilidade bem definida."),
    ]
    cols = st.columns(2)
    for i, (title, desc) in enumerate(practices):
        with cols[i % 2]:
            st.markdown(f"""<div class="cr-card" style="padding:.85rem 1.1rem;margin-bottom:.6rem">
                <strong style="color:#FFD740;font-size:.88rem">✓ {title}</strong>
                <p style="color:#5A7A99;font-size:.82rem;margin:.2rem 0 0">{desc}</p>
            </div>""", unsafe_allow_html=True)

    section_divider("stack tecnológico")
    stack = {
        "Python 3.10+": "Linguagem base com suporte completo a type hints modernos.",
        "Pandas":        "Manipulação e análise de dados tabulares.",
        "NumPy":         "Operações vetorizadas e tipos de dados eficientes.",
        "Scikit-Learn":  "Pipeline de ML, RandomForest, métricas e cross-validation.",
        "Matplotlib":    "Visualizações customizadas com tema escuro Clash Royale.",
        "Streamlit":     "Interface web interativa sem necessidade de HTML/JS manual.",
    }
    stack_df = pd.DataFrame(
        [{"Biblioteca": k, "Papel": v} for k, v in stack.items()]
    )
    st.dataframe(stack_df, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Roteador principal
# ---------------------------------------------------------------------------

def main() -> None:
    if not MODULES_OK:
        st.error(f"❌ Erro ao importar módulos do projeto: {MODULE_ERROR}")
        st.info("Certifique-se de que preprocessing.py, feature_engineering.py, analysis.py, visualization.py e model.py estão na mesma pasta que app.py.")
        return

    page, df = render_sidebar()

    page_map = {
        "🏠 Home":              page_home,
        "📊 Análise Geral":     page_analise_geral,
        "🃏 Análise de Cartas": page_analise_cartas,
        "🏆 Análise de Troféus":page_analise_trofeus,
        "🤖 Modelo Preditivo":  page_modelo,
        "📝 Conclusões":        page_conclusoes,
    }

    render_fn = page_map.get(page, page_home)
    render_fn(df)


if __name__ == "__main__":
    main()