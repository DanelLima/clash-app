from __future__ import annotations

import io
import os
import tempfile

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Clash Royale Análises",
    layout="wide",
    initial_sidebar_state="expanded",
)

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


# ------------------------------------------------------------------
# Funções auxiliares
# ------------------------------------------------------------------

def exibir_figura(fig: plt.Figure) -> None:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    st.image(buf, use_container_width=True)
    plt.close(fig)


def linha_kpi(itens: list[tuple]) -> None:
    colunas = st.columns(len(itens))
    for col, (rotulo, valor, delta, cor_delta) in zip(colunas, itens):
        with col:
            st.metric(
                label=rotulo,
                value=valor,
                delta=delta or None,
                delta_color=cor_delta or "normal",
            )


# ------------------------------------------------------------------
# Cache
# ------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def carregar_dados(conteudo: bytes, nome: str) -> pd.DataFrame:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(conteudo)
        caminho = tmp.name
    try:
        df = preprocess(caminho)
        df = create_features(df)
    finally:
        os.unlink(caminho)
    return df


@st.cache_data(show_spinner=False)
def obter_analise(_df: pd.DataFrame) -> dict:
    return an.run_all(_df)


@st.cache_resource(show_spinner=False)
def obter_modelo(_df: pd.DataFrame):
    return train_model(_df)


# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------

def sidebar() -> tuple[str, pd.DataFrame | None]:
    with st.sidebar:
        st.title("CR Analytics")
        st.caption("Clash Royale · Dashboard")
        st.divider()

        arquivo = st.file_uploader(
            "Carregar CSV de partidas",
            type=["csv"],
            help="CSV exportado da API do Clash Royale.",
        )

        df = None
        if arquivo:
            with st.spinner("Processando..."):
                try:
                    df = carregar_dados(arquivo.read(), arquivo.name)
                    st.success(f"{len(df):,} partidas carregadas")
                except Exception as e:
                    st.error(f"Erro ao processar: {e}")

        st.divider()

        pagina = st.radio(
            "Navegação",
            ["Início", "Análise Geral", "Cartas", "Troféus", "Modelo", "Conclusões"],
        )

        if df is not None:
            st.divider()
            st.markdown(f"**Partidas:** {len(df):,}")
            st.markdown(f"**Colunas:** {len(df.columns)}")
            if "battle_time" in df.columns:
                try:
                    inicio = df["battle_time"].min().strftime("%d/%m/%Y")
                    fim = df["battle_time"].max().strftime("%d/%m/%Y")
                    st.markdown(f"**Período:** {inicio} → {fim}")
                except Exception:
                    pass

        st.divider()
        st.caption("Python · Streamlit · Scikit-Learn")

    return pagina, df


# ------------------------------------------------------------------
# Páginas
# ------------------------------------------------------------------

def pagina_inicio(df: pd.DataFrame | None) -> None:
    st.title("Clash Royale Analytics")
    st.write("Dashboard de análise estatística e preditiva de partidas.")
    st.divider()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Análise Geral")
        st.write("Taxa de vitória, distribuição de resultados e padrões por hora do dia.")
    with col2:
        st.subheader("Cartas")
        st.write("Cartas mais usadas no meta e quais têm o maior win rate.")
    with col3:
        st.subheader("Modelo Preditivo")
        st.write("Random Forest treinado para prever vitórias com base em features de batalha.")

    st.divider()
    st.subheader("Como usar")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**1. Carregue o CSV**")
        st.write("Use o painel lateral para importar o dataset de partidas.")
    with c2:
        st.markdown("**2. Explore as páginas**")
        st.write("Navegue pelo menu para ver análises de resultados, cartas, troféus e o modelo.")
    with c3:
        st.markdown("**3. Interprete os resultados**")
        st.write("Cada seção traz métricas, gráficos e tabelas para orientar sua estratégia.")

    if df is None:
        st.info("Carregue um arquivo CSV no painel lateral para começar.")

    st.divider()
    st.subheader("Colunas esperadas no CSV")

    colunas = [
        "battle_time", "battle_type", "game_mode", "player_tag", "player_name",
        "player_crowns", "player_starting_trophies", "opponent_tag", "opponent_name",
        "opponent_crowns", "opponent_starting_trophies", "result",
        "player_deck", "opponent_deck", "player_deck_hash", "opponent_deck_hash",
        "player_king_tower_hp", "opponent_king_tower_hp",
    ]
    tipos = [
        "datetime", "str", "str", "str", "str",
        "int", "int", "str", "str", "int", "int",
        "str", "str", "str", "str", "str", "int", "int",
    ]
    st.dataframe(
        pd.DataFrame({"Coluna": colunas, "Tipo": tipos}),
        use_container_width=True,
        hide_index=True,
    )


def pagina_analise_geral(df: pd.DataFrame | None) -> None:
    st.title("Análise Geral")

    if df is None:
        st.warning("Carregue um dataset para visualizar esta seção.")
        return

    resultados = obter_analise(df)
    wr = resultados["overall_win_rate"].iloc[0]
    rd = resultados["result_distribution"]
    cr = resultados["avg_crowns_by_result"]

    st.subheader("Indicadores principais")
    linha_kpi([
        ("Taxa de Vitória",    f"{wr['win_rate_pct']:.1f}%",       None, None),
        ("Total de Partidas",  f"{int(wr['total_battles']):,}",    None, None),
        ("Vitórias",           f"{int(wr['total_wins']):,}",       None, None),
        ("Derrotas / Empates", f"{int(wr['total_losses']):,}",     None, None),
    ])

    st.divider()
    st.subheader("Distribuição de resultados")

    col1, col2 = st.columns([3, 2])
    with col1:
        fig = viz.plot_result_distribution(rd)
        exibir_figura(fig)
    with col2:
        tabela_rd = rd.copy()
        if "result" not in tabela_rd.columns:
            tabela_rd = tabela_rd.reset_index()
        tabela_rd.columns = [c.replace("_", " ").title() for c in tabela_rd.columns]
        st.dataframe(tabela_rd, use_container_width=True, hide_index=True)

        st.markdown("**Coroas médias por resultado**")
        tabela_cr = cr[["result", "avg_player_crowns", "avg_opponent_crowns"]].copy()
        tabela_cr.columns = ["Resultado", "Coroas Jogador", "Coroas Oponente"]
        st.dataframe(tabela_cr, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Win rate por hora do dia")

    wh = resultados["win_rate_by_hour"]
    if not wh.empty:
        melhor = wh.loc[wh["win_rate_pct"].idxmax()]
        pior = wh.loc[wh["win_rate_pct"].idxmin()]
        linha_kpi([
            ("Melhor horário", f"{int(melhor['battle_hour'])}h", f"{melhor['win_rate_pct']:.1f}% win rate", "normal"),
            ("Pior horário",   f"{int(pior['battle_hour'])}h",   f"{pior['win_rate_pct']:.1f}% win rate",   "inverse"),
            ("Amplitude",      f"{melhor['win_rate_pct'] - pior['win_rate_pct']:.1f} pp", "pico menos vale", None),
        ])
        fig2 = viz.plot_win_rate_by_hour(wh)
        exibir_figura(fig2)
    else:
        st.info("Coluna 'battle_hour' não encontrada. Execute o feature_engineering primeiro.")


def pagina_cartas(df: pd.DataFrame | None) -> None:
    st.title("Análise de Cartas")

    if df is None:
        st.warning("Carregue um dataset para visualizar esta seção.")
        return

    resultados = obter_analise(df)
    top_n = st.slider("Número de cartas exibidas", min_value=5, max_value=30, value=15, step=5)

    st.subheader("Cartas mais utilizadas")
    mu = resultados["most_used_cards"].head(top_n)

    col1, col2 = st.columns([3, 2])
    with col1:
        fig = viz.plot_most_used_cards(resultados["most_used_cards"], top_n=top_n)
        exibir_figura(fig)
    with col2:
        tabela_mu = mu[["card", "usage_count", "usage_rate_pct"]].copy()
        tabela_mu.columns = ["Carta", "Usos", "Taxa (%)"]
        st.dataframe(tabela_mu, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Win rate por carta")
    cwr = resultados["cards_win_rate"].head(top_n)

    col3, col4 = st.columns([3, 2])
    with col3:
        fig2 = viz.plot_cards_win_rate(resultados["cards_win_rate"], top_n=top_n)
        exibir_figura(fig2)
    with col4:
        tabela_cwr = cwr[["card", "win_rate_pct", "appearances", "wins"]].copy()
        tabela_cwr.columns = ["Carta", "Win Rate (%)", "Aparições", "Vitórias"]
        st.dataframe(tabela_cwr, use_container_width=True, hide_index=True)

        if not cwr.empty:
            melhor = cwr.iloc[0]
            st.success(f"Melhor carta: **{melhor['card']}** com **{melhor['win_rate_pct']:.1f}%** de win rate")

    st.divider()
    st.subheader("Uso vs. win rate")
    if not mu.empty and not cwr.empty:
        comparativo = pd.merge(
            mu[["card", "usage_rate_pct"]],
            cwr[["card", "win_rate_pct"]],
            on="card",
            how="inner",
        ).sort_values("win_rate_pct", ascending=False).head(20)
        comparativo.columns = ["Carta", "Taxa de Uso (%)", "Win Rate (%)"]
        st.dataframe(comparativo, use_container_width=True, hide_index=True)


def pagina_trofeus(df: pd.DataFrame | None) -> None:
    st.title("Análise de Troféus")

    if df is None:
        st.warning("Carregue um dataset para visualizar esta seção.")
        return

    resultados = obter_analise(df)

    st.subheader("Distribuição de troféus")
    col_hist, col_stats = st.columns([3, 1])

    with col_hist:
        fig = viz.plot_trophy_histogram(df)
        exibir_figura(fig)

    with col_stats:
        coluna = "player_starting_trophies"
        if coluna in df.columns:
            dados = df[coluna].dropna()
            st.metric("Média",     f"{dados.mean():,.0f}")
            st.metric("Mediana",   f"{dados.median():,.0f}")
            st.metric("Mínimo",    f"{dados.min():,.0f}")
            st.metric("Máximo",    f"{dados.max():,.0f}")
            st.metric("Desvio P.", f"{dados.std():,.0f}")

    st.divider()
    st.subheader("Win rate por faixa de troféus")
    tr = resultados["win_rate_by_trophy_range"]

    if not tr.empty:
        melhor = tr.loc[tr["win_rate_pct"].idxmax()]
        pior = tr.loc[tr["win_rate_pct"].idxmin()]
        linha_kpi([
            ("Melhor faixa",      melhor["trophy_range"], f"{melhor['win_rate_pct']:.1f}% win rate", "normal"),
            ("Pior faixa",        pior["trophy_range"],   f"{pior['win_rate_pct']:.1f}% win rate",   "inverse"),
            ("Faixas com dados",  str(len(tr[tr["total_battles"] > 0])),                             None, None),
        ])

        col_tabela, col_grafico = st.columns([2, 3])
        with col_tabela:
            tabela_tr = tr[["trophy_range", "total_battles", "wins", "win_rate_pct"]].copy()
            tabela_tr.columns = ["Faixa", "Partidas", "Vitórias", "Win Rate (%)"]
            st.dataframe(tabela_tr, use_container_width=True, hide_index=True)

        with col_grafico:
            validos = tr[tr["total_battles"] > 0]
            fig2, ax = plt.subplots(figsize=(8, 4))
            cores = ["#2ca02c" if v >= 50 else "#1f77b4" for v in validos["win_rate_pct"]]
            ax.bar(validos["trophy_range"].astype(str), validos["win_rate_pct"], color=cores)
            ax.axhline(50, color="red", linestyle="--", linewidth=1, label="50%")
            ax.set_title("Win Rate por Faixa de Troféus")
            ax.set_xlabel("Faixa de Troféus")
            ax.set_ylabel("Win Rate (%)")
            ax.tick_params(axis="x", rotation=35)
            ax.legend()
            fig2.tight_layout()
            exibir_figura(fig2)

    st.divider()
    st.subheader("Diferença de troféus entre jogador e oponente")
    if "trophy_difference" in df.columns:
        diff = df["trophy_difference"]
        linha_kpi([
            ("Média",          f"{diff.mean():+.0f}",  None, None),
            ("Mediana",        f"{diff.median():+.0f}", None, None),
            ("Máx. vantagem",  f"{diff.max():+,.0f}",  None, None),
            ("Máx. desvant.",  f"{diff.min():+,.0f}",  None, None),
        ])


def pagina_modelo(df: pd.DataFrame | None) -> None:
    st.title("Modelo Preditivo")

    if df is None:
        st.warning("Carregue um dataset para treinar o modelo.")
        return

    st.write(
        "Um **Random Forest Classifier** é treinado para prever vitórias com base em "
        "features de batalha. O treino usa 80% dos dados e a avaliação os 20% restantes, "
        "com validação cruzada 5-fold para checar overfitting."
    )

    st.subheader("Features utilizadas")
    descricoes = [
        "Troféus iniciais do jogador",
        "Troféus iniciais do oponente",
        "Diferença de troféus (jogador − oponente)",
        "Hora do dia da partida (0–23)",
        "Diferença de coroas (jogador − oponente)",
    ]
    st.dataframe(
        pd.DataFrame({"Feature": MODEL_FEATURES, "Descrição": descricoes}),
        use_container_width=True,
        hide_index=True,
    )

    if st.button("Treinar modelo"):
        with st.spinner("Treinando..."):
            try:
                resultado = obter_modelo(df)
                st.session_state["resultado_modelo"] = resultado
            except Exception as e:
                st.error(f"Erro no treinamento: {e}")
                return

    resultado = st.session_state.get("resultado_modelo")
    if resultado is None:
        return

    m = resultado.metrics

    st.divider()
    st.subheader("Métricas de avaliação")
    linha_kpi([
        ("Acurácia",  f"{m.accuracy * 100:.2f}%",  None, None),
        ("Precisão",  f"{m.precision * 100:.2f}%", None, None),
        ("Recall",    f"{m.recall * 100:.2f}%",    None, None),
        ("F1-Score",  f"{m.f1 * 100:.2f}%",        None, None),
    ])

    col_cv, col_cm = st.columns(2)
    with col_cv:
        st.markdown("**Validação cruzada (5-fold F1)**")
        st.metric("Média",  f"{m.cv_mean * 100:.2f}%")
        st.metric("Desvio", f"± {m.cv_std * 100:.2f} pp")
        st.caption("Calculado no conjunto de treino para detectar overfitting.")

    with col_cm:
        st.markdown("**Matriz de confusão**")
        cm_df = pd.DataFrame(
            m.confusion_matrix,
            index=["Real: Derrota", "Real: Vitória"],
            columns=["Pred: Derrota", "Pred: Vitória"],
        )
        st.dataframe(cm_df, use_container_width=True)

    st.divider()
    st.subheader("Importância das features")

    imp_df = pd.DataFrame(
        sorted(m.feature_importances.items(), key=lambda x: x[1], reverse=True),
        columns=["Feature", "Importância"],
    )

    col_grafico, col_tabela = st.columns([3, 2])
    with col_grafico:
        ordenado = imp_df.sort_values("Importância")
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.barh(ordenado["Feature"], ordenado["Importância"], color="#1f77b4")
        ax.set_title("Importância das Features")
        ax.set_xlabel("Importância")
        fig.tight_layout()
        exibir_figura(fig)

    with col_tabela:
        imp_df["Importância (%)"] = (imp_df["Importância"] * 100).round(2)
        st.dataframe(imp_df[["Feature", "Importância (%)"]], use_container_width=True, hide_index=True)

    st.divider()
    with st.expander("Ver relatório de classificação completo"):
        st.code(m.classification_report, language=None)

    st.subheader("Informações do treinamento")
    linha_kpi([
        ("Amostras de treino", f"{m.train_size:,}", None, None),
        ("Amostras de teste",  f"{m.test_size:,}",  None, None),
        ("Split",              "80% / 20%",          None, None),
    ])


def pagina_conclusoes(df: pd.DataFrame | None) -> None:
    st.title("Conclusões")

    st.write(
        "Este dashboard foi desenvolvido para transformar dados brutos de partidas de "
        "Clash Royale em análises práticas. Cada módulo tem uma responsabilidade "
        "bem definida, separando coleta, processamento, análise, visualização e modelagem."
    )

    st.divider()
    st.subheader("Estrutura do projeto")

    modulos = [
        ("preprocessing.py",       "Carrega, limpa e padroniza o dataset bruto."),
        ("feature_engineering.py", "Cria features derivadas: trophy_difference, battle_hour, win_binary."),
        ("analysis.py",            "Calcula métricas agregadas e retorna DataFrames prontos para exibição."),
        ("visualization.py",       "Gera gráficos Matplotlib e retorna objetos Figure."),
        ("model.py",               "Treina o Random Forest, avalia com CV 5-fold e retorna modelo e métricas."),
        ("app.py",                 "Orquestra tudo via Streamlit com cache e navegação por páginas."),
    ]
    for nome, descricao in modulos:
        st.markdown(f"- **{nome}** — {descricao}")

    st.divider()
    st.subheader("Boas práticas aplicadas")

    praticas = [
        ("Type hints", "Anotações de tipo em todas as funções (PEP 484)."),
        ("Docstrings", "Documentação no estilo Google nas funções públicas."),
        ("Cache", "@st.cache_data e @st.cache_resource evitam reprocessamento desnecessário."),
        ("Imutabilidade", "Funções operam em df.copy() — o DataFrame original nunca é alterado."),
        ("Pipeline sklearn", "StandardScaler + RandomForest em pipeline evita data leakage."),
        ("Validação antecipada", "Erros de schema são detectados cedo com mensagens claras."),
        ("Logging", "Módulos usam logging com timestamp em vez de print()."),
        ("SRP", "Cada módulo tem uma única responsabilidade."),
    ]

    col1, col2 = st.columns(2)
    for i, (titulo, descricao) in enumerate(praticas):
        with (col1 if i % 2 == 0 else col2):
            st.markdown(f"**{titulo}**")
            st.caption(descricao)

    st.divider()
    st.subheader("Tecnologias utilizadas")

    stack = {
        "Python 3.10+": "Linguagem base com type hints modernos.",
        "Pandas":        "Manipulação e análise de dados tabulares.",
        "NumPy":         "Operações vetorizadas e tipos numéricos eficientes.",
        "Scikit-Learn":  "Pipeline de ML, Random Forest, métricas e cross-validation.",
        "Matplotlib":    "Visualizações com estilo consistente.",
        "Streamlit":     "Interface web sem necessidade de HTML ou JavaScript.",
    }
    st.dataframe(
        pd.DataFrame([{"Biblioteca": k, "Papel": v} for k, v in stack.items()]),
        use_container_width=True,
        hide_index=True,
    )


# ------------------------------------------------------------------
# Roteador
# ------------------------------------------------------------------

def main() -> None:
    if not MODULES_OK:
        st.error(f"Erro ao importar módulos: {MODULE_ERROR}")
        st.info("Verifique se a pasta src/ com todos os módulos está no mesmo diretório que app.py.")
        return

    pagina, df = sidebar()

    rotas = {
        "Início":        pagina_inicio,
        "Análise Geral": pagina_analise_geral,
        "Cartas":        pagina_cartas,
        "Troféus":       pagina_trofeus,
        "Modelo":        pagina_modelo,
        "Conclusões":    pagina_conclusoes,
    }

    rotas.get(pagina, pagina_inicio)(df)


if __name__ == "__main__":
    main()