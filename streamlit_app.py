from __future__ import annotations

import io
import os
import tempfile

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# importa os modulos do pacote src
from src.preprocessamento import preprocessar
from src.engenharia_features import criar_features
import src.analise as analise
import src.visualizacao as viz
from src.modelo import treinar_modelo, FEATURES as FEATURES_MODELO

st.set_page_config(
    page_title="Clash Royale - Analise de Batalhas",
    layout="wide",
    initial_sidebar_state="expanded",
)


# Funcoes auxiliares

def exibir_figura(fig: plt.Figure) -> None:
    """Salva a figura em memoria e exibe no Streamlit."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    st.image(buf, use_container_width=True)
    plt.close(fig)


def linha_kpi(itens: list[tuple]) -> None:
    """Exibe uma linha de metricas (KPIs) em colunas."""
    colunas = st.columns(len(itens))
    for col, (rotulo, valor) in zip(colunas, itens):
        col.metric(label=rotulo, value=valor)


# Cache: evita reprocessamento a cada interacao

@st.cache_data(show_spinner=False)
def carregar_dados(conteudo: bytes, nome: str) -> pd.DataFrame:
    """Carrega, limpa e cria features a partir do CSV enviado."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(conteudo)
        caminho = tmp.name
    try:
        df = preprocessar(caminho)
        df = criar_features(df)
    finally:
        os.unlink(caminho)
    return df


@st.cache_data(show_spinner=False)
def obter_analise(_df: pd.DataFrame) -> dict:
    """Executa todas as analises e cacheia o resultado."""
    return analise.executar_todas(_df)


@st.cache_resource(show_spinner=False)
def obter_modelo(_df: pd.DataFrame):
    """Treina o modelo e cacheia o resultado."""
    return treinar_modelo(_df)



# Sidebar


def sidebar() -> tuple[str, pd.DataFrame | None]:
    """Renderiza o painel lateral e retorna a pagina selecionada e o df."""
    with st.sidebar:
        st.title("CR Analise")
        st.caption("Clash Royale - Dashboard Academico")
        st.divider()

        arquivo = st.file_uploader(
            "Carregar CSV de batalhas",
            type=["csv"],
            help="CSV com o historico de batalhas do jogador.",
        )

        df = None
        if arquivo:
            with st.spinner("Processando..."):
                try:
                    df = carregar_dados(arquivo.read(), arquivo.name)
                    st.success(f"{len(df):,} batalhas carregadas")
                except Exception as erro:
                    st.error(f"Erro: {erro}")

        st.divider()

        pagina = st.radio(
            "Paginas",
            ["Inicio", "Analise Geral", "Cartas", "Trofeus", "Modelo", "Conclusoes"],
        )

        # resumo rapido do dataset na sidebar
        if df is not None:
            st.divider()
            st.markdown(f"**Batalhas:** {len(df):,}")
            st.markdown(f"**Colunas:** {len(df.columns)}")
            if "battle_time" in df.columns:
                try:
                    inicio = df["battle_time"].min().strftime("%d/%m/%Y")
                    fim    = df["battle_time"].max().strftime("%d/%m/%Y")
                    st.markdown(f"**Periodo:** {inicio} a {fim}")
                except Exception:
                    pass

        st.divider()
        st.caption("Python | Streamlit | Scikit-learn")

    return pagina, df



# Paginas


def pagina_inicio(df: pd.DataFrame | None) -> None:
    st.title("Clash Royale - Analise de Batalhas")
    st.write(
        "Dashboard desenvolvido como trabalho academico para analisar o historico "
        "de batalhas de um jogador no modo Path of Legend do Clash Royale."
    )
    st.divider()

    st.subheader("O que e analisado?")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Resultados**")
        st.write("Taxa de vitoria, distribuicao de resultados e comportamento por hora do dia.")
    with col2:
        st.markdown("**Cartas**")
        st.write("Cartas mais usadas e quais tem o melhor win rate no meta do jogador.")
    with col3:
        st.markdown("**Modelo**")
        st.write("Random Forest para prever vitorias a partir de dados da batalha.")

    st.divider()
    st.subheader("Como usar")
    st.markdown("1. Carregue o arquivo CSV pelo painel lateral.")
    st.markdown("2. Navegue pelas paginas de analise.")
    st.markdown("3. Na pagina Modelo, clique em 'Treinar modelo' para ver as predicoes.")

    if df is None:
        st.info("Carregue o CSV no painel lateral para comecar.")


def pagina_analise_geral(df: pd.DataFrame | None) -> None:
    st.title("Analise Geral")

    if df is None:
        st.warning("Carregue um dataset para ver esta pagina.")
        return

    resultados = obter_analise(df)
    tv = resultados["taxa_vitoria_geral"].iloc[0]
    dr = resultados["distribuicao_resultados"]

    st.subheader("Indicadores gerais")
    linha_kpi([
        ("Taxa de Vitoria",    f"{tv['taxa_vitoria_pct']:.1f}%"),
        ("Total de Batalhas",  f"{int(tv['total_partidas']):,}"),
        ("Vitorias",           f"{int(tv['vitorias']):,}"),
        ("Derrotas / Empates", f"{int(tv['derrotas_empates']):,}"),
    ])

    st.divider()
    st.subheader("Distribuicao de resultados")

    col1, col2 = st.columns([2, 1])
    with col1:
        exibir_figura(viz.grafico_distribuicao_resultados(dr))
    with col2:
        tabela = dr.copy()
        tabela.columns = ["Resultado", "Partidas", "Percentual (%)"]
        st.dataframe(tabela, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Taxa de vitoria por hora do dia")
    st.caption("Identifica horarios com maior ou menor desempenho.")
    exibir_figura(viz.grafico_vitoria_por_hora(resultados["vitoria_por_hora"]))

    st.divider()
    st.subheader("Coroas por resultado")
    st.caption("Compara a media de coroas do jogador e do oponente em cada tipo de resultado.")
    exibir_figura(viz.grafico_coroas_por_resultado(df))


def pagina_cartas(df: pd.DataFrame | None) -> None:
    st.title("Analise de Cartas")

    if df is None:
        st.warning("Carregue um dataset para ver esta pagina.")
        return

    resultados = obter_analise(df)

    st.subheader("Cartas mais usadas")
    st.caption("Frequencia de aparicao nos decks do jogador ao longo das batalhas.")
    exibir_figura(viz.grafico_cartas_mais_usadas(resultados["cartas_mais_usadas"]))

    st.divider()
    st.subheader("Win rate por carta")
    st.caption("Apenas cartas com ao menos 20 aparicoes. Verde = acima de 50% de vitoria.")
    exibir_figura(viz.grafico_win_rate_cartas(resultados["win_rate_por_carta"]))

    st.divider()
    st.subheader("Tabela completa de cartas")
    tabela = resultados["win_rate_por_carta"].copy()
    tabela.columns = ["Carta", "Aparicoes", "Vitorias", "Win Rate (%)"]
    st.dataframe(tabela, use_container_width=True, hide_index=True)


def pagina_trofeus(df: pd.DataFrame | None) -> None:
    st.title("Analise de Trofeus")

    if df is None:
        st.warning("Carregue um dataset para ver esta pagina.")
        return

    resultados = obter_analise(df)
    est = resultados["estatisticas_trofeus"].iloc[0]

    st.subheader("Estatisticas de trofeus")
    linha_kpi([
        ("Media",         f"{est['media']:.0f}"),
        ("Mediana",       f"{est['mediana']:.0f}"),
        ("Minimo",        f"{int(est['minimo']):,}"),
        ("Maximo",        f"{int(est['maximo']):,}"),
        ("Desvio Padrao", f"{est['desvio_padrao']:.1f}"),
    ])

    st.divider()
    st.subheader("Distribuicao de trofeus")
    exibir_figura(viz.grafico_trofeus_histograma(df))

    st.divider()
    st.subheader("Win rate por faixa de trofeus")
    st.caption("Verde = faixas onde o jogador ganhou mais de 50% das batalhas.")

    col1, col2 = st.columns([3, 2])
    with col1:
        exibir_figura(viz.grafico_win_rate_por_faixa(resultados["vitoria_por_faixa"]))
    with col2:
        tabela = resultados["vitoria_por_faixa"][["faixa_trofeus", "partidas", "vitorias", "taxa_vitoria_pct"]].copy()
        tabela.columns = ["Faixa", "Partidas", "Vitorias", "Win Rate (%)"]
        st.dataframe(tabela, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Diferenca de trofeus entre jogador e oponente")
    st.caption("Mostra se o jogador enfrentou oponentes mais fortes ou mais fracos em cada resultado.")
    exibir_figura(viz.grafico_diferenca_trofeus_boxplot(df))

    if "diferenca_trofeus" in df.columns:
        diff = df["diferenca_trofeus"]
        linha_kpi([
            ("Media",          f"{diff.mean():+.0f}"),
            ("Mediana",        f"{diff.median():+.0f}"),
            ("Max vantagem",   f"{diff.max():+,.0f}"),
            ("Max desvantagem", f"{diff.min():+,.0f}"),
        ])


def pagina_modelo(df: pd.DataFrame | None) -> None:
    st.title("Modelo Preditivo")

    if df is None:
        st.warning("Carregue um dataset para treinar o modelo.")
        return

    st.write(
        "Um Random Forest Classifier e treinado para prever vitorias. "
        "O treino usa 80% dos dados e a avaliacao usa os 20% restantes. "
        "A validacao cruzada 5-fold verifica se ha overfitting."
    )

    st.subheader("Features utilizadas")
    descricoes = [
        "Trofeus iniciais do jogador",
        "Trofeus iniciais do oponente",
        "Diferenca de trofeus (jogador - oponente)",
        "Hora do dia da batalha (0-23)",
        "Razao entre HP da torre do jogador e do oponente",
    ]
    st.dataframe(
        pd.DataFrame({"Feature": FEATURES_MODELO, "Descricao": descricoes}),
        use_container_width=True,
        hide_index=True,
    )

    if st.button("Treinar modelo"):
        with st.spinner("Treinando..."):
            try:
                resultado = obter_modelo(df)
                st.session_state["resultado_modelo"] = resultado
            except Exception as erro:
                st.error(f"Erro no treinamento: {erro}")
                return

    resultado = st.session_state.get("resultado_modelo")
    if resultado is None:
        return

    m = resultado.metricas

    st.divider()
    st.subheader("Metricas de avaliacao")
    linha_kpi([
        ("Acuracia",  f"{m.acuracia * 100:.2f}%"),
        ("Precisao",  f"{m.precisao * 100:.2f}%"),
        ("Recall",    f"{m.recall * 100:.2f}%"),
        ("F1-Score",  f"{m.f1 * 100:.2f}%"),
    ])

    col_cv, col_cm = st.columns(2)
    with col_cv:
        st.markdown("**Validacao cruzada (5-fold F1)**")
        st.metric("Media",  f"{m.media_cv * 100:.2f}%")
        st.metric("Desvio", f"+-{m.desvio_cv * 100:.2f} pp")
        st.caption("Calculado no treino para detectar overfitting.")

    with col_cm:
        st.markdown("**Matriz de confusao**")
        cm_df = pd.DataFrame(
            m.matriz_confusao,
            index=["Real: Derrota", "Real: Vitoria"],
            columns=["Pred: Derrota", "Pred: Vitoria"],
        )
        st.dataframe(cm_df, use_container_width=True)

    st.divider()
    st.subheader("Importancia das features")

    imp_df = pd.DataFrame(
        sorted(m.importancia_features.items(), key=lambda x: x[1], reverse=True),
        columns=["Feature", "Importancia"],
    )

    col_g, col_t = st.columns([3, 2])
    with col_g:
        # grafico de barras horizontais com importancia de cada feature
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ordenado = imp_df.sort_values("Importancia")
        ax.barh(ordenado["Feature"], ordenado["Importancia"], color="#1f77b4")
        ax.set_title("Importancia das Features")
        ax.set_xlabel("Importancia (Gini)")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        exibir_figura(fig)

    with col_t:
        imp_df["Importancia (%)"] = (imp_df["Importancia"] * 100).round(2)
        st.dataframe(imp_df[["Feature", "Importancia (%)"]], use_container_width=True, hide_index=True)

    st.divider()
    with st.expander("Ver relatorio de classificacao completo"):
        st.code(m.relatorio_classificacao, language=None)

    st.subheader("Dados do treinamento")
    linha_kpi([
        ("Amostras de treino", f"{m.tamanho_treino:,}"),
        ("Amostras de teste",  f"{m.tamanho_teste:,}"),
        ("Split",              "80% / 20%"),
    ])


def pagina_conclusoes(df: pd.DataFrame | None) -> None:
    st.title("Conclusoes")

    st.write(
        "Este projeto analisou o historico de batalhas de um jogador no modo "
        "Path of Legend do Clash Royale, transformando dados brutos em insights "
        "sobre desempenho, uso de cartas e padroes de jogo."
    )

    st.divider()
    st.subheader("Estrutura do projeto")
    modulos = [
        ("preprocessamento.py",   "Carrega, limpa e padroniza o dataset."),
        ("engenharia_features.py", "Cria features derivadas como diferenca_trofeus e hora_batalha."),
        ("analise.py",             "Calcula metricas agregadas por resultado, hora, carta e faixa."),
        ("visualizacao.py",        "Gera graficos Matplotlib e retorna objetos Figure."),
        ("modelo.py",              "Treina o Random Forest com validacao cruzada 5-fold."),
        ("app.py",                 "Orquestra tudo via Streamlit com cache e paginas separadas."),
    ]
    for nome, descricao in modulos:
        st.markdown(f"- **{nome}** — {descricao}")

    st.divider()
    st.subheader("Tecnologias")
    stack = {
        "Python 3.10+":  "Linguagem base.",
        "Pandas":         "Manipulacao e analise de dados tabulares.",
        "NumPy":          "Operacoes numericas eficientes.",
        "Scikit-learn":   "Random Forest, pipeline e validacao cruzada.",
        "Matplotlib":     "Visualizacoes estaticas.",
        "Streamlit":      "Interface web sem HTML ou JavaScript.",
    }
    st.dataframe(
        pd.DataFrame([{"Biblioteca": k, "Papel": v} for k, v in stack.items()]),
        use_container_width=True,
        hide_index=True,
    )



# Roteador principal


def main() -> None:
    pagina, df = sidebar()

    rotas = {
        "Inicio":       pagina_inicio,
        "Analise Geral": pagina_analise_geral,
        "Cartas":       pagina_cartas,
        "Trofeus":      pagina_trofeus,
        "Modelo":       pagina_modelo,
        "Conclusoes":   pagina_conclusoes,
    }

    rotas.get(pagina, pagina_inicio)(df)


if __name__ == "__main__":
    main()
