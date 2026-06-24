import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

# paleta de cores simples e consistente
COR_PRINCIPAL  = "#1f77b4"  # azul
COR_DESTAQUE   = "#2ca02c"  # verde (vitoria)
COR_ALERTA     = "#d62728"  # vermelho (derrota / limite)
COR_NEUTRA     = "#7f7f7f"  # cinza


def _aplicar_estilo(ax: plt.Axes, titulo: str, xlabel: str = "", ylabel: str = "") -> None:
    """Aplica titulo, eixos e grade padrao em todos os graficos."""
    ax.set_title(titulo, fontsize=13, pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.spines[["top", "right"]].set_visible(False)


def grafico_distribuicao_resultados(df_resultados: pd.DataFrame) -> plt.Figure:
    """Grafico de barras com a distribuicao de resultados (WIN/LOSS/DRAW)."""
    fig, ax = plt.subplots(figsize=(6, 4))

    cores = {"WIN": COR_DESTAQUE, "LOSS": COR_ALERTA, "DRAW": COR_NEUTRA}
    barras_cores = [cores.get(r, COR_PRINCIPAL) for r in df_resultados["resultado"]]

    ax.bar(df_resultados["resultado"], df_resultados["quantidade"], color=barras_cores)

    # adiciona o percentual acima de cada barra
    for barra, pct in zip(ax.patches, df_resultados["percentual"]):
        ax.text(
            barra.get_x() + barra.get_width() / 2,
            barra.get_height() + 10,
            f"{pct:.1f}%",
            ha="center", fontsize=10,
        )

    _aplicar_estilo(ax, "Distribuicao de Resultados", "Resultado", "Partidas")
    fig.tight_layout()
    return fig


def grafico_vitoria_por_hora(df_horas: pd.DataFrame) -> plt.Figure:
    """Grafico de linha com a taxa de vitoria por hora do dia."""
    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(
        df_horas["hora_batalha"],
        df_horas["taxa_vitoria_pct"],
        marker="o", linewidth=2, color=COR_PRINCIPAL,
    )
    # linha de referencia em 50%
    ax.axhline(50, color=COR_ALERTA, linestyle="--", linewidth=1, label="50%")
    ax.set_xticks(range(0, 24))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.legend(fontsize=9)

    _aplicar_estilo(ax, "Taxa de Vitoria por Hora do Dia", "Hora (UTC)", "Win Rate")
    fig.tight_layout()
    return fig


def grafico_cartas_mais_usadas(df_cartas: pd.DataFrame) -> plt.Figure:
    """Grafico de barras horizontais com as cartas mais usadas."""
    fig, ax = plt.subplots(figsize=(8, 5))

    ordenado = df_cartas.sort_values("aparicoes")
    ax.barh(ordenado["carta"], ordenado["aparicoes"], color=COR_PRINCIPAL)

    _aplicar_estilo(ax, "Cartas Mais Usadas pelo Jogador", "Aparicoes")
    ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


def grafico_win_rate_cartas(df_cartas: pd.DataFrame, top_n: int = 12) -> plt.Figure:
    """Grafico de barras horizontais com o win rate das principais cartas."""
    fig, ax = plt.subplots(figsize=(8, 5))

    dados = df_cartas.head(top_n).sort_values("win_rate_pct")
    cores = [COR_DESTAQUE if v >= 50 else COR_ALERTA for v in dados["win_rate_pct"]]

    ax.barh(dados["carta"], dados["win_rate_pct"], color=cores)
    ax.axvline(50, color=COR_NEUTRA, linestyle="--", linewidth=1, label="50%")
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.legend(fontsize=9)

    _aplicar_estilo(ax, "Win Rate por Carta (min. 20 aparicoes)", "Win Rate (%)")
    ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


def grafico_trofeus_histograma(df: pd.DataFrame) -> plt.Figure:
    """Histograma da distribuicao de trofeus do jogador."""
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.hist(df["player_starting_trophies"].dropna(), bins=20, color=COR_PRINCIPAL, edgecolor="white")

    _aplicar_estilo(ax, "Distribuicao de Trofeus do Jogador", "Trofeus", "Partidas")
    fig.tight_layout()
    return fig


def grafico_win_rate_por_faixa(df_faixas: pd.DataFrame) -> plt.Figure:
    """Grafico de barras com o win rate por faixa de trofeus."""
    fig, ax = plt.subplots(figsize=(9, 4))

    cores = [COR_DESTAQUE if v >= 50 else COR_PRINCIPAL for v in df_faixas["taxa_vitoria_pct"]]
    ax.bar(df_faixas["faixa_trofeus"], df_faixas["taxa_vitoria_pct"], color=cores)
    ax.axhline(50, color=COR_ALERTA, linestyle="--", linewidth=1, label="50%")
    ax.tick_params(axis="x", rotation=35)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.legend(fontsize=9)

    _aplicar_estilo(ax, "Win Rate por Faixa de Trofeus", "Faixa de Trofeus", "Win Rate (%)")
    fig.tight_layout()
    return fig


def grafico_diferenca_trofeus_boxplot(df: pd.DataFrame) -> plt.Figure:
    """Boxplot da diferenca de trofeus separado por resultado."""
    fig, ax = plt.subplots(figsize=(7, 4))

    # agrupa os valores de diferenca por resultado
    grupos = [
        df.loc[df["result"] == "WIN",  "diferenca_trofeus"].dropna().values,
        df.loc[df["result"] == "LOSS", "diferenca_trofeus"].dropna().values,
        df.loc[df["result"] == "DRAW", "diferenca_trofeus"].dropna().values,
    ]
    ax.boxplot(grupos, tick_labels=["Vitoria", "Derrota", "Empate"], patch_artist=True,
               boxprops=dict(facecolor=COR_PRINCIPAL, alpha=0.5))
    ax.axhline(0, color=COR_ALERTA, linestyle="--", linewidth=1)

    _aplicar_estilo(ax, "Diferenca de Trofeus por Resultado", "Resultado", "Diferenca (jogador - oponente)")
    fig.tight_layout()
    return fig


def grafico_coroas_por_resultado(df: pd.DataFrame) -> plt.Figure:
    """Media de coroas do jogador e do oponente por resultado."""
    fig, ax = plt.subplots(figsize=(7, 4))

    media = df.groupby("result")[["player_crowns", "opponent_crowns"]].mean().reset_index()
    largura = 0.3
    posicoes = range(len(media))

    ax.bar([p - largura / 2 for p in posicoes], media["player_crowns"],  width=largura, label="Jogador",  color=COR_DESTAQUE)
    ax.bar([p + largura / 2 for p in posicoes], media["opponent_crowns"], width=largura, label="Oponente", color=COR_ALERTA)
    ax.set_xticks(list(posicoes))
    ax.set_xticklabels(media["result"])
    ax.legend(fontsize=9)

    _aplicar_estilo(ax, "Media de Coroas por Resultado", "Resultado", "Coroas (media)")
    fig.tight_layout()
    return fig
