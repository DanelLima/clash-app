from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
from matplotlib.figure import Figure
from cycler import cycler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

PALETTE = {
    "win":        "#4FC3F7",   # azul arena — vitória
    "loss":       "#EF5350",   # vermelho sangue — derrota
    "draw":       "#B0BEC5",   # cinza pedra — empate
    "accent":     "#FFD740",   # ouro troféu
    "accent_alt": "#AB47BC",   # roxo carta épica
    "neutral":    "#78909C",   # cinza slate
    "bg":         "#0D1B2A",   # azul marinho profundo (fundo)
    "surface":    "#16263B",   # superfície dos painéis
    "grid":       "#1E3450",   # linhas de grade
    "text":       "#E8EDF2",   # texto principal
    "text_dim":   "#7A92A8",   # texto secundário
}

# Mapeamento de resultado → cor
RESULT_COLOR_MAP: dict[str, str] = {
    "win":  PALETTE["win"],
    "loss": PALETTE["loss"],
    "draw": PALETTE["draw"],
}

FONT_TITLE = {"fontsize": 15, "fontweight": "bold", "color": PALETTE["text"]}
FONT_LABEL = {"fontsize": 11, "color": PALETTE["text_dim"]}
FONT_TICK  = {"labelsize": 10, "colors": PALETTE["text_dim"]}
DEFAULT_COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
]

# Aplica tema escuro globalmente a todas as figuras do módulo
def _apply_global_style() -> None:
    plt.rcParams.update({
        "figure.facecolor":  PALETTE["bg"],
        "axes.facecolor":    PALETTE["surface"],
        "axes.edgecolor":    PALETTE["grid"],
        "axes.labelcolor":   PALETTE["text_dim"],
        "axes.grid":         True,
        "grid.color":        PALETTE["grid"],
        "grid.linewidth":    0.7,
        "grid.alpha":        0.8,
        "xtick.color":       PALETTE["text_dim"],
        "ytick.color":       PALETTE["text_dim"],
        "text.color":        PALETTE["text"],
        "legend.facecolor":  PALETTE["surface"],
        "legend.edgecolor":  PALETTE["grid"],
        "legend.labelcolor": PALETTE["text"],
        "font.family":       "DejaVu Sans",
        "axes.prop_cycle":   cycler(color=DEFAULT_COLORS),
    })

_apply_global_style()

def _require_columns(df: pd.DataFrame, *columns: str, context: str = "") -> None:
    """Valida colunas obrigatórias antes de plotar.

    Args:
        df: DataFrame a verificar.
        *columns: Colunas que devem existir.
        context: Nome da função chamadora para mensagens de erro claras.

    Raises:
        KeyError: Se alguma coluna estiver ausente.
    """
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(
            f"[{context}] Colunas ausentes no DataFrame: {missing}."
        )


def _add_value_labels(
    ax: plt.Axes,
    fmt: str = "{:.1f}%",
    padding: float = 0.5,
    color: str | None = None,
) -> None:
    """Adiciona rótulos de valor no topo de cada barra.

    Args:
        ax: Eixo Matplotlib com barras desenhadas.
        fmt: Formato da string para o rótulo (padrão: percentual com 1 decimal).
        padding: Espaçamento acima da barra (padrão: 0.5 unidades).
        color: Cor do texto. Se None, usa PALETTE['text'].
    """
    label_color = color or PALETTE["text"]
    for bar in ax.patches:
        height = bar.get_height()
        if height == 0:
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + padding,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color=label_color,
        )


def _style_axes(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    """Aplica estilo padronizado de título e labels ao eixo.

    Args:
        ax: Eixo a estilizar.
        title: Título do gráfico.
        xlabel: Label do eixo X.
        ylabel: Label do eixo Y.
    """
    ax.set_title(title, pad=16, **FONT_TITLE)
    ax.set_xlabel(xlabel, labelpad=10, **FONT_LABEL)
    ax.set_ylabel(ylabel, labelpad=10, **FONT_LABEL)
    ax.tick_params(axis="both", **FONT_TICK)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

def plot_result_distribution(df_results: pd.DataFrame) -> Figure:
    """Gera um gráfico de barras com a distribuição dos resultados das partidas.

    Args:
        df_results: DataFrame retornado por ``analysis.result_distribution()``.
            Deve conter as colunas 'result' (ou index) e 'percentage'.

    Returns:
        Figure do Matplotlib com o gráfico.

    Raises:
        KeyError: Se colunas necessárias estiverem ausentes.

    Example:
        >>> from analysis import result_distribution
        >>> from visualization import plot_result_distribution
        >>> fig = plot_result_distribution(result_distribution(df))
        >>> fig.savefig("result_distribution.png", dpi=150)
    """
    _require_columns(df_results, "percentage", context="plot_result_distribution")

    # Normaliza: suporta tanto index nomeado quanto coluna 'result'
    if "result" in df_results.columns:
        labels = df_results["result"].astype(str).str.lower()
        percentages = df_results["percentage"].values
        counts = df_results.get("count", pd.Series([None] * len(df_results))).values
    else:
        labels = df_results.index.astype(str).str.lower()
        percentages = df_results["percentage"].values
        counts = df_results.get("count", pd.Series([None] * len(df_results))).values

    colors = [RESULT_COLOR_MAP.get(lbl, PALETTE["neutral"]) for lbl in labels]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, percentages, color=colors, width=0.55, zorder=3,
                  edgecolor=PALETTE["bg"], linewidth=1.2)

    _add_value_labels(ax, fmt="{:.1f}%", padding=0.4)
    _style_axes(
        ax,
        title="Distribuição de Resultados",
        xlabel="Resultado",
        ylabel="Percentual de Partidas (%)",
    )
    ax.set_ylim(0, max(percentages) * 1.18)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))

    # Legenda com contagem absoluta (se disponível)
    if counts[0] is not None:
        legend_labels = [f"{lbl.capitalize()} ({int(c):,})" for lbl, c in zip(labels, counts)]
        legend_patches = [
            plt.Rectangle((0, 0), 1, 1, fc=c, ec=PALETTE["bg"], lw=1.2)
            for c in colors
        ]
        ax.legend(legend_patches, legend_labels, loc="upper right",
                  framealpha=0.85, fontsize=9)

    fig.tight_layout()
    logger.info("Gráfico 'result_distribution' gerado.")
    return fig

def plot_win_rate_by_hour(df_hours: pd.DataFrame) -> Figure:
    """Gera um gráfico de linha com a taxa de vitória por hora do dia.

    Args:
        df_hours: DataFrame retornado por ``analysis.win_rate_by_hour()``.
            Deve conter 'battle_hour' e 'win_rate_pct'.

    Returns:
        Figure do Matplotlib com o gráfico.

    Raises:
        KeyError: Se colunas necessárias estiverem ausentes.
    """
    _require_columns(df_hours, "battle_hour", "win_rate_pct",
                     context="plot_win_rate_by_hour")

    hours = df_hours["battle_hour"].values
    wr    = df_hours["win_rate_pct"].values

    fig, ax = plt.subplots(figsize=(12, 5))

    # Área preenchida sob a linha
    ax.fill_between(hours, wr, alpha=0.15, color=PALETTE["win"], zorder=2)

    # Linha principal
    ax.plot(hours, wr, color=PALETTE["win"], linewidth=2.5,
            marker="o", markersize=6, markerfacecolor=PALETTE["accent"],
            markeredgecolor=PALETTE["bg"], markeredgewidth=1.5,
            zorder=3, label="Win Rate (%)")

    # Linha de referência: média geral
    mean_wr = wr.mean()
    ax.axhline(mean_wr, color=PALETTE["accent"], linewidth=1.2,
               linestyle="--", alpha=0.75, label=f"Média: {mean_wr:.1f}%", zorder=2)

    # Anotação do pico
    peak_idx = wr.argmax()
    ax.annotate(
        f"Pico: {wr[peak_idx]:.1f}%",
        xy=(hours[peak_idx], wr[peak_idx]),
        xytext=(hours[peak_idx] + 0.6, wr[peak_idx] + 1.5),
        fontsize=9,
        color=PALETTE["accent"],
        arrowprops={"arrowstyle": "->", "color": PALETTE["accent"], "lw": 1.2},
    )

    _style_axes(
        ax,
        title="Taxa de Vitória por Hora do Dia",
        xlabel="Hora (0–23h)",
        ylabel="Win Rate (%)",
    )
    ax.set_xticks(range(0, 24))
    ax.set_xlim(-0.5, 23.5)
    ax.set_ylim(max(0, wr.min() - 5), min(100, wr.max() + 8))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))
    ax.legend(loc="lower right", fontsize=9, framealpha=0.85)

    fig.tight_layout()
    logger.info("Gráfico 'win_rate_by_hour' gerado.")
    return fig

def plot_most_used_cards(df_cards: pd.DataFrame, top_n: int = 15) -> Figure:
    """Gera um gráfico de barras horizontais com as cartas mais utilizadas.

    Args:
        df_cards: DataFrame retornado por ``analysis.most_used_cards()``.
            Deve conter 'card' e 'usage_count'.
        top_n: Quantidade de cartas a exibir (padrão: 15).

    Returns:
        Figure do Matplotlib com o gráfico.

    Raises:
        KeyError: Se colunas necessárias estiverem ausentes.
        ValueError: Se ``top_n`` for menor ou igual a zero.
    """
    _require_columns(df_cards, "card", "usage_count",
                     context="plot_most_used_cards")

    if top_n <= 0:
        raise ValueError(f"'top_n' deve ser > 0. Recebido: {top_n}.")

    df = df_cards.head(top_n).sort_values("usage_count", ascending=True)
    cards  = df["card"].values
    counts = df["usage_count"].values

    # Gradiente de cor: roxo épico → azul arena
    n = len(cards)
    bar_colors = [
        plt.cm.cool(i / max(n - 1, 1))  # type: ignore[attr-defined]
        for i in range(n)
    ]

    fig, ax = plt.subplots(figsize=(10, max(5, n * 0.48)))
    bars = ax.barh(cards, counts, color=bar_colors, height=0.7,
                   edgecolor=PALETTE["bg"], linewidth=0.8, zorder=3)

    # Rótulos inline à direita de cada barra
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_width() + max(counts) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{count:,}",
            va="center", ha="left",
            fontsize=9, fontweight="bold",
            color=PALETTE["text"],
        )

    _style_axes(
        ax,
        title=f"Top {top_n} Cartas Mais Utilizadas",
        xlabel="Número de Usos",
        ylabel="Carta",
    )
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.set_xlim(0, max(counts) * 1.15)
    ax.grid(axis="y", alpha=0)  # remove grade horizontal para gráfico horizontal

    fig.tight_layout()
    logger.info("Gráfico 'most_used_cards' gerado (top %d).", top_n)
    return fig

def plot_cards_win_rate(df_cards: pd.DataFrame, top_n: int = 15) -> Figure:
    """Gera um gráfico de barras horizontais com o win rate por carta.

    Args:
        df_cards: DataFrame retornado por ``analysis.cards_win_rate()``.
            Deve conter 'card', 'win_rate_pct' e 'appearances'.
        top_n: Quantidade de cartas a exibir (padrão: 15).

    Returns:
        Figure do Matplotlib com o gráfico.

    Raises:
        KeyError: Se colunas necessárias estiverem ausentes.
    """
    _require_columns(df_cards, "card", "win_rate_pct",
                     context="plot_cards_win_rate")

    df = df_cards.head(top_n).sort_values("win_rate_pct", ascending=True)
    cards   = df["card"].values
    wr_vals = df["win_rate_pct"].values
    appears = df["appearances"].values if "appearances" in df.columns else [None] * len(df)

    # Cor condicional: ouro para cartas acima de 55 %, azul para o restante
    bar_colors = [
        PALETTE["accent"] if v >= 55 else PALETTE["win"]
        for v in wr_vals
    ]

    fig, ax = plt.subplots(figsize=(10, max(5, len(cards) * 0.48)))
    bars = ax.barh(cards, wr_vals, color=bar_colors, height=0.7,
                   edgecolor=PALETTE["bg"], linewidth=0.8, zorder=3)

    # Linha de referência em 50 %
    ax.axvline(50, color=PALETTE["loss"], linewidth=1.2,
               linestyle="--", alpha=0.7, label="50% (equilíbrio)")

    # Rótulo com win rate e aparições
    for bar, wr, app in zip(bars, wr_vals, appears):
        label = f"{wr:.1f}%" + (f"  ({int(app):,} usos)" if app is not None else "")
        ax.text(
            bar.get_width() + 0.3,
            bar.get_y() + bar.get_height() / 2,
            label,
            va="center", ha="left",
            fontsize=9, fontweight="bold",
            color=PALETTE["text"],
        )

    _style_axes(
        ax,
        title=f"Top {top_n} Cartas por Win Rate",
        xlabel="Win Rate (%)",
        ylabel="Carta",
    )
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))
    ax.set_xlim(0, min(100, max(wr_vals) + 12))
    ax.legend(loc="lower right", fontsize=9, framealpha=0.85)
    ax.grid(axis="y", alpha=0)

    fig.tight_layout()
    logger.info("Gráfico 'cards_win_rate' gerado (top %d).", top_n)
    return fig


# ---------------------------------------------------------------------------
# 5. Histograma de troféus
# ---------------------------------------------------------------------------

def plot_trophy_histogram(
    df: pd.DataFrame,
    column: str = "player_starting_trophies",
    bins: int = 40,
) -> Figure:
    """Gera um histograma da distribuição de troféus iniciais dos jogadores.

    Args:
        df: DataFrame com a coluna de troféus (saída de preprocessing/feature_engineering).
        column: Nome da coluna de troféus (padrão: 'player_starting_trophies').
        bins: Número de intervalos do histograma (padrão: 40).

    Returns:
        Figure do Matplotlib com o gráfico.

    Raises:
        KeyError: Se a coluna não existir no DataFrame.
        ValueError: Se ``bins`` for menor ou igual a zero.
    """
    _require_columns(df, column, context="plot_trophy_histogram")

    if bins <= 0:
        raise ValueError(f"'bins' deve ser > 0. Recebido: {bins}.")

    data = df[column].dropna()

    fig, ax = plt.subplots(figsize=(11, 5))

    n, bin_edges, patches = ax.hist(
        data, bins=bins,
        color=PALETTE["accent"],
        edgecolor=PALETTE["bg"],
        linewidth=0.6,
        zorder=3,
        alpha=0.9,
    )

    # Gradiente de cor do histograma: mais escuro nas pontas
    max_n = n.max()
    for patch, height in zip(patches, n):
        intensity = height / max_n if max_n else 0
        patch.set_facecolor(plt.cm.YlOrRd(0.3 + intensity * 0.65))  # type: ignore[attr-defined]

    # Linhas de referência: média e mediana
    mean_val   = data.mean()
    median_val = data.median()

    ax.axvline(mean_val, color=PALETTE["win"], linewidth=1.8,
               linestyle="--", label=f"Média: {mean_val:,.0f}")
    ax.axvline(median_val, color=PALETTE["accent_alt"], linewidth=1.8,
               linestyle="-.", label=f"Mediana: {median_val:,.0f}")

    _style_axes(
        ax,
        title="Distribuição de Troféus Iniciais",
        xlabel="Troféus",
        ylabel="Número de Partidas",
    )
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.legend(loc="upper right", fontsize=9, framealpha=0.85)

    # Caixa de estatísticas
    stats_text = (
        f"N:      {len(data):,}\n"
        f"Mín:    {data.min():,.0f}\n"
        f"Máx:    {data.max():,.0f}\n"
        f"Desvio: {data.std():,.0f}"
    )
    ax.text(
        0.99, 0.97, stats_text,
        transform=ax.transAxes,
        fontsize=8.5,
        verticalalignment="top",
        horizontalalignment="right",
        color=PALETTE["text_dim"],
        bbox={"boxstyle": "round,pad=0.4", "fc": PALETTE["surface"],
              "ec": PALETTE["grid"], "alpha": 0.85},
    )

    fig.tight_layout()
    logger.info("Histograma 'trophy_histogram' gerado (%d bins).", bins)
    return fig

def plot_all(
    analysis_results: dict[str, pd.DataFrame],
    df_raw: pd.DataFrame | None = None,
    top_n: int = 15,
) -> dict[str, Figure]:
    """Gera todos os gráficos disponíveis e retorna um dicionário de Figures.

    Args:
        analysis_results: Dicionário retornado por ``analysis.run_all()``.
        df_raw: DataFrame com dados brutos/enriquecidos (necessário para o histograma).
            Se None, o histograma é ignorado.
        top_n: Quantidade de itens nos rankings de cartas (padrão: 15).

    Returns:
        Dicionário ``{nome: Figure}`` com as chaves:
            - ``"result_distribution"``
            - ``"win_rate_by_hour"``
            - ``"most_used_cards"``
            - ``"cards_win_rate"``
            - ``"trophy_histogram"`` (somente se df_raw for fornecido)

    Example:
        >>> from analysis import run_all
        >>> from visualization import plot_all
        >>> figs = plot_all(run_all(df), df_raw=df)
        >>> figs["result_distribution"].savefig("results.png", dpi=150)
    """
    logger.info("=== Gerando todos os gráficos ===")
    figures: dict[str, Figure] = {}

    if "result_distribution" in analysis_results:
        figures["result_distribution"] = plot_result_distribution(
            analysis_results["result_distribution"]
        )
    if "win_rate_by_hour" in analysis_results:
        figures["win_rate_by_hour"] = plot_win_rate_by_hour(
            analysis_results["win_rate_by_hour"]
        )
    if "most_used_cards" in analysis_results:
        figures["most_used_cards"] = plot_most_used_cards(
            analysis_results["most_used_cards"], top_n=top_n
        )
    if "cards_win_rate" in analysis_results:
        figures["cards_win_rate"] = plot_cards_win_rate(
            analysis_results["cards_win_rate"], top_n=top_n
        )
    if df_raw is not None:
        figures["trophy_histogram"] = plot_trophy_histogram(df_raw)

    logger.info("=== %d gráfico(s) gerado(s). ===", len(figures))
    return figures


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from preprocessing import preprocess                 # type: ignore[import]
    from feature_engineering import create_features      # type: ignore[import]
    from analysis import run_all                         # type: ignore[import]

    if len(sys.argv) < 2:
        print("Uso: python visualization.py <caminho_do_csv> [pasta_saída]")
        sys.exit(1)

    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(".")
    output_dir.mkdir(parents=True, exist_ok=True)

    df_clean  = preprocess(sys.argv[1])
    df_feat   = create_features(df_clean)
    results   = run_all(df_feat)
    figures   = plot_all(results, df_raw=df_feat)

    for name, fig in figures.items():
        out_path = output_dir / f"{name}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"  Salvo: {out_path}")

    print(f"\n{len(figures)} gráfico(s) exportado(s) para '{output_dir}'.")