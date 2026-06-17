"""
analysis.py
-----------
Módulo de análise estatística de partidas de Clash Royale.

Responsabilidades:
    - Calcular métricas agregadas a partir de um DataFrame enriquecido.
    - Retornar DataFrames prontos para visualização ou exportação.
    - Não gerar gráficos (responsabilidade do módulo de visualização).

Dependências:
    - pandas >= 1.5

Pipeline esperado:
    preprocessing.preprocess() → feature_engineering.create_features() → analysis.*
"""

from __future__ import annotations

import logging
from typing import Literal

import pandas as pd

# ---------------------------------------------------------------------------
# Configuração de logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

# Limites (inclusive) das faixas de troféus
TROPHY_BINS: list[int] = [0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 99_999]
TROPHY_LABELS: list[str] = [
    "0–1k", "1k–2k", "2k–3k", "3k–4k", "4k–5k",
    "5k–6k", "6k–7k", "7k–8k", "8k–9k", "9k+",
]

RESULT_COLUMN = "result"
WIN_BINARY_COLUMN = "win_binary"
HOUR_COLUMN = "battle_hour"
PLAYER_CROWNS_COLUMN = "player_crowns"
OPPONENT_CROWNS_COLUMN = "opponent_crowns"
PLAYER_DECK_COLUMN = "player_deck"
PLAYER_TROPHIES_COLUMN = "player_starting_trophies"


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------

def _require_columns(df: pd.DataFrame, *columns: str) -> None:
    """Verifica se todas as colunas necessárias existem no DataFrame.

    Args:
        df: DataFrame a ser verificado.
        *columns: Nomes das colunas obrigatórias.

    Raises:
        KeyError: Se alguma coluna estiver ausente.
    """
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(
            f"Colunas obrigatórias ausentes no DataFrame: {missing}. "
            "Verifique se o pipeline de pré-processamento e feature engineering foi executado."
        )


def _parse_deck(deck_series: pd.Series) -> pd.Series:
    """Converte a coluna de deck (string separada por vírgula) em listas de cartas.

    Args:
        deck_series: Série com decks no formato 'carta1,carta2,...,carta8'.

    Returns:
        Série com listas de strings (nomes de cartas).
    """
    return deck_series.astype(str).str.strip().str.split(",")


# ---------------------------------------------------------------------------
# 1. Taxa geral de vitória
# ---------------------------------------------------------------------------

def overall_win_rate(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula a taxa geral de vitória do jogador.

    Args:
        df: DataFrame enriquecido com a coluna 'win_binary'.

    Returns:
        DataFrame com colunas:
            - ``total_battles``  : número total de partidas.
            - ``total_wins``     : número de vitórias.
            - ``total_losses``   : número de derrotas/empates.
            - ``win_rate``       : taxa de vitória (0.0 – 1.0).
            - ``win_rate_pct``   : taxa de vitória em percentual (0.0 – 100.0).

    Raises:
        KeyError: Se 'win_binary' não estiver presente.

    Example:
        >>> df_rate = overall_win_rate(df)
        >>> print(df_rate["win_rate_pct"].iloc[0])
        57.3
    """
    _require_columns(df, WIN_BINARY_COLUMN)

    total = len(df)
    wins = int(df[WIN_BINARY_COLUMN].sum())

    result = pd.DataFrame([{
        "total_battles": total,
        "total_wins": wins,
        "total_losses": total - wins,
        "win_rate": round(wins / total, 4) if total else 0.0,
        "win_rate_pct": round(wins / total * 100, 2) if total else 0.0,
    }])

    logger.info(
        "Taxa geral de vitória: %.2f%% (%d/%d partidas).",
        result["win_rate_pct"].iloc[0], wins, total,
    )
    return result


# ---------------------------------------------------------------------------
# 2. Distribuição de resultados
# ---------------------------------------------------------------------------

def result_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula a distribuição absoluta e relativa dos resultados das partidas.

    Args:
        df: DataFrame com a coluna 'result'.

    Returns:
        DataFrame indexado pelo valor de 'result' com colunas:
            - ``count``       : número de ocorrências.
            - ``percentage``  : percentual sobre o total (0.0 – 100.0).

    Raises:
        KeyError: Se 'result' não estiver presente.

    Example:
        >>> dist = result_distribution(df)
        >>> dist.loc["win", "percentage"]
        57.3
    """
    _require_columns(df, RESULT_COLUMN)

    counts = df[RESULT_COLUMN].value_counts()
    pct = (counts / len(df) * 100).round(2)

    result = pd.DataFrame({
        "count": counts,
        "percentage": pct,
    })
    result.index.name = RESULT_COLUMN

    logger.info("Distribuição de resultados calculada: %s.", result["count"].to_dict())
    return result


# ---------------------------------------------------------------------------
# 3. Taxa de vitória por faixa de troféus
# ---------------------------------------------------------------------------

def win_rate_by_trophy_range(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula a taxa de vitória segmentada por faixa de troféus iniciais.

    As faixas são definidas pela constante ``TROPHY_BINS`` (intervalos de 1.000 troféus).

    Args:
        df: DataFrame com 'player_starting_trophies' e 'win_binary'.

    Returns:
        DataFrame com colunas:
            - ``trophy_range``  : rótulo da faixa (ex.: '3k–4k').
            - ``total_battles`` : partidas na faixa.
            - ``wins``          : vitórias na faixa.
            - ``win_rate``      : taxa de vitória (0.0 – 1.0).
            - ``win_rate_pct``  : taxa de vitória em percentual.

    Raises:
        KeyError: Se colunas necessárias estiverem ausentes.
    """
    _require_columns(df, PLAYER_TROPHIES_COLUMN, WIN_BINARY_COLUMN)

    df = df.copy()
    df["trophy_range"] = pd.cut(
        df[PLAYER_TROPHIES_COLUMN],
        bins=TROPHY_BINS,
        labels=TROPHY_LABELS,
        right=False,
    )

    grouped = (
        df.groupby("trophy_range", observed=True)[WIN_BINARY_COLUMN]
        .agg(total_battles="count", wins="sum")
        .reset_index()
    )
    grouped["win_rate"] = (grouped["wins"] / grouped["total_battles"]).round(4)
    grouped["win_rate_pct"] = (grouped["win_rate"] * 100).round(2)
    grouped = grouped.rename(columns={"trophy_range": "trophy_range"})

    logger.info("Taxa de vitória calculada para %d faixa(s) de troféus.", len(grouped))
    return grouped


# ---------------------------------------------------------------------------
# 4. Taxa de vitória por hora
# ---------------------------------------------------------------------------

def win_rate_by_hour(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula a taxa de vitória agrupada pela hora do dia (0–23).

    Args:
        df: DataFrame com 'battle_hour' e 'win_binary'.

    Returns:
        DataFrame com colunas:
            - ``battle_hour``   : hora do dia (0–23).
            - ``total_battles`` : partidas nessa hora.
            - ``wins``          : vitórias nessa hora.
            - ``win_rate``      : taxa de vitória (0.0 – 1.0).
            - ``win_rate_pct``  : taxa de vitória em percentual.

    Raises:
        KeyError: Se colunas necessárias estiverem ausentes.
    """
    _require_columns(df, HOUR_COLUMN, WIN_BINARY_COLUMN)

    grouped = (
        df.groupby(HOUR_COLUMN)[WIN_BINARY_COLUMN]
        .agg(total_battles="count", wins="sum")
        .reset_index()
        .rename(columns={HOUR_COLUMN: "battle_hour"})
    )
    grouped["win_rate"] = (grouped["wins"] / grouped["total_battles"]).round(4)
    grouped["win_rate_pct"] = (grouped["win_rate"] * 100).round(2)
    grouped = grouped.sort_values("battle_hour").reset_index(drop=True)

    logger.info("Taxa de vitória calculada para %d hora(s) distintas.", len(grouped))
    return grouped


# ---------------------------------------------------------------------------
# 5. Cartas mais utilizadas
# ---------------------------------------------------------------------------

def most_used_cards(
    df: pd.DataFrame,
    top_n: int = 20,
    deck_column: Literal["player_deck", "opponent_deck"] = "player_deck",
) -> pd.DataFrame:
    """Identifica as cartas mais utilizadas nos decks.

    Expande a coluna de deck (string separada por vírgula) e conta
    a frequência de cada carta individualmente.

    Args:
        df: DataFrame com a coluna de deck especificada.
        top_n: Número de cartas a retornar (padrão: 20).
        deck_column: Coluna de deck a analisar ('player_deck' ou 'opponent_deck').

    Returns:
        DataFrame com colunas:
            - ``card``            : nome da carta.
            - ``usage_count``     : número de aparições nos decks.
            - ``usage_rate``      : taxa de aparição sobre o total de partidas (0.0 – 1.0).
            - ``usage_rate_pct``  : taxa de aparição em percentual.

    Raises:
        KeyError: Se a coluna de deck não estiver presente.
        ValueError: Se ``top_n`` for menor ou igual a zero.
    """
    _require_columns(df, deck_column)

    if top_n <= 0:
        raise ValueError(f"'top_n' deve ser maior que zero. Recebido: {top_n}.")

    cards = _parse_deck(df[deck_column]).explode().str.strip()
    cards = cards[cards != ""]  # remove entradas vazias

    counts = cards.value_counts().head(top_n).reset_index()
    counts.columns = pd.Index(["card", "usage_count"])
    counts["usage_rate"] = (counts["usage_count"] / len(df)).round(4)
    counts["usage_rate_pct"] = (counts["usage_rate"] * 100).round(2)

    logger.info(
        "Top %d cartas mais usadas calculadas a partir de '%s'.", top_n, deck_column
    )
    return counts.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 6. Cartas com maior win rate
# ---------------------------------------------------------------------------

def cards_win_rate(
    df: pd.DataFrame,
    top_n: int = 20,
    min_appearances: int = 50,
    deck_column: Literal["player_deck", "opponent_deck"] = "player_deck",
) -> pd.DataFrame:
    """Calcula o win rate individual de cada carta presente nos decks.

    Para cada partida, associa o resultado (win_binary) a cada carta do deck.
    Filtra cartas com poucas aparições para evitar resultados estatisticamente
    não representativos.

    Args:
        df: DataFrame com a coluna de deck e 'win_binary'.
        top_n: Número de cartas a retornar ordenadas por win rate (padrão: 20).
        min_appearances: Número mínimo de aparições para incluir uma carta
            (padrão: 50). Evita ruído estatístico em cartas raras.
        deck_column: Coluna de deck a analisar ('player_deck' ou 'opponent_deck').

    Returns:
        DataFrame com colunas:
            - ``card``           : nome da carta.
            - ``appearances``    : total de aparições nos decks.
            - ``wins``           : partidas vencidas com a carta no deck.
            - ``win_rate``       : taxa de vitória (0.0 – 1.0).
            - ``win_rate_pct``   : taxa de vitória em percentual.

    Raises:
        KeyError: Se colunas necessárias estiverem ausentes.
        ValueError: Se ``min_appearances`` for negativo.
    """
    _require_columns(df, deck_column, WIN_BINARY_COLUMN)

    if min_appearances < 0:
        raise ValueError(
            f"'min_appearances' deve ser >= 0. Recebido: {min_appearances}."
        )

    # Expande deck em linhas individuais mantendo o win_binary correspondente
    expanded = df[[deck_column, WIN_BINARY_COLUMN]].copy()
    expanded["card"] = _parse_deck(expanded[deck_column])
    expanded = expanded.explode("card")
    expanded["card"] = expanded["card"].str.strip()
    expanded = expanded[expanded["card"] != ""]

    grouped = (
        expanded.groupby("card")[WIN_BINARY_COLUMN]
        .agg(appearances="count", wins="sum")
        .reset_index()
    )

    # Filtra cartas com poucas aparições
    grouped = grouped[grouped["appearances"] >= min_appearances].copy()

    grouped["win_rate"] = (grouped["wins"] / grouped["appearances"]).round(4)
    grouped["win_rate_pct"] = (grouped["win_rate"] * 100).round(2)
    grouped = (
        grouped.sort_values("win_rate", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    logger.info(
        "Win rate calculado para %d carta(s) (mín. %d aparições).",
        len(grouped), min_appearances,
    )
    return grouped


# ---------------------------------------------------------------------------
# 7. Média de coroas por resultado
# ---------------------------------------------------------------------------

def avg_crowns_by_result(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula a média de coroas obtidas e sofridas por tipo de resultado.

    Args:
        df: DataFrame com 'result', 'player_crowns' e 'opponent_crowns'.

    Returns:
        DataFrame com colunas:
            - ``result``               : tipo de resultado (win, loss, etc.).
            - ``avg_player_crowns``    : média de coroas do jogador.
            - ``avg_opponent_crowns``  : média de coroas do oponente.
            - ``avg_crown_difference`` : diferença média (player − opponent).
            - ``battle_count``         : número de partidas no grupo.

    Raises:
        KeyError: Se colunas necessárias estiverem ausentes.
    """
    _require_columns(df, RESULT_COLUMN, PLAYER_CROWNS_COLUMN, OPPONENT_CROWNS_COLUMN)

    grouped = (
        df.groupby(RESULT_COLUMN)
        .agg(
            avg_player_crowns=(PLAYER_CROWNS_COLUMN, "mean"),
            avg_opponent_crowns=(OPPONENT_CROWNS_COLUMN, "mean"),
            battle_count=(PLAYER_CROWNS_COLUMN, "count"),
        )
        .reset_index()
    )
    grouped["avg_player_crowns"] = grouped["avg_player_crowns"].round(3)
    grouped["avg_opponent_crowns"] = grouped["avg_opponent_crowns"].round(3)
    grouped["avg_crown_difference"] = (
        grouped["avg_player_crowns"] - grouped["avg_opponent_crowns"]
    ).round(3)

    logger.info(
        "Média de coroas calculada para %d resultado(s) distintos.", len(grouped)
    )
    return grouped


# ---------------------------------------------------------------------------
# Função agregadora: executa todas as análises de uma vez
# ---------------------------------------------------------------------------

def run_all(
    df: pd.DataFrame,
    top_n_cards: int = 20,
    min_card_appearances: int = 50,
) -> dict[str, pd.DataFrame]:
    """Executa todas as análises disponíveis e retorna um dicionário de DataFrames.

    Conveniente para relatórios ou exportações em lote.

    Args:
        df: DataFrame enriquecido (saída de ``create_features``).
        top_n_cards: Número de cartas a retornar nos rankings (padrão: 20).
        min_card_appearances: Mínimo de aparições para o ranking de win rate
            de cartas (padrão: 50).

    Returns:
        Dicionário com as chaves:
            - ``"overall_win_rate"``
            - ``"result_distribution"``
            - ``"win_rate_by_trophy_range"``
            - ``"win_rate_by_hour"``
            - ``"most_used_cards"``
            - ``"cards_win_rate"``
            - ``"avg_crowns_by_result"``

    Example:
        >>> results = run_all(df_features)
        >>> results["overall_win_rate"]
           total_battles  total_wins  ...  win_rate_pct
        0          10000        5730  ...          57.3
    """
    logger.info("=== Executando todas as análises ===")

    analyses: dict[str, pd.DataFrame] = {
        "overall_win_rate": overall_win_rate(df),
        "result_distribution": result_distribution(df),
        "win_rate_by_trophy_range": win_rate_by_trophy_range(df),
        "win_rate_by_hour": win_rate_by_hour(df),
        "most_used_cards": most_used_cards(df, top_n=top_n_cards),
        "cards_win_rate": cards_win_rate(df, top_n=top_n_cards, min_appearances=min_card_appearances),
        "avg_crowns_by_result": avg_crowns_by_result(df),
    }

    logger.info("=== Todas as análises concluídas: %d módulos. ===", len(analyses))
    return analyses


# ---------------------------------------------------------------------------
# Entry point para testes rápidos via CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from preprocessing import preprocess          # type: ignore[import]
    from feature_engineering import create_features  # type: ignore[import]

    if len(sys.argv) < 2:
        print("Uso: python analysis.py <caminho_do_csv>")
        sys.exit(1)

    df_clean = preprocess(sys.argv[1])
    df_feat = create_features(df_clean)
    results = run_all(df_feat)

    for name, frame in results.items():
        print(f"\n{'=' * 60}")
        print(f" {name.upper()}")
        print("=" * 60)
        print(frame.to_string(index=False))