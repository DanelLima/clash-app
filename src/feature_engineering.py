"""
feature_engineering.py
-----------------------
Módulo de engenharia de features para análise de partidas de Clash Royale.

Responsabilidades:
    - Criar features derivadas a partir de um DataFrame já limpo.
    - Enriquecer o DataFrame com informações contextuais e estatísticas.

Dependências:
    - pandas >= 1.5
    - numpy >= 1.23
"""

from __future__ import annotations

import logging

import numpy as np
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

# Mapeamento de número do dia da semana para nome (en → pt-BR)
WEEKDAY_NAMES: dict[int, str] = {
    0: "segunda",
    1: "terça",
    2: "quarta",
    3: "quinta",
    4: "sexta",
    5: "sábado",
    6: "domingo",
}

WIN_VALUE = "win"          # valor esperado na coluna 'result' para vitória


# ---------------------------------------------------------------------------
# Funções de feature individuais
# ---------------------------------------------------------------------------

def add_trophy_difference(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula a diferença de troféus iniciais entre jogador e oponente.

    Feature: ``trophy_difference = player_starting_trophies - opponent_starting_trophies``

    Valores positivos indicam que o jogador entrou na partida com mais troféus;
    valores negativos indicam desvantagem relativa.

    Args:
        df: DataFrame limpo contendo 'player_starting_trophies' e
            'opponent_starting_trophies'.

    Returns:
        DataFrame com a coluna 'trophy_difference' adicionada (int32).
    """
    df = df.copy()
    df["trophy_difference"] = (
        df["player_starting_trophies"].astype(np.int32)
        - df["opponent_starting_trophies"].astype(np.int32)
    )
    logger.info("Feature criada: 'trophy_difference'.")
    return df


def add_battle_hour(df: pd.DataFrame) -> pd.DataFrame:
    """Extrai a hora do dia (0–23) a partir de 'battle_time'.

    Feature: ``battle_hour = battle_time.dt.hour``

    Útil para identificar padrões de comportamento por período do dia
    (madrugada, manhã, tarde, noite).

    Args:
        df: DataFrame com a coluna 'battle_time' no tipo datetime.

    Returns:
        DataFrame com a coluna 'battle_hour' adicionada (int8, 0–23).

    Raises:
        TypeError: Se 'battle_time' não for do tipo datetime.
    """
    df = df.copy()

    if not pd.api.types.is_datetime64_any_dtype(df["battle_time"]):
        raise TypeError(
            "A coluna 'battle_time' deve ser do tipo datetime. "
            "Execute convert_datetime() antes desta etapa."
        )

    df["battle_hour"] = df["battle_time"].dt.hour.astype(np.int8)
    logger.info("Feature criada: 'battle_hour'.")
    return df


def add_battle_weekday(df: pd.DataFrame) -> pd.DataFrame:
    """Extrai o dia da semana (0 = segunda … 6 = domingo) de 'battle_time'.

    Features criadas:
        - ``battle_weekday``      → inteiro 0–6 (int8).
        - ``battle_weekday_name`` → nome em português (categoria).

    Args:
        df: DataFrame com a coluna 'battle_time' no tipo datetime.

    Returns:
        DataFrame com as colunas 'battle_weekday' e 'battle_weekday_name' adicionadas.

    Raises:
        TypeError: Se 'battle_time' não for do tipo datetime.
    """
    df = df.copy()

    if not pd.api.types.is_datetime64_any_dtype(df["battle_time"]):
        raise TypeError(
            "A coluna 'battle_time' deve ser do tipo datetime. "
            "Execute convert_datetime() antes desta etapa."
        )

    df["battle_weekday"] = df["battle_time"].dt.dayofweek.astype(np.int8)
    df["battle_weekday_name"] = (
        df["battle_weekday"].map(WEEKDAY_NAMES).astype("category")
    )
    logger.info("Features criadas: 'battle_weekday', 'battle_weekday_name'.")
    return df


def add_win_binary(df: pd.DataFrame) -> pd.DataFrame:
    """Converte o resultado da partida em variável binária numérica.

    Feature: ``win_binary = 1`` se ``result == 'win'``, caso contrário ``0``.

    Ideal para uso como variável-alvo em modelos de classificação.

    Args:
        df: DataFrame com a coluna 'result' (string ou categoria).

    Returns:
        DataFrame com a coluna 'win_binary' adicionada (int8: 0 ou 1).
    """
    df = df.copy()
    df["win_binary"] = (
        df["result"].astype(str).str.lower().str.strip() == WIN_VALUE
    ).astype(np.int8)

    wins = df["win_binary"].sum()
    total = len(df)
    logger.info(
        "Feature criada: 'win_binary' — %d vitórias de %d partidas (%.1f%%).",
        wins,
        total,
        (wins / total * 100) if total else 0.0,
    )
    return df


def add_crown_difference(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula a diferença de coroas obtidas entre jogador e oponente.

    Feature: ``crown_difference = player_crowns - opponent_crowns``

    Reflete a dominância do jogador na partida:
        - ``> 0``: jogador destruiu mais torres.
        - ``= 0``: empate em coroas.
        - ``< 0``: oponente foi mais dominante.

    Args:
        df: DataFrame com 'player_crowns' e 'opponent_crowns'.

    Returns:
        DataFrame com a coluna 'crown_difference' adicionada (int8).
    """
    df = df.copy()
    df["crown_difference"] = (
        df["player_crowns"].astype(np.int8)
        - df["opponent_crowns"].astype(np.int8)
    )
    logger.info("Feature criada: 'crown_difference'.")
    return df


def add_tower_hp_difference(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula a diferença de HP da torre do rei entre jogador e oponente.

    Feature: ``tower_hp_difference = player_king_tower_hp - opponent_king_tower_hp``

    Indica quanto HP de vantagem o jogador teve ao final da partida:
        - Valores positivos → jogador sobreviveu com mais HP.
        - Valores negativos → oponente estava em melhor situação de HP.
        - ``0``             → HP de torre idêntico no encerramento.

    Args:
        df: DataFrame com 'player_king_tower_hp' e 'opponent_king_tower_hp'.

    Returns:
        DataFrame com a coluna 'tower_hp_difference' adicionada (int32).
    """
    df = df.copy()
    df["tower_hp_difference"] = (
        df["player_king_tower_hp"].astype(np.int32)
        - df["opponent_king_tower_hp"].astype(np.int32)
    )
    logger.info("Feature criada: 'tower_hp_difference'.")
    return df


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Executa o pipeline completo de engenharia de features.

    Aplica, em sequência, todas as transformações de enriquecimento sobre
    o DataFrame limpo recebido como entrada. O DataFrame original não é
    modificado (operações são feitas em cópias).

    Features adicionadas:
        - ``trophy_difference``    : diferença de troféus iniciais.
        - ``battle_hour``          : hora do dia da partida (0–23).
        - ``battle_weekday``       : dia da semana (0 = seg … 6 = dom).
        - ``battle_weekday_name``  : nome do dia da semana em pt-BR.
        - ``win_binary``           : vitória codificada como 1/0.
        - ``crown_difference``     : diferença de coroas obtidas.
        - ``tower_hp_difference``  : diferença de HP da torre do rei.

    Args:
        df: DataFrame limpo produzido pelo módulo ``preprocessing``.
            Deve conter as colunas base do dataset de Clash Royale com
            'battle_time' já convertida para datetime.

    Returns:
        DataFrame enriquecido com todas as features acima.

    Raises:
        TypeError: Se 'battle_time' não for do tipo datetime.
        KeyError:  Se alguma coluna base necessária estiver ausente.

    Example:
        >>> from preprocessing import preprocess
        >>> from feature_engineering import create_features
        >>>
        >>> df_clean = preprocess("data/clash_royale_battles.csv")
        >>> df_features = create_features(df_clean)
        >>> df_features[["trophy_difference", "win_binary", "crown_difference"]].head()
    """
    logger.info("=== Iniciando pipeline de feature engineering ===")
    logger.info("Shape de entrada: %d linhas × %d colunas.", *df.shape)

    pipeline = [
        add_trophy_difference,
        add_battle_hour,
        add_battle_weekday,
        add_win_binary,
        add_crown_difference,
        add_tower_hp_difference,
    ]

    df_enriched = df.copy()
    for step in pipeline:
        df_enriched = step(df_enriched)

    new_cols = df_enriched.columns.difference(df.columns).tolist()
    logger.info(
        "=== Pipeline concluído — %d feature(s) criada(s): %s ===",
        len(new_cols),
        new_cols,
    )
    return df_enriched


# ---------------------------------------------------------------------------
# Entry point para testes rápidos via CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent))
    from preprocessing import preprocess  # type: ignore[import]

    if len(sys.argv) < 2:
        print("Uso: python feature_engineering.py <caminho_do_csv>")
        sys.exit(1)

    df_clean = preprocess(sys.argv[1])
    df_final = create_features(df_clean)

    new_features = [
        "trophy_difference",
        "battle_hour",
        "battle_weekday",
        "battle_weekday_name",
        "win_binary",
        "crown_difference",
        "tower_hp_difference",
    ]
    print("\n--- Features criadas ---")
    print(df_final[new_features].head(10).to_string(index=False))
    print("\n--- Estatísticas descritivas ---")
    print(df_final[new_features].describe(include="all"))