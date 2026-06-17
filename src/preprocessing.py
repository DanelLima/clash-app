"""
preprocessing.py
----------------
Módulo de pré-processamento de dados para análise de partidas de Clash Royale.

Responsabilidades:
    - Carregamento de CSV
    - Remoção de duplicatas
    - Tratamento de valores nulos
    - Conversão e padronização de tipos
    - Padronização de nomes de colunas
"""

from __future__ import annotations

import logging
from pathlib import Path

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
DATETIME_COLUMN = "battle_time"
DATETIME_FORMAT = "%Y%m%dT%H%M%S.%fZ"  # ex.: 20240315T183045.000Z

EXPECTED_COLUMNS: list[str] = [
    "battle_time",
    "battle_type",
    "game_mode",
    "player_tag",
    "player_name",
    "player_crowns",
    "player_starting_trophies",
    "opponent_tag",
    "opponent_name",
    "opponent_crowns",
    "opponent_starting_trophies",
    "result",
    "player_deck",
    "opponent_deck",
    "player_deck_hash",
    "opponent_deck_hash",
    "player_king_tower_hp",
    "opponent_king_tower_hp",
]

# Valores padrão para preenchimento de nulos por tipo de coluna
NULL_FILL_DEFAULTS: dict[str, object] = {
    "player_crowns": 0,
    "opponent_crowns": 0,
    "player_starting_trophies": 0,
    "opponent_starting_trophies": 0,
    "player_king_tower_hp": 0,
    "opponent_king_tower_hp": 0,
    "player_name": "Unknown",
    "opponent_name": "Unknown",
    "battle_type": "unknown",
    "game_mode": "unknown",
    "result": "unknown",
    "player_deck": "",
    "opponent_deck": "",
    "player_deck_hash": "",
    "opponent_deck_hash": "",
}

# ---------------------------------------------------------------------------
# Funções auxiliares
# ---------------------------------------------------------------------------

def _validate_columns(df: pd.DataFrame) -> None:
    """Valida se o DataFrame contém todas as colunas esperadas.

    Args:
        df: DataFrame a ser validado.

    Raises:
        ValueError: Se alguma coluna esperada estiver ausente.
    """
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(
            f"As seguintes colunas esperadas estão ausentes no arquivo: {missing}"
        )
    logger.info("Validação de colunas concluída — todas as %d colunas presentes.", len(EXPECTED_COLUMNS))


# ---------------------------------------------------------------------------
# Funções principais
# ---------------------------------------------------------------------------

def load_csv(filepath: str | Path, encoding: str = "utf-8") -> pd.DataFrame:
    """Carrega um arquivo CSV em um DataFrame do Pandas.

    Args:
        filepath: Caminho para o arquivo CSV.
        encoding: Codificação do arquivo (padrão: 'utf-8').

    Returns:
        DataFrame com os dados brutos do arquivo.

    Raises:
        FileNotFoundError: Se o arquivo não existir no caminho informado.
        ValueError: Se o arquivo estiver vazio ou com colunas inválidas.
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {filepath}")

    logger.info("Carregando arquivo: %s", filepath)
    df = pd.read_csv(filepath, encoding=encoding)

    if df.empty:
        raise ValueError(f"O arquivo '{filepath}' está vazio.")

    logger.info("Arquivo carregado — %d linhas × %d colunas.", *df.shape)
    _validate_columns(df)
    return df


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Padroniza os nomes das colunas para snake_case minúsculo.

    Remove espaços extras, converte para minúsculas e substitui
    espaços/hífens por underscores.

    Args:
        df: DataFrame com colunas originais.

    Returns:
        DataFrame com nomes de colunas padronizados.
    """
    original_columns = df.columns.tolist()

    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"[\s\-]+", "_", regex=True)
        .str.replace(r"[^\w]", "", regex=True)
    )

    renamed = {old: new for old, new in zip(original_columns, df.columns) if old != new}
    if renamed:
        logger.info("Colunas renomeadas: %s", renamed)
    else:
        logger.info("Nenhuma coluna precisou ser renomeada.")

    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove linhas duplicadas do DataFrame.

    Considera duplicatas como linhas com mesma combinação de
    'battle_time', 'player_tag' e 'opponent_tag'.

    Args:
        df: DataFrame possivelmente com duplicatas.

    Returns:
        DataFrame sem linhas duplicadas.
    """
    subset = ["battle_time", "player_tag", "opponent_tag"]
    # Usa subset apenas se todas as colunas estiverem presentes
    available_subset = [col for col in subset if col in df.columns]

    before = len(df)
    df = df.drop_duplicates(subset=available_subset or None, keep="first").copy()
    removed = before - len(df)

    if removed:
        logger.info("Duplicatas removidas: %d linha(s).", removed)
    else:
        logger.info("Nenhuma duplicata encontrada.")

    return df


def handle_null_values(df: pd.DataFrame) -> pd.DataFrame:
    """Trata valores nulos no DataFrame.

    Estratégia por coluna:
        - Colunas numéricas de HP e troféus → preenchidas com 0.
        - Colunas de crowns                 → preenchidas com 0.
        - Colunas de texto/categóricas      → preenchidas com 'Unknown' ou 'unknown'.
        - Linhas sem 'player_tag' ou 'opponent_tag' → removidas (chaves primárias).

    Args:
        df: DataFrame com possíveis valores nulos.

    Returns:
        DataFrame com valores nulos tratados.
    """
    df = df.copy()

    # Remove linhas sem identificadores essenciais
    essential_cols = [c for c in ["player_tag", "opponent_tag"] if c in df.columns]
    before = len(df)
    df = df.dropna(subset=essential_cols)
    dropped = before - len(df)
    if dropped:
        logger.warning(
            "%d linha(s) removida(s) por ausência de 'player_tag' ou 'opponent_tag'.",
            dropped,
        )

    # Preenchimento de nulos por coluna
    for column, default_value in NULL_FILL_DEFAULTS.items():
        if column in df.columns and df[column].isna().any():
            null_count = df[column].isna().sum()
            df[column] = df[column].fillna(default_value)
            logger.info(
                "Coluna '%s': %d valor(es) nulo(s) preenchido(s) com '%s'.",
                column,
                null_count,
                default_value,
            )

    remaining_nulls = df.isna().sum().sum()
    if remaining_nulls:
        logger.warning(
            "%d valor(es) nulo(s) restantes não cobertos pela política padrão.",
            remaining_nulls,
        )

    return df


def convert_datetime(
    df: pd.DataFrame,
    column: str = DATETIME_COLUMN,
    fmt: str | None = DATETIME_FORMAT,
) -> pd.DataFrame:
    """Converte a coluna de data/hora para o tipo datetime do Pandas.

    Args:
        df: DataFrame com a coluna de data/hora como string.
        column: Nome da coluna a ser convertida (padrão: 'battle_time').
        fmt: Formato strptime esperado. Se None, infere automaticamente.

    Returns:
        DataFrame com a coluna convertida para datetime64[ns].

    Raises:
        KeyError: Se a coluna informada não existir no DataFrame.
    """
    if column not in df.columns:
        raise KeyError(f"Coluna '{column}' não encontrada no DataFrame.")

    df = df.copy()

    try:
        df[column] = pd.to_datetime(df[column], format=fmt, utc=True, errors="raise")
        logger.info("Coluna '%s' convertida para datetime com formato '%s'.", column, fmt)
    except Exception:
        logger.warning(
            "Formato '%s' falhou para '%s'. Tentando inferência automática...",
            fmt,
            column,
        )
        df[column] = pd.to_datetime(df[column], infer_datetime_format=True, utc=True, errors="coerce")
        coerced = df[column].isna().sum()
        if coerced:
            logger.warning(
                "%d valor(es) em '%s' não puderam ser convertidos e foram definidos como NaT.",
                coerced,
                column,
            )

    return df


def set_optimal_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Converte colunas para tipos de dados mais eficientes em memória.

    Args:
        df: DataFrame pré-processado.

    Returns:
        DataFrame com tipos otimizados.
    """
    df = df.copy()

    integer_cols = [
        "player_crowns",
        "opponent_crowns",
        "player_starting_trophies",
        "opponent_starting_trophies",
        "player_king_tower_hp",
        "opponent_king_tower_hp",
    ]
    for col in integer_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("int32")

    categorical_cols = ["battle_type", "game_mode", "result"]
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")

    logger.info("Tipos de dados otimizados.")
    return df


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def preprocess(filepath: str | Path, encoding: str = "utf-8") -> pd.DataFrame:
    """Executa o pipeline completo de pré-processamento.

    Etapas (em ordem):
        1. Carregamento do CSV.
        2. Padronização dos nomes de colunas.
        3. Remoção de duplicatas.
        4. Tratamento de valores nulos.
        5. Conversão de battle_time para datetime.
        6. Otimização de tipos de dados.

    Args:
        filepath: Caminho para o arquivo CSV de partidas.
        encoding: Codificação do arquivo CSV (padrão: 'utf-8').

    Returns:
        DataFrame limpo e pronto para análise.

    Example:
        >>> df = preprocess("data/clash_royale_battles.csv")
        >>> df.dtypes
        battle_time    datetime64[ns, UTC]
        result                    category
        ...
    """
    logger.info("=== Iniciando pipeline de pré-processamento ===")

    df = load_csv(filepath, encoding=encoding)
    df = standardize_column_names(df)
    df = remove_duplicates(df)
    df = handle_null_values(df)
    df = convert_datetime(df)
    df = set_optimal_dtypes(df)

    logger.info(
        "=== Pipeline concluído — DataFrame final: %d linhas × %d colunas ===",
        *df.shape,
    )
    return df


# ---------------------------------------------------------------------------
# Entry point para testes rápidos via CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Uso: python preprocessing.py <caminho_do_csv>")
        sys.exit(1)

    clean_df = preprocess(sys.argv[1])
    print("\n--- Informações do DataFrame limpo ---")
    print(clean_df.info())
    print("\n--- Primeiras linhas ---")
    print(clean_df.head())