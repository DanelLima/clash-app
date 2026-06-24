import logging
import pandas as pd

# configuracao do log para rastrear o processamento
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# colunas obrigatorias esperadas no CSV
COLUNAS_OBRIGATORIAS = [
    "battle_time",
    "player_crowns",
    "player_starting_trophies",
    "opponent_crowns",
    "opponent_starting_trophies",
    "result",
    "player_deck",
    "player_king_tower_hp",
    "opponent_king_tower_hp",
]


def carregar_csv(caminho: str) -> pd.DataFrame:
    """Le o CSV e valida as colunas obrigatorias."""
    df = pd.read_csv(caminho)
    logger.info("CSV carregado: %d linhas, %d colunas", len(df), len(df.columns))

    # verifica se as colunas necessarias existem
    faltando = [c for c in COLUNAS_OBRIGATORIAS if c not in df.columns]
    if faltando:
        raise ValueError(f"Colunas ausentes no CSV: {faltando}")

    return df


def limpar_dados(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicatas, trata nulos e corrige tipos."""
    df = df.copy()

    # remove linhas completamente duplicadas
    qtd_antes = len(df)
    df = df.drop_duplicates()
    logger.info("Duplicatas removidas: %d", qtd_antes - len(df))

    # preenche trofeus nulos com a mediana da coluna
    for coluna in ["player_starting_trophies", "opponent_starting_trophies"]:
        nulos = df[coluna].isna().sum()
        if nulos > 0:
            mediana = df[coluna].median()
            df[coluna] = df[coluna].fillna(mediana)
            logger.info("Coluna '%s': %d nulos preenchidos com mediana (%.0f)", coluna, nulos, mediana)

    # converte a data da batalha para datetime
    df["battle_time"] = pd.to_datetime(df["battle_time"], utc=True, errors="coerce")

    # remove linhas com data invalida
    invalidas = df["battle_time"].isna().sum()
    if invalidas > 0:
        df = df.dropna(subset=["battle_time"])
        logger.info("Linhas com data invalida removidas: %d", invalidas)

    # padroniza o resultado para maiusculas
    df["result"] = df["result"].str.upper().str.strip()

    logger.info("Limpeza concluida: %d linhas validas", len(df))
    return df


def preprocessar(caminho: str) -> pd.DataFrame:
    """Pipeline completo: carrega e limpa o dataset."""
    df = carregar_csv(caminho)
    df = limpar_dados(df)
    return df
