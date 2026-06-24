import logging
import pandas as pd

logger = logging.getLogger(__name__)


def criar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # vitoria como valor numerico para facilitar calculos
    df["vitoria_binaria"] = (df["result"] == "WIN").astype(int)

    # diferenca de trofeus: positivo = jogador e mais forte
    df["diferenca_trofeus"] = (
        df["player_starting_trophies"] - df["opponent_starting_trophies"]
    )

    # hora em que a batalha ocorreu
    df["hora_batalha"] = df["battle_time"].dt.hour

    # diferenca de coroas apos a batalha
    df["diferenca_coroas"] = df["player_crowns"] - df["opponent_crowns"]

    # razao de hp das torres (indica quem estava mais danificado)
    df["hp_torre_ratio"] = df["player_king_tower_hp"] / df["opponent_king_tower_hp"].replace(0, 1)

    # faixa de trofeus em intervalos de 100
    df["faixa_trofeus"] = (
        (df["player_starting_trophies"] // 100 * 100).astype(int).astype(str)
        + "-"
        + ((df["player_starting_trophies"] // 100 * 100) + 100).astype(int).astype(str)
    )

    logger.info("Features criadas: vitoria_binaria, diferenca_trofeus, hora_batalha, diferenca_coroas, hp_torre_ratio, faixa_trofeus")
    return df
