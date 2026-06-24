import logging
import pandas as pd

logger = logging.getLogger(__name__)


def taxa_vitoria_geral(df: pd.DataFrame) -> pd.DataFrame:
    """Retorna taxa de vitoria, total de partidas, vitorias e derrotas."""
    total = len(df)
    vitorias = df["vitoria_binaria"].sum()
    derrotas_empates = total - vitorias
    taxa = vitorias / total * 100 if total > 0 else 0

    return pd.DataFrame([{
        "total_partidas": total,
        "vitorias": int(vitorias),
        "derrotas_empates": int(derrotas_empates),
        "taxa_vitoria_pct": round(taxa, 2),
    }])


def distribuicao_resultados(df: pd.DataFrame) -> pd.DataFrame:
    """Conta quantas vitorias, derrotas e empates existem."""
    contagem = df["result"].value_counts().reset_index()
    contagem.columns = ["resultado", "quantidade"]
    contagem["percentual"] = (contagem["quantidade"] / len(df) * 100).round(2)
    return contagem


def vitoria_por_hora(df: pd.DataFrame) -> pd.DataFrame:
    """Taxa de vitoria agrupada por hora do dia."""
    agrupado = (
        df.groupby("hora_batalha")["vitoria_binaria"]
        .agg(partidas="count", vitorias="sum")
        .reset_index()
    )
    agrupado["taxa_vitoria_pct"] = (
        agrupado["vitorias"] / agrupado["partidas"] * 100
    ).round(2)
    return agrupado


def cartas_mais_usadas(df: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """Conta quantas vezes cada carta aparece nos decks do jogador."""
    # separa as cartas de cada deck (formato: 'Carta1,Carta2,...')
    todas_cartas = df["player_deck"].dropna().str.split(",").explode().str.strip()
    contagem = todas_cartas.value_counts().head(top_n).reset_index()
    contagem.columns = ["carta", "aparicoes"]
    return contagem


def win_rate_por_carta(df: pd.DataFrame, min_aparicoes: int = 20) -> pd.DataFrame:
    """
    Calcula o win rate de cada carta usada pelo jogador.
    Considera apenas cartas com pelo menos min_aparicoes ocorrencias.
    """
    registros = []

    # expande cada linha em uma linha por carta
    for _, linha in df[["player_deck", "vitoria_binaria"]].iterrows():
        if pd.isna(linha["player_deck"]):
            continue
        cartas = [c.strip() for c in linha["player_deck"].split(",")]
        for carta in cartas:
            registros.append({"carta": carta, "vitoria": linha["vitoria_binaria"]})

    df_cartas = pd.DataFrame(registros)

    agrupado = (
        df_cartas.groupby("carta")["vitoria"]
        .agg(aparicoes="count", vitorias="sum")
        .reset_index()
    )
    # filtra cartas com poucas aparicoes para evitar estatisticas enganosas
    agrupado = agrupado[agrupado["aparicoes"] >= min_aparicoes].copy()
    agrupado["win_rate_pct"] = (agrupado["vitorias"] / agrupado["aparicoes"] * 100).round(2)
    agrupado = agrupado.sort_values("win_rate_pct", ascending=False)

    return agrupado


def vitoria_por_faixa_trofeus(df: pd.DataFrame) -> pd.DataFrame:
    """Taxa de vitoria agrupada por faixa de trofeus."""
    agrupado = (
        df.groupby("faixa_trofeus")["vitoria_binaria"]
        .agg(partidas="count", vitorias="sum")
        .reset_index()
    )
    agrupado["taxa_vitoria_pct"] = (
        agrupado["vitorias"] / agrupado["partidas"] * 100
    ).round(2)
    agrupado = agrupado.sort_values("faixa_trofeus")
    return agrupado


def estatisticas_trofeus(df: pd.DataFrame) -> pd.DataFrame:
    """Estatisticas descritivas dos trofeus do jogador."""
    col = df["player_starting_trophies"]
    return pd.DataFrame([{
        "media": round(col.mean(), 1),
        "mediana": round(col.median(), 1),
        "minimo": int(col.min()),
        "maximo": int(col.max()),
        "desvio_padrao": round(col.std(), 1),
    }])


def executar_todas(df: pd.DataFrame) -> dict:
    """Executa todas as analises e retorna um dicionario de DataFrames."""
    logger.info("Executando todas as analises...")
    return {
        "taxa_vitoria_geral":       taxa_vitoria_geral(df),
        "distribuicao_resultados":  distribuicao_resultados(df),
        "vitoria_por_hora":         vitoria_por_hora(df),
        "cartas_mais_usadas":       cartas_mais_usadas(df),
        "win_rate_por_carta":       win_rate_por_carta(df),
        "vitoria_por_faixa":        vitoria_por_faixa_trofeus(df),
        "estatisticas_trofeus":     estatisticas_trofeus(df),
    }
