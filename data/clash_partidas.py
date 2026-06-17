import pandas as pd
import requests
import time
import hashlib
from datetime import datetime

TOKEN = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiIsImtpZCI6IjI4YTMxOGY3LTAwMDAtYTFlYi03ZmExLTJjNzQzM2M2Y2NhNSJ9.eyJpc3MiOiJzdXBlcmNlbGwiLCJhdWQiOiJzdXBlcmNlbGw6Z2FtZWFwaSIsImp0aSI6IjUyNWY1ODNmLWM2NWYtNDA3Yy04ZTk3LThiNmI0MWVmZGM2NCIsImlhdCI6MTc4MTQ3MTU2OSwic3ViIjoiZGV2ZWxvcGVyL2UyZTdlM2ZiLTkyYjMtNTZmNy1hYjgzLWU3NWU3NjgxNjU5MiIsInNjb3BlcyI6WyJyb3lhbGUiXSwibGltaXRzIjpbeyJ0aWVyIjoiZGV2ZWxvcGVyL3NpbHZlciIsInR5cGUiOiJ0aHJvdHRsaW5nIn0seyJjaWRycyI6WyIxNzcuMi42OC4yMDAiXSwidHlwZSI6ImNsaWVudCJ9XX0.1LWf8x3H5D67CnQQAzgm_OiGVfPX2idjZz2eclS0H0Sffyq6oX7FtRs2LJLiGThQ4daKW0llhcgnpvhg1CIscQ"

HEADERS = {
    "Authorization": f"Bearer {TOKEN}"
}

BASE_URL = "https://api.clashroyale.com/v1"

# Tenta carregar o arquivo gerado anteriormente
try:
    df = pd.read_csv("decks_ultimate_champion_2026_06_14.csv")
except FileNotFoundError:
    # Fallback caso o nome do arquivo mude ligeiramente
    import glob
    arquivos = glob.glob("decks_ultimate_champion_*.csv")
    if arquivos:
        df = pd.read_csv(arquivos[0])
        print(f"Carregando arquivo encontrado: {arquivos[0]}")
    else:
        raise FileNotFoundError("Não foi possível encontrar o arquivo CSV de jogadores.")

resultados = []
ARQUIVO_SAIDA = "partidas_ranked_detalhadas.csv"

def gerar_deck_hash(deck):
    """
    Gera um identificador único para o deck.
    """
    deck_ordenado = sorted(deck)
    texto = "|".join(deck_ordenado)
    return hashlib.md5(texto.encode()).hexdigest()

print(f"Iniciando a busca de partidas para {len(df)} jogadores...")

for idx, row in df.iterrows():
    tag = str(row["tag"]).strip()
    print(f"[{idx+1}/{len(df)}] Buscando partidas de {row.get('player', tag)}")

    tag_api = tag.replace("#", "%23")
    url = f"{BASE_URL}/players/{tag_api}/battlelog"

    try:
        response = requests.get(url, headers=HEADERS, timeout=15)

        if response.status_code == 429:
            print("Rate limit atingido... Aguardando 10 segundos.")
            time.sleep(10)
            continue

        if response.status_code != 200:
            print(f"Erro {response.status_code} para {tag}")
            continue

        battles = response.json()
        partidas_adicionadas_do_jogador = 0

        for battle in battles:
            # FILTRO CRÍTICO: Garante que estamos pegando apenas partidas da Rota das Lendas (Ranked)
            # Modos válidos na API: 'pathOfLegend' ou 'ranked'
            tipo_batalha = battle.get("type", "")
            if tipo_batalha not in ["pathOfLegend", "ranked"]:
                continue

            try:
                if len(battle.get("team", [])) != 1 or len(battle.get("opponent", [])) != 1:
                    continue

                player = battle["team"][0]
                opponent = battle["opponent"][0]

                player_deck = [card["name"] for card in player.get("cards", [])]
                opponent_deck = [card["name"] for card in opponent.get("cards", [])]

                # Se por algum motivo o deck veio incompleto, ignora
                if not player_deck or not opponent_deck:
                    continue

                player_crowns = player.get("crowns", 0)
                opponent_crowns = opponent.get("crowns", 0)

                if player_crowns > opponent_crowns:
                    result = "WIN"
                elif player_crowns < opponent_crowns:
                    result = "LOSS"
                else:
                    result = "DRAW"

                resultados.append({
                    # Informações da Batalha
                    "battle_time": battle.get("battleTime"),
                    "battle_type": tipo_batalha,
                    "game_mode": battle.get("gameMode", {}).get("name"),
                    
                    # Dados do Jogador Base
                    "player_tag": player.get("tag"),
                    "player_name": player.get("name"),
                    "player_crowns": player_crowns,
                    "player_starting_trophies": player.get("startingTrophies"), # Troféus/Passos no momento da partida
                    
                    # Dados do Oponente
                    "opponent_tag": opponent.get("tag"),
                    "opponent_name": opponent.get("name"),
                    "opponent_crowns": opponent_crowns,
                    "opponent_starting_trophies": opponent.get("startingTrophies"),

                    # Resultado
                    "result": result,

                    # Decks e Códigos Únicos
                    "player_deck": ",".join(player_deck),
                    "opponent_deck": ",".join(opponent_deck),
                    "player_deck_hash": gerar_deck_hash(player_deck),
                    "opponent_deck_hash": gerar_deck_hash(opponent_deck),

                    # HP das Torres do Rei
                    "player_king_tower_hp": player.get("kingTowerHitPoints"),
                    "opponent_king_tower_hp": opponent.get("kingTowerHitPoints")
                })
                
                partidas_adicionadas_do_jogador += 1

            except Exception as e:
                print(f"Erro ao processar dados internos da batalha: {e}")

        print(f"-> Adicionadas {partidas_adicionadas_do_jogador} partidas ranqueadas para este jogador.")
        
        # Salvamento Progressivo de segurança
        if resultados:
            pd.DataFrame(resultados).to_csv(ARQUIVO_SAIDA, index=False, encoding="utf-8-sig")

        # Pausa para respeitar as diretrizes de rate limit da API
        time.sleep(0.4)

    except Exception as e:
        print(f"Erro de conexão ao buscar {tag}: {e}")
        time.sleep(2)

print(f"\nProcesso concluído com sucesso! Total de {len(resultados)} partidas ranqueadas salvas em '{ARQUIVO_SAIDA}'.")