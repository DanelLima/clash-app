import requests
import pandas as pd
import time
from datetime import datetime

# NOTA: Lembre-se que tokens da Supercell expiram ou dependem do IP cadastrado.
TOKEN = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiIsImtpZCI6IjI4YTMxOGY3LTAwMDAtYTFlYi03ZmExLTJjNzQzM2M2Y2NhNSJ9.eyJpc3MiOiJzdXBlcmNlbGwiLCJhdWQiOiJzdXBlcmNlbGw6Z2FtZWFwaSIsImp0aSI6IjUyNWY1ODNmLWM2NWYtNDA3Yy04ZTk3LThiNmI0MWVmZGM2NCIsImlhdCI6MTc4MTQ3MTU2OSwic3ViIjoiZGV2ZWxvcGVyL2UyZTdlM2ZiLTkyYjMtNTZmNy1hYjgzLWU3NWU3NjgxNjU5MiIsInNjb3BlcyI6WyJyb3lhbGUiXSwibGltaXRzIjpbeyJ0aWVyIjoiZGV2ZWxvcGVyL3NpbHZlciIsInR5cGUiOiJ0aHJvdHRsaW5nIn0seyJjaWRycyI6WyIxNzcuMi42OC4yMDAiXSwidHlwZSI6ImNsaWVudCJ9XX0.1LWf8x3H5D67CnQQAzgm_OiGVfPX2idjZz2eclS0H0Sffyq6oX7FtRs2LJLiGThQ4daKW0llhcgnpvhg1CIscQ"

HEADERS = {
    "Authorization": f"Bearer {TOKEN}"
}

BASE_URL = "https://api.clashroyale.com/v1"

def get_top_players(limit=100):
    """
    Busca os melhores jogadores globais limitando a quantidade na própria API.
    """
    # Adicionado o parâmetro ?limit= na URL para a API saber quantos registros enviar
    url = f"{BASE_URL}/locations/global/pathoflegend/players?limit={limit}"
    response = requests.get(url, headers=HEADERS)

    if response.status_code != 200:
        print("Erro ao buscar ranking:", response.text)
        return []

    return response.json().get("items", [])


def get_player(tag):
    """
    Busca dados completos do jogador.
    """
    # Transforma o '#' em '%23' para a URL funcionar
    tag = tag.replace("#", "%23")
    url = f"{BASE_URL}/players/{tag}"

    response = requests.get(url, headers=HEADERS)

    if response.status_code != 200:
        print(f"Erro ao buscar jogador {tag}: {response.status_code}")
        return None

    return response.json()


def extract_deck(player):
    """
    Extrai o deck atual do jogador.
    """
    deck = player.get("currentDeck", [])

    return {
        "player": player.get("name", "Desconhecido"),
        "tag": player.get("tag"),
        "trophies": player.get("trophies"),
        "deck": ", ".join(card.get("name", "") for card in deck)
    }


def save_decks():
    # Agora puxando os 100 jogadores atualizados
    top_players = get_top_players(100)
    
    print(f"Total de jogadores encontrados no ranking: {len(top_players)}")

    data = []

    for idx, p in enumerate(top_players, start=1):
        print(f"[{idx}/100] Buscando {p['name']} ({p['tag']})")

        player = get_player(p["tag"])

        if player:
            data.append(extract_deck(player))
        else:
            # Caso falhe o fetch detalhado, salva o que veio do ranking básico
            data.append({
                "player": p.get("name"),
                "tag": p.get("tag"),
                "trophies": p.get("trophies"),
                "deck": "Erro ao carregar deck"
            })

        # Respeitar o limite de requisições por segundo da API
        time.sleep(0.2)

    if data:
        df = pd.DataFrame(data)
        file_name = f"decks_ultimate_champion_{datetime.now().strftime('%Y_%m_%d')}.csv"
        df.to_csv(file_name, index=False, encoding="utf-8-sig")
        print(f"\nArquivo salvo com sucesso: {file_name}")
    else:
        print("\nNenhum dado foi coletado.")


if __name__ == "__main__":
    save_decks()