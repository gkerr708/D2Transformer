# src/d2transformer/steam_api.py
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
import pandas as pd
import requests
import time
import json

# ────────────────────────────────────────────────────────────────────────────
OPENDOTA_URL = "https://api.opendota.com/api/matches/{}"  # <─ NEW
# ────────────────────────────────────────────────────────────────────────────


class SteamAPI:
    def __init__(self) -> None:
        api_key_path = Path(r"C:\Users\gkerr\code\steam_api_key.txt")
        if api_key_path.exists():
            with open(api_key_path, "r") as file:
                self.API_KEY: str = file.read().strip()
        else:
            raise FileNotFoundError(f"API key file not found at {api_key_path}")

        self.match_url = (
            "https://api.steampowered.com/IDOTA2Match_570/GetMatchHistory/v1/"
        )
        self.hero_url = (
            "https://api.steampowered.com/IEconDOTA2_570/GetHeroes/v1/"
        )
        self.json_match_data: list[dict] = [{}]
        self.df: pd.DataFrame = pd.DataFrame()

        response = requests.get(self.hero_url, params={"key": self.API_KEY})
        self.hero_map = response.json().get("result", {}).get("heroes", [])

    # ───────────────────────────────────────────────────────────────────
    # REPLACED: now uses OpenDota, no api_key arg needed
    # ───────────────────────────────────────────────────────────────────
    
    def get_winner(self, match_id: int) -> int:
        url = OPENDOTA_URL.format(match_id)
        for attempt in range(3):                       # ≤ 3 tries
            r = requests.get(url, timeout=6)
            if r.status_code == 200:
                js = r.json()
                return 0 if js["radiant_win"] else 1
            if r.status_code == 429:                   # hit rate‑limit
                wait = 1.0 * (attempt + 1)             # 1 s, then 2 s
                time.sleep(wait)
                continue
            raise RuntimeError(f"OpenDota HTTP {r.status_code} for {match_id}")
        raise RuntimeError(f"OpenDota 429 throttled for {match_id}")

    # ───────────────────────────────────────────────────────────────────
    def _index2hero(self, index: int) -> str:
        for hero in self.hero_map:
            if hero["id"] == index:
                return hero["name"]
        return "Unknown Hero"

    def request_matches(self, N_matches: int, game_mode: int = 2) -> list[dict]:
        params = {
            "key": self.API_KEY,
            "matches_requested": N_matches,
            "game_mode": game_mode,
        }

        response = requests.get(self.match_url, params=params)
        self.json_response: list[dict] = response.json().get("result", {}).get(
            "matches", []
        )
        return self.json_response

    def convert_to_dataframe(self) -> pd.DataFrame:
        if not self.json_response:
            return pd.DataFrame()

        for i, match in enumerate(self.json_response):
            print(f"Processing match ID: {i+1}/{len(self.json_response)}")
            for player in match.get("players", []):
                name = self._index2hero(player["hero_id"])
                localized_name = name.replace("npc_dota_hero_", "")
                player["hero_name"] = name
                player["hero_localized_name"] = localized_name

            radiant_draft = [
                p["hero_id"] for p in match["players"] if p["team_number"] == 0
            ]
            dire_draft = [
                p["hero_id"] for p in match["players"] if p["team_number"] == 1
            ]

            match["radiant_draft"] = radiant_draft
            match["dire_draft"] = dire_draft

            # convert start time to Halifax TZ for convenience
            ts = match.get("start_time")
            match["start_time_alt"] = str(
                pd.to_datetime(ts, unit="s", utc=True).tz_convert("America/Halifax")
            )

            # ------------- NEW: fetch win label via OpenDota -------------
            try:
                match["radiant_wins"] = self.get_winner(match["match_id"])
            except Exception as e:
                print(f"  ↳ skipped (winner fetch failed: {e})")
                continue
            # -------------------------------------------------------------

            if match["radiant_wins"] not in {0, 1}:
                print(
                    f"Invalid radiant_wins in match {match['match_id']}: "
                    f"{match['radiant_wins']}"
                )
                continue

            if len(radiant_draft) != 5 or len(dire_draft) != 5:
                print(
                    f"Invalid draft length in match {match['match_id']}: "
                    f"Radiant {len(radiant_draft)}, Dire {len(dire_draft)}"
                )
                continue

        self.df = pd.DataFrame(self.json_response)
        return self.df

    def save_to_parquet(self, file_path: str) -> None:
        self.df.to_parquet(file_path, index=False)
        print(f"DataFrame saved to {file_path}")


# ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    dota = SteamAPI()
    dota.request_matches(100, 22)  # ranked AP queue
    print(f"Number of matches requested: {len(dota.json_response)}")
    dota.convert_to_dataframe()
    dota.save_to_parquet(r"C:\Users\gkerr\code\D2Transformer\data\dota_matches.parquet")
    print(dota.df.head())
