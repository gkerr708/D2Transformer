# src/d2transformer/steam_api.py
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
import pandas as pd
import requests
import time
import json
import os

OPENDOTA_URL = "https://api.opendota.com/api/matches/{}"
MAX_STEAM_MATCHES = 99


class SteamAPI:
    def __init__(self) -> None:
        api_key_path = Path(r"C:\Users\gkerr\code\steam_api_key.txt")
        if api_key_path.exists():
            with open(api_key_path, "r") as file:
                self.API_KEY: str = file.read().strip()
        else:
            raise FileNotFoundError(f"API key file not found at {api_key_path}")

        self.match_url = "https://api.steampowered.com/IDOTA2Match_570/GetMatchHistory/v1/"
        self.hero_url = "https://api.steampowered.com/IEconDOTA2_570/GetHeroes/v1/"
        self.json_response: list[dict] = []
        self.df: pd.DataFrame = pd.DataFrame()

        response = requests.get(self.hero_url, params={"key": self.API_KEY})
        self.hero_map = response.json().get("result", {}).get("heroes", [])

    def get_winner(self, match_id: int) -> int:
        url = OPENDOTA_URL.format(match_id)
        for attempt in range(3):
            r = requests.get(url, timeout=6)
            if r.status_code == 200:
                js = r.json()
                return 0 if js["radiant_win"] else 1
            if r.status_code == 429:
                time.sleep(1.0 * (attempt + 1))
                continue
            raise RuntimeError(f"OpenDota HTTP {r.status_code} for {match_id}")
        raise RuntimeError(f"OpenDota 429 throttled for {match_id}")

    def _index2hero(self, index: int) -> str:
        for hero in self.hero_map:
            if hero["id"] == index:
                return hero["name"]
        return "Unknown Hero"

    def request_matches(self, total_matches: int, game_mode: int = 2) -> list[dict]:
        all_matches = []
        requested = 0

        while requested < total_matches:
            count = min(MAX_STEAM_MATCHES, total_matches - requested)
            params = {
                "key": self.API_KEY,
                "matches_requested": count,
                "game_mode": game_mode,
            }
            response = requests.get(self.match_url, params=params)
            chunk = response.json().get("result", {}).get("matches", [])
            if not chunk:
                break
            all_matches.extend(chunk)
            requested += len(chunk)
            print(f"Requested {requested}/{total_matches} matches so far.")
            time.sleep(1)

        self.json_response = all_matches
        return all_matches

    def convert_to_dataframe(self) -> pd.DataFrame:
        if not self.json_response:
            return pd.DataFrame()

        valid_matches = []

        for i, match in enumerate(self.json_response):
            print(f"Processing match ID: {i+1}/{len(self.json_response)}")

            for player in match.get("players", []):
                name = self._index2hero(player["hero_id"])
                player["hero_name"] = name
                player["hero_localized_name"] = name.replace("npc_dota_hero_", "")

            radiant_draft = [p["hero_id"] for p in match["players"] if p["team_number"] == 0]
            dire_draft = [p["hero_id"] for p in match["players"] if p["team_number"] == 1]

            match["radiant_draft"] = radiant_draft
            match["dire_draft"] = dire_draft

            ts = match.get("start_time")
            match["start_time_alt"] = str(
                pd.to_datetime(ts, unit="s", utc=True).tz_convert("America/Halifax")
            )

            try:
                match["radiant_wins"] = self.get_winner(match["match_id"])
            except Exception as e:
                print(f"  ↳ skipped (winner fetch failed: {e})")
                continue

            valid_matches.append(match)

        if not valid_matches:
            print("No valid matches to convert.")
            return pd.DataFrame()

        df = pd.DataFrame(valid_matches)
        df = df[df["radiant_wins"].isin([0, 1])]
        df = df.dropna(subset=["radiant_draft", "dire_draft"])

        lens_ok = (df["radiant_draft"].str.len() == 5) & (df["dire_draft"].str.len() == 5)
        unique_ok = (df["radiant_draft"].apply(lambda x: len(set(x)) == 5) &
                     df["dire_draft"].apply(lambda x: len(set(x)) == 5))
        df = df[lens_ok & unique_ok]

        valid_ids = {h["id"] for h in self.hero_map}
        def ids_ok(lst): return all(i in valid_ids for i in lst)
        mask = df["radiant_draft"].apply(ids_ok) & df["dire_draft"].apply(ids_ok)
        df = df[mask]

        df["radiant_wins"] = df["radiant_wins"].astype("float32")
        self.df = df.reset_index(drop=True)
        return self.df

    def save_to_parquet(self, file_path: str) -> None:
        file_path = Path(file_path)
        if file_path.exists():
            old_df = pd.read_parquet(file_path)
            combined_df = pd.concat([old_df, self.df]).drop_duplicates(subset="match_id")
            print(f"Old DataFrame loaded with {len(old_df)} rows.")
            print(f"New DataFrame has {len(self.df)} rows.")
        else:
            combined_df = self.df
            print(f"New DataFrame has {len(self.df)} rows, saving as new file.")
        combined_df.reset_index(drop=True).to_parquet(file_path, index=False)
        print(f"Combined DataFrame saved to {file_path}")


# ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    dota = SteamAPI()
    dota.request_matches(10000, 22)  # ranked AP queue
    print(f"Number of matches fetched: {len(dota.json_response)}")
    dota.convert_to_dataframe()
    save_path = Path(r"C:\Users\gkerr\code\D2Transformer\data\dota_matches4.parquet")
    dota.save_to_parquet(save_path)
    print(dota.df.head())
