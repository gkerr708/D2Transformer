from pathlib import Path
import pandas as pd
import click as ck

# Define the current patch
current_patch_for_model = "7_39b" 
current_patch_for_parquet = "7_39c"

#KEY = "21bcc3a2-28fc-4331-a1ad-5e68401b8f71"
KEY = None

# Define constants for shared configurations
# Smart thing to write the same thing twice.
PROJECT_DIR = Path(__file__).parent.parent.parent
MATCH_DATA_PATH = PROJECT_DIR / "data" / f"dota_matches.parquet"
MODEL_PATH = PROJECT_DIR / "src" / "d2draftnet" / "models" / f"{current_patch_for_model}_model.pth"

# Define the list of heroes
HEROS_ = ['Anti-Mage', 'Axe', 'Bane', 'Bloodseeker', 'Crystal Maiden', 'Drow Ranger', 'Earthshaker', 'Juggernaut', 'Mirana', 'Morphling', 'Shadow Fiend', 
         'Phantom Lancer', 'Puck', 'Pudge', 'Razor', 'Sand King', 'Storm Spirit', 'Sven', 'Tiny', 'Vengeful Spirit', 'Windranger', 'Zeus', 'Kunkka', 'Lina', 
         'Lion', 'Shadow Shaman', 'Slardar', 'Tidehunter', 'Witch Doctor', 'Lich', 'Riki', 'Enigma', 'Tinker', 'Sniper', 'Necrophos', 'Warlock', 'Beastmaster', 
         'Queen of Pain', 'Venomancer', 'Faceless Void', 'Wraith King', 'Death Prophet', 'Phantom Assassin', 'Pugna', 'Templar Assassin', 'Viper', 'Luna', 
         'Dragon Knight', 'Dazzle', 'Clockwerk', 'Leshrac', "Nature's Prophet", 'Lifestealer', 'Dark Seer', 'Clinkz', 'Omniknight', 'Enchantress', 'Huskar', 
         'Night Stalker', 'Broodmother', 'Bounty Hunter', 'Weaver', 'Jakiro', 'Batrider', 'Chen', 'Spectre', 'Ancient Apparition', 'Doom', 'Ursa', 
         'Spirit Breaker', 'Gyrocopter', 'Alchemist', 'Invoker', 'Silencer', 'Outworld Destroyer', 'Lycan', 'Brewmaster', 'Shadow Demon', 'Lone Druid', 
         'Chaos Knight', 'Meepo', 'Treant Protector', 'Ogre Magi', 'Undying', 'Rubick', 'Disruptor', 'Nyx Assassin', 'Naga Siren', 'Keeper of the Light', 
         'Io', 'Visage', 'Slark', 'Medusa', 'Troll Warlord', 'Centaur Warrunner', 'Magnus', 'Timbersaw', 'Bristleback', 'Tusk', 'Skywrath Mage', 'Abaddon', 
         'Elder Titan', 'Legion Commander', 'Techies', 'Ember Spirit', 'Earth Spirit', 'Underlord', 'Terrorblade', 'Phoenix', 'Oracle', 'Winter Wyvern', 
         'Arc Warden', 'Monkey King', 'Dark Willow', 'Pangolier', 'Grimstroke', 'Hoodwink', 'Void Spirit', 'Snapfire', 'Mars', 'Ringmaster', 'Dawnbreaker', 
         'Marci', 'Primal Beast', 'Muerta', 'Kez']

HEROS = sorted(HEROS_)
DIRE_HEROS = sorted(["Dire_" + hero for hero in HEROS])
RADIANT_HEROS = sorted(["Radiant_" + hero for hero in HEROS])
HERO_MAP = {hero: i + 1 for i, hero in enumerate(HEROS)}

NUM_HEROS= len(HERO_MAP) + 1  # Ensure consistency with training
LAYERS = [32, 16]  # Layers for the current model (7.37e)
EMBEDDING_DIM = 3  # Embedding dimension for the current model (7.37e)


def load_data():
    """
    Load the dataset from the Parquet file.
    """
    return pd.read_parquet(MATCH_DATA_PATH)

if __name__ == "__main__":
    directories = [PROJECT_DIR, MATCH_DATA_PATH, MODEL_PATH]
    _= [
        ck.secho(f"OK: {d}", fg="green") 
        if d.exists() 
        else ck.secho(f"Missing: {d}", fg="red")
        for d in directories
    ]





