import pandas as pd
from .config import MATCH_DATA_PATH


def load_match_data() -> pd.DataFrame:
    """
    Load the match data from the Parquet file.
    
    Returns:
        pd.DataFrame: DataFrame containing match data.
    """
    try:
        df = pd.read_parquet(MATCH_DATA_PATH)
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Match data file not found at {MATCH_DATA_PATH}")
    except Exception as e:
        raise RuntimeError(f"Error loading match data: {e}")



if __name__ == "__main__":
    try:
        match_data = load_match_data()
        print(f"Loaded match data with {len(match_data)} records.")
    except Exception as e:
        print(f"Failed to load match data: {e}")

    print(match_data.head())  # Display the first few rows of the DataFrame
