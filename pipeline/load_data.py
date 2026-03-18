import pandas as pd
import os

def load_data(file_path):
    """
    Reads a CSV file and returns a pandas DataFrame.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame or None if loading failed
    """
    
    print(f"[load_data] Attempting to load: {file_path}")
    
    # ── Check 1: Does the file exist? ─────────────────────────────────────────
    if not os.path.exists(file_path):
        print(f"[load_data] ERROR: File not found → {file_path}")
        return None
    
    # ── Check 2: Is it actually a .csv file? ──────────────────────────────────
    if not file_path.endswith(".csv"):
        print(f"[load_data] ERROR: Expected a .csv file, got → {file_path}")
        return None
    
    # ── Check 3: Is the file empty? ───────────────────────────────────────────
    if os.path.getsize(file_path) == 0:
        print(f"[load_data] ERROR: File is empty → {file_path}")
        return None
    
    # ── Try reading the file ──────────────────────────────────────────────────
    try:
        df = pd.read_csv(file_path)
        
        print(f"[load_data] SUCCESS: Loaded {len(df)} rows × {len(df.columns)} columns")
        print(f"[load_data] Columns: {list(df.columns)}")
        
        return df
    
    except pd.errors.EmptyDataError:
        print(f"[load_data] ERROR: File has no data or headers → {file_path}")
        return None
    
    except pd.errors.ParserError:
        print(f"[load_data] ERROR: Could not parse file — is it a valid CSV? → {file_path}")
        return None
    
    except Exception as e:
        print(f"[load_data] ERROR: Unexpected error → {e}")
        return None