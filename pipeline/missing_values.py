import pandas as pd
import numpy as np

def handle_missing_values(df):
    df = df.copy()
    filled = 0
    for col in df.columns:
        missing = int(df[col].isnull().sum())
        if missing == 0:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].mean())
        else:
            mode = df[col].mode()
            df[col] = df[col].fillna(mode[0] if not mode.empty else "Unknown")
        filled += missing
    return df, f"handle_missing_values → filled {filled} missing value(s)."
