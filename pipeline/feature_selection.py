import numpy as np

def select_features(df, variance_threshold=0.01):
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    dropped = [c for c in num_cols if df[c].var() < variance_threshold]
    df = df.drop(columns=dropped)
    return df, f"select_features → dropped {len(dropped)} low-variance column(s). {df.shape[1]} kept."
