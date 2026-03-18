import numpy as np

def add_noise(df):
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in num_cols:
        std = df[col].std()
        if std > 0:
            df[col] = df[col] + np.random.normal(0, std * 0.01, size=len(df))
    cat_cols = df.select_dtypes(include=["object","category"]).columns.tolist()
    for col in cat_cols:
        unique = df[col].dropna().unique()
        if len(unique) < 2:
            continue
        n = max(1, int(len(df) * 0.05))
        rows = np.random.choice(df.index, size=n, replace=False)
        for idx in rows:
            others = [v for v in unique if v != df.at[idx, col]]
            if others:
                df.at[idx, col] = np.random.choice(others)
    return df, f"add_noise → added noise to {len(num_cols)} numerical, {len(cat_cols)} categorical column(s)."
