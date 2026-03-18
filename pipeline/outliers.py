import pandas as pd
import numpy as np

def remove_outliers(df):
    df = df.copy()
    before = len(df)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        mask = pd.Series(True, index=df.index)
        for col in num_cols:
            Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            IQR = Q3 - Q1
            mask &= (df[col] >= Q1 - 1.5*IQR) & (df[col] <= Q3 + 1.5*IQR)
        df = df[mask].reset_index(drop=True)
    return df, f"remove_outliers → removed {before - len(df)} outlier row(s). {len(df)} remain."
