import numpy as np
from sklearn.preprocessing import MinMaxScaler

def normalize_data(df):
    df = df.copy()
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if df[c].std() > 0]
    if num_cols:
        df[num_cols] = MinMaxScaler().fit_transform(df[num_cols])
    return df, f"normalize_data → scaled {len(num_cols)} column(s) to [0, 1]."
