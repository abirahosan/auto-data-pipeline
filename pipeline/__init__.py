from .missing_values import handle_missing_values
from .noise import add_noise
from .validation import validate_data
from .normalization import normalize_data
from .outliers import remove_outliers
from .feature_selection import select_features
from .split import split_data

def run_pipeline(df):
    summary = []
    for i, step in enumerate([handle_missing_values, add_noise, validate_data,
                               normalize_data, remove_outliers, select_features, split_data], 1):
        df, msg = step(df)
        summary.append({"step": i, "name": step.__name__, "message": msg})
    return df, summary
