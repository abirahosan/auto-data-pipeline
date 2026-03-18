def validate_data(df):
    df = df.copy()
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    removed = before - len(df)
    return df, f"validate_data → removed {removed} duplicate(s). {len(df)} row(s) remain."
