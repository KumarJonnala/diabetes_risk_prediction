import pandas as pd
from sklearn.model_selection import train_test_split

def build_processed_data(
    raw_path: str = "data/diabetes.csv",
    out_path: str = "data/diabetes_processed.csv"
):
    """Run cleaning + FE on raw CSV and save processed version."""
    from src.preprocess import clean_data, make_target
    from src.features import engineer_features

    df = pd.read_csv(raw_path)
    df = clean_data(df)
    df = make_target(df)
    df = engineer_features(df)
    df.to_csv(out_path, index=False)
    print(f"Saved processed data → {out_path}  shape: {df.shape}")


def load_data(path: str = "data/diabetes_processed.csv"):
    """Load processed dataset and return train/test split."""
    df = pd.read_csv(path)
    X = df.drop(columns=["Diabetes_Binary"])
    y = df["Diabetes_Binary"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=12, stratify=y
    )
    return X_train, X_test, y_train, y_test