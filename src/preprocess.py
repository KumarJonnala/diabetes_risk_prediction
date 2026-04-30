import pandas as pd

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicates and cap BMI outliers."""
    df = df.drop_duplicates().reset_index(drop=True)
    df = df[df['BMI'] <= 60].reset_index(drop=True)
    return df

def make_target(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse 3-class target to binary and rename column."""
    df = df.copy()
    df['Diabetes_Binary'] = (df['Diabetes_012'] > 0).astype(int)
    df = df.drop(columns=['Diabetes_012'])
    return df