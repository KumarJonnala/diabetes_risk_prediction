import pandas as pd
from src.preprocess import clean_data, make_target


# clean_data

def test_clean_data_removes_duplicates():
    df = pd.DataFrame({"BMI": [25.0, 25.0, 30.0]})
    result = clean_data(df)
    assert len(result) == 2

def test_clean_data_caps_bmi_outliers():
    """Rows with BMI > 60 should be dropped."""
    df = pd.DataFrame({"BMI": [22.0, 61.0, 45.0, 99.0]})
    result = clean_data(df)
    assert len(result) == 2
    assert result["BMI"].max() <= 60

def test_clean_data_resets_index():
    df = pd.DataFrame({"BMI": [22.0, 61.0, 30.0]})
    result = clean_data(df)
    assert list(result.index) == list(range(len(result)))


# make_target

def test_make_target_collapses_to_binary():
    """0 stays 0; 1 and 2 (pre-diabetes / diabetes) both become 1."""
    df = pd.DataFrame({"Diabetes_012": [0, 1, 2, 0, 2]})
    result = make_target(df)
    assert list(result["Diabetes_Binary"]) == [0, 1, 1, 0, 1]

def test_make_target_drops_original_column():
    df = pd.DataFrame({"Diabetes_012": [0, 1, 2]})
    result = make_target(df)
    assert "Diabetes_012" not in result.columns
    assert "Diabetes_Binary" in result.columns

def test_make_target_does_not_modify_original():
    df = pd.DataFrame({"Diabetes_012": [0, 1, 2]})
    make_target(df)
    assert "Diabetes_012" in df.columns
