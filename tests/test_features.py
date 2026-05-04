import pandas as pd
from src.features import bmi_category, engineer_features


# bmi_category

def test_bmi_underweight():
    assert bmi_category(17.0) == 0

def test_bmi_normal():
    assert bmi_category(22.0) == 1

def test_bmi_overweight():
    assert bmi_category(27.5) == 2

def test_bmi_obese():
    assert bmi_category(35.0) == 3

def test_bmi_boundaries():
    """Exact boundary values must map to the correct category."""
    assert bmi_category(18.5) == 1   # first Normal value
    assert bmi_category(25.0) == 2   # first Overweight value
    assert bmi_category(30.0) == 3   # first Obese value

# engineer_features

def test_engineer_features_adds_bmi_cat():
    df = pd.DataFrame({"BMI": [17.0, 22.0, 27.5, 35.0]})
    result = engineer_features(df)
    assert "BMI_cat" in result.columns
    assert list(result["BMI_cat"]) == [0, 1, 2, 3]

def test_engineer_features_does_not_modify_original():
    """engineer_features must return a copy, not mutate the input."""
    df = pd.DataFrame({"BMI": [22.0]})
    engineer_features(df)
    assert "BMI_cat" not in df.columns
