def bmi_category(bmi):
    """Map a BMI value to an ordinal category."""
    if bmi < 18.5: return 0 # Underweight
    if bmi < 25:   return 1 # Normal (18.5 to < 25)
    if bmi < 30:   return 2 # Overweight (25 to < 30)
    return 3 # Obese (>= 30)

def engineer_features(df):
    """Return DataFrame with engineered features added.
     Adds:
    - BMI_cat: ordinal BMI category derived from BMI.
    """
    df = df.copy()
    df['BMI_cat'] = df['BMI'].apply(bmi_category)
    return df