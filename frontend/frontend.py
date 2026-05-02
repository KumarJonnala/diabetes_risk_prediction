import os
import streamlit as st
import requests

API_URL = os.getenv("API_URL", "http://localhost:8000/predict")

st.title("Diabetes Risk Predictor")
st.write("Fill in your health information below to assess your diabetes risk.")

# Health Conditions
st.subheader("Health Conditions")
col1, col2 = st.columns(2)

with col1:
    HighBP               = st.selectbox("High Blood Pressure",          [0, 1], format_func=lambda x: "Yes" if x else "No")
    HighChol             = st.selectbox("High Cholesterol",             [0, 1], format_func=lambda x: "Yes" if x else "No")
    CholCheck            = st.selectbox("Cholesterol Check (5 yrs)",    [0, 1], format_func=lambda x: "Yes" if x else "No")

with col2:
    Stroke               = st.selectbox("Ever had a Stroke",            [0, 1], format_func=lambda x: "Yes" if x else "No")
    HeartDiseaseorAttack = st.selectbox("Heart Disease or Attack",      [0, 1], format_func=lambda x: "Yes" if x else "No")

# Lifestyle
st.subheader("Lifestyle")
col3, col4 = st.columns(2)

with col3:
    Smoker               = st.selectbox("Smoker (100+ cigarettes ever)", [0, 1], format_func=lambda x: "Yes" if x else "No")
    PhysActivity         = st.selectbox("Physical Activity (past 30d)",  [0, 1], format_func=lambda x: "Yes" if x else "No")
    Fruits               = st.selectbox("Eat Fruit daily",               [0, 1], format_func=lambda x: "Yes" if x else "No")

with col4:
    Veggies              = st.selectbox("Eat Vegetables daily",          [0, 1], format_func=lambda x: "Yes" if x else "No")
    HvyAlcoholConsump    = st.selectbox("Heavy Alcohol Consumption",     [0, 1], format_func=lambda x: "Yes" if x else "No")

# Healthcare Access
st.subheader("Healthcare Access")
col5, col6 = st.columns(2)

with col5:
    AnyHealthcare        = st.selectbox("Have Health Coverage",          [0, 1], format_func=lambda x: "Yes" if x else "No")

with col6:
    NoDocbcCost          = st.selectbox("Avoided Doctor due to Cost",    [0, 1], format_func=lambda x: "Yes" if x else "No")

# General Health
st.subheader("General Health")
col7, col8 = st.columns(2)

with col7:
    BMI                  = st.number_input("BMI", min_value=10.0, max_value=100.0, value=25.0, step=0.1)
    GenHlth              = st.slider("General Health (1=Excellent, 5=Poor)", 1, 5, 3)
    DiffWalk             = st.selectbox("Difficulty Walking",            [0, 1], format_func=lambda x: "Yes" if x else "No")

with col8:
    MentHlth             = st.slider("Poor Mental Health Days (past 30d)", 0, 30, 0)
    PhysHlth             = st.slider("Poor Physical Health Days (past 30d)", 0, 30, 0)

# Demographics
st.subheader("Demographics")
col9, col10 = st.columns(2)

with col9:
    Sex                  = st.selectbox("Sex",       [0, 1], format_func=lambda x: "Male" if x else "Female")
    Age                  = st.slider("Age Category (1=18-24, 13=80+)", 1, 13, 5)

with col10:
    Education            = st.slider("Education Level (1=None, 6=College grad)", 1, 6, 4)
    Income               = st.slider("Income Level (1=<$10k, 8=>$75k)", 1, 8, 4)

# Predict
st.divider()

if st.button("Predict Diabetes Risk", type="primary"):
    payload = {
        "HighBP": HighBP, "HighChol": HighChol, "CholCheck": CholCheck,
        "Smoker": Smoker, "Stroke": Stroke, "HeartDiseaseorAttack": HeartDiseaseorAttack,
        "PhysActivity": PhysActivity, "Fruits": Fruits, "Veggies": Veggies,
        "HvyAlcoholConsump": HvyAlcoholConsump, "AnyHealthcare": AnyHealthcare,
        "NoDocbcCost": NoDocbcCost, "DiffWalk": DiffWalk, "Sex": Sex,
        "BMI": BMI, "GenHlth": GenHlth, "MentHlth": MentHlth,
        "PhysHlth": PhysHlth, "Age": Age, "Education": Education, "Income": Income,
    }

    with st.spinner("Analysing..."):
        try:
            response = requests.post(API_URL, json=payload)
            if response.status_code == 200:
                result = response.json()
                score = result["diabetes_risk_score"]
                label = result["risk_label"]

                st.divider()
                if label == "High Risk":
                    st.error(f"**{label}** — Risk Score: {score}%")
                else:
                    st.success(f"**{label}** — Risk Score: {score}%")

                st.caption("This is a statistical prediction based on CDC BRFSS data. Consult a medical professional for a proper diagnosis.")
            else:
                st.error(f"API error {response.status_code}: {response.text}")
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the API. Make sure the FastAPI server is running on localhost:8000.")
