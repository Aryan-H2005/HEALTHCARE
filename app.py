import streamlit as st
import pandas as pd
import joblib

# ---------------- LOAD FILES ----------------
model = joblib.load("healthcare_model.pkl")
encoders = joblib.load("encoders.pkl")

st.title("Healthcare Admission Type Predictor")


# ---------------- USER INPUT ----------------
age = st.slider("Age", 1, 100, 30)

gender = st.selectbox("Gender", ["Male","Female"])

blood_type = st.selectbox("Blood Type",["A","B","AB","O"])

condition = st.text_input("Medical Condition","Diabetes")

insurance = st.text_input("Insurance Provider","XYZ Insurance")

billing = st.number_input("Billing Amount",1000,100000,5000)

medication = st.text_input("Medication","None")

test_results = st.selectbox("Test Results",["Normal","Abnormal","Critical"])

length_stay = st.slider("Length of Stay",1,60,5)

admission_day = st.selectbox("Admission Day",
["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])


# ---------------- FEATURE ENGINEERING ----------------

# exact same bins used during training
age_group = pd.cut(
    [age],
    bins=[0,18,35,50,65,100],
    labels=['Child','Young','Adult','Senior','Old']
)[0]


# billing category safe logic
if billing < 20000:
    billing_category = "Low"
elif billing < 50000:
    billing_category = "Medium"
else:
    billing_category = "High"


# ---------------- CREATE INPUT DATAFRAME ----------------
df = pd.DataFrame([{
    "Age":age,
    "Gender":gender,
    "Blood Type":blood_type,
    "Medical Condition":condition,
    "Insurance Provider":insurance,
    "Billing Amount":billing,
    "Medication":medication,
    "Test Results":test_results,
    "Length of Stay":length_stay,
    "Age Group":age_group,
    "Billing Category":billing_category,
    "Admission Day":admission_day
}])


# ---------------- ENCODING ----------------
for col in df.columns:

    # if encoder exists → use it
    if col in encoders:
        try:
            df[col] = encoders[col].transform(df[col].astype(str))
        except:
            df[col] = -1

    # if encoder missing → auto encode
    else:
        if df[col].dtype == "object":
            df[col] = df[col].astype("category").cat.codes

# ---------------- MATCH COLUMN ORDER ----------------
df = df[model.feature_names_in_]


# ---------------- PREDICTION ----------------
if st.button("Predict"):
    prediction = model.predict(df)[0]
    st.success(f"Predicted Admission Type: {prediction}")