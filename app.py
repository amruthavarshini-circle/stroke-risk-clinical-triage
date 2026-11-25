import joblib
import pandas as pd
import streamlit as st
from src.config import MODEL_PATH, FEATURES

st.set_page_config(page_title="Stroke Risk Screening", page_icon="🩺",
                   layout="wide")

st.title("Stroke Risk Factors & Clinical Triage App")
st.caption("Educational data-science screening demonstration")

st.warning(
    "This application is not a medical diagnostic tool. It cannot determine "
    "whether a person will have a stroke. Do not use its output for medical "
    "decisions. Seek professional medical advice for health concerns."
)

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)

model = load_model()
if model is None:
    st.error("Trained model not found. First run: python -m src.train_model")
    st.stop()

st.subheader("Enter example record")
with st.form("patient_form"):
    c1, c2 = st.columns(2)
    with c1:
        gender = st.selectbox("Gender", ["Female", "Male", "Other"])
        age = st.number_input("Age (years)", min_value=0.0, max_value=120.0,
                              value=45.0, step=1.0)
        hypertension = st.selectbox("Hypertension", [0, 1],
                                    format_func=lambda x: "Yes" if x else "No")
        heart_disease = st.selectbox("Heart disease", [0, 1],
                                     format_func=lambda x: "Yes" if x else "No")
        ever_married = st.selectbox("Ever married", ["No", "Yes"])
        work_type = st.selectbox("Work type", [
            "Private", "Self-employed", "Govt_job", "children", "Never_worked"
        ])
    with c2:
        residence_type = st.selectbox("Residence type", ["Urban", "Rural"])
        glucose = st.number_input("Average glucose level", min_value=0.0,
                                  max_value=500.0, value=100.0, step=0.1)
        bmi = st.number_input("BMI", min_value=0.0, max_value=100.0,
                              value=25.0, step=0.1)
        smoking = st.selectbox("Smoking status", [
            "never smoked", "formerly smoked", "smokes", "Unknown"
        ])
    submitted = st.form_submit_button("Run educational screening")

if submitted:
    row = pd.DataFrame([{
        "gender": gender,
        "age": age,
        "hypertension": hypertension,
        "heart_disease": heart_disease,
        "ever_married": ever_married,
        "work_type": work_type,
        "Residence_type": residence_type,
        "avg_glucose_level": glucose,
        "bmi": bmi,
        "smoking_status": smoking
    }], columns=FEATURES)

    prediction = int(model.predict(row)[0])
    probability = float(model.predict_proba(row)[0, 1])

    st.subheader("Model output")
    st.metric("Model-estimated probability for the positive class",
              f"{probability:.1%}")
    st.write("Predicted class:", "Positive (1)" if prediction == 1
             else "Negative (0)")
    st.info(
        "This is a model output learned from a limited historical dataset, "
        "not an individual's medically validated probability of stroke. "
        "The result must not be interpreted as a diagnosis or reassurance."
    )

st.divider()
st.caption("Project 49 | Stroke Risk Factors & Clinical Triage App")
