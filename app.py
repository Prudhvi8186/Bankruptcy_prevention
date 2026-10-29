from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Bankruptcy Risk Prediction", page_icon="📊", layout="centered")

st.title("Bankruptcy Prevention Prediction")
st.write("Enter the six company risk indicators to estimate the model's prediction.")

MODEL_PATH = Path(__file__).resolve().parent / "bankruptcy_model.pkl"

@st.cache_resource
def load_model(path: str):
    return joblib.load(path)

if not MODEL_PATH.exists():
    st.error(f"Model file not found: {MODEL_PATH.name}. Upload bankruptcy_model.pkl into the same folder as app.py.")
    st.stop()

try:
    model = load_model(str(MODEL_PATH))
except Exception as exc:
    st.error("Could not load bankruptcy_model.pkl. Check that the model was saved correctly and that scikit-learn/package versions are compatible.")
    st.exception(exc)
    st.stop()

FEATURES = [
    "industrial_risk",
    "management_risk",
    "financial_flexibility",
    "credibility",
    "competitiveness",
    "operating_risk",
]

st.caption("Choose 0, 0.5, or 1 for each indicator, as used by the training dataset.")
with st.form("prediction_form"):
    values = {}
    labels = {
        "industrial_risk": "Industrial risk",
        "management_risk": "Management risk",
        "financial_flexibility": "Financial flexibility",
        "credibility": "Credibility",
        "competitiveness": "Competitiveness",
        "operating_risk": "Operating risk",
    }
    for feature in FEATURES:
        values[feature] = st.select_slider(
            labels[feature], options=[0.0, 0.5, 1.0], value=0.5,
            format_func=lambda x: str(x)
        )
    submitted = st.form_submit_button("Predict bankruptcy risk", type="primary")

if submitted:
    input_df = pd.DataFrame([[values[f] for f in FEATURES]], columns=FEATURES)
    try:
        prediction = int(model.predict(input_df)[0])
        probability = None
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_df)[0]
            classes = list(model.classes_)
            if 1 in classes:
                probability = float(probabilities[classes.index(1)])

        if prediction == 1:
            st.error("Model prediction: Bankruptcy")
        else:
            st.success("Model prediction: Non-bankruptcy")
        if probability is not None:
            st.metric("Estimated bankruptcy probability", f"{probability:.1%}")
        st.warning("This is a machine-learning estimate, not financial or legal advice. Validate the model before making business decisions.")
    except Exception as exc:
        st.error("Prediction failed. Confirm that bankruptcy_model.pkl was trained using these six features in this order.")
        st.exception(exc)
