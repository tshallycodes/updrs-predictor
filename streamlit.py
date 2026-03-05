import streamlit as st
import numpy as np
import joblib
import pandas as pd

# import your shared preprocessing functions
from preprocessing import pad_signal, extract_features_from_signal

st.title("Parkinson’s Finger Tapping Classifier")

# =========================
# Load Models
# =========================

LEFT_MODEL_PATH = "models/left_svm.pkl"
RIGHT_MODEL_PATH = "models/right_svm.pkl"

left_model = joblib.load(LEFT_MODEL_PATH)
right_model = joblib.load(RIGHT_MODEL_PATH)

# =========================
# Sidebar Inputs
# =========================

hand = st.sidebar.selectbox("Select Hand", ["Left", "Right"])

updrs_raw = st.sidebar.selectbox(
    "UPDRS Finger Tapping Score",
    options=[0, 1, 2, 3, 4]
)

st.sidebar.write("Upload or paste signals")

amplitude_input = st.text_area(
    "Amplitude values (comma separated)",
    height=150
)

time_input = st.text_area(
    "Time values (comma separated)",
    height=150
)

# =========================
# Helpers
# =========================

def parse_signal(text):
    return np.array([float(x.strip()) for x in text.split(",") if x.strip() != ""])

# =========================
# Prediction
# =========================

if st.button("Predict"):

    if amplitude_input == "" or time_input == "":
        st.error("Please provide amplitude and time values.")

    else:
        amplitude = parse_signal(amplitude_input)
        time = parse_signal(time_input)

        amplitude, time = pad_signal(amplitude, time)

        features = extract_features_from_signal(amplitude, time)

        if features is None:
            st.error("Not enough valid taps detected.")

        else:
            # encode UPDRS
            updrs_encoded = 1 if updrs_raw >= 3 else 0

            features["updrs"] = updrs_encoded

            feature_vector = pd.DataFrame([features])

            model = left_model if hand == "Left" else right_model

            prob = model.predict_proba(feature_vector)[0][1]

            if prob < 0.40:
                prediction = "Negative (No Parkinson’s)"
            elif prob > 0.55:
                prediction = "Positive (Parkinson’s)"
            else:
                prediction = "Uncertain, further evaluation needed"

            st.subheader("Prediction")
            st.write(prediction)
            st.write(f"Probability: {prob:.3f}")
            st.write("Encoded UPDRS:", updrs_encoded)