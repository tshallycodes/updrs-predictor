import streamlit as st
import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn

# Import your shared preprocessing functions
from preprocessing import pad_signal, extract_features_from_signal

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="UPDRS Predictor", page_icon="🧠", layout="centered")
st.title("🧠 UPDRS Severity Predictor")
st.caption("Predicts Parkinson's severity (Mild / Moderate / Severe) from finger tapping signals")

# ─────────────────────────────────────────────────────────────
# CLASS LABELS
# ─────────────────────────────────────────────────────────────
CLASS_LABELS = {0: "🟢 Mild (UPDRS 0–1)", 1: "🟡 Moderate (UPDRS 2–3)", 2: "🔴 Severe (UPDRS 4)"}

# ─────────────────────────────────────────────────────────────
# CNN ARCHITECTURE — must match training definition exactly
# ─────────────────────────────────────────────────────────────
class UPDRSModel(nn.Module):
    def __init__(self, input_dim, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# ─────────────────────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_sklearn_models():
    models = {}
    model_paths = {
        "Random Forest": {"Left": "saved_models/rf_left.pkl",  "Right": "saved_models/rf_right.pkl"},
        "SVC":           {"Left": "saved_models/svc_left.pkl", "Right": "saved_models/svc_right.pkl"},
        "SVR":           {"Left": "saved_models/svr_left.pkl", "Right": "saved_models/svr_right.pkl"},
    }
    for model_name, sides in model_paths.items():
        models[model_name] = {}
        for side, path in sides.items():
            with open(path, "rb") as f:
                models[model_name][side] = pickle.load(f)
    return models

@st.cache_resource
def load_cnn_models(input_dim):
    cnn = {}
    for side, path in [("Left", "saved_models/cnn_left.pth"), ("Right", "saved_models/cnn_right.pth")]:
        model = UPDRSModel(input_dim=input_dim, num_classes=3)
        model.load_state_dict(torch.load(path, map_location=torch.device("cpu")))
        model.eval()
        cnn[side] = model
    return cnn

# ─────────────────────────────────────────────────────────────
# SIDEBAR — signal inputs only, no UPDRS input
# ─────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Configuration")

hand = st.sidebar.selectbox("Hand", ["Left", "Right"])

model_choice = st.sidebar.selectbox(
    "Model",
    ["Random Forest", "SVC", "SVR", "CNN"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Signal Data**")

amplitude_input = st.sidebar.text_area("Amplitude values (comma separated)", height=120)
time_input      = st.sidebar.text_area("Time values (comma separated)",      height=120)

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def parse_signal(text):
    return np.array([float(x.strip()) for x in text.split(",") if x.strip() != ""])

def predict_sklearn(model, feature_vector):
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(feature_vector)[0]
        pred  = int(np.argmax(probs))
        return pred, probs
    else:
        # SVR — round continuous output to nearest class
        raw  = model.predict(feature_vector)[0]
        pred = int(np.clip(round(raw), 0, 2))
        return pred, None

def predict_cnn(model, feature_vector):
    tensor = torch.tensor(feature_vector.values, dtype=torch.float32)
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1).numpy()[0]
        pred   = int(np.argmax(probs))
    return pred, probs

# ─────────────────────────────────────────────────────────────
# MAIN — prediction on button click
# ─────────────────────────────────────────────────────────────
if st.button("🔍 Predict", use_container_width=True):

    if not amplitude_input or not time_input:
        st.error("Please provide both amplitude and time values.")

    else:
        amplitude = parse_signal(amplitude_input)
        time      = parse_signal(time_input)

        amplitude, time = pad_signal(amplitude, time)
        features        = extract_features_from_signal(amplitude, time)

        if features is None:
            st.error("Not enough valid taps detected. Please check your signal.")

        else:
            # Features come purely from the signal — no UPDRS input
            feature_vector = pd.DataFrame([features])

            # Load models
            sklearn_models = load_sklearn_models()
            cnn_models     = load_cnn_models(feature_vector.shape[1])

            # Run prediction
            if model_choice == "CNN":
                pred, probs = predict_cnn(cnn_models[hand], feature_vector)
            else:
                pred, probs = predict_sklearn(sklearn_models[model_choice][hand], feature_vector)

            # ── Results ───────────────────────────────────────
            st.markdown("---")
            st.subheader("📊 Prediction Result")

            st.markdown("""
                <style>
                    [data-testid="stMetricValue"] {
                        font-size: 1rem;
                        white-space: normal;
                        word-break: break-word;
                    }
                </style>
            """, unsafe_allow_html=True)

            col1, col2 = st.columns(2)
            col1.metric("Predicted Severity", CLASS_LABELS[pred])
            col2.metric("Model Used", f"{model_choice} ({hand} Hand)")

            if probs is not None:
                st.markdown("**Class Probabilities:**")
                prob_df = pd.DataFrame({
                    "Class":       list(CLASS_LABELS.values()),
                    "Probability": [f"{p:.3f}" for p in probs]
                })
                st.dataframe(prob_df, use_container_width=True, hide_index=True)
                st.progress(float(probs[pred]), text=f"Confidence: {probs[pred]:.1%}")

            else:
                st.info("SVR does not output class probabilities — prediction is based on rounded continuous output.")