import streamlit as st
import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
import json

from preprocessing import pad_signal, extract_features_from_signal

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="UPDRS Predictor", page_icon="🧠", layout="centered")

st.markdown("""
    <style>
        [data-testid="stMetricValue"] {
            font-size: 1rem;
            white-space: normal;
            word-break: break-word;
        }

        /* Hand button styling */
        div[data-testid="column"] button {
            width: 100%;
            padding: 0.75rem;
            font-size: 1rem;
            font-weight: 600;
            border-radius: 8px;
            border: 2px solid transparent;
            transition: all 0.2s ease;
        }

        /* Glowing selected state */
        div[data-testid="column"] button[kind="primary"] {
            border-color: #7C3AED;
            box-shadow: 0 0 12px rgba(124, 58, 237, 0.6);
            background-color: #7C3AED;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🧠 UPDRS Severity Predictor")
st.markdown("---")
st.markdown("### 📚 Supporting Research")

st.markdown("""
- [**Hand Resting Tremor — SVC, RF, and Bilateral Assessment**  
  Iosa et al. (2020) — Frontiers in Bioengineering and Biotechnology](https://pmc.ncbi.nlm.nih.gov/articles/PMC7381229/)

- [**Automatic Classification of the Severity Level of Parkinson's Disease**  
  Kodali et al. (2023) — Computer Speech and Language](https://www.sciencedirect.com/science/article/pii/S0885230823000670)

- [**UPDRS-Based Multiclass Classification with Small Datasets**  
  Benmalek et al. (2018) — International Journal of Speech Technology](https://dl.acm.org/doi/10.1007/s10772-017-9401-9)
""")
st.markdown("---")
st.caption("Predicts Parkinson's severity (Mild / Moderate / Severe) from finger tapping signals")

# ─────────────────────────────────────────────────────────────
# CLASS LABELS
# ─────────────────────────────────────────────────────────────
CLASS_LABELS = {0: "🟢 Mild (UPDRS 0–1)", 1: "🟡 Moderate (UPDRS 2–3)", 2: "🔴 Severe (UPDRS 4)"}

# ─────────────────────────────────────────────────────────────
# UPDATED CNN ARCHITECTURE — matches Optuna trained model
# ─────────────────────────────────────────────────────────────
class UPDRSModel(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, num_classes)
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
def load_cnn_models():                          # ← remove input_dim parameter
    cnn = {}
    for side in ["Left", "Right"]:
        with open(f"saved_models/cnn_{side.lower()}_params.json") as f:
            params = json.load(f)

        input_dim = len(params['features'])     # ← derive from saved feature list

        model = UPDRSModel(
            input_dim=input_dim,
            hidden_dim1=params['hidden_dim1'],
            hidden_dim2=params['hidden_dim2']
        )
        model.load_state_dict(torch.load(
            f"saved_models/cnn_{side.lower()}.pth",
            map_location=torch.device("cpu")
        ))
        model.eval()
        cnn[side] = {"model": model, "features": params['features']}
    return cnn

# ─────────────────────────────────────────────────────────────
# HAND SELECTION — two glowing buttons side by side
# ─────────────────────────────────────────────────────────────
st.markdown("### ✋ Select Hand")

if "hand" not in st.session_state:
    st.session_state.hand = "Left"

col_left, col_right = st.columns(2)

with col_left:
    if st.button(
        "Left Hand",
        type="primary" if st.session_state.hand == "Left" else "secondary",
        use_container_width=True
    ):
        st.session_state.hand = "Left"

with col_right:
    if st.button(
        "Right Hand",
        type="primary" if st.session_state.hand == "Right" else "secondary",
        use_container_width=True
    ):
        st.session_state.hand = "Right"

hand = st.session_state.hand
st.markdown(f"Selected: **{hand} Hand**")

# ─────────────────────────────────────────────────────────────
# SIGNAL INPUTS — on page, no sidebar
# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 📈 Signal Data")

amplitude_input = st.text_area("Amplitude values (comma separated)", height=120)
time_input      = st.text_area("Time values (comma separated)",      height=120)

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def parse_signal(text):
    return np.array([float(x.strip()) for x in text.split(",") if x.strip() != ""])

def predict_sklearn(model, feature_vector):
    if hasattr(model, "predict_proba"):
        raw_probs = model.predict_proba(feature_vector)[0]
        classes   = model.classes_                          # e.g. [0, 1] or [0, 1, 2]

        # Pad to always have 3 class probabilities
        probs = np.zeros(3)
        for i, cls in enumerate(classes):
            probs[cls] = raw_probs[i]

        pred = int(np.argmax(probs))
        return pred, probs
    else:
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

@st.cache_resource
def load_sklearn_features():
    features = {}
    for side in ["Left", "Right"]:
        with open(f"saved_models/sklearn_features_{side.lower()}.json") as f:
            features[side] = json.load(f)
    return features

# ─────────────────────────────────────────────────────────────
# PREDICT — all models at once
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
            feature_vector = pd.DataFrame([features])
            sklearn_models   = load_sklearn_models()
            sklearn_features = load_sklearn_features()
            cnn_models       = load_cnn_models()

            # ── Filter features per model type ────────────────────────
            feature_vector_sklearn = feature_vector[sklearn_features[hand]]
            feature_vector_cnn     = feature_vector[cnn_models[hand]["features"]]

            # ── Run all models ─────────────────────────────────────────
            results = {
                "Random Forest": predict_sklearn(sklearn_models["Random Forest"][hand], feature_vector_sklearn),
                "SVC":           predict_sklearn(sklearn_models["SVC"][hand],           feature_vector_sklearn),
                "SVR":           predict_sklearn(sklearn_models["SVR"][hand],           feature_vector_sklearn),
                "CNN":           predict_cnn(cnn_models[hand]["model"],                 feature_vector_cnn),
            }

            # ── Display results ────────────────────────────
            st.markdown("---")
            st.subheader(f"📊 Predictions — {hand} Hand")

            for model_name, (pred, probs) in results.items():
                with st.expander(f"**{model_name}** → {CLASS_LABELS[pred]}", expanded=True):
                    if probs is not None:
                        prob_df = pd.DataFrame({
                            "Class":       list(CLASS_LABELS.values()),
                            "Probability": [f"{p:.3f}" for p in probs]
                        })
                        st.dataframe(prob_df, use_container_width=True, hide_index=True)
                        st.progress(float(probs[pred]), text=f"Confidence: {probs[pred]:.1%}")
                    else:
                        st.info("SVR does not output class probabilities.")