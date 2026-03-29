import streamlit as st
import numpy as np
import pandas as pd
import pickle
import torch
import torch.nn as nn
import json
import cv2
import av
import time

from scipy.interpolate import interp1d
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

import mediapipe as mp
from mediapipe.tasks.python        import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions

from preprocessing import pad_signal, extract_features_from_signal

# ── Singleton signal store — survives all Streamlit reruns ──
import signal_store as ss

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
        div[data-testid="column"] button {
            width: 100%;
            padding: 0.75rem;
            font-size: 1rem;
            font-weight: 600;
            border-radius: 8px;
            border: 2px solid transparent;
            transition: all 0.2s ease;
        }
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
# CONSTANTS
# ─────────────────────────────────────────────────────────────
CLASS_LABELS          = {0: "🟢 Mild (UPDRS 0–1)", 1: "🟡 Moderate (UPDRS 2–3)", 2: "🔴 Severe (UPDRS 4)"}
MIN_RECORDING_SECONDS = 30
TARGET_HZ             = 60   # must match training data sample rate

# ─────────────────────────────────────────────────────────────
# CNN ARCHITECTURE
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
def load_cnn_models():
    cnn = {}
    for side in ["Left", "Right"]:
        with open(f"saved_models/cnn_{side.lower()}_params.json") as f:
            params = json.load(f)
        input_dim = len(params['features'])
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

@st.cache_resource
def load_sklearn_features():
    features = {}
    for side in ["Left", "Right"]:
        with open(f"saved_models/sklearn_features_{side.lower()}.json") as f:
            features[side] = json.load(f)
    return features

# ─────────────────────────────────────────────────────────────
# HAND SELECTION
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
# INTERPOLATION HELPER
# Resamples sparse webcam signal (~10Hz) to training rate (60Hz)
# ─────────────────────────────────────────────────────────────
def interpolate_to_target_hz(amps, times, target_hz=TARGET_HZ):
    times_arr = np.array(times)
    amps_arr  = np.array(amps)

    duration      = times_arr[-1] - times_arr[0]
    n_target      = int(duration * target_hz)
    times_uniform = np.linspace(times_arr[0], times_arr[-1], n_target)

    # Cubic interpolation — smoother than linear for signal data
    interpolator  = interp1d(times_arr, amps_arr, kind='cubic', fill_value='extrapolate')
    amps_uniform  = interpolator(times_uniform)

    return amps_uniform.tolist(), times_uniform.tolist()

# ─────────────────────────────────────────────────────────────
# VIDEO PROCESSOR
# Fix 1: Recording starts automatically when stream connects —
#         no separate Start Recording button needed.
# Fix 2: Samples written to ss.store (module singleton) so they
#         survive Streamlit reruns and button clicks.
# ─────────────────────────────────────────────────────────────
class FingerTapProcessor(VideoProcessorBase):
    def __init__(self):
        self.detector = HandLandmarker.create_from_options(
            HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
                num_hands=1,
                min_hand_detection_confidence=0.7,
                min_tracking_confidence=0.7,
            )
        )
        # Auto-start recording as soon as stream connects
        with ss.lock:
            ss.store["recording"]  = True
            ss.store["amplitudes"] = []
            ss.store["timestamps"] = []
            ss.store["start_time"] = None

    def recv(self, frame):
        img     = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        results  = self.detector.detect(mp_image)

        hand_detected = False

        if results.hand_landmarks:
            hand_detected = True
            for hand_lms in results.hand_landmarks:

                # Draw all 21 landmarks
                for lm in hand_lms:
                    cv2.circle(img, (int(lm.x * w), int(lm.y * h)), 4, (0, 255, 0), -1)

                # Index fingertip = 8, Thumb tip = 4
                index_tip = hand_lms[8]
                thumb_tip = hand_lms[4]
                amplitude = abs(index_tip.y - thumb_tip.y) * h

                ix, iy = int(index_tip.x * w), int(index_tip.y * h)
                tx, ty = int(thumb_tip.x * w), int(thumb_tip.y * h)
                cv2.line(img, (ix, iy), (tx, ty), (255, 100, 0), 2)
                cv2.putText(img, f"Amp: {amplitude:.1f}px", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                # ── Write to singleton store ───────────────
                with ss.lock:
                    if ss.store["recording"]:
                        now = time.time()
                        if ss.store["start_time"] is None:
                            ss.store["start_time"] = now

                        elapsed   = now - ss.store["start_time"]
                        remaining = max(0, MIN_RECORDING_SECONDS - elapsed)

                        cv2.putText(
                            img,
                            f"Min remaining: {remaining:.1f}s" if remaining > 0 else "Min reached ✓",
                            (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 200, 255) if remaining > 0 else (0, 255, 0), 2,
                        )

                        ss.store["amplitudes"].append(amplitude)
                        ss.store["timestamps"].append(round(elapsed, 4))

        # Warn user if hand not detected
        if not hand_detected:
            cv2.putText(img, "No hand detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # REC indicator
        with ss.lock:
            is_recording = ss.store["recording"]
            n_samples    = len(ss.store["amplitudes"])

        if is_recording:
            cv2.circle(img, (w - 30, 30), 12, (0, 0, 255), -1)
            cv2.putText(img, "REC", (w - 70, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(img, f"Samples: {n_samples}", (10, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ─────────────────────────────────────────────────────────────
# VIDEO CAPTURE SECTION
# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 🎥 Live Video Capture")

st.info(
    "⏱ **Minimum recording duration: 30 seconds.** "
    "Recording starts automatically when the stream connects. "
    "Tap your index finger and thumb together repeatedly at a steady pace. "
    "Click **💾 Save Signal** BEFORE stopping the stream."
)

RTC_CONFIG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

ctx = webrtc_streamer(
    key="finger-tap",
    video_processor_factory=FingerTapProcessor,
    rtc_configuration=RTC_CONFIG,
    media_stream_constraints={"video": True, "audio": False},
)

if ctx.video_processor:

    col_save, col_reset = st.columns(2)

    with col_save:
        if st.button("💾 Save Signal", use_container_width=True):
            with ss.lock:
                amps  = list(ss.store["amplitudes"])
                times = list(ss.store["timestamps"])
                ss.store["recording"] = False

            if len(amps) < 2:
                st.warning("Not enough samples yet — keep tapping.")
            else:
                # ── Fix 2: Interpolate sparse webcam signal to 60Hz ──
                amps_interp, times_interp = interpolate_to_target_hz(amps, times)

                st.session_state["raw_amps"]  = amps_interp
                st.session_state["raw_times"] = times_interp
                st.success(
                    f"✅ Saved {len(amps)} raw samples → "
                    f"interpolated to {len(amps_interp)} samples at {TARGET_HZ}Hz "
                    f"({times[-1]:.1f}s). You can now stop the stream."
                )

    with col_reset:
        if st.button("🔄 Reset", use_container_width=True):
            with ss.lock:
                ss.store["amplitudes"] = []
                ss.store["timestamps"] = []
                ss.store["recording"]  = True   # keep recording after reset
                ss.store["start_time"] = None
            st.session_state.pop("raw_amps",          None)
            st.session_state.pop("raw_times",         None)
            st.session_state.pop("amplitude_input",   None)
            st.session_state.pop("time_input",        None)
            st.success("Signal cleared — recording restarted.")

    # Live duration + sample count display
    with ss.lock:
        times_so_far = list(ss.store["timestamps"])
        n_so_far     = len(ss.store["amplitudes"])

    if times_so_far:
        duration = times_so_far[-1]
        colour   = "green" if duration >= MIN_RECORDING_SECONDS else "orange"
        st.markdown(
            f"**Recorded so far:** :{colour}[{duration:.1f}s]"
            + (" ✅" if duration >= MIN_RECORDING_SECONDS else f" — need {MIN_RECORDING_SECONDS - duration:.1f}s more")
        )
        est_interp = int(duration * TARGET_HZ)
        st.caption(f"Raw samples: {n_so_far} → will interpolate to ~{est_interp} at {TARGET_HZ}Hz")

# ─────────────────────────────────────────────────────────────
# TRIM + LOAD
# ─────────────────────────────────────────────────────────────
if "raw_amps" in st.session_state and "raw_times" in st.session_state:
    amps  = st.session_state["raw_amps"]
    times = st.session_state["raw_times"]

    st.markdown(f"**Saved signal:** {len(amps)} samples — {times[-1]:.1f}s at {TARGET_HZ}Hz")
    st.markdown("#### ✂️ Trim Signal")
    st.caption("Drag to remove unwanted sections at the start or end.")

    trim_start, trim_end = st.slider(
        "Trim range (seconds)",
        min_value=0.0,
        max_value=float(round(times[-1], 1)),
        value=(0.0, float(round(times[-1], 1))),
        step=0.1,
    )

    times_arr     = np.array(times)
    amps_arr      = np.array(amps)
    mask          = (times_arr >= trim_start) & (times_arr <= trim_end)
    trimmed_times = times_arr[mask]
    trimmed_amps  = amps_arr[mask]
    trimmed_dur   = float(trimmed_times[-1] - trimmed_times[0]) if len(trimmed_times) > 1 else 0.0

    if trimmed_dur < MIN_RECORDING_SECONDS:
        st.warning(f"⚠️ Trimmed signal is {trimmed_dur:.1f}s — minimum is {MIN_RECORDING_SECONDS}s.")
    else:
        st.success(f"✅ Trimmed signal: {trimmed_dur:.1f}s ({len(trimmed_amps)} samples) — ready to load.")

    if st.button("📥 Load Signal into Fields", use_container_width=True):
        if trimmed_dur < MIN_RECORDING_SECONDS:
            st.error(f"Cannot load — signal is {trimmed_dur:.1f}s, minimum is {MIN_RECORDING_SECONDS}s.")
        else:
            # Normalise amplitude to 0–1 range
            a_min, a_max = trimmed_amps.min(), trimmed_amps.max()
            amps_norm    = (trimmed_amps - a_min) / (a_max - a_min + 1e-8)
            times_norm   = trimmed_times - trimmed_times[0]

            st.session_state["amplitude_input"] = ", ".join([f"{a:.6f}" for a in amps_norm])
            st.session_state["time_input"]      = ", ".join([f"{t:.6f}" for t in times_norm])
            st.success(f"✅ Loaded {len(amps_norm)} samples into signal fields below.")

# ─────────────────────────────────────────────────────────────
# SIGNAL INPUTS
# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 📈 Signal Data")
st.caption("Auto-filled from video capture, or paste manually.")

amplitude_input = st.text_area(
    "Amplitude values (comma separated)",
    value=st.session_state.get("amplitude_input", ""),
    height=120,
    key="amplitude_input",
)
time_input = st.text_area(
    "Time values (comma separated)",
    value=st.session_state.get("time_input", ""),
    height=120,
    key="time_input",
)

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def parse_signal(text):
    return np.array([float(x.strip()) for x in text.split(",") if x.strip() != ""])

def predict_sklearn(model, feature_vector):
    if hasattr(model, "predict_proba"):
        raw_probs = model.predict_proba(feature_vector)[0]
        classes   = model.classes_
        probs     = np.zeros(3)
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

# ─────────────────────────────────────────────────────────────
# PREDICT — all models at once
# ─────────────────────────────────────────────────────────────
if st.button("🔍 Predict", use_container_width=True):

    if not amplitude_input or not time_input:
        st.error("Please provide both amplitude and time values.")

    else:
        amplitude = parse_signal(amplitude_input)
        time_vals = parse_signal(time_input)

        if len(time_vals) > 1:
            manual_duration = time_vals[-1] - time_vals[0]
            if manual_duration < MIN_RECORDING_SECONDS:
                st.error(
                    f"⚠️ Signal duration is **{manual_duration:.1f}s** — "
                    f"minimum required is **{MIN_RECORDING_SECONDS}s**."
                )
                st.stop()

        amplitude, time_vals = pad_signal(amplitude, time_vals)
        features             = extract_features_from_signal(amplitude, time_vals)

        if features is None:
            st.error("Not enough valid taps detected. Please check your signal.")

        else:
            feature_vector   = pd.DataFrame([features])
            sklearn_models   = load_sklearn_models()
            sklearn_features = load_sklearn_features()
            cnn_models       = load_cnn_models()

            feature_vector_sklearn = feature_vector[sklearn_features[hand]]
            feature_vector_cnn     = feature_vector[cnn_models[hand]["features"]]

            results = {
                "Random Forest": predict_sklearn(sklearn_models["Random Forest"][hand], feature_vector_sklearn),
                "SVC":           predict_sklearn(sklearn_models["SVC"][hand],           feature_vector_sklearn),
                "SVR":           predict_sklearn(sklearn_models["SVR"][hand],           feature_vector_sklearn),
                "CNN":           predict_cnn(cnn_models[hand]["model"],                 feature_vector_cnn),
            }

            st.markdown("---")
            st.subheader(f"📊 Predictions — {hand} Hand")

            for model_name, (pred, probs) in results.items():
                with st.expander(f"**{model_name}** → {CLASS_LABELS[pred]}", expanded=True):
                    if probs is not None:
                        prob_df = pd.DataFrame({
                            "Class":       list(CLASS_LABELS.values()),
                            "Probability": [f"{p:.3f}" for p in probs],
                        })
                        st.dataframe(prob_df, use_container_width=True, hide_index=True)
                        st.progress(float(probs[pred]), text=f"Confidence: {probs[pred]:.1%}")
                    else:
                        st.info("SVR does not output class probabilities.")