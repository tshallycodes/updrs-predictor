# signal_store.py
# ─────────────────────────────────────────────────────────────
# Module-level singleton — Python only loads this once,
# so ss.store is the exact same object across all Streamlit
# reruns and across the video processor thread.
# ─────────────────────────────────────────────────────────────
import threading

lock = threading.Lock()

store = {
    "amplitudes": [],
    "timestamps": [],
    "recording":  False,
    "start_time": None,
}