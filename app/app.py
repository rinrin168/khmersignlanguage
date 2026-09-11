# ============================================================
# Stage 8 — app.py  (Streamlit Web Interface)
# Purpose : Web UI to run live webcam Khmer Sign Language
#           Recognition and browse the LSTM / GRU / Transformer
#           comparison results.
#
# How to run:
#   venv\Scripts\streamlit.exe run app\app.py
# ============================================================

import sys
import os
import collections

import streamlit as st
import numpy as np
import cv2
import torch
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from landmarks import extract_landmarks  # noqa: E402
from models import build_model  # noqa: E402
from utils import get_device  # noqa: E402

st.set_page_config(
    page_title="Khmer Sign Language Recogniser",
    page_icon="🤟",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main-header {
    text-align: center;
    padding: 1.5rem;
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border-radius: 16px;
    margin-bottom: 1.5rem;
    color: white;
}
.main-header h1 {
    font-size: 2.2rem;
    font-weight: 700;
    margin: 0;
    background: linear-gradient(90deg, #00d2ff, #a855f7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.prediction-box {
    background: linear-gradient(135deg, #0f3460, #533483);
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    margin-top: 1rem;
}
.prediction-word {
    font-size: 2.5rem;
    font-weight: 700;
    color: #00d2ff;
}
</style>
""", unsafe_allow_html=True)

MODELS_DIR = "models"
DATA_DIR = "data"
RESULTS_DIR = "results"
HOLISTIC_PATH = os.path.join(MODELS_DIR, "holistic_landmarker.task")
SEQUENCE_LENGTH = 30
APPROACHES = ["lstm", "gru", "transformer"]


@st.cache_resource
def load_assets(approach: str):
    model_path = os.path.join(MODELS_DIR, f"{approach}_khmer_sign.pt")
    labels_path = os.path.join(DATA_DIR, "labels.npy")
    if not os.path.exists(model_path) or not os.path.exists(labels_path):
        return None, None
    labels = np.load(labels_path, allow_pickle=True)
    device = get_device()
    model = build_model(approach, input_size=258, num_classes=len(labels)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, labels


st.markdown("""
<div class="main-header">
    <h1>🤟 Khmer Sign Language Recogniser</h1>
    <p>Real-time Cambodian Sign Language recognition — MediaPipe Holistic + LSTM / GRU / Transformer (PyTorch)</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## ⚙️ Settings")
    approach = st.selectbox("Approach", APPROACHES, index=2)
    confidence_threshold = st.slider("Confidence Threshold", 0.4, 1.0, 0.70, 0.05)
    st.markdown("---")
    st.markdown("### 📋 Quick Commands")
    st.code("venv\\Scripts\\python.exe src\\collect_data.py", language="bash")
    st.code(f"venv\\Scripts\\python.exe src\\train.py --model {approach}", language="bash")
    st.code(f"venv\\Scripts\\python.exe src\\predict.py --model {approach}", language="bash")

tab1, tab2, tab3 = st.tabs(["🎥 Live Recognition", "📊 Model Comparison", "📖 Architecture"])

with tab1:
    device = get_device()
    model, labels = load_assets(approach)
    if model is None:
        st.warning(f"⚠️ No trained '{approach}' model found yet. "
                    f"Run `src/collect_data.py`, `src/preprocess.py`, then "
                    f"`src/train.py --model {approach}`.")
    else:
        st.success(f"✅ {approach.upper()} model ready! Recognizing **{len(labels)}** signs: "
                    + ", ".join(f"`{w}`" for w in labels))
        col1, col2 = st.columns([3, 2])

        with col1:
            st.markdown("### Camera Stream")
            run = st.toggle("▶️ Start Live Camera", key="webcam_toggle")
            frame_box = st.empty()

        with col2:
            st.markdown("### Prediction")
            pred_box = st.empty()

        if run:
            base_options = python.BaseOptions(model_asset_path=HOLISTIC_PATH)
            options = vision.HolisticLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE
            )
            cap = cv2.VideoCapture(0)
            sequence = collections.deque(maxlen=SEQUENCE_LENGTH)
            current_word = "Waiting..."
            current_conf = 0.0

            with vision.HolisticLandmarker.create_from_options(options) as landmarker:
                while run:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Failed to read webcam.")
                        break

                    frame = cv2.flip(frame, 1)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    results = landmarker.detect(mp_img)

                    landmarks = extract_landmarks(results)
                    sequence.append(landmarks)

                    if len(sequence) == SEQUENCE_LENGTH:
                        X = torch.tensor(np.array(sequence), dtype=torch.float32).unsqueeze(0).to(device)
                        with torch.no_grad():
                            probs = torch.softmax(model(X), dim=1)[0].cpu().numpy()
                        idx = int(np.argmax(probs))
                        current_conf = float(probs[idx])
                        if current_conf >= confidence_threshold:
                            current_word = labels[idx]

                    frame_box.image(frame, channels="BGR", use_container_width=True)
                    pred_box.markdown(f"""
                    <div class="prediction-box">
                        <div class="prediction-word">{current_word.upper().replace("_", " ")}</div>
                        <p style="color:#94a3b8; margin-top:0.5rem;">Confidence: <strong style="color:#00d2ff;">{current_conf*100:.1f}%</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                    run = st.session_state.get("webcam_toggle", False)
            cap.release()

with tab2:
    st.markdown("### Side-by-side comparison across all trained approaches")
    table_path = os.path.join(RESULTS_DIR, "comparison_table.csv")
    if os.path.exists(table_path):
        import pandas as pd
        st.dataframe(pd.read_csv(table_path), use_container_width=True)
    else:
        st.info("Run `src/compare.py` after training/evaluating each approach to populate this table.")

    fig_dir = os.path.join(RESULTS_DIR, "figures")
    col_a, col_b = st.columns(2)
    acc_fig = os.path.join(fig_dir, "accuracy_comparison.png")
    curves_fig = os.path.join(fig_dir, "learning_curves.png")
    if os.path.exists(acc_fig):
        col_a.image(acc_fig, caption="Test Accuracy & F1 by Approach")
    if os.path.exists(curves_fig):
        col_b.image(curves_fig, caption="Validation Accuracy Learning Curves")

    st.markdown(f"#### {approach.upper()} details")
    hist_path = os.path.join(RESULTS_DIR, approach, "training_history.png")
    cm_path = os.path.join(RESULTS_DIR, approach, "confusion_matrix.png")
    col_c, col_d = st.columns(2)
    if os.path.exists(hist_path):
        col_c.image(hist_path, caption=f"{approach.upper()} Accuracy & Loss Curves")
    if os.path.exists(cm_path):
        col_d.image(cm_path, caption=f"{approach.upper()} Confusion Matrix")

with tab3:
    st.markdown("""
### Pipeline Overview
```
Webcam Frame
  ↓
MediaPipe Holistic Landmarker (258 values/frame: 132 Pose + 63 Left Hand + 63 Right Hand)
  ↓
Rolling 30-Frame Sequence Buffer (30 × 258)
  ↓
One of three PyTorch classifiers, trained from scratch on the identical split:
  • LSTM        — stacked LSTM(64→128→64)
  • GRU         — stacked GRU(64→128→64)
  • Transformer — self-attention encoder over the 30 frames
  ↓
Dense Classification Head (64 units → Softmax)
  ↓
Recognized Khmer Sign Word
```
""")
