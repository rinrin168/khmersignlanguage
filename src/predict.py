# ============================================================
# Stage 7 — predict.py
# Purpose : Real-time prediction using Webcam and a trained
#           PyTorch model (LSTM / GRU / Transformer).
#
# How to run:
#   venv\Scripts\python.exe src\predict.py --model transformer
# Press  Q  to quit.
# ============================================================

import argparse
import os
import sys
import collections

import cv2
import numpy as np
import torch
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

sys.path.insert(0, os.path.dirname(__file__))
from landmarks import extract_landmarks  # noqa: E402
from models import build_model  # noqa: E402
from utils import get_device  # noqa: E402

MODELS_DIR = "models"
DATA_DIR = "data"
HOLISTIC_PATH = os.path.join(MODELS_DIR, "holistic_landmarker.task")
SEQUENCE_LENGTH = 30
CONFIDENCE_THRESH = 0.70


def parse_args():
    p = argparse.ArgumentParser(description="Real-time Khmer Sign Language recognition.")
    p.add_argument("--model", default="transformer", choices=["lstm", "gru", "transformer"])
    return p.parse_args()


def main():
    args = parse_args()
    device = get_device()

    model_path = os.path.join(MODELS_DIR, f"{args.model}_khmer_sign.pt")
    if not os.path.exists(model_path):
        print(f"Trained model not found at {model_path}. Run train.py --model {args.model} first.")
        return

    labels = np.load(os.path.join(DATA_DIR, "labels.npy"), allow_pickle=True)
    net = build_model(args.model, input_size=258, num_classes=len(labels)).to(device)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()
    print(f"Loaded '{args.model}' model. Classes ({len(labels)}): {list(labels)}")

    base_options = python.BaseOptions(model_asset_path=HOLISTIC_PATH)
    options = vision.HolisticLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera.")
        return

    sequence = collections.deque(maxlen=SEQUENCE_LENGTH)
    current_word = "Waiting..."
    current_conf = 0.0

    print("=" * 60)
    print("  Real-time Khmer Sign Language Recognition started!")
    print("  Press 'Q' to quit.")
    print("=" * 60)

    with vision.HolisticLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
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
                    probs = torch.softmax(net(X), dim=1)[0].cpu().numpy()
                idx = int(np.argmax(probs))
                current_conf = float(probs[idx])
                if current_conf >= CONFIDENCE_THRESH:
                    current_word = labels[idx]

            h, w, _ = frame.shape
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, h - 90), (w, h), (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

            bar_w = int(w * len(sequence) / SEQUENCE_LENGTH)
            cv2.rectangle(frame, (0, h - 6), (bar_w, h), (0, 230, 120), -1)

            if current_word != "Waiting...":
                cv2.putText(frame, current_word.upper().replace("_", " "), (20, h - 45),
                            cv2.FONT_HERSHEY_DUPLEX, 1.3, (0, 255, 200), 2)
                cv2.putText(frame, f"{current_conf*100:.1f}% confidence", (20, h - 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 255, 200), 1)
            else:
                cv2.putText(frame, "Waiting for sign...", (20, h - 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.85, (150, 150, 150), 2)

            cv2.imshow("Khmer Sign Language Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
