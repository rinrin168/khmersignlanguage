# ============================================================
# Stage 2 — collect_data.py
# Purpose : Record webcam sequences and save landmark arrays
#           for each Khmer Sign Language word.
#
# How to run:
#   venv\Scripts\python.exe src\collect_data.py
#
# What happens:
#   For each word in WORDS list:
#     1. Shows a 3-second countdown so you can get ready.
#     2. Records NUM_SEQUENCES short clips (e.g. 30 clips).
#     3. Each clip = SEQUENCE_LENGTH frames (e.g. 30 frames ≈ 1 second).
#     4. Saves the landmarks as a .npy file in data/<word>/<seq_num>/landmarks.npy
# ============================================================

import os
import time
import urllib.request
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from landmarks import extract_landmarks

# ─── SETTINGS (edit words as desired) ───────────────────────
WORDS = [
    "hello",       # sour sdei     (ស្ួស្ដី)
    "thank_you",   # orkun         (អរគុណ)
    "yes",         # baht/jas      (បាទ/ចាស)
    "no",          # te            (ទេ)
    "please",      # soum          (សូម)
    "sorry",       # som_tos       (សូមទោស)
    "help",        # chuoy         (ជួយ)
    "water",       # tuk           (ទឹក)
    "eat",         # si_bai        (ស៊ីបាយ)
    "name",        # chhmous       (ឈ្មោះ)
]

NUM_SEQUENCES    = 30   # Clips to record per word
SEQUENCE_LENGTH  = 30   # Frames per clip (~1 second at 30fps)
DATA_DIR         = "data"
MODEL_DIR        = "models"
MODEL_PATH       = os.path.join(MODEL_DIR, "holistic_landmarker.task")
MODEL_URL        = "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task"
# ────────────────────────────────────────────────────────────

def ensure_model():
    os.makedirs(MODEL_DIR, exist_ok=True)
    if not os.path.exists(MODEL_PATH):
        print("Downloading holistic model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

def create_folders():
    for word in WORDS:
        for seq in range(NUM_SEQUENCES):
            path = os.path.join(DATA_DIR, word, str(seq))
            os.makedirs(path, exist_ok=True)

def draw_hud(frame, word, seq_num, frame_num, countdown=None):
    h, w, _ = frame.shape
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 65), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    if countdown:
        text = f"GET READY for '{word.upper()}' ({countdown}s)"
        color = (0, 215, 255)
    else:
        text = f"RECORDING: {word.upper()} | Clip {seq_num+1}/{NUM_SEQUENCES} | Frame {frame_num+1}/{SEQUENCE_LENGTH}"
        color = (0, 255, 120)

    cv2.putText(frame, text, (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

def main():
    ensure_model()
    create_folders()

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HolisticLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam.")
        return

    with vision.HolisticLandmarker.create_from_options(options) as landmarker:
        for word in WORDS:
            print(f"\n==========================================")
            print(f"  >>> RECORDING WORD: '{word}'")
            print(f"==========================================")

            for seq in range(NUM_SEQUENCES):
                # 3-second countdown
                start = time.time()
                while time.time() - start < 3:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = cv2.flip(frame, 1)
                    rem = int(3 - (time.time() - start)) + 1
                    draw_hud(frame, word, seq, 0, countdown=rem)
                    cv2.imshow("Khmer Sign Language - Data Collector", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        cap.release()
                        cv2.destroyAllWindows()
                        return

                # Record SEQUENCE_LENGTH frames
                sequence_data = []
                for frame_idx in range(SEQUENCE_LENGTH):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = cv2.flip(frame, 1)

                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    results = landmarker.detect(mp_img)

                    landmarks = extract_landmarks(results)
                    sequence_data.append(landmarks)

                    draw_hud(frame, word, seq, frame_idx)
                    cv2.imshow("Khmer Sign Language - Data Collector", frame)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        cap.release()
                        cv2.destroyAllWindows()
                        return

                # Save .npy file
                save_path = os.path.join(DATA_DIR, word, str(seq), "landmarks.npy")
                np.save(save_path, np.array(sequence_data))
                print(f"  Saved clip {seq+1}/{NUM_SEQUENCES} for '{word}'")

    cap.release()
    cv2.destroyAllWindows()
    print("\nData collection finished! Next step: run extract_landmarks.py")

if __name__ == "__main__":
    main()
