# ============================================================
# Stage 2 — collect_data.py
# Purpose : Record webcam sequences and save landmark arrays
#           for each Khmer Sign Language word.
#
# How to run:
#   venv\Scripts\python.exe src\collect_data.py                 # record every word not finished yet
#   venv\Scripts\python.exe src\collect_data.py --words delicious sorry   # record only these
#   venv\Scripts\python.exe src\collect_data.py --list          # show progress per word
#
# What happens:
#   For each word to record:
#     1. Shows a "READY?" screen and WAITS for you (press SPACE to start,
#        S to skip the word, Q to quit) — take as long as you need to
#        look up the sign.
#     2. Records NUM_SEQUENCES short clips (e.g. 30 clips), each with a
#        COUNTDOWN_SECONDS countdown (4s).
#     3. Each clip = SEQUENCE_LENGTH frames (90 frames ≈ 3 seconds).
#     4. Saves the landmarks as a .npy file in data/<word>/<seq_num>/landmarks.npy
#
# Re-running is safe: words that already have all clips are skipped
# (unless you name them with --words, which re-records them from clip 0).
# ============================================================

import argparse
import os
import time
import urllib.request
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from landmarks import extract_landmarks, SEQUENCE_LENGTH, FEATURE_SIZE

# ─── SETTINGS (edit words as desired) ───────────────────────
WORDS = [
    "how_are_you",
    "again",
    "correct",          # correct / right
    "incorrect",        # incorrect / wrong
    "dont_understand",
    "understand",
    "deaf_person",
    "hearing_person",   # person who can hear
    "delicious",
    "sorry",
    "thank_you",
]

NUM_SEQUENCES    = 30   # Clips to record per word
COUNTDOWN_SECONDS = 4   # Get-ready time before each clip
# Frames per clip (SEQUENCE_LENGTH = 90, ~3 seconds) is set in landmarks.py
DATA_DIR        = "data"
MODEL_DIR        = "models"
MODEL_PATH       = os.path.join(MODEL_DIR, "holistic_landmarker.task")
MODEL_URL        = "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task"
WINDOW           = "Khmer Sign Language - Data Collector"
# ────────────────────────────────────────────────────────────

def ensure_model():
    os.makedirs(MODEL_DIR, exist_ok=True)
    if not os.path.exists(MODEL_PATH):
        print("Downloading holistic model...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

def clips_recorded(word):
    """Count saved clips that have the current shape (old-length clips don't count)."""
    count = 0
    for seq in range(NUM_SEQUENCES):
        path = os.path.join(DATA_DIR, word, str(seq), "landmarks.npy")
        if os.path.exists(path) and np.load(path, mmap_mode="r").shape == (SEQUENCE_LENGTH, FEATURE_SIZE):
            count += 1
    return count

def print_progress():
    print(f"\nProgress (target {NUM_SEQUENCES} clips per word):")
    for i, word in enumerate(WORDS, 1):
        n = clips_recorded(word)
        status = "done" if n >= NUM_SEQUENCES else ("not started" if n == 0 else "partial")
        print(f"  {i:2d}. {word:<16} {n:2d}/{NUM_SEQUENCES}  {status}")

def draw_banner(frame, text, color):
    h, w, _ = frame.shape
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 65), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    cv2.putText(frame, text, (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

def draw_hud(frame, word, seq_num, frame_num, countdown=None):
    if countdown:
        draw_banner(frame, f"GET READY for '{word.upper()}' ({countdown}s)", (0, 215, 255))
    else:
        draw_banner(
            frame,
            f"RECORDING: {word.upper()} | Clip {seq_num+1}/{NUM_SEQUENCES} | Frame {frame_num+1}/{SEQUENCE_LENGTH}",
            (0, 255, 120),
        )

def wait_for_start(cap, word, position, total):
    """Wait until the user presses SPACE. Returns 'start', 'skip' or 'quit'."""
    while True:
        ret, frame = cap.read()
        if not ret:
            return "quit"
        frame = cv2.flip(frame, 1)
        draw_banner(frame, f"Word {position}/{total}: '{word.upper()}'", (0, 215, 255))
        h = frame.shape[0]
        cv2.putText(frame, "SPACE = start recording   S = skip this word   Q = quit",
                    (20, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow(WINDOW, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            return "start"
        if key == ord('s'):
            return "skip"
        if key == ord('q'):
            return "quit"

def record_word(cap, landmarker, word):
    """Record all clips for one word. Returns False if the user quit."""
    for seq in range(NUM_SEQUENCES):
        # Countdown before each clip
        start = time.time()
        while time.time() - start < COUNTDOWN_SECONDS:
            ret, frame = cap.read()
            if not ret:
                return False
            frame = cv2.flip(frame, 1)
            rem = int(COUNTDOWN_SECONDS - (time.time() - start)) + 1
            draw_hud(frame, word, seq, 0, countdown=rem)
            cv2.imshow(WINDOW, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return False

        # Record SEQUENCE_LENGTH frames
        sequence_data = []
        for frame_idx in range(SEQUENCE_LENGTH):
            ret, frame = cap.read()
            if not ret:
                return False
            frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            results = landmarker.detect(mp_img)

            sequence_data.append(extract_landmarks(results))

            draw_hud(frame, word, seq, frame_idx)
            cv2.imshow(WINDOW, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return False

        # Create the clip folder only now, so quitting never leaves empty folders behind
        clip_dir = os.path.join(DATA_DIR, word, str(seq))
        os.makedirs(clip_dir, exist_ok=True)
        np.save(os.path.join(clip_dir, "landmarks.npy"), np.array(sequence_data))
        print(f"  Saved clip {seq+1}/{NUM_SEQUENCES} for '{word}'")
    return True

def parse_args():
    p = argparse.ArgumentParser(description="Record Khmer Sign Language clips.")
    p.add_argument("--words", nargs="+", metavar="WORD",
                   help="record only these words (re-records them from clip 0)")
    p.add_argument("--list", action="store_true", help="show progress per word and exit")
    return p.parse_args()

def main():
    args = parse_args()

    if args.list:
        print_progress()
        return

    if args.words:
        unknown = [w for w in args.words if w not in WORDS]
        if unknown:
            print(f"Unknown word(s): {unknown}\nChoose from: {WORDS}")
            return
        to_record = args.words
    else:
        to_record = [w for w in WORDS if clips_recorded(w) < NUM_SEQUENCES]

    if not to_record:
        print("All words already have all clips. Next step: run extract_landmarks.py")
        print("(To re-record a word, use: --words <word>)")
        return

    print(f"Words to record this session: {to_record}")
    ensure_model()

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
        for position, word in enumerate(to_record, 1):
            print(f"\n==========================================")
            print(f"  >>> NEXT WORD: '{word}'  ({position}/{len(to_record)})")
            print(f"==========================================")

            action = wait_for_start(cap, word, position, len(to_record))
            if action == "quit":
                break
            if action == "skip":
                print(f"  Skipped '{word}'.")
                continue

            if not record_word(cap, landmarker, word):
                print(f"  Stopped during '{word}' ({clips_recorded(word)}/{NUM_SEQUENCES} clips saved).")
                break

    cap.release()
    cv2.destroyAllWindows()
    print_progress()
    print("\nWhen every word shows 'done', run: extract_landmarks.py")

if __name__ == "__main__":
    main()
