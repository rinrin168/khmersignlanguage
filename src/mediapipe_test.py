# ============================================================
# Stage 3 — mediapipe_test.py
# Purpose : MediaPipe Holistic Live Demo (Python 3.11 / 3.13 compatible)
#           - Opens Webcam
#           - Detects Face (Mesh & Contours)
#           - Detects Left Hand (21 keypoints)
#           - Detects Right Hand (21 keypoints)
#           - Detects Body / Pose (33 keypoints)
#           - Draws all landmarks and connections in real time
#
# How to run:
#   venv\Scripts\python.exe src\mediapipe_test.py
# Press  Q  on your keyboard to quit.
# ============================================================

import os
import urllib.request
import time
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ─── 1. Ensure Model Asset Exists ─────────────────────────────
MODEL_DIR  = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "holistic_landmarker.task")
MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task"

os.makedirs(MODEL_DIR, exist_ok=True)
if not os.path.exists(MODEL_PATH):
    print("Downloading holistic_landmarker.task model (~13MB)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model downloaded successfully!")

# ─── 2. Landmark Connection Maps ──────────────────────────────
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index finger
    (5, 9), (9, 10), (10, 11), (11, 12),   # Middle finger
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring finger
    (13, 17), (17, 18), (18, 19), (19, 20),# Pinky
    (0, 17)                                # Palm base
]

POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15),          # Left arm & shoulders
    (12, 14), (14, 16),                    # Right arm
    (11, 23), (12, 24), (23, 24),          # Torso
    (23, 25), (25, 27),                    # Left leg
    (24, 26), (26, 28)                     # Right leg
]

def draw_hand(image, landmarks, point_color, line_color):
    """Draw hand joints and connecting bones."""
    if not landmarks:
        return
    h, w, _ = image.shape
    coords = []
    for lm in landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        coords.append((cx, cy))
        cv2.circle(image, (cx, cy), 4, point_color, -1)
        cv2.circle(image, (cx, cy), 5, (255, 255, 255), 1)

    for start_idx, end_idx in HAND_CONNECTIONS:
        if start_idx < len(coords) and end_idx < len(coords):
            cv2.line(image, coords[start_idx], coords[end_idx], line_color, 2)


def draw_pose(image, landmarks):
    """Draw body skeleton connections and joints."""
    if not landmarks:
        return
    h, w, _ = image.shape
    coords = []
    for lm in landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        coords.append((cx, cy))
        # Draw upper body points
        if lm.visibility is None or lm.visibility > 0.5:
            cv2.circle(image, (cx, cy), 5, (0, 255, 255), -1)

    for start_idx, end_idx in POSE_CONNECTIONS:
        if start_idx < len(coords) and end_idx < len(coords):
            p1, p2 = coords[start_idx], coords[end_idx]
            cv2.line(image, p1, p2, (0, 200, 255), 2)


def draw_face(image, landmarks):
    """Draw subtle facial points (eyes, nose, mouth contours)."""
    if not landmarks:
        return
    h, w, _ = image.shape
    # Draw sample key face points for performance and clarity
    step = 4  # skip some points for clean display
    for i in range(0, len(landmarks), step):
        lm = landmarks[i]
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(image, (cx, cy), 1, (200, 200, 255), -1)


# ─── 3. Main Camera Loop ──────────────────────────────────────
def main():
    print("=" * 60)
    print("  Starting MediaPipe Holistic Landmarker...")
    print("  Press 'Q' in the camera window anytime to quit.")
    print("=" * 60)

    # Initialize Landmarker
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HolisticLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ ERROR: Cannot access webcam. Make sure it is connected.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    prev_time = 0

    with vision.HolisticLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame.")
                break

            # Mirror selfie view
            frame = cv2.flip(frame, 1)

            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # Run Holistic Detection
            detection_result = landmarker.detect(mp_image)

            # Draw Landmarks
            # 1. Face
            if detection_result.face_landmarks:
                draw_face(frame, detection_result.face_landmarks)

            # 2. Pose / Body
            if detection_result.pose_landmarks:
                draw_pose(frame, detection_result.pose_landmarks)

            # 3. Left Hand (Magenta/Purple)
            if detection_result.left_hand_landmarks:
                draw_hand(frame, detection_result.left_hand_landmarks,
                          point_color=(255, 0, 255), line_color=(180, 50, 200))

            # 4. Right Hand (Cyan/Blue)
            if detection_result.right_hand_landmarks:
                draw_hand(frame, detection_result.right_hand_landmarks,
                          point_color=(0, 255, 255), line_color=(0, 180, 230))

            # Calculate FPS
            curr_time = time.time()
            fps = int(1 / (curr_time - prev_time)) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time

            # Status HUD
            face_st = "YES" if detection_result.face_landmarks else "NO"
            pose_st = "YES" if detection_result.pose_landmarks else "NO"
            lh_st   = "YES" if detection_result.left_hand_landmarks else "NO"
            rh_st   = "YES" if detection_result.right_hand_landmarks else "NO"

            # Draw HUD Box
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (460, 115), (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

            cv2.putText(frame, f"FPS: {fps} | MediaPipe Holistic", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            cv2.putText(frame, f"Face: {face_st}  |  Pose: {pose_st}", (20, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.putText(frame, f"Left Hand: {lh_st}  |  Right Hand: {rh_st}", (20, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

            cv2.imshow("Khmer Sign Language - MediaPipe Holistic Test", frame)

            # Press 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("MediaPipe Holistic test finished.")


if __name__ == "__main__":
    main()
