# ============================================================
# landmarks.py — shared MediaPipe landmark extraction, used by
# collect_data.py and predict.py so the feature vector definition
# lives in exactly one place.
# ============================================================

import numpy as np

POSE_LANDMARKS = 33
HAND_LANDMARKS = 21
FEATURE_SIZE = POSE_LANDMARKS * 4 + HAND_LANDMARKS * 3 + HAND_LANDMARKS * 3  # 258


def extract_landmarks(result) -> np.ndarray:
    """
    Extract 258 numbers per frame from a MediaPipe HolisticLandmarker result:
      Pose: 33 points x 4 (x,y,z,visibility) = 132
      Left Hand: 21 points x 3 (x,y,z)        = 63
      Right Hand: 21 points x 3 (x,y,z)       = 63
      Total = 258
    """
    if result.pose_landmarks:
        pose = np.array([[lm.x, lm.y, lm.z, lm.visibility or 0.0]
                         for lm in result.pose_landmarks]).flatten()
    else:
        pose = np.zeros(POSE_LANDMARKS * 4)

    if result.left_hand_landmarks:
        lh = np.array([[lm.x, lm.y, lm.z]
                       for lm in result.left_hand_landmarks]).flatten()
    else:
        lh = np.zeros(HAND_LANDMARKS * 3)

    if result.right_hand_landmarks:
        rh = np.array([[lm.x, lm.y, lm.z]
                       for lm in result.right_hand_landmarks]).flatten()
    else:
        rh = np.zeros(HAND_LANDMARKS * 3)

    return np.concatenate([pose, lh, rh])
