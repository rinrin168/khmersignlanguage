# ============================================================
# Khmer Sign Language Project - Environment Test
# Run this script to confirm all packages are working.
# How to run:
#   venv\Scripts\python.exe test_environment.py
# ============================================================

import sys
import os

PASS = "[OK]"
FAIL = "[FAIL]"

print("=" * 55)
print("  Khmer Sign Language - Environment Check")
print("=" * 55)

# --- 1. Python Version ---
print(f"\n{PASS} Python version : {sys.version.split()[0]}")

# --- 2. NumPy ---
try:
    import numpy as np
    print(f"{PASS} NumPy          : {np.__version__}")
except ImportError as e:
    print(f"{FAIL} NumPy          : {e}")

# --- 3. Pandas ---
try:
    import pandas as pd
    print(f"{PASS} Pandas         : {pd.__version__}")
except ImportError as e:
    print(f"{FAIL} Pandas         : {e}")

# --- 4. Matplotlib ---
try:
    import matplotlib
    print(f"{PASS} Matplotlib     : {matplotlib.__version__}")
except ImportError as e:
    print(f"{FAIL} Matplotlib     : {e}")

# --- 5. scikit-learn ---
try:
    import sklearn
    print(f"{PASS} scikit-learn   : {sklearn.__version__}")
except ImportError as e:
    print(f"{FAIL} scikit-learn   : {e}")

# --- 6. OpenCV ---
try:
    import cv2
    print(f"{PASS} OpenCV         : {cv2.__version__}")
except ImportError as e:
    print(f"{FAIL} OpenCV         : {e}")

# --- 7. MediaPipe ---
try:
    import mediapipe as mp
    print(f"{PASS} MediaPipe      : {mp.__version__}")
except ImportError as e:
    print(f"{FAIL} MediaPipe      : {e}")

# --- 8. TensorFlow / Keras ---
try:
    import tensorflow as tf
    print(f"{PASS} TensorFlow     : {tf.__version__}")
    print(f"{PASS} Keras (built-in): {tf.keras.__version__}")
except ImportError as e:
    print(f"{FAIL} TensorFlow     : {e}")

# --- 9. Streamlit ---
try:
    import streamlit
    print(f"{PASS} Streamlit      : {streamlit.__version__}")
except ImportError as e:
    print(f"{FAIL} Streamlit      : {e}")

# --- Final Result ---
print("\n" + "=" * 55)
print("  If all lines above show [OK], you are ready to go!")
print("=" * 55)
