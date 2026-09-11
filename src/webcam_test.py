# ============================================================
# Stage 1 — webcam_test.py
# Purpose : Make sure OpenCV can open your webcam.
# How to run:
#   venv\Scripts\python.exe src\webcam_test.py
# Press  Q  on the keyboard to quit.
# ============================================================

import cv2  # OpenCV for camera access and image display

print("Opening webcam... press Q to quit.")

# 0 = first webcam.  Try 1 or 2 if nothing shows up.
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Cannot open webcam. Check it is plugged in.")
else:
    print("Webcam opened successfully!")

while True:
    # Read one frame from the camera
    ret, frame = cap.read()          # ret = True if the read worked
    if not ret:
        print("Failed to grab frame")
        break

    # Show the frame in a window called "Webcam Test"
    cv2.imshow("Webcam Test", frame)

    # Wait 1 ms; if the user pressed Q, exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Always release the camera and close windows when done
cap.release()
cv2.destroyAllWindows()
print("Webcam test finished.")
