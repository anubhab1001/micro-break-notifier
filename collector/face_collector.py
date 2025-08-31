# collectors/face_collector.py
import cv2
import mediapipe as mp
import time
import csv
from datetime import datetime

# MediaPipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

def aspect_ratio(p1, p2, p3, p4, p5, p6):
    # eye aspect ratio formula
    return (abs(p2[1] - p6[1]) + abs(p3[1] - p5[1])) / (2.0 * abs(p1[0] - p4[0]))

def mouth_aspect_ratio(top_lip, bottom_lip):
    return abs(top_lip[1] - bottom_lip[1])

cap = cv2.VideoCapture(0)

with open("../data/face_data.csv", mode="a", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["timestamp", "EAR", "MAR", "head_tilt", "label"])  # label: 0 alert, 1 fatigued

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        EAR, MAR, tilt = 0, 0, 0
        if results.multi_face_landmarks:
            h, w, _ = frame.shape
            landmarks = results.multi_face_landmarks[0].landmark

            # Example: EAR using eye landmarks (just left eye here)
            left_eye = [
                (int(landmarks[33].x * w), int(landmarks[33].y * h)),  # left corner
                (int(landmarks[160].x * w), int(landmarks[160].y * h)),  # upper
                (int(landmarks[158].x * w), int(landmarks[158].y * h)),  # upper
                (int(landmarks[133].x * w), int(landmarks[133].y * h)),  # right corner
                (int(landmarks[153].x * w), int(landmarks[153].y * h)),  # lower
                (int(landmarks[144].x * w), int(landmarks[144].y * h))   # lower
            ]
            EAR = aspect_ratio(*left_eye)

            # MAR (mouth aspect ratio) using top/bottom lips
            top_lip = (int(landmarks[13].x * w), int(landmarks[13].y * h))
            bottom_lip = (int(landmarks[14].x * w), int(landmarks[14].y * h))
            MAR = mouth_aspect_ratio(top_lip, bottom_lip)

            # head tilt using eyes y difference
            left_eye_center = (landmarks[33].x * w, landmarks[33].y * h)
            right_eye_center = (landmarks[263].x * w, landmarks[263].y * h)
            tilt = abs(left_eye_center[1] - right_eye_center[1])

        # Ask user to enter state once (0 = alert, 1 = tired)
        label = int(input("Enter label (0 alert / 1 tired): "))

        writer.writerow([datetime.now(), EAR, MAR, tilt, label])
        print(f"Logged EAR={EAR:.3f}, MAR={MAR:.3f}, tilt={tilt:.3f}")

        cv2.imshow("Face Collector", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
