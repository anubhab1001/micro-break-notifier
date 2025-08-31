import cv2
import time
import threading
import csv
import os
from deepface import DeepFace
from pynput import keyboard, mouse
import numpy as np
import mediapipe as mp

# ================================
# CSV Setup
# ================================
csv_file = "user_activity_log.csv"
write_header = not os.path.exists(csv_file)

with open(csv_file, "a", newline="") as f:
    writer = csv.writer(f)
    if write_header:
        writer.writerow([
            "Timestamp",
            "Emotion", "BlinkRate", "EyeRatio", "MouthRatio",
            "AvgTypingInterval", "MaxPause", "AvgMouseIdle"
        ])

# ================================
# Keyboard Tracking
# ================================
key_times = []
last_key_time = None
pause_durations = []

def on_key_press(key):
    global last_key_time, key_times, pause_durations
    now = time.time()
    if last_key_time is not None:
        interval = now - last_key_time
        key_times.append(interval)

        # Detect pauses > 2 sec
        if interval > 2.0:
            pause_durations.append(interval)

    last_key_time = now

def start_keyboard_listener():
    with keyboard.Listener(on_press=on_key_press) as listener:
        listener.join()

# ================================
# Mouse Tracking
# ================================
mouse_idle_time = []
last_mouse_move = time.time()

def on_move(x, y):
    global last_mouse_move, mouse_idle_time
    now = time.time()
    idle_duration = now - last_mouse_move
    if idle_duration > 2.0:  # More than 2 sec idle
        mouse_idle_time.append(idle_duration)
    last_mouse_move = now

def start_mouse_listener():
    with mouse.Listener(on_move=on_move) as listener:
        listener.join()

# ================================
# Blink + Eye/Mouth Ratio (Mediapipe)
# ================================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

def get_ratios(frame):
    """Returns blink (EAR), mouth aspect ratio (MAR)"""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    EAR, MAR = 0, 0
    blink = 0

    if results.multi_face_landmarks:
        h, w, _ = frame.shape
        landmarks = results.multi_face_landmarks[0].landmark

        # Eye indices (left eye example)
        left_eye = [33, 160, 158, 133, 153, 144]
        right_eye = [362, 385, 387, 263, 373, 380]

        def eye_ratio(indices):
            p1, p2, p3, p4, p5, p6 = [landmarks[i] for i in indices]
            p1, p4 = np.array([p1.x * w, p1.y * h]), np.array([p4.x * w, p4.y * h])
            p2, p6 = np.array([p2.x * w, p2.y * h]), np.array([p6.x * w, p6.y * h])
            p3, p5 = np.array([p3.x * w, p3.y * h]), np.array([p5.x * w, p5.y * h])
            return (np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)) / (2.0 * np.linalg.norm(p1 - p4))

        EAR = (eye_ratio(left_eye) + eye_ratio(right_eye)) / 2.0
        if EAR < 0.20:  # Blink threshold
            blink = 1

        # Mouth indices
        top_lip = landmarks[13]
        bottom_lip = landmarks[14]
        left_lip = landmarks[78]
        right_lip = landmarks[308]

        MAR = (np.linalg.norm([top_lip.x*w - bottom_lip.x*w, top_lip.y*h - bottom_lip.y*h])) / (
                np.linalg.norm([left_lip.x*w - right_lip.x*w, left_lip.y*h - right_lip.y*h]) + 1e-6)

    return EAR, MAR, blink

# ================================
# Main Webcam + Data Logging Loop
# ================================
def start_webcam_and_log():
    cap = cv2.VideoCapture(0)
    last_check = time.time()
    interval = 10  # log every 10 sec
    blink_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        EAR, MAR, blink = get_ratios(frame)
        if blink:
            blink_count += 1

        detected_emotion = "Analyzing..."
        if time.time() - last_check >= interval:
            # DeepFace Emotion Detection
            try:
                result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                detected_emotion = result[0]['dominant_emotion']
            except Exception:
                detected_emotion = "Error"

            # Typing Metrics
            avg_typing = sum(key_times)/len(key_times) if key_times else 0
            max_pause = max(pause_durations) if pause_durations else 0

            # Mouse Metrics
            avg_mouse_idle = sum(mouse_idle_time)/len(mouse_idle_time) if mouse_idle_time else 0

            # Blink Rate per 10 sec
            blink_rate = blink_count
            blink_count = 0  # reset

            # Save to CSV
            with open(csv_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    time.strftime("%Y-%m-%d %H:%M:%S"),
                    detected_emotion, blink_rate, round(EAR, 3), round(MAR, 3),
                    round(avg_typing, 2), round(max_pause, 2), round(avg_mouse_idle, 2)
                ])

            print(f"[LOGGED] Emotion={detected_emotion}, Blinks={blink_rate}, EAR={EAR:.2f}, MAR={MAR:.2f}, "
                  f"Typing={avg_typing:.2f}, Pause={max_pause:.2f}, MouseIdle={avg_mouse_idle:.2f}")

            last_check = time.time()

        # Display
        cv2.putText(frame, f"Emotion: {detected_emotion}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Integrated Monitor", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

# ================================
# Run Everything
# ================================
if __name__ == "__main__":
    threading.Thread(target=start_keyboard_listener, daemon=True).start()
    threading.Thread(target=start_mouse_listener, daemon=True).start()
    start_webcam_and_log()
