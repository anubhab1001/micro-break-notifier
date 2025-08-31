# collector_predict.py
import cv2
import time
import threading
import csv
import os
import numpy as np
import mediapipe as mp
from deepface import DeepFace
from pynput import keyboard, mouse
import joblib
import pandas as pd

# ================================
# Config
# ================================
CSV_FILE = "user_activity_log.csv"
MODEL_FILE = "stress_rf_model.pkl"
LOG_INTERVAL_SEC = 10           # take a snapshot + log every 10s
IDLE_THRESHOLD_SEC = 2.0        # idle pause threshold
BLINK_EAR_THRESHOLD = 0.20
BOOTSTRAP_LABELS = True         # weak auto-labeling rules (can turn off)

# Columns used by the model (keep stable order)
EMO_COLS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
FEATURE_COLS = [
    "BlinkRate", "EyeRatio", "MouthRatio",
    "AvgTypingInterval", "MaxPause", "AvgMouseIdle",
    *[f"emo_{c}" for c in EMO_COLS]
]

# ================================
# CSV Setup (with new emotion-prob columns)
# ================================
write_header = not os.path.exists(CSV_FILE)
with open(CSV_FILE, "a", newline="") as f:
    writer = csv.writer(f)
    if write_header:
        writer.writerow([
            "Timestamp",
            "Emotion",  # dominant emotion (string for your reference)
            "BlinkRate", "EyeRatio", "MouthRatio",
            "AvgTypingInterval", "MaxPause", "AvgMouseIdle",
            # numeric emotion probabilities
            *[f"emo_{c}" for c in EMO_COLS],
            "StressLevel"  # label 0/1/2
        ])

# ================================
# Globals
# ================================
key_times = []
last_key_time = None
pause_durations = []
mouse_idle_time = []
last_mouse_move = time.time()

current_stress_level = 0        # manual label set by user (0/1/2)
rf_model = None                 # loaded RF model if exists

# ================================
# Keyboard Listener (typing + labels)
# ================================
def on_key_press(key):
    global last_key_time, key_times, pause_durations, current_stress_level
    now = time.time()

    # typing interval
    if last_key_time is not None:
        interval = now - last_key_time
        key_times.append(interval)
        if interval > IDLE_THRESHOLD_SEC:
            pause_durations.append(interval)
    last_key_time = now

    # manual label 0/1/2
    try:
        if key.char in ["0", "1", "2"]:
            current_stress_level = int(key.char)
            print(f"[LABEL] Stress level set to {current_stress_level}")
    except AttributeError:
        pass

def start_keyboard_listener():
    with keyboard.Listener(on_press=on_key_press) as listener:
        listener.join()

# ================================
# Mouse Listener (idle)
# ================================
def on_move(x, y):
    global last_mouse_move, mouse_idle_time
    now = time.time()
    idle_duration = now - last_mouse_move
    if idle_duration > IDLE_THRESHOLD_SEC:
        mouse_idle_time.append(idle_duration)
    last_mouse_move = now

def start_mouse_listener():
    with mouse.Listener(on_move=on_move) as listener:
        listener.join()

# ================================
# Mediapipe Face Mesh (EAR/MAR)
# ================================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

def get_ratios(frame):
    """Return EAR, MAR, blink_flag(0/1) for this frame."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    EAR, MAR, blink = 0.0, 0.0, 0

    if results.multi_face_landmarks:
        h, w, _ = frame.shape
        lm = results.multi_face_landmarks[0].landmark

        left_eye = [33, 160, 158, 133, 153, 144]
        right_eye = [362, 385, 387, 263, 373, 380]

        def eye_ratio(idxs):
            p1, p2, p3, p4, p5, p6 = [lm[i] for i in idxs]
            p1, p4 = np.array([p1.x*w, p1.y*h]), np.array([p4.x*w, p4.y*h])
            p2, p6 = np.array([p2.x*w, p2.y*h]), np.array([p6.x*w, p6.y*h])
            p3, p5 = np.array([p3.x*w, p3.y*h]), np.array([p5.x*w, p5.y*h])
            return (np.linalg.norm(p2-p6) + np.linalg.norm(p3-p5)) / (2.0 * np.linalg.norm(p1-p4))

        EAR = (eye_ratio(left_eye) + eye_ratio(right_eye)) / 2.0
        if EAR < BLINK_EAR_THRESHOLD:
            blink = 1

        top_lip, bottom_lip = lm[13], lm[14]
        left_lip, right_lip = lm[78], lm[308]
        num = np.linalg.norm([top_lip.x*w - bottom_lip.x*w, top_lip.y*h - bottom_lip.y*h])
        den = np.linalg.norm([left_lip.x*w - right_lip.x*w, left_lip.y*h - right_lip.y*h]) + 1e-6
        MAR = num / den

    return float(EAR), float(MAR), int(blink)

# ================================
# Helpers
# ================================
def get_typing_mouse_metrics():
    avg_typing = sum(key_times)/len(key_times) if key_times else 0.0
    max_pause = max(pause_durations) if pause_durations else 0.0
    avg_mouse_idle = sum(mouse_idle_time)/len(mouse_idle_time) if mouse_idle_time else 0.0
    return float(avg_typing), float(max_pause), float(avg_mouse_idle)

def weak_auto_label(dominant_emotion, emo_probs, avg_typing, max_pause):
    """
    Very simple bootstrapping rule to generate weak labels when user didn't press 0/1/2.
    """
    if not BOOTSTRAP_LABELS:
        return current_stress_level

    sad_angry = emo_probs.get("sad", 0) + emo_probs.get("angry", 0)
    happy_neutral = emo_probs.get("happy", 0) + emo_probs.get("neutral", 0)

    if sad_angry >= 60 and max_pause > 3.0:
        return 2
    if happy_neutral >= 60 and avg_typing <= 1.5:
        return 0
    return current_stress_level

def build_feature_row(blink_rate, ear, mar, avg_typing, max_pause, avg_mouse_idle, emo_probs):
    row = {
        "BlinkRate": blink_rate,
        "EyeRatio": round(ear, 3),
        "MouthRatio": round(mar, 3),
        "AvgTypingInterval": round(avg_typing, 2),
        "MaxPause": round(max_pause, 2),
        "AvgMouseIdle": round(avg_mouse_idle, 2),
    }
    for k in EMO_COLS:
        row[f"emo_{k}"] = float(emo_probs.get(k, 0.0))
    return row

def predict_if_model_exists(feat_row_df):
    global rf_model
    if rf_model is None:
        if os.path.exists(MODEL_FILE):
            try:
                rf_model = joblib.load(MODEL_FILE)
                print("[MODEL] Loaded stress_rf_model.pkl")
            except Exception as e:
                print("[MODEL] Load error:", e)
                rf_model = None
        else:
            return None

    # align columns
    needed = getattr(rf_model, "feature_names_in_", None)
    if needed is not None:
        for col in needed:
            if col not in feat_row_df:
                feat_row_df[col] = 0.0
        feat_row_df = feat_row_df[needed]

    try:
        pred = int(rf_model.predict(feat_row_df)[0])
        return pred
    except Exception as e:
        print("[MODEL] Predict error:", e)
        return None

# ================================
# Webcam Loop
# ================================
def start_webcam_and_log():
    cap = cv2.VideoCapture(0)
    last_check = time.time()
    blink_count = 0
    detected_emotion = "Analyzing..."

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        ear, mar, blink = get_ratios(frame)
        if blink:
            blink_count += 1

        now = time.time()
        if now - last_check >= LOG_INTERVAL_SEC:
            # DeepFace emotion + probabilities
            emo_probs = {k: 0.0 for k in EMO_COLS}
            try:
                # enforce_detection=False avoids exceptions when face not found
                result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                if isinstance(result, list):
                    result = result[0]
                detected_emotion = result.get('dominant_emotion', 'unknown')
                emotions = result.get('emotion', {})
                # normalize keys to lower
                for k, v in emotions.items():
                    k_low = k.lower()
                    if k_low in emo_probs:
                        emo_probs[k_low] = float(v)  # DeepFace gives 0-100 scores
            except Exception:
                detected_emotion = "Error"

            # typing & mouse
            avg_typing, max_pause, avg_mouse_idle = get_typing_mouse_metrics()

            # blink rate over the interval
            blink_rate = blink_count
            blink_count = 0

            # label (manual or weak auto)
            auto_label = weak_auto_label(detected_emotion, emo_probs, avg_typing, max_pause)

            # build numeric feature row (for model & csv)
            feat_row = build_feature_row(
                blink_rate, ear, mar,
                avg_typing, max_pause, avg_mouse_idle,
                emo_probs
            )

            # write to csv
            with open(CSV_FILE, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    time.strftime("%Y-%m-%d %H:%M:%S"),
                    detected_emotion,
                    feat_row["BlinkRate"], feat_row["EyeRatio"], feat_row["MouthRatio"],
                    feat_row["AvgTypingInterval"], feat_row["MaxPause"], feat_row["AvgMouseIdle"],
                    feat_row["emo_angry"], feat_row["emo_disgust"], feat_row["emo_fear"],
                    feat_row["emo_happy"], feat_row["emo_neutral"], feat_row["emo_sad"],
                    feat_row["emo_surprise"],
                    auto_label
                ])

            # live prediction if model exists
            pred = predict_if_model_exists(pd.DataFrame([feat_row]))
            if pred is not None:
                current_stress_level = pred  # display predicted level

            print(f"[LOGGED] emo={detected_emotion} pred={pred} label={auto_label} "
                  f"blink={feat_row['BlinkRate']} EAR={feat_row['EyeRatio']} MAR={feat_row['MouthRatio']} "
                  f"typing={feat_row['AvgTypingInterval']} pause={feat_row['MaxPause']} mouseIdle={feat_row['AvgMouseIdle']}")

            last_check = now

        # overlay
        cv2.putText(frame, f"Emotion: {detected_emotion} | Stress: {current_stress_level}",
                    (18, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
        cv2.imshow("Stress Monitor (Collector + Live Predict)", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

# ================================
# Run
# ================================
if __name__ == "__main__":
    threading.Thread(target=start_keyboard_listener, daemon=True).start()
    threading.Thread(target=start_mouse_listener, daemon=True).start()
    start_webcam_and_log()
