from flask import Flask, jsonify, request
from flask_cors import CORS
import threading

from stress_monitor_2 import StressMonitoringSystem  # import your class (rename file if needed)

app = Flask(__name__)
CORS(app)

monitor = StressMonitoringSystem(emotion_interval=10, feature_analysis_interval=15)
monitor_thread = None
is_running = False

@app.route("/start", methods=["POST"])
def start_monitor():
    global monitor_thread, is_running
    if not is_running:
        monitor_thread = threading.Thread(target=monitor.run, daemon=True)
        monitor_thread.start()
        is_running = True
        return jsonify({"status": "started"}), 200
    return jsonify({"status": "already running"}), 200

@app.route("/status", methods=["GET"])
def get_status():
    return jsonify({
        "emotion": monitor.detected_emotion,
        "stress_level": monitor.stress_level,
        "confidence": monitor.emotion_confidence
    })

@app.route("/stop", methods=["POST"])
def stop_monitor():
    global is_running
    if is_running:
        monitor.cleanup()
        is_running = False
        return jsonify({"status": "stopped"}), 200
    return jsonify({"status": "not running"}), 200

if __name__ == "__main__":
    app.run(debug=True)
