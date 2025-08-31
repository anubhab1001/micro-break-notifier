
import cv2
import time
import logging
import threading
import json
import os
from datetime import datetime
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import tkinter as tk
from tkinter import messagebox, ttk
from deepface import DeepFace
from pynput import keyboard, mouse
import warnings
warnings.filterwarnings('ignore')

class StressMonitoringSystem:
    def __init__(self, emotion_interval=10, feature_analysis_interval=15):
        # Initialize parameters
        self.emotion_interval = emotion_interval
        self.feature_analysis_interval = feature_analysis_interval
        
        # Webcam and emotion detection
        self.cap = cv2.VideoCapture(0)
        self.last_emotion_check = time.time()
        self.detected_emotion = "neutral"
        self.emotion_confidence = 0.5
        
        # Keyboard tracking
        self.key_times = []
        self.last_key_time = None
        self.pause_durations = []
        self.typing_speed = 0
        
        # Mouse tracking
        self.last_mouse_move = time.time()
        self.mouse_idle_times = []
        self.mouse_movements = []
        
        # Feature extraction and ML
        self.features = {}
        self.stress_level = 0
        self.model = None
        self.scaler = StandardScaler()
        self.setup_ml_model()
        
        # Data logging
        self.log_file = "stress_monitoring_log.json"
        self.session_data = []
        
        # Notification system
        self.last_notification_time = 0
        self.notification_cooldown = 300  # 5 minutes
        
        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
        
    def setup_ml_model(self):
        """Initialize or load the ML model for stress prediction"""
        model_file = "stress_model.joblib"
        scaler_file = "stress_scaler.joblib"
        
        if os.path.exists(model_file) and os.path.exists(scaler_file):
            # Load existing model
            self.model = joblib.load(model_file)
            self.scaler = joblib.load(scaler_file)
            logging.info("Loaded existing ML model")
        else:
            # Create a simple rule-based model initially
            self.model = self.create_initial_model()
            logging.info("Created initial rule-based model")
    
    def create_initial_model(self):
        """Create initial model with synthetic data"""
        # Generate synthetic training data
        np.random.seed(42)
        n_samples = 1000
        
        # Features: [emotion_stress_score, avg_typing_interval, max_pause, mouse_idle_avg, blink_rate]
        X = np.random.rand(n_samples, 5)
        
        # Create labels based on simple rules
        y = np.zeros(n_samples)
        for i in range(n_samples):
            emotion_stress = X[i, 0]
            typing_interval = X[i, 1]
            max_pause = X[i, 2]
            mouse_idle = X[i, 3]
            
            stress_score = (emotion_stress * 0.4 + typing_interval * 0.3 + 
                          max_pause * 0.2 + mouse_idle * 0.1)
            
            if stress_score > 0.7:
                y[i] = 2  # High stress
            elif stress_score > 0.4:
                y[i] = 1  # Mild stress
            else:
                y[i] = 0  # Normal
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        X_scaled = self.scaler.fit_transform(X)
        model.fit(X_scaled, y)
        
        # Save model
        joblib.dump(model, "stress_model.joblib")
        joblib.dump(self.scaler, "stress_scaler.joblib")
        
        return model
    
    def analyze_emotion(self, frame):
        """Analyze emotion from webcam frame"""
        try:
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            self.detected_emotion = result[0]['dominant_emotion']
            emotions = result[0]['emotion']
            self.emotion_confidence = emotions[self.detected_emotion] / 100.0
            logging.info(f"Detected Emotion: {self.detected_emotion} ({self.emotion_confidence:.2f})")
        except Exception as e:
            logging.error(f"Error analyzing emotion: {e}")
            self.detected_emotion = "neutral"
            self.emotion_confidence = 0.5
    
    def on_key_press(self, key):
        """Handle keyboard press events"""
        now = time.time()
        if self.last_key_time is not None:
            interval = now - self.last_key_time
            self.key_times.append(interval)
            
            # Detect pauses longer than 2 seconds
            if interval > 2.0:
                self.pause_durations.append(interval)
                
            # Keep only recent data (last 100 keystrokes)
            if len(self.key_times) > 100:
                self.key_times = self.key_times[-100:]
            if len(self.pause_durations) > 20:
                self.pause_durations = self.pause_durations[-20:]
        
        self.last_key_time = now
    
    def on_mouse_move(self, x, y):
        """Handle mouse movement events"""
        now = time.time()
        idle_duration = now - self.last_mouse_move
        
        if idle_duration > 2.0:  # More than 2 sec idle
            self.mouse_idle_times.append(idle_duration)
            
            # Keep only recent data
            if len(self.mouse_idle_times) > 50:
                self.mouse_idle_times = self.mouse_idle_times[-50:]
        
        self.mouse_movements.append((x, y, now))
        if len(self.mouse_movements) > 100:
            self.mouse_movements = self.mouse_movements[-100:]
            
        self.last_mouse_move = now
    
    def extract_features(self):
        """Extract features for stress prediction"""
        features = {}
        
        # Emotion-based features
        emotion_stress_mapping = {
            'angry': 0.9, 'disgust': 0.7, 'fear': 0.8, 'sad': 0.8,
            'surprise': 0.3, 'happy': 0.1, 'neutral': 0.4
        }
        features['emotion_stress_score'] = emotion_stress_mapping.get(self.detected_emotion, 0.4)
        features['emotion_confidence'] = self.emotion_confidence
        
        # Typing behavior features
        if len(self.key_times) > 0:
            features['avg_typing_interval'] = np.mean(self.key_times)
            features['typing_variance'] = np.var(self.key_times)
        else:
            features['avg_typing_interval'] = 0.5
            features['typing_variance'] = 0.1
            
        if len(self.pause_durations) > 0:
            features['max_pause_duration'] = max(self.pause_durations)
            features['avg_pause_duration'] = np.mean(self.pause_durations)
        else:
            features['max_pause_duration'] = 0
            features['avg_pause_duration'] = 0
        
        # Mouse behavior features
        if len(self.mouse_idle_times) > 0:
            features['avg_mouse_idle'] = np.mean(self.mouse_idle_times)
            features['max_mouse_idle'] = max(self.mouse_idle_times)
        else:
            features['avg_mouse_idle'] = 0
            features['max_mouse_idle'] = 0
        
        # Calculate mouse movement speed
        if len(self.mouse_movements) >= 2:
            speeds = []
            for i in range(1, len(self.mouse_movements)):
                x1, y1, t1 = self.mouse_movements[i-1]
                x2, y2, t2 = self.mouse_movements[i]
                distance = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                time_diff = t2 - t1
                if time_diff > 0:
                    speeds.append(distance / time_diff)
            
            if speeds:
                features['avg_mouse_speed'] = np.mean(speeds)
                features['mouse_speed_variance'] = np.var(speeds)
            else:
                features['avg_mouse_speed'] = 0
                features['mouse_speed_variance'] = 0
        else:
            features['avg_mouse_speed'] = 0
            features['mouse_speed_variance'] = 0
        
        # Simple blink rate estimation (placeholder)
        features['estimated_blink_rate'] = 0.3  # This would need proper eye tracking
        
        self.features = features
        return features
    
    def predict_stress_level(self, features):
        """Predict stress level using ML model"""
        try:
            # Prepare feature vector for model
            feature_vector = np.array([
                features.get('emotion_stress_score', 0.4),
                features.get('avg_typing_interval', 0.5),
                features.get('max_pause_duration', 0),
                features.get('avg_mouse_idle', 0),
                features.get('estimated_blink_rate', 0.3)
            ]).reshape(1, -1)
            
            # Scale features
            feature_vector_scaled = self.scaler.transform(feature_vector)
            
            # Predict
            prediction = self.model.predict(feature_vector_scaled)[0]
            prediction_proba = self.model.predict_proba(feature_vector_scaled)[0]
            
            confidence = max(prediction_proba)
            
            logging.info(f"Stress Level Prediction: {prediction} (confidence: {confidence:.2f})")
            
            return int(prediction), confidence
            
        except Exception as e:
            logging.error(f"Error predicting stress level: {e}")
            return 0, 0.5
    
    def show_break_notification(self, stress_level):
        """Show desktop notification for break suggestion"""
        current_time = time.time()
        if current_time - self.last_notification_time < self.notification_cooldown:
            return  # Too soon for another notification
        
        def show_popup():
            root = tk.Tk()
            root.withdraw()  # Hide main window
            
            messages = {
                1: {
                    'title': 'Mild Stress Detected',
                    'message': 'Take a moment to breathe deeply.\n\nSuggestions:\n• Look away from screen for 20 seconds\n• Do some neck stretches\n• Drink some water'
                },
                2: {
                    'title': 'High Stress Detected', 
                    'message': 'Time for a proper break!\n\nSuggestions:\n• Take a 5-minute walk\n• Practice deep breathing\n• Step away from your workspace\n• Do some light stretching'
                }
            }
            
            msg = messages.get(stress_level, messages[1])
            
            # Custom dialog
            popup = tk.Toplevel(root)
            popup.title(msg['title'])
            popup.geometry("400x300")
            popup.configure(bg='#f0f0f0')
            
            # Message
            label = tk.Label(popup, text=msg['message'], bg='#f0f0f0', 
                           font=('Arial', 11), justify='left', wraplength=350)
            label.pack(pady=20, padx=20)
            
            # Buttons
            button_frame = tk.Frame(popup, bg='#f0f0f0')
            button_frame.pack(pady=10)
            
            def took_break():
                self.log_user_feedback('took_break', stress_level)
                popup.destroy()
                root.destroy()
            
            def snooze():
                self.log_user_feedback('snoozed', stress_level)
                popup.destroy()
                root.destroy()
            
            tk.Button(button_frame, text="I'll take a break", 
                     command=took_break, bg='#4CAF50', fg='white',
                     font=('Arial', 10), padx=20).pack(side='left', padx=10)
            
            tk.Button(button_frame, text="Remind me later", 
                     command=snooze, bg='#ff9800', fg='white',
                     font=('Arial', 10), padx=20).pack(side='left', padx=10)
            
            popup.transient(root)
            popup.grab_set()
            popup.lift()
            popup.focus_force()
            
            root.mainloop()
        
        # Show popup in separate thread
        threading.Thread(target=show_popup, daemon=True).start()
        self.last_notification_time = current_time
    
    def log_user_feedback(self, action, stress_level):
        """Log user feedback for model improvement"""
        feedback_entry = {
            'timestamp': datetime.now().isoformat(),
            'stress_level': stress_level,
            'user_action': action,
            'features': self.features.copy()
        }
        
        self.session_data.append(feedback_entry)
        logging.info(f"User feedback logged: {action} for stress level {stress_level}")
    
    def log_session_data(self):
        """Save session data to file"""
        try:
            # Load existing data
            existing_data = []
            if os.path.exists(self.log_file):
                with open(self.log_file, 'r') as f:
                    existing_data = json.load(f)
            
            # Append new data
            existing_data.extend(self.session_data)
            
            # Save back to file
            with open(self.log_file, 'w') as f:
                json.dump(existing_data, f, indent=2)
            
            logging.info(f"Session data saved to {self.log_file}")
            self.session_data = []  # Clear session data
            
        except Exception as e:
            logging.error(f"Error saving session data: {e}")
    
    def start_input_listeners(self):
        """Start keyboard and mouse listeners"""
        def keyboard_listener():
            with keyboard.Listener(on_press=self.on_key_press) as listener:
                listener.join()
        
        def mouse_listener():
            with mouse.Listener(on_move=self.on_mouse_move) as listener:
                listener.join()
        
        threading.Thread(target=keyboard_listener, daemon=True).start()
        threading.Thread(target=mouse_listener, daemon=True).start()
        logging.info("Input listeners started")
    
    def run_monitoring_cycle(self):
        """Main monitoring cycle"""
        last_feature_analysis = time.time()
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    logging.error("Failed to capture frame from webcam")
                    break
                
                current_time = time.time()
                
                # Emotion detection every 10 seconds
                if current_time - self.last_emotion_check >= self.emotion_interval:
                    self.analyze_emotion(frame)
                    self.last_emotion_check = current_time
                
                # Feature analysis and stress prediction every 15 seconds
                if current_time - last_feature_analysis >= self.feature_analysis_interval:
                    features = self.extract_features()
                    stress_level, confidence = self.predict_stress_level(features)
                    self.stress_level = stress_level
                    
                    # Log current state
                    log_entry = {
                        'timestamp': datetime.now().isoformat(),
                        'emotion': self.detected_emotion,
                        'stress_level': stress_level,
                        'confidence': confidence,
                        'features': features
                    }
                    self.session_data.append(log_entry)
                    
                    # Show notification if needed
                    if stress_level >= 1:
                        self.show_break_notification(stress_level)
                    
                    last_feature_analysis = current_time
                    
                    # Print status
                    print(f"\n--- Stress Monitor Status ---")
                    print(f"Time: {datetime.now().strftime('%H:%M:%S')}")
                    print(f"Emotion: {self.detected_emotion} ({self.emotion_confidence:.2f})")
                    print(f"Stress Level: {stress_level} ({confidence:.2f} confidence)")
                    print(f"Features: {features}")
                    print("----------------------------")
                
                # Display frame with overlay
                overlay_text = f"Emotion: {self.detected_emotion} | Stress: {self.stress_level}"
                cv2.putText(frame, overlay_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                
                cv2.imshow("Stress Monitoring System", frame)
                
                # Save session data periodically (every 5 minutes)
                if len(self.session_data) > 20:
                    self.log_session_data()
                
                # Exit on 'q' key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logging.info("Exiting stress monitoring system")
                    break
                    
        except KeyboardInterrupt:
            logging.info("System interrupted by user")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        logging.info("Cleaning up resources...")
        
        # Save final session data
        if self.session_data:
            self.log_session_data()
        
        # Release webcam
        if self.cap.isOpened():
            self.cap.release()
        
        cv2.destroyAllWindows()
        logging.info("Cleanup completed")
    
    def run(self):
        """Start the complete stress monitoring system"""
        logging.info("Starting Stress Monitoring System...")
        
        # Start input listeners
        self.start_input_listeners()
        
        # Wait a moment for listeners to initialize
        time.sleep(2)
        
        # Start main monitoring cycle
        self.run_monitoring_cycle()

if __name__ == "__main__":
    # Create and run the stress monitoring system
    monitor = StressMonitoringSystem(emotion_interval=10, feature_analysis_interval=15)
    monitor.run()