
#!/usr/bin/env python3
"""
HACKATHON SUBMISSION: Complete Stress Monitoring System
Integrates your emotion detection + behavior tracking + stress prediction
"""

import cv2
import time
import logging
import threading
import numpy as np
import json
from datetime import datetime
from collections import deque
import tkinter as tk
from tkinter import messagebox

# Your imports
from deepface import DeepFace
from pynput import keyboard, mouse

class HackathonStressMonitor:
    def __init__(self):
        print("🚀 HACKATHON STRESS MONITOR STARTING...")
        
        # ===== STEP 1: INITIALIZE WEBCAM & TRACKING =====
        self.cap = cv2.VideoCapture(0)
        self.detected_emotion = "neutral"
        self.emotion_confidence = 0.0
        self.last_emotion_check = time.time()
        
        # ===== KEYBOARD & MOUSE TRACKING (Your Code) =====
        self.key_times = []
        self.last_key_time = None
        self.pause_durations = []
        self.mouse_idle_time = []
        self.last_mouse_move = time.time()
        
        # ===== BASELINE ESTABLISHMENT =====
        self.baseline_typing_speed = 1.0
        self.baseline_established = False
        self.start_time = time.time()
        self.typing_errors = 0  # Simulated for demo
        
        # ===== STRESS DETECTION =====
        self.stress_conditions = 0
        self.overload_event = False
        self.current_stress_level = 0
        
        # ===== NOTIFICATION SYSTEM =====
        self.last_notification = 0
        self.notification_cooldown = 30  # 30 seconds for hackathon demo
        
        # ===== DATA LOGGING =====
        self.session_data = []
        
        # ===== BREAK SUGGESTIONS =====
        self.break_suggestions = [
            "🫁 Take 5 deep breaths",
            "👀 Look at something 20 feet away for 20 seconds",
            "💪 Roll your shoulders and stretch your neck",
            "💧 Drink a glass of water",
            "🚶 Stand up and walk for 1 minute",
            "😌 Close your eyes and relax for 30 seconds"
        ]
        
        # Start behavior tracking
        self.start_behavior_tracking()
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
        
        print("✅ System initialized! Ready for demo!")

    # ========================================
    # STEP 2: WEBCAM EMOTION DETECTION (Every 10 sec)
    # ========================================
    def analyze_emotion(self, frame):
        """Your emotion detection code enhanced"""
        try:
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            self.detected_emotion = result[0]['dominant_emotion']
            
            # Get confidence score
            emotions = result[0]['emotion']
            self.emotion_confidence = emotions[self.detected_emotion] / 100.0
            
            logging.info(f"Detected Emotion: {self.detected_emotion} (conf: {self.emotion_confidence:.2f})")
            return self.detected_emotion, self.emotion_confidence
            
        except Exception as e:
            logging.error(f"Error analyzing emotion: {e}")
            self.detected_emotion = "neutral"
            return "neutral", 0.0

    # ========================================
    # STEP 3: BEHAVIOR TRACKING (Real-time)
    # ========================================
    def on_key_press(self, key):
        """Enhanced keyboard tracking from your code"""
        now = time.time()
        
        if self.last_key_time is not None:
            interval = now - self.last_key_time
            self.key_times.append(interval)
            
            # Detect pauses longer than 2 sec (your logic)
            if interval > 2.0:
                self.pause_durations.append(interval)
                
        self.last_key_time = now
        
        # Establish baseline after 60 seconds
        if not self.baseline_established and (now - self.start_time > 60):
            if len(self.key_times) > 5:
                self.baseline_typing_speed = sum(self.key_times) / len(self.key_times)
                self.baseline_established = True
                print(f"📊 Baseline established: {self.baseline_typing_speed:.2f}s per key")

    def on_mouse_move(self, x, y):
        """Enhanced mouse tracking from your code"""
        now = time.time()
        idle_duration = now - self.last_mouse_move
        
        if idle_duration > 2.0:  # More than 2 sec idle
            self.mouse_idle_time.append(idle_duration)
            
        self.last_mouse_move = now

    def start_behavior_tracking(self):
        """Start your keyboard and mouse listeners"""
        def keyboard_thread():
            with keyboard.Listener(on_press=self.on_key_press) as listener:
                listener.join()
                
        def mouse_thread():
            with mouse.Listener(on_move=self.on_mouse_move) as listener:
                listener.join()
        
        threading.Thread(target=keyboard_thread, daemon=True).start()
        threading.Thread(target=mouse_thread, daemon=True).start()

    # ========================================
    # STEP 4: FEATURE EXTRACTION
    # ========================================
    def extract_features(self):
        """Extract all features as per your workflow"""
        
        # Calculate metrics (your logic)
        avg_typing_interval = sum(self.key_times) / len(self.key_times) if self.key_times else 0.0
        max_pause = max(self.pause_durations) if self.pause_durations else 0.0
        avg_mouse_idle = sum(self.mouse_idle_time) / len(self.mouse_idle_time) if self.mouse_idle_time else 0.0
        
        # Calculate typing speed change from baseline
        typing_speed_change = 0.0
        if self.baseline_established and avg_typing_interval > 0:
            typing_speed_change = ((avg_typing_interval - self.baseline_typing_speed) / self.baseline_typing_speed) * 100
        
        # Simulate error rate for demo (in real app, track actual errors)
        error_rate_multiplier = 1.0 + (np.random.random() * 0.5)  # 1.0 to 1.5x
        
        return {
            'emotion': self.detected_emotion,
            'emotion_confidence': self.emotion_confidence,
            'avg_typing_interval': avg_typing_interval,
            'typing_speed_change': typing_speed_change,
            'error_rate_multiplier': error_rate_multiplier,
            'max_pause_duration': max_pause,
            'avg_mouse_idle': avg_mouse_idle,
            'timestamp': datetime.now().isoformat()
        }

    # ========================================
    # STEP 5: STRESS LEVEL PREDICTION
    # ========================================
    def predict_stress_level(self, features):
        """Predict stress using your specified conditions"""
        
        self.stress_conditions = 0
        conditions_met = []
        
        # CONDITION 1: Typing speed ↓ 40% below baseline → stress
        if features['typing_speed_change'] > 40:
            self.stress_conditions += 1
            conditions_met.append(f"Typing 40% slower ({features['typing_speed_change']:+.1f}%)")
        
        # CONDITION 2: Error rate ↑ 2× normal → stress  
        if features['error_rate_multiplier'] >= 2.0:
            self.stress_conditions += 1
            conditions_met.append(f"Error rate doubled ({features['error_rate_multiplier']:.1f}x)")
        
        # CONDITION 3: Idle gaps > 10s repeatedly → fatigue/distraction
        long_idles = [p for p in self.pause_durations if p > 10.0]
        if len(long_idles) >= 2:  # 2+ long pauses recently
            self.stress_conditions += 1
            conditions_met.append(f"Repeated long pauses ({len(long_idles)} times)")
        
        # CONDITION 4: High stress emotions
        if features['emotion'] in ['angry', 'sad', 'fear'] and features['emotion_confidence'] > 0.6:
            self.stress_conditions += 1
            conditions_met.append(f"High stress emotion: {features['emotion']}")
        
        # If ≥2 conditions true → flag overload event
        self.overload_event = self.stress_conditions >= 2
        
        # Determine stress level
        if self.overload_event or self.stress_conditions >= 3:
            stress_level = 2  # High stress
        elif self.stress_conditions >= 1:
            stress_level = 1  # Mild stress  
        else:
            stress_level = 0  # Normal
        
        # Print conditions for demo
        if conditions_met:
            print(f"⚠️ Stress conditions: {', '.join(conditions_met)}")
        
        if self.overload_event:
            print("🚨 OVERLOAD EVENT DETECTED!")
        
        return stress_level

    # ========================================
    # STEP 6: MICRO-BREAK NOTIFICATIONS
    # ========================================
    def should_show_notification(self, stress_level):
        """Check if we should show notification"""
        current_time = time.time()
        return (stress_level >= 1 and 
                current_time - self.last_notification > self.notification_cooldown)

    def show_break_notification(self, stress_level):
        """Show desktop popup with break suggestion"""
        try:
            root = tk.Tk()
            root.withdraw()  # Hide main window
            
            stress_messages = {
                1: "😐 Mild stress detected - Time for a quick break?",
                2: "😰 High stress detected - You need a micro-break!"
            }
            
            message = stress_messages.get(stress_level, "Take a break")
            suggestion = np.random.choice(self.break_suggestions)
            
            result = messagebox.askyesno(
                "🧠 Stress Monitor - Micro-Break",
                f"{message}\n\n💡 Suggestion: {suggestion}\n\n⏰ This will only take 1-2 minutes.\n\nTake a micro-break now?"
            )
            
            root.destroy()
            self.last_notification = time.time()
            
            user_action = "took_break" if result else "snoozed"
            print(f"📱 User {user_action} break suggestion")
            
            return user_action
            
        except Exception as e:
            print(f"Notification error: {e}")
            return "error"

    # ========================================
    # STEP 7: DATA LOGGING & USER FEEDBACK
    # ========================================
    def log_session_data(self, features, stress_level, user_feedback=None):
        """Log stress predictions and user reactions"""
        log_entry = {
            **features,
            'stress_level': stress_level,
            'stress_conditions_met': self.stress_conditions,
            'overload_event': self.overload_event,
            'user_feedback': user_feedback,
            'session_duration_sec': time.time() - self.start_time
        }
        
        self.session_data.append(log_entry)
        
        # Save to file every 5 entries
        if len(self.session_data) % 5 == 0:
            with open(f'stress_log_{datetime.now().strftime("%Y%m%d_%H%M")}.json', 'w') as f:
                json.dump(self.session_data, f, indent=2)

    # ========================================
    # STEP 8: MAIN MONITORING LOOP (Repeat every few seconds)
    # ========================================
    def run_monitoring(self):
        """Main loop - continues monitoring passively in background"""
        print("🧠 HACKATHON DEMO: Stress Monitor Running!")
        print("📹 Webcam active | ⌨️ Keyboard tracked | 🖱️ Mouse tracked")
        print("🎯 Demo mode: Quick notifications for testing")
        print("Press 'q' on webcam window to quit\n")
        
        last_analysis_time = 0
        analysis_interval = 5  # Analyze every 5 seconds for demo
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("❌ Webcam error")
                    break
                
                current_time = time.time()
                
                # CORE MONITORING LOGIC (Every 5 seconds)
                if current_time - last_analysis_time >= analysis_interval:
                    
                    # Step 2: Webcam analysis (every 10 sec)
                    if current_time - self.last_emotion_check >= 10:
                        self.analyze_emotion(frame)
                        self.last_emotion_check = current_time
                    
                    # Step 4: Extract features
                    features = self.extract_features()
                    
                    # Step 5: Predict stress level  
                    stress_level = self.predict_stress_level(features)
                    self.current_stress_level = stress_level
                    
                    # Step 6: Show notification if needed
                    user_feedback = None
                    if self.should_show_notification(stress_level):
                        user_feedback = self.show_break_notification(stress_level)
                    
                    # Step 7: Log data
                    self.log_session_data(features, stress_level, user_feedback)
                    
                    # Print status for demo
                    self.print_demo_status(features, stress_level)
                    
                    last_analysis_time = current_time
                
                # Display webcam with overlay
                self.display_webcam(frame)
                
                # Exit condition
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            print("\n🛑 Stopping demo...")
        finally:
            self.cleanup()

    def print_demo_status(self, features, stress_level):
        """Print status for hackathon demo"""
        stress_icons = {0: "😊 NORMAL", 1: "😐 MILD STRESS", 2: "😰 HIGH STRESS"}
        status = stress_icons.get(stress_level, "❓")
        
        print(f"\n{status}")
        print(f"😀 Emotion: {features['emotion']} ({features['emotion_confidence']:.2f})")
        print(f"⌨️ Typing: {features['avg_typing_interval']:.2f}s avg, {features['typing_speed_change']:+.1f}% vs baseline")
        print(f"⏸️ Max pause: {features['max_pause_duration']:.1f}s")
        print(f"🖱️ Mouse idle: {features['avg_mouse_idle']:.1f}s avg")
        print(f"⚠️ Stress conditions: {self.stress_conditions}/4")
        if self.overload_event:
            print("🚨 OVERLOAD EVENT!")
        print("-" * 60)

    def display_webcam(self, frame):
        """Display webcam with stress overlay"""
        # Color based on stress level
        colors = {0: (0, 255, 0), 1: (0, 255, 255), 2: (0, 0, 255)}
        color = colors.get(self.current_stress_level, (255, 255, 255))
        
        # Add overlays
        cv2.putText(frame, f"Emotion: {self.detected_emotion}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"Stress Level: {self.current_stress_level}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"Conditions: {self.stress_conditions}/4", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        if self.overload_event:
            cv2.putText(frame, "OVERLOAD EVENT!", (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.putText(frame, "Press 'q' to quit | HACKATHON DEMO", (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow("🧠 Hackathon Stress Monitor", frame)

    def cleanup(self):
        """Clean up resources"""
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        
        # Save final log
        if self.session_data:
            with open(f'final_stress_log_{datetime.now().strftime("%Y%m%d_%H%M")}.json', 'w') as f:
                json.dump(self.session_data, f, indent=2)
        
        print("\n✅ Hackathon demo completed!")
        print(f"📊 Collected {len(self.session_data)} data points")
        print("📁 Data saved to JSON file")

# ========================================
# HACKATHON MAIN EXECUTION
# ========================================
if __name__ == "__main__":
    print("🎯 HACKATHON SUBMISSION: Real-time Stress Monitoring System")
    print("=" * 60)
    print("📋 Features:")
    print("   ✅ Webcam emotion detection (every 10s)")
    print("   ✅ Real-time keyboard/mouse behavior tracking") 
    print("   ✅ Baseline establishment & comparison")
    print("   ✅ 4 stress condition detection")
    print("   ✅ Overload event flagging")
    print("   ✅ Smart micro-break notifications")
    print("   ✅ Complete data logging")
    print("   ✅ User feedback tracking")
    print("=" * 60)
    
    monitor = HackathonStressMonitor()
    
    try:
        monitor.run_monitoring()
    except Exception as e:
        print(f"❌ Error: {e}")
        monitor.cleanup()