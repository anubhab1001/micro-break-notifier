import cv2
import time
import logging
from deepface import DeepFace

class EmotionDetector:
    def __init__(self, interval=10):
        self.interval = interval
        self.last_check_time = time.time()
        self.detected_emotion = "Analyzing..."
        self.cap = cv2.VideoCapture(0)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

    def analyze_emotion(self, frame):
        try:
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            self.detected_emotion = result[0]['dominant_emotion']
            logging.info(f"Detected Emotion: {self.detected_emotion}")
        except Exception as e:
            logging.error(f"Error analyzing emotion: {e}")
            self.detected_emotion = "Error"

    def run(self):
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    logging.error("Failed to capture frame from webcam.")
                    break

                current_time = time.time()
                if current_time - self.last_check_time >= self.interval:
                    self.analyze_emotion(frame)
                    self.last_check_time = current_time

                # Display the detected emotion on the frame
                cv2.putText(frame, f"Emotion: {self.detected_emotion}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow("Emotion Detector", frame)

                # Stop the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logging.info("Exiting Emotion Detector.")
                    break
        finally:
            self.cleanup()

    def cleanup(self):
        logging.info("Releasing resources.")
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = EmotionDetector(interval=10)
    detector.run()