import cv2
import logging
from ultralytics import YOLO
import numpy as np
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PersonDetector:
    def __init__(self, model_path='yolov8n.pt'):
        try:
            self.model = YOLO(model_path)
            logging.info(f"YOLOv8 model '{model_path}' loaded/initialized successfully.")
        except Exception as e:
            logging.error(f"Error loading YOLOv8 model '{model_path}': {e}", exc_info=True)
            raise

    def detect_persons(self, frame, confidence_threshold=0.5):
        if frame is None:
            logging.warning("Input frame is None. Cannot perform detection.")
            return []

        persons_detected = []
        try:
            results = self.model(frame, classes=[0], verbose=False)
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    confidence = float(box.conf.item())
                    if confidence >= confidence_threshold:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        persons_detected.append({
                            'bbox': (x1, y1, x2, y2),
                            'confidence': confidence,
                            'class_id': int(box.cls.item())
                        })
            if persons_detected:
                logging.debug(f"Detected {len(persons_detected)} persons in the frame.")
        except Exception as e:
            logging.error(f"Error during person detection: {e}", exc_info=True)
        return persons_detected

if __name__ == '__main__':
    logging.info("Starting PersonDetector example usage...")
    detector = None
    try:
        logging.info("Initializing PersonDetector with 'yolov8n.pt'. This may download the model.")
        detector = PersonDetector('yolov8n.pt')
        logging.info("PersonDetector initialized successfully.")
    except Exception as e:
        logging.error(f"Failed to initialize PersonDetector: {e}. Example cannot continue robustly.")

    if detector:
        dummy_frame_height, dummy_frame_width = 480, 640
        dummy_frame = np.full((dummy_frame_height, dummy_frame_width, 3), (128, 128, 128), dtype=np.uint8)
        cv2.putText(dummy_frame, "Test Frame (No People Expected)", (50, dummy_frame_height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        logging.info(f"Performing detection on a dummy frame of shape: {dummy_frame.shape}")
        detected_persons_dummy = detector.detect_persons(dummy_frame, confidence_threshold=0.25)
        if detected_persons_dummy:
            logging.info(f"Detected {len(detected_persons_dummy)} persons in the dummy frame:")
            for i, person in enumerate(detected_persons_dummy):
                logging.info(f"  Person {i+1}: BBox={person['bbox']}, Confidence={person['confidence']:.2f}, ClassID={person['class_id']}")
        else:
            logging.info("No persons detected in the dummy frame, as expected.")
    else:
        logging.warning("PersonDetector was not initialized. Skipping detection tests.")
    logging.info("PersonDetector example usage finished.")
