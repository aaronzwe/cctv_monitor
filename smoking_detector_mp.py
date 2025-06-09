import cv2
import mediapipe as mp
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SmokingGestureDetector:
    def __init__(self, static_image_mode=False, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        '''
        Initializes the SmokingGestureDetector with MediaPipe Pose.
        :param static_image_mode: Whether to treat input images as a batch of static images or a video stream.
        :param model_complexity: Complexity of the pose landmark model (0, 1, or 2).
        :param min_detection_confidence: Minimum confidence value for person detection to be considered successful.
        :param min_tracking_confidence: Minimum confidence value for pose landmarks to be considered tracked successfully.
        '''
        try:
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=static_image_mode,
                model_complexity=model_complexity,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence)

            # Define keypoint indices for hands and mouth (from MediaPipe Pose documentation)
            # Hands (Wrists, Index MCP or Fingertips for more specific gesture)
            self.hand_keypoints_indices = {
                "LEFT_WRIST": self.mp_pose.PoseLandmark.LEFT_WRIST.value, #15
                "RIGHT_WRIST": self.mp_pose.PoseLandmark.RIGHT_WRIST.value, #16
                "LEFT_INDEX_FINGERTIP": self.mp_pose.PoseLandmark.LEFT_INDEX.value, #19 (MediaPipe shows this as INDEX_FINGER_MCP, actual fingertip is 20 for index)
                "RIGHT_INDEX_FINGERTIP": self.mp_pose.PoseLandmark.RIGHT_INDEX.value, #20
            }
            # Mouth (Simplified as a region between left and right mouth corners)
            self.mouth_keypoints_indices = {
                "MOUTH_LEFT": self.mp_pose.PoseLandmark.MOUTH_LEFT.value, #9
                "MOUTH_RIGHT": self.mp_pose.PoseLandmark.MOUTH_RIGHT.value, #10
            }
            logging.info("MediaPipe Pose initialized successfully for SmokingGestureDetector.")
        except Exception as e:
            logging.error(f"Error initializing MediaPipe Pose: {e}", exc_info=True)
            raise

    def _calculate_distance(self, point1, point2):
        '''Helper to calculate Euclidean distance between two 2D points (x,y).'''
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

    def detect_hand_to_mouth_gesture(self, frame_rgb, person_bbox=None, distance_threshold_factor=0.1):
        '''
        Detects hand-to-mouth gestures in a given RGB frame within a person's bounding box.
        :param frame_rgb: Input RGB frame (NumPy array).
        :param person_bbox: Optional bounding box (x1, y1, x2, y2) of the person to focus on.
                            If None, processes the full frame.
        :param distance_threshold_factor: Factor of person height to determine proximity threshold.
                                           e.g., 0.1 means 10% of person height.
        :return: Dictionary with 'gesture_detected' (bool), 'confidence' (float),
                 'involved_hand' (str or None), and 'debug_landmarks' (dict of relevant keypoints).
        '''
        if frame_rgb is None or frame_rgb.size == 0:
            logging.warning("Input frame_rgb is None or empty.")
            return {'gesture_detected': False, 'confidence': 0.0, 'involved_hand': None, 'debug_landmarks': {}}

        roi_frame_rgb = frame_rgb
        offset_x, offset_y = 0, 0
        person_height = frame_rgb.shape[0] # Default to frame height

        if person_bbox:
            x1, y1, x2, y2 = person_bbox
            # Ensure bbox coordinates are within frame dimensions
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame_rgb.shape[1], x2), min(frame_rgb.shape[0], y2)

            if x1 >= x2 or y1 >= y2: # If bbox is invalid or results in empty crop
                logging.warning("Invalid person_bbox for ROI. Processing full frame instead.")
            else:
                roi_frame_rgb = frame_rgb[y1:y2, x1:x2]
                offset_x, offset_y = x1, y1
                person_height = y2 - y1

        if roi_frame_rgb.size == 0: # If ROI is empty somehow
             logging.warning("ROI frame is empty. Cannot detect gesture.")
             return {'gesture_detected': False, 'confidence': 0.0, 'involved_hand': None, 'debug_landmarks': {}}


        # Process the ROI/frame with MediaPipe Pose
        results = self.pose.process(roi_frame_rgb)
        gesture_info = {'gesture_detected': False, 'confidence': 0.0, 'involved_hand': None, 'debug_landmarks': {}}

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            h, w = roi_frame_rgb.shape[:2] # Height, Width of the ROI

            # Get mouth coordinates (average for simplicity)
            mouth_left_lm = landmarks[self.mouth_keypoints_indices["MOUTH_LEFT"]]
            mouth_right_lm = landmarks[self.mouth_keypoints_indices["MOUTH_RIGHT"]]

            if mouth_left_lm.visibility < 0.5 or mouth_right_lm.visibility < 0.5:
                logging.debug("Mouth landmarks not clearly visible.")
                return gesture_info # Not enough info

            mouth_center_roi_x = (mouth_left_lm.x + mouth_right_lm.x) / 2 * w
            mouth_center_roi_y = (mouth_left_lm.y + mouth_right_lm.y) / 2 * h
            mouth_point_roi = (mouth_center_roi_x, mouth_center_roi_y)

            gesture_info['debug_landmarks']['mouth_center'] = (mouth_center_roi_x + offset_x, mouth_center_roi_y + offset_y)

            # Define proximity threshold based on person's height in the ROI
            # This is a heuristic and might need tuning
            proximity_threshold = person_height * distance_threshold_factor

            for hand_name, hand_idx in self.hand_keypoints_indices.items():
                hand_lm = landmarks[hand_idx]
                if hand_lm.visibility < 0.3: # Lower threshold for hands as they might be partially occluded
                    logging.debug(f"{hand_name} landmark not clearly visible (visibility: {hand_lm.visibility:.2f}).")
                    continue

                hand_point_roi_x = hand_lm.x * w
                hand_point_roi_y = hand_lm.y * h
                hand_point_roi = (hand_point_roi_x, hand_point_roi_y)

                gesture_info['debug_landmarks'][hand_name] = (hand_point_roi_x + offset_x, hand_point_roi_y + offset_y)

                distance = self._calculate_distance(hand_point_roi, mouth_point_roi)
                logging.debug(f"Distance between {hand_name} and mouth_center: {distance:.2f}px. Threshold: {proximity_threshold:.2f}px")

                if distance < proximity_threshold:
                    gesture_info['gesture_detected'] = True
                    gesture_info['confidence'] = 1.0 # Simple confidence for now
                    gesture_info['involved_hand'] = hand_name.split('_')[0].lower() # 'left' or 'right'
                    logging.info(f"Hand-to-mouth gesture detected: {hand_name} near mouth.")
                    break # Assuming one hand is enough to confirm gesture for now
        else:
            logging.debug("No pose landmarks detected in the ROI/frame.")

        return gesture_info

    def close(self):
        '''Releases MediaPipe Pose resources.'''
        if hasattr(self, 'pose') and self.pose:
            # The 'close' method is standard for MediaPipe solutions
            self.pose.close()
            logging.info("MediaPipe Pose resources released.")


if __name__ == '__main__':
    logging.info("Starting SmokingGestureDetector example usage...")

    detector = None
    try:
        detector = SmokingGestureDetector()
    except Exception as e:
        logging.error(f"Failed to initialize SmokingGestureDetector: {e}. Example will be skipped.")

    if detector:
        # Create a dummy RGB frame (e.g., a gray image)
        frame_height, frame_width = 480, 640
        # Create a BGR frame first, then convert to RGB, as OpenCV often handles BGR.
        dummy_frame_bgr = np.full((frame_height, frame_width, 3), (100, 100, 100), dtype=np.uint8)
        cv2.putText(dummy_frame_bgr, "Test Area", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
        dummy_frame_rgb = cv2.cvtColor(dummy_frame_bgr, cv2.COLOR_BGR2RGB)

        # 1. Test with full frame (no person_bbox)
        logging.info("Testing gesture detection on full dummy frame (no person expected)...")
        # MediaPipe Pose needs an image with a person to detect landmarks effectively.
        # A blank frame is unlikely to yield pose landmarks.
        gesture_result_full = detector.detect_hand_to_mouth_gesture(dummy_frame_rgb)
        logging.info(f"Full frame gesture detection result: {gesture_result_full}")
        assert not gesture_result_full['gesture_detected'], "Gesture should not be detected in a blank dummy frame."

        # 2. Test with a person_bbox (still on dummy frame, so no actual person)
        # This mainly tests the ROI cropping logic.
        dummy_bbox = (frame_width // 4, frame_height // 4,
                      3 * frame_width // 4, 3 * frame_height // 4) # A central box
        cv2.rectangle(dummy_frame_bgr, (dummy_bbox[0], dummy_bbox[1]), (dummy_bbox[2], dummy_bbox[3]), (0,255,0), 2) # Draw bbox
        dummy_frame_rgb_with_box_drawn = cv2.cvtColor(dummy_frame_bgr, cv2.COLOR_BGR2RGB)


        logging.info(f"Testing gesture detection with a dummy bbox: {dummy_bbox} (no actual person in ROI)...")
        gesture_result_bbox = detector.detect_hand_to_mouth_gesture(dummy_frame_rgb_with_box_drawn, person_bbox=dummy_bbox)
        logging.info(f"Dummy bbox gesture detection result: {gesture_result_bbox}")
        assert not gesture_result_bbox['gesture_detected'], "Gesture should not be detected in a blank ROI."

        # (Optional) Save the frame with bbox for visual inspection
        # cv2.imwrite("dummy_gesture_test_frame.jpg", cv2.cvtColor(dummy_frame_rgb_with_box_drawn, cv2.COLOR_RGB2BGR))

        # Note: To truly test gesture detection, an image/frame with a person performing the gesture is needed.
        # The current dummy test primarily checks if the code runs without errors and if ROI logic is hit.
        logging.info("To effectively test gesture detection, an image containing a person performing a hand-to-mouth gesture is required.")

        detector.close() # Clean up MediaPipe resources

    logging.info("SmokingGestureDetector example usage finished.")
