import cv2
import argparse
import torch
from ultralytics import YOLO
import mediapipe as mp
import numpy as np

# MediaPipe setup
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="CCTV Monitor - Process video file.")
    parser.add_argument("-i", "--input", required=True, help="Path to the input video file.")
    parser.add_argument("-o", "--output", required=True, help="Path for the output video file.")

    args = parser.parse_args()

    # Open the input video file
    cap = cv2.VideoCapture(args.input)

    # Check if video capture object was successfully opened
    if not cap.isOpened():
        print(f"Error: Could not open input video file: {args.input}")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Define the codec and create VideoWriter object
    # Using MP4V codec, which is widely compatible.
    # You might need to change this based on your system or desired output format.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (frame_width, frame_height))

    if not out.isOpened():
        print(f"Error: Could not open output video file for writing: {args.output}")
        cap.release()
        return

    # Load YOLOv8 model
    try:
        model = YOLO('yolov8n.pt')
        print("YOLOv8 model loaded successfully.")
    except Exception as e:
        print(f"Error loading YOLOv8 model: {e}")
        cap.release()
        out.release()
        return

    # Initialize MediaPipe Pose and Hands
    # min_detection_confidence and min_tracking_confidence can be tuned
    pose_detector = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    hands_detector = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_hands=2)

    print(f"Processing video: {args.input}")
    print(f"Output will be saved to: {args.output}")

    while True:
        ret, frame = cap.read()

        if ret:
            # Perform detection for persons (class 0) and cigarettes (class 79)
            # Increase confidence for cigarettes slightly as they might be harder to detect accurately
            yolo_results = model.predict(frame, classes=[0, 79], conf=0.25)

            # Create a list of detected persons for MediaPipe processing
            persons_rois = []
            person_boxes_orig_coords = []

            # First, draw YOLO detections and store person ROIs
            for result in yolo_results:
                for box in result.boxes:
                    x1_orig, y1_orig, x2_orig, y2_orig = map(int, box.xyxy[0])
                    confidence = box.conf[0]
                    class_id = int(box.cls[0])

                    if class_id == 0: # Person
                        persons_rois.append(frame[y1_orig:y2_orig, x1_orig:x2_orig])
                        person_boxes_orig_coords.append((x1_orig, y1_orig, x2_orig, y2_orig))

                        # Draw person box (will be default green)
                        color = (0, 255, 0)
                        label = f"Person: {confidence:.2f}"
                        cv2.rectangle(frame, (x1_orig, y1_orig), (x2_orig, y2_orig), color, 2)
                        cv2.putText(frame, label, (x1_orig, y1_orig - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    elif class_id == 79: # Cigarette
                        color = (0, 0, 255) # Red for cigarette
                        label = f"Cigarette: {confidence:.2f}"
                        cv2.rectangle(frame, (x1_orig, y1_orig), (x2_orig, y2_orig), color, 1)
                        cv2.putText(frame, label, (x1_orig, y1_orig - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            detected_persons_info = [] # Store info: [box, confidence, gesture_status]
            detected_cigarettes_info = [] # Store info: [box, confidence]

            # First, draw YOLO detections and store person ROIs and cigarette boxes
            for result in yolo_results:
                for box in result.boxes:
                    x1_orig, y1_orig, x2_orig, y2_orig = map(int, box.xyxy[0])
                    confidence = box.conf[0]
                    class_id = int(box.cls[0])

                    if class_id == 0: # Person
                        # Initial color and label for person
                        color = (0, 255, 0) # Green
                        label = f"Person: {confidence:.2f}"
                        # Store person data for later processing (MediaPipe and smoking logic)
                        detected_persons_info.append({
                            "box_orig": (x1_orig, y1_orig, x2_orig, y2_orig),
                            "roi": frame[y1_orig:y2_orig, x1_orig:x2_orig],
                            "confidence": confidence,
                            "has_gesture": False, # Will be updated by MediaPipe
                            "is_smoking": False   # Will be updated by combined logic
                        })
                        # Initial drawing - might be updated later by gesture/smoking logic
                        cv2.rectangle(frame, (x1_orig, y1_orig), (x2_orig, y2_orig), color, 2)
                        cv2.putText(frame, label, (x1_orig, y1_orig - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    elif class_id == 79: # Cigarette
                        detected_cigarettes_info.append({
                            "box_orig": (x1_orig, y1_orig, x2_orig, y2_orig),
                            "confidence": confidence
                        })
                        # Cigarette drawing can be done here or later if we want to avoid drawing if part of smoking event
                        # For now, draw all cigarettes initially
                        color = (0, 0, 255) # Red for cigarette
                        label = f"Cigarette: {confidence:.2f}"
                        cv2.rectangle(frame, (x1_orig, y1_orig), (x2_orig, y2_orig), color, 1)
                        cv2.putText(frame, label, (x1_orig, y1_orig - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            # Process each detected person with MediaPipe for gesture
            for person_info in detected_persons_info:
                person_roi = person_info["roi"]
                if person_roi.size == 0: continue

                # Convert ROI to RGB
                rgb_roi = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)
                rgb_roi.flags.writeable = False

                pose_results = pose_detector.process(rgb_roi)
                hands_results = hands_detector.process(rgb_roi)

                rgb_roi.flags.writeable = True

                current_gesture_detected = False
                min_distance_threshold = 30

                if pose_results.pose_landmarks:
                    mouth_left = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.MOUTH_LEFT]
                    mouth_right = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.MOUTH_RIGHT]

                    mouth_x_roi = int((mouth_left.x + mouth_right.x) / 2 * person_roi.shape[1])
                    mouth_y_roi = int((mouth_left.y + mouth_right.y) / 2 * person_roi.shape[0])

                    if hands_results.multi_hand_landmarks:
                        for hand_landmarks in hands_results.multi_hand_landmarks:
                            key_hand_landmarks_indices = [
                                mp_hands.HandLandmark.WRIST, mp_hands.HandLandmark.THUMB_TIP,
                                mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                                mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.PINKY_TIP
                            ]
                            for lm_idx in key_hand_landmarks_indices:
                                hand_lm = hand_landmarks.landmark[lm_idx]
                                hand_x_roi = int(hand_lm.x * person_roi.shape[1])
                                hand_y_roi = int(hand_lm.y * person_roi.shape[0])
                                distance = np.sqrt((hand_x_roi - mouth_x_roi)**2 + (hand_y_roi - mouth_y_roi)**2)
                                if distance < min_distance_threshold:
                                    current_gesture_detected = True
                                    break
                            if current_gesture_detected:
                                break
                person_info["has_gesture"] = current_gesture_detected

            # Combined Smoking Logic and Re-drawing
            # Helper function for IoU
            def calculate_iou(boxA, boxB):
                # Determine the (x, y)-coordinates of the intersection rectangle
                xA = max(boxA[0], boxB[0])
                yA = max(boxA[1], boxB[1])
                xB = min(boxA[2], boxB[2])
                yB = min(boxA[3], boxB[3])

                # Compute the area of intersection rectangle
                interArea = max(0, xB - xA) * max(0, yB - yA)
                if interArea == 0: return 0

                # Compute the area of both the prediction and ground-truth rectangles
                boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
                boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

                iou = interArea / float(boxAArea + boxBArea - interArea)
                return iou

            # Clear previous drawings for persons (or draw over them)
            # This is a bit inefficient; ideally, drawing happens once after all logic.
            # For simplicity now, we might redraw some boxes.
            # A better approach would be to collect all drawing commands and execute at the end.

            for person_info in detected_persons_info:
                p_box = person_info["box_orig"]
                is_smoking = False
                if person_info["has_gesture"]:
                    for cig_info in detected_cigarettes_info:
                        c_box = cig_info["box_orig"]
                        iou = calculate_iou(p_box, c_box)
                        # Also check if cigarette center is within person box (looser check)
                        c_center_x = (c_box[0] + c_box[2]) / 2
                        c_center_y = (c_box[1] + c_box[3]) / 2
                        is_cig_center_in_person = (p_box[0] < c_center_x < p_box[2] and \
                                                   p_box[1] < c_center_y < p_box[3])

                        if iou > 0.01 or is_cig_center_in_person: # IoU threshold or center check
                            is_smoking = True
                            person_info["is_smoking"] = True
                            break # Found a cigarette for this gesturing person

                # Visualization update based on final status
                final_color = (0, 255, 0) # Default Green for person
                final_label_text = f"Person: {person_info['confidence']:.2f}"
                label_y_offset = -10

                if person_info["is_smoking"]:
                    final_color = (255, 0, 255)  # Magenta for SMOKING
                    final_label_text = "SMOKING DETECTED"
                    label_y_offset = -25 # Make space if multiple lines or more prominent
                    # Optionally, remove the specific cigarette box that triggered this, or change its color
                elif person_info["has_gesture"]:
                    final_color = (0, 165, 255)  # Orange for Hand-Mouth Gesture
                    final_label_text = "Hand-Mouth Gesture"

                # Redraw person box with final status
                cv2.rectangle(frame, (p_box[0], p_box[1]), (p_box[2], p_box[3]), final_color, 2)
                cv2.putText(frame, final_label_text, (p_box[0], p_box[1] + label_y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, final_color, 2)

            # Note: Cigarettes not associated with smoking are already drawn.
            # If we wanted to hide them if they are part of a smoking event, that logic would be added here.

            out.write(frame)

            # Optional: Display the frame
            # cv2.imshow('Frame', frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'): # Press 'q' to quit display
            #     break
        else:
            # End of video or error
            break

    # Release everything when job is finished
    cap.release()
    out.release()
    pose_detector.close()
    hands_detector.close()
    # cv2.destroyAllWindows() # Uncomment if using cv2.imshow

    print(f"Video processing complete. Output saved to {args.output}")

if __name__ == "__main__":
    main()
