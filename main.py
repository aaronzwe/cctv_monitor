import logging
import time
import os
import cv2
import numpy as np

from video_stream import VideoStream
from person_detector import PersonDetector
from person_reid import PersonReID
from smoking_detector_mp import SmokingGestureDetector
from cigarette_detector_yolo import CigaretteDetector
from anomaly_detector import calculate_optical_flow # Import for optical flow

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
PRIMARY_VIDEO_SOURCE = "rtsp://example.com/media/stream1"
FALLBACK_VIDEO_SOURCE = "test_video.mp4"
MAX_FRAMES_TO_PROCESS = 100
PROCESSING_LOOP_DURATION_SECONDS = 30

def ensure_video_source_or_create_dummy(primary_source, fallback_source):
    video_source_to_use = primary_source
    is_primary_file = not (primary_source.startswith("rtsp://") or primary_source.startswith("rtmp://"))

    if is_primary_file and not os.path.exists(primary_source):
        logger.warning(f"Primary video source '{primary_source}' not found.")
        if os.path.exists(fallback_source):
            logger.info(f"Using fallback video source: '{fallback_source}'")
            video_source_to_use = fallback_source
        else:
            logger.warning(f"Fallback video source '{fallback_source}' also not found.")
            try:
                logger.info(f"Attempting to create dummy video: {fallback_source}")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out_dummy = cv2.VideoWriter(fallback_source, fourcc, 10, (320,240))
                for i in range(50):
                    frame_dummy_np = np.full((240,320,3), (np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255)), dtype=np.uint8)
                    cv2.putText(frame_dummy_np, f"Dummy {i+1}", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                    out_dummy.write(frame_dummy_np)
                out_dummy.release()
                logger.info(f"Dummy video '{fallback_source}' created successfully.")
                video_source_to_use = fallback_source
            except Exception as e:
                logger.error(f"Failed to create dummy video '{fallback_source}': {e}. Exiting as no video source is available.", exc_info=True)
                return None
    elif not is_primary_file:
         logger.info(f"Using primary video source (network stream): {primary_source}")
    else:
        logger.info(f"Using primary video source (local file): {primary_source}")
    return video_source_to_use


def main():
    logger.info("Starting main application.")

    video_stream = None
    person_detector = None
    person_reid = None
    smoking_gesture_detector = None
    cigarette_detector = None
    prev_frame_gray = None # For optical flow

    try:
        video_source_to_use = ensure_video_source_or_create_dummy(PRIMARY_VIDEO_SOURCE, FALLBACK_VIDEO_SOURCE)
        if not video_source_to_use:
            return

        logger.info(f"Initializing video stream from: {video_source_to_use}")
        video_stream = VideoStream(stream_url=video_source_to_use, buffer_size=60, stream_name="main_feed")

        logger.info("Initializing PersonDetector...")
        person_detector = PersonDetector(model_path='yolov8n.pt')

        logger.info("Initializing PersonReID...")
        person_reid = PersonReID()

        logger.info("Initializing SmokingGestureDetector...")
        smoking_gesture_detector = SmokingGestureDetector()

        logger.info("Initializing CigaretteDetector (Placeholder)...")
        cigarette_detector = CigaretteDetector(model_path='yolov8n.pt')

        if not video_stream.connect():
            logger.error("Failed to connect to video stream. Exiting.")
            return

        logger.info("All components initialized and stream connected successfully.")

        frames_processed = 0
        start_time = time.time()
        person_features_db = {}
        next_person_id = 0

        while True:
            current_time = time.time()
            if frames_processed >= MAX_FRAMES_TO_PROCESS or \
               current_time - start_time >= PROCESSING_LOOP_DURATION_SECONDS:
                logger.info("Reached processing limit (frames or time).")
                break

            frame = video_stream.read_frame(timeout=1.0)
            if frame is None:
                if not video_stream.is_opened() and video_stream.buffer.empty():
                    logger.info("Stream ended or buffer empty.")
                    # Reset prev_frame_gray if stream ends, to avoid processing stale flow on restart
                    prev_frame_gray = None
                    break
                logger.debug("No frame, continuing...")
                time.sleep(0.05)
                continue

            frames_processed += 1

            # --- Optical Flow Calculation ---
            current_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_frame_gray is not None:
                flow = calculate_optical_flow(prev_frame_gray, current_frame_gray)
                if flow is not None:
                    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                    mean_flow_magnitude = np.mean(magnitude)
                    logger.info(f"Frame {frames_processed}: Mean optical flow magnitude: {mean_flow_magnitude:.2f}")
                else:
                    logger.warning(f"Frame {frames_processed}: Optical flow calculation failed.")
            prev_frame_gray = current_frame_gray.copy() # Update prev_frame_gray for the next iteration
            # --- End Optical Flow ---

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # For MediaPipe

            detected_persons_data = person_detector.detect_persons(frame, confidence_threshold=0.4)

            if detected_persons_data:
                logger.info(f"Frame {frames_processed}: Detected {len(detected_persons_data)} persons.")

                for p_data in detected_persons_data:
                    person_id_for_log = "new_person"
                    bbox = p_data['bbox']
                    x1, y1, x2, y2 = bbox

                    h_frame, w_frame = frame.shape[:2]
                    crop_x1, crop_y1 = max(0, x1), max(0, y1)
                    crop_x2, crop_y2 = min(w_frame, x2), min(h_frame, y2)

                    if crop_x1 >= crop_x2 or crop_y1 >= crop_y2:
                        logger.warning(f"Person {person_id_for_log}: Invalid bbox for crop. Skipping.")
                        continue

                    person_crop_bgr = frame[crop_y1:crop_y2, crop_x1:crop_x2]

                    if person_crop_bgr.size == 0:
                        logger.warning(f"Person {person_id_for_log}: Cropped image (BGR) is empty. Skipping.")
                        continue

                    features = person_reid.extract_features(person_crop_bgr)
                    if features is not None:
                        current_person_id = next_person_id
                        person_features_db[current_person_id] = features
                        next_person_id += 1
                        person_id_for_log = f"PersonID-{current_person_id}"
                        logger.info(f"  {person_id_for_log}: Re-ID features extracted (shape: {features.shape}).")
                    else:
                        logger.warning(f"  {person_id_for_log}: Re-ID feature extraction failed.")

                    gesture_info = smoking_gesture_detector.detect_hand_to_mouth_gesture(frame_rgb, person_bbox=bbox)
                    if gesture_info['gesture_detected']:
                        logger.info(f"  {person_id_for_log}: Hand-to-mouth gesture DETECTED. Hand: {gesture_info.get('involved_hand', 'N/A')}, Conf: {gesture_info['confidence']:.2f}")
                    else:
                        logger.info(f"  {person_id_for_log}: No hand-to-mouth gesture detected.")

                    cigarette_detections = cigarette_detector.detect_cigarettes(person_crop_bgr)
                    if cigarette_detections:
                        logger.info(f"  {person_id_for_log}: Placeholder cigarette detector found {len(cigarette_detections)} items (unexpected).")
                    else:
                        logger.info(f"  {person_id_for_log}: Placeholder cigarette detector found no cigarettes, as expected.")
            else:
                logger.info(f"Frame {frames_processed}: No persons detected.")

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt. Shutting down.")
    except Exception as e:
        logger.error(f"Unhandled exception in main loop: {e}", exc_info=True)
    finally:
        logger.info("Cleaning up resources...")
        if video_stream:
            video_stream.stop()
        if smoking_gesture_detector:
            smoking_gesture_detector.close()
        logger.info(f"Main application finished. Total unique persons processed for Re-ID: {next_person_id}")

if __name__ == '__main__':
    main()
