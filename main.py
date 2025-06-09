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
from anomaly_detector import calculate_optical_flow, detect_loitering, detect_crowd_formation
from clothing_detector import ClothingFeatureExtractor, ClothingComparator
from alert_manager import AlertManager # Import AlertManager

# Configure basic logging
# Main's basicConfig will set the root logger. AlertManager's logger will inherit this.
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__) # Logger for main.py
logging.getLogger('AlertManager').setLevel(logging.WARNING) # Ensure AlertManager's specific logs are at least WARNING

# Configuration
PRIMARY_VIDEO_SOURCE = "rtsp://example.com/media/stream1"
FALLBACK_VIDEO_SOURCE = "test_video.mp4"
MAX_FRAMES_TO_PROCESS = 200
PROCESSING_LOOP_DURATION_SECONDS = 60
CLOTHING_CHECK_INTERVAL_FRAMES = 50
ANOMALY_CHECK_INTERVAL_FRAMES = 100
MAX_TRACK_LENGTH = 50
HIGH_FLOW_THRESHOLD = 10.0 # Example threshold for optical flow alert

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
                for i in range(ANOMALY_CHECK_INTERVAL_FRAMES + 50):
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
    clothing_feature_extractor = None
    alert_manager = None # Initialize AlertManager variable
    prev_frame_gray = None

    person_tracks_data = {}
    person_clothing_db = {}
    next_person_id_global = 0

    try:
        video_source_to_use = ensure_video_source_or_create_dummy(PRIMARY_VIDEO_SOURCE, FALLBACK_VIDEO_SOURCE)
        if not video_source_to_use: return

        video_stream = VideoStream(stream_url=video_source_to_use, buffer_size=60, stream_name="main_feed")
        person_detector = PersonDetector(model_path='yolov8n.pt')
        person_reid = PersonReID()
        smoking_gesture_detector = SmokingGestureDetector()
        cigarette_detector = CigaretteDetector(model_path='yolov8n.pt')
        clothing_feature_extractor = ClothingFeatureExtractor()
        alert_manager = AlertManager() # Initialize AlertManager

        logger.info("Core components initialized.")
        if not video_stream.connect():
            logger.error("Failed to connect to video stream. Exiting.")
            return
        logger.info("All components initialized and stream connected successfully.")

        frames_processed_total = 0
        start_time = time.time()

        while True:
            current_loop_time = time.time()
            if frames_processed_total >= MAX_FRAMES_TO_PROCESS or \
               current_loop_time - start_time >= PROCESSING_LOOP_DURATION_SECONDS:
                logger.info("Reached global processing limit.")
                break

            frame = video_stream.read_frame(timeout=1.0)
            if frame is None:
                if not video_stream.is_opened() and video_stream.buffer.empty():
                    prev_frame_gray = None; break
                time.sleep(0.05); continue

            frames_processed_total += 1
            cam_id = video_stream.stream_name # Get camera ID from stream object

            current_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mean_flow_magnitude = 0.0 # Default value
            if prev_frame_gray is not None:
                flow = calculate_optical_flow(prev_frame_gray, current_frame_gray)
                if flow is not None:
                    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                    mean_flow_magnitude = np.mean(magnitude)
                    logger.info(f"Frame {frames_processed_total}: Mean optical flow: {mean_flow_magnitude:.2f}")
                    if mean_flow_magnitude > HIGH_FLOW_THRESHOLD:
                        alert_manager.generate_alert(
                            event_type="HIGH_MOTION_DETECTED",
                            camera_id=cam_id,
                            details={'mean_flow_magnitude': round(mean_flow_magnitude,2), 'frame_number': frames_processed_total}
                        )
            prev_frame_gray = current_frame_gray.copy()

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detected_persons_data = person_detector.detect_persons(frame, confidence_threshold=0.4)
            current_bboxes_for_crowd_check = [p['bbox'] for p in detected_persons_data] if detected_persons_data else []

            if detected_persons_data:
                logger.debug(f"Frame {frames_processed_total}: Detected {len(detected_persons_data)} persons.")

                for p_idx, p_data in enumerate(detected_persons_data):
                    bbox = p_data['bbox']
                    x1,y1,x2,y2 = bbox
                    h_frame, w_frame = frame.shape[:2]
                    crop_x1, crop_y1, crop_x2, crop_y2 = max(0,x1), max(0,y1), min(w_frame,x2), min(h_frame,y2)

                    if crop_x1 >= crop_x2 or crop_y1 >= crop_y2: continue
                    person_crop_bgr = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                    if person_crop_bgr.size == 0: continue

                    assigned_person_id = None
                    reid_features_current_person = person_reid.extract_features(person_crop_bgr)

                    if reid_features_current_person is not None:
                        assigned_person_id = next_person_id_global
                        person_reid_db[assigned_person_id] = {'features': reid_features_current_person, 'last_seen_frame': frames_processed_total, 'bbox': bbox}
                        next_person_id_global += 1
                        logger.info(f"  Assigned PersonID-{assigned_person_id} (new ReID).")
                    else:
                        logger.warning(f"  Detection idx {p_idx}: Re-ID feature extraction failed.")
                        continue

                    current_timestamp = time.time()
                    if assigned_person_id not in person_tracks_data: person_tracks_data[assigned_person_id] = []
                    person_tracks_data[assigned_person_id].append((bbox, current_timestamp))
                    person_tracks_data[assigned_person_id] = person_tracks_data[assigned_person_id][-MAX_TRACK_LENGTH:]

                    last_clothing_check_frame = person_clothing_db.get(assigned_person_id, {}).get('last_check_frame', 0)
                    if (frames_processed_total - last_clothing_check_frame) >= CLOTHING_CHECK_INTERVAL_FRAMES or last_clothing_check_frame == 0:
                        logger.info(f"  PersonID-{assigned_person_id}: Checking clothing features.")
                        current_clothing_feats = clothing_feature_extractor.extract_clothing_features(person_crop_bgr)
                        if current_clothing_feats:
                            combined_similarity_score = -1.0 # Default if no previous features
                            if assigned_person_id in person_clothing_db and 'features' in person_clothing_db[assigned_person_id]:
                                prev_clothing_feats = person_clothing_db[assigned_person_id]['features']
                                combined_similarity_score, change = ClothingComparator.compare_clothing_features(prev_clothing_feats, current_clothing_feats)
                                if change:
                                    alert_manager.generate_alert(
                                        event_type="CLOTHING_CHANGE_DETECTED",
                                        person_id=assigned_person_id,
                                        camera_id=cam_id,
                                        details={
                                            'previous_features_timestamp': person_clothing_db[assigned_person_id].get('timestamp'),
                                            'similarity_score': round(combined_similarity_score, 4),
                                            'current_bbox': bbox
                                        }
                                    )
                                else: logger.info(f"  PersonID-{assigned_person_id}: Clothing consistent (Sim: {combined_similarity_score:.4f}).")
                            else: logger.info(f"  PersonID-{assigned_person_id}: Storing initial clothing features.")
                            person_clothing_db[assigned_person_id] = {
                                'features': current_clothing_feats,
                                'last_check_frame': frames_processed_total,
                                'timestamp': current_loop_time,
                                'last_similarity': combined_similarity_score
                            }
                        else: logger.warning(f"  PersonID-{assigned_person_id}: Clothing feature extraction failed.")

                    gesture = smoking_gesture_detector.detect_hand_to_mouth_gesture(frame_rgb, person_bbox=bbox)
                    if gesture['gesture_detected']:
                        alert_manager.generate_alert(
                            "SMOKING_GESTURE_DETECTED",
                            confidence=gesture['confidence'],
                            person_id=assigned_person_id,
                            camera_id=cam_id,
                            details={'hand': gesture.get('involved_hand'), 'bbox': bbox} # Removed landmarks for brevity
                        )

                    # Placeholder cigarette detection alert (currently detect_cigarettes returns empty)
                    # cigarettes = cigarette_detector.detect_cigarettes(person_crop_bgr)
                    # if cigarettes: # This list will be empty based on current placeholder
                    #     alert_manager.generate_alert(
                    #         "CIGARETTE_DETECTED_PLACEHOLDER",
                    #         person_id=assigned_person_id,
                    #         camera_id=cam_id,
                    #         details={'detections': cigarettes, 'bbox': bbox}
                    #     )

            if frames_processed_total > 0 and frames_processed_total % ANOMALY_CHECK_INTERVAL_FRAMES == 0:
                logger.info(f"--- Frame {frames_processed_total}: Performing global anomaly checks ---")
                fps_estimate = video_stream.cap.get(cv2.CAP_PROP_FPS) if video_stream and video_stream.cap and video_stream.cap.isOpened() and video_stream.cap.get(cv2.CAP_PROP_FPS) > 0 else 10.0

                loitering_events = detect_loitering(person_tracks_data, time_threshold_seconds=10, max_displacement_pixels=30, fps=fps_estimate)
                for event in loitering_events: # Will be empty from stub
                    alert_manager.generate_alert("LOITERING_DETECTED_PLACEHOLDER", person_id=event['person_id'], camera_id=cam_id, details=event)
                if not loitering_events: logger.info("No loitering events (placeholder).")

                crowd_events = detect_crowd_formation(current_bboxes_for_crowd_check, min_persons_for_crowd=3, proximity_threshold_pixels=50)
                for event in crowd_events: # Will be empty from stub
                    alert_manager.generate_alert("CROWD_FORMATION_PLACEHOLDER", camera_id=cam_id, details=event)
                if not crowd_events: logger.info("No crowd events (placeholder).")

    except KeyboardInterrupt: logger.info("Keyboard interrupt. Shutting down.")
    except Exception as e: logger.error(f"Unhandled exception in main loop: {e}", exc_info=True)
    finally:
        logger.info("Cleaning up resources...")
        if video_stream: video_stream.stop()
        if smoking_gesture_detector: smoking_gesture_detector.close()
        logger.info(f"Main application finished. Total unique person IDs considered: {next_person_id_global}")
        if alert_manager and alert_manager.get_all_alerts():
            logger.info("\n--- All Generated Alerts ---")
            for alert_item in alert_manager.get_all_alerts():
                logger.info(alert_item) # Alert object has __str__

if __name__ == '__main__':
    main()
