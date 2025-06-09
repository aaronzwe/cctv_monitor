import cv2
import numpy as np
import logging
import time # For timestamps in loitering example

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_optical_flow(prev_frame_gray, current_frame_gray, flow_params=None):
    '''
    Calculates dense optical flow using the Farneback method.
    :param prev_frame_gray: Previous frame (grayscale, NumPy array).
    :param current_frame_gray: Current frame (grayscale, NumPy array).
    :param flow_params: Optional dictionary of parameters for cv2.calcOpticalFlowFarneback.
    :return: The calculated flow field (NumPy array), or None if inputs are invalid.
    '''
    if prev_frame_gray is None or current_frame_gray is None:
        logging.warning("Previous or current frame is None. Cannot calculate optical flow.")
        return None

    if prev_frame_gray.shape != current_frame_gray.shape:
        logging.warning(f"Frame shapes differ: prev {prev_frame_gray.shape}, curr {current_frame_gray.shape}. Cannot calculate flow.")
        return None

    if len(prev_frame_gray.shape) != 2 or len(current_frame_gray.shape) != 2:
        logging.warning("Input frames must be grayscale (2D).")
        return None

    default_params = {
        'pyr_scale': 0.5,
        'levels': 3,
        'winsize': 15,
        'iterations': 3,
        'poly_n': 5,
        'poly_sigma': 1.2,
        'flags': 0
    }

    current_params = {**default_params, **(flow_params or {})}

    try:
        flow = cv2.calcOpticalFlowFarneback(prev_frame_gray, current_frame_gray, None, **current_params)
        return flow
    except cv2.error as e:
        logging.error(f"OpenCV error during Farneback optical flow calculation: {e}", exc_info=True)
        return None
    except Exception as e:
        logging.error(f"Unexpected error during Farneback optical flow calculation: {e}", exc_info=True)
        return None

def detect_loitering(person_tracks, time_threshold_seconds=60,
                     max_displacement_pixels=50, fps=10):
    '''
    (Placeholder) Detects loitering behavior from person tracks.
    :param person_tracks: Dict {person_id: [(bbox, timestamp), ...]} where bbox is (x1,y1,x2,y2).
    :param time_threshold_seconds: Minimum duration for a track to be considered loitering.
    :param max_displacement_pixels: Maximum displacement of bbox center for loitering.
    :param fps: Frames per second, to estimate time if timestamps are frame numbers (not used if timestamps are actual time).
    :return: List of person_ids detected loitering (empty for placeholder).
    '''
    logging.info("[Placeholder] detect_loitering called.")
    loitering_events = []

    if not person_tracks:
        logging.debug("No person tracks provided for loitering detection.")
        return loitering_events

    for person_id, track in person_tracks.items():
        if not track or len(track) < 2:
            logging.debug(f"Track for person {person_id} is too short for loitering analysis (length: {len(track)}).")
            continue

        first_bbox, first_timestamp = track[0]
        last_bbox, last_timestamp = track[-1]

        # Ensure timestamps are numeric for subtraction
        if not (isinstance(first_timestamp, (int, float)) and isinstance(last_timestamp, (int, float))):
            logging.warning(f"Person {person_id}: Timestamps are not numeric ({type(first_timestamp)}, {type(last_timestamp)}). Skipping.")
            continue

        duration_seconds = (last_timestamp - first_timestamp)

        center_first_x = (first_bbox[0] + first_bbox[2]) / 2
        center_first_y = (first_bbox[1] + first_bbox[3]) / 2
        center_last_x = (last_bbox[0] + last_bbox[2]) / 2
        center_last_y = (last_bbox[1] + last_bbox[3]) / 2

        displacement = np.sqrt((center_last_x - center_first_x)**2 + (center_last_y - center_first_y)**2)

        logging.info(f"  Loitering check for person ID {person_id}: "
                     f"Duration={duration_seconds:.2f}s (threshold: {time_threshold_seconds}s), "
                     f"Displacement={displacement:.2f}px (threshold: {max_displacement_pixels}px). "
                     "(Placeholder - actual logic TBD)")

        # Example actual logic (commented out for placeholder):
        # if duration_seconds >= time_threshold_seconds and displacement <= max_displacement_pixels:
        #     loitering_events.append({'person_id': person_id, 'duration': duration_seconds, 'displacement': displacement, 'bbox': last_bbox})
        #     logging.info(f"    EVENT (Placeholder Simulated): Loitering detected for person {person_id}")

    return loitering_events

def detect_crowd_formation(current_person_bboxes, min_persons_for_crowd=5,
                           proximity_threshold_pixels=100, area_bbox=None):
    '''
    (Placeholder) Detects crowd formation from current person bounding boxes.
    :param current_person_bboxes: List of current bboxes [(x1,y1,x2,y2), ...].
    :param min_persons_for_crowd: Minimum number of persons to be considered a crowd.
    :param proximity_threshold_pixels: Proximity for DBSCAN epsilon (conceptual).
    :param area_bbox: Optional ROI bounding box (x1,y1,x2,y2) to check for crowds within.
    :return: List of crowd event dicts (empty for placeholder).
    '''
    logging.info("[Placeholder] detect_crowd_formation called.")
    crowd_events = []

    bboxes_to_check = current_person_bboxes
    if area_bbox:
        logging.info(f"  Checking for crowds within area: {area_bbox}")
        # Placeholder: filter bboxes to only include those mostly within area_bbox
        # For simplicity, we'll just log and continue with all bboxes for this placeholder.
        ax1, ay1, ax2, ay2 = area_bbox
        filtered_bboxes = []
        for bx1, by1, bx2, by2 in current_person_bboxes:
            # Check if center of bbox is within area_bbox
            bcx, bcy = (bx1+bx2)/2, (by1+by2)/2
            if ax1 <= bcx < ax2 and ay1 <= bcy < ay2:
                filtered_bboxes.append((bx1,by1,bx2,by2))
        logging.info(f"  {len(filtered_bboxes)} persons are within the specified area.")
        bboxes_to_check = filtered_bboxes

    num_persons = len(bboxes_to_check)
    logging.info(f"  Crowd formation check: {num_persons} persons in relevant area "
                 f"(min for crowd: {min_persons_for_crowd}). "
                 f"(Placeholder - DBSCAN/clustering TBD)")

    # Example actual logic (commented out for placeholder):
    # if num_persons >= min_persons_for_crowd:
    #    # Conceptual: apply clustering (e.g. DBSCAN) using proximity_threshold_pixels
    #    # to see if these people form a dense group.
    #    logging.info(f"    EVENT (Placeholder Simulated): Potential crowd detected with {num_persons} persons.")
    #    # crowd_events.append({'num_persons': num_persons, 'bboxes': bboxes_to_check, 'area_focus': area_bbox if area_bbox else "full_frame"})

    return crowd_events

if __name__ == '__main__':
    logging.info("Starting AnomalyDetector module example usage...")

    # --- Optical Flow Tests ---
    logging.info("\n--- Testing Optical Flow Calculation ---")
    frame_height, frame_width = 240, 320
    prev_gray = np.full((frame_height, frame_width), 128, dtype=np.uint8)
    current_gray = prev_gray.copy()
    rect_x, rect_y, rect_w, rect_h = frame_width // 4, frame_height // 4, frame_width // 2, frame_height // 2
    cv2.rectangle(current_gray, (rect_x, rect_y), (rect_x + rect_w, rect_y + rect_h), 255, -1)
    flow_field = calculate_optical_flow(prev_gray, current_gray)
    if flow_field is not None:
        logging.info(f"Optical flow calculated successfully. Flow field shape: {flow_field.shape}")
        assert flow_field.shape == (frame_height, frame_width, 2)
    else:
        logging.error("Optical flow calculation failed for dummy frames.")
    # (Other optical flow tests like None inputs, mismatched shapes, color inputs remain relevant but omitted here for brevity)

    # --- Loitering Detection Placeholder Tests ---
    logging.info("\n--- Testing Loitering Detection Placeholder ---")
    current_abs_time = time.time() # Use absolute time for more realistic timestamps
    dummy_tracks = {
        1: [((10,10,30,60), current_abs_time - 70.0), ((15,12,35,62), current_abs_time)], # Loitering candidate
        2: [((50,50,70,90), current_abs_time - 30.0), ((150,150,170,190), current_abs_time)], # Not loitering (moved too much)
        3: [((10,10,30,60), current_abs_time - 10.0)], # Too short track
        4: [((10,10,30,60), "not_a_timestamp"), ((15,12,35,62), current_abs_time)], # Invalid timestamp
    }
    loitering_results = detect_loitering(dummy_tracks, time_threshold_seconds=60, max_displacement_pixels=20)
    logging.info(f"Loitering detection results (placeholder): {loitering_results}")
    assert isinstance(loitering_results, list) and not loitering_results, "Placeholder should return empty list"

    # --- Crowd Formation Placeholder Tests ---
    logging.info("\n--- Testing Crowd Formation Placeholder ---")
    dummy_bboxes_crowd = [(i*10 + 50, i*10 + 50, i*10+20 + 50, i*10+40 + 50) for i in range(6)] # 6 close persons
    dummy_bboxes_sparse = [(i*100, i*100, i*100+20, i*100+40) for i in range(3)] # 3 sparse persons

    crowd_results_crowd = detect_crowd_formation(dummy_bboxes_crowd, min_persons_for_crowd=5)
    logging.info(f"Crowd detection for dense bboxes (placeholder): {crowd_results_crowd}")
    assert isinstance(crowd_results_crowd, list) and not crowd_results_crowd, "Placeholder should return empty list"

    crowd_results_sparse = detect_crowd_formation(dummy_bboxes_sparse, min_persons_for_crowd=5)
    logging.info(f"Crowd detection for sparse bboxes (placeholder): {crowd_results_sparse}")
    assert isinstance(crowd_results_sparse, list) and not crowd_results_sparse

    # Test crowd detection with an area_bbox
    test_area = (60, 60, 100, 100) # A small area where some of dummy_bboxes_crowd should fall
    crowd_results_area = detect_crowd_formation(dummy_bboxes_crowd, min_persons_for_crowd=2, area_bbox=test_area)
    logging.info(f"Crowd detection for dense bboxes within area {test_area} (placeholder): {crowd_results_area}")
    assert isinstance(crowd_results_area, list) and not crowd_results_area

    logging.info("AnomalyDetector module example usage finished.")
