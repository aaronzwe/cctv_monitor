import cv2
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def calculate_optical_flow(prev_frame_gray, current_frame_gray, flow_params=None):
    '''
    Calculates dense optical flow using the Farneback method.
    :param prev_frame_gray: Previous frame (grayscale, NumPy array).
    :param current_frame_gray: Current frame (grayscale, NumPy array).
    :param flow_params: Optional dictionary of parameters for cv2.calcOpticalFlowFarneback.
                        Example: dict(pyr_scale=0.5, levels=3, winsize=15, iterations=3,
                                      poly_n=5, poly_sigma=1.2, flags=0)
    :return: The calculated flow field (NumPy array), or None if inputs are invalid.
    '''
    if prev_frame_gray is None or current_frame_gray is None:
        logging.warning("Previous or current frame is None. Cannot calculate optical flow.")
        return None

    if prev_frame_gray.shape != current_frame_gray.shape:
        logging.warning(f"Frame shapes differ: prev {prev_frame_gray.shape}, curr {current_frame_gray.shape}. Cannot calculate flow.")
        # Optionally, one might try to resize current_frame_gray to prev_frame_gray.shape here,
        # but for now, we'll require them to be the same.
        return None

    if len(prev_frame_gray.shape) != 2 or len(current_frame_gray.shape) != 2:
        logging.warning("Input frames must be grayscale (2D).")
        return None

    default_params = {
        'pyr_scale': 0.5,  # Pyramid scale, < 1 means a classical pyramid, where each layer is half the size of the previous one.
        'levels': 3,       # Number of pyramid layers including the initial image.
        'winsize': 15,     # Averaging window size; larger values increase robustness but also blur motion.
        'iterations': 3,   # Number of iterations at each pyramid level.
        'poly_n': 5,       # Size of the pixel neighborhood used to find polynomial expansion; typically 5 or 7.
        'poly_sigma': 1.2, # Standard deviation of the Gaussian that is used to smooth derivatives; typically 1.1 for poly_n=5 and 1.5 for poly_n=7.
        'flags': 0         # Operation flags. Can be 0, OPTFLOW_USE_INITIAL_FLOW, or OPTFLOW_FARNEBACK_GAUSSIAN.
    }

    if flow_params:
        current_params = {**default_params, **flow_params} # Merge user params with defaults, user params take precedence
    else:
        current_params = default_params

    try:
        flow = cv2.calcOpticalFlowFarneback(prev_frame_gray, current_frame_gray, None, **current_params)
        return flow
    except cv2.error as e: # More specific OpenCV error
        logging.error(f"OpenCV error during Farneback optical flow calculation: {e}", exc_info=True)
        return None
    except Exception as e:
        logging.error(f"Unexpected error during Farneback optical flow calculation: {e}", exc_info=True)
        return None

if __name__ == '__main__':
    logging.info("Starting AnomalyDetector (Optical Flow) example usage...")

    # Create two dummy grayscale frames to simulate motion
    frame_height, frame_width = 240, 320

    # Previous frame: a gray background
    prev_gray = np.full((frame_height, frame_width), 128, dtype=np.uint8)

    # Current frame: same gray background with a white rectangle that has "moved"
    # (or appears, simulating change)
    current_gray = prev_gray.copy()
    # Draw a white rectangle on the current frame
    rect_x, rect_y, rect_w, rect_h = frame_width // 4, frame_height // 4, frame_width // 2, frame_height // 2
    cv2.rectangle(current_gray, (rect_x, rect_y), (rect_x + rect_w, rect_y + rect_h), 255, -1) # Filled white rectangle

    logging.info("Calculating optical flow between two dummy grayscale frames...")
    flow_field = calculate_optical_flow(prev_gray, current_gray)

    if flow_field is not None:
        logging.info(f"Optical flow calculated successfully. Flow field shape: {flow_field.shape}")
        # Flow field contains 2 channels for dx and dy vectors
        assert flow_field.shape == (frame_height, frame_width, 2), \
            f"Expected flow field shape ({frame_height}, {frame_width}, 2), got {flow_field.shape}"

        # Optional: Visualize the flow (convert to HSV and then BGR for display/saving)
        # magnitude, angle = cv2.cartToPolar(flow_field[..., 0], flow_field[..., 1])
        # hsv_mask = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        # hsv_mask[..., 0] = angle * 180 / np.pi / 2 # Angle mapped to hue
        # hsv_mask[..., 1] = 255 # Saturation full
        # hsv_mask[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX) # Magnitude mapped to value
        # bgr_flow_viz = cv2.cvtColor(hsv_mask, cv2.COLOR_HSV2BGR)
        # try:
        #    cv2.imwrite("optical_flow_visualization.jpg", bgr_flow_viz)
        #    logging.info("Optical flow visualization saved to optical_flow_visualization.jpg")
        # except Exception as e_img:
        #    logging.error(f"Failed to save optical flow visualization: {e_img}")

    else:
        logging.error("Optical flow calculation failed for dummy frames.")

    # Test with None inputs
    logging.info("Testing optical flow with None inputs...")
    assert calculate_optical_flow(None, current_gray) is None, "Flow with None prev_frame should be None"
    assert calculate_optical_flow(prev_gray, None) is None, "Flow with None current_frame should be None"
    logging.info("None input tests successful.")

    # Test with mismatched shapes
    logging.info("Testing optical flow with mismatched shapes...")
    different_shape_gray = np.full((100,100), 128, dtype=np.uint8)
    assert calculate_optical_flow(prev_gray, different_shape_gray) is None, "Flow with mismatched shapes should be None"
    logging.info("Mismatched shapes test successful.")

    # Test with color images (should fail or warn as it expects grayscale)
    logging.info("Testing optical flow with color inputs (expecting failure/None)...")
    prev_color = cv2.cvtColor(prev_gray, cv2.COLOR_GRAY2BGR)
    current_color = cv2.cvtColor(current_gray, cv2.COLOR_GRAY2BGR)
    assert calculate_optical_flow(prev_color, current_color) is None, "Flow with color inputs should be None"
    logging.info("Color input test successful (returned None as frames were not 2D).")


    logging.info("AnomalyDetector (Optical Flow) example usage finished.")
