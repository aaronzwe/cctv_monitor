import logging
import time
import os

from video_stream import VideoStream
from person_detector import PersonDetector

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
PRIMARY_VIDEO_SOURCE = "rtsp://example.com/media/stream1" # Replace with actual stream or leave for fallback
FALLBACK_VIDEO_SOURCE = "test_video.mp4" # Assumed to be in the same directory or accessible path
MAX_FRAMES_TO_PROCESS = 100 # Limit the number of frames for this example
PROCESSING_LOOP_DURATION_SECONDS = 30 # Or limit by time

def main():
    logger.info("Starting main application.")

    video_source_to_use = PRIMARY_VIDEO_SOURCE

    # Check if primary source is a file and exists, or if it's a network stream
    is_primary_file = not (PRIMARY_VIDEO_SOURCE.startswith("rtsp://") or PRIMARY_VIDEO_SOURCE.startswith("rtmp://"))

    if is_primary_file and not os.path.exists(PRIMARY_VIDEO_SOURCE):
        logger.warning(f"Primary video source '{PRIMARY_VIDEO_SOURCE}' not found or not a stream URL.")
        if os.path.exists(FALLBACK_VIDEO_SOURCE):
            logger.info(f"Using fallback video source: '{FALLBACK_VIDEO_SOURCE}'")
            video_source_to_use = FALLBACK_VIDEO_SOURCE
        else:
            logger.error(f"Fallback video source '{FALLBACK_VIDEO_SOURCE}' also not found. Exiting.")
            # Before exiting, let's try to create a dummy test_video.mp4 for demonstration if cv2 is available
            # This part is more for ensuring the example can run in a test environment
            try:
                import cv2
                import numpy as np
                logger.info(f"Attempting to create dummy video: {FALLBACK_VIDEO_SOURCE}")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out_dummy = cv2.VideoWriter(FALLBACK_VIDEO_SOURCE, fourcc, 10, (640,480))
                for i in range(50): # 5 seconds of video at 10fps
                    frame = np.zeros((480,640,3), dtype=np.uint8)
                    cv2.putText(frame, f"Dummy Frame {i+1}", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
                    out_dummy.write(frame)
                out_dummy.release()
                logger.info(f"Dummy video '{FALLBACK_VIDEO_SOURCE}' created successfully.")
                video_source_to_use = FALLBACK_VIDEO_SOURCE
            except Exception as e:
                logger.error(f"Failed to create dummy video '{FALLBACK_VIDEO_SOURCE}': {e}. Please ensure a video file is available. Exiting.")
                return
    elif not is_primary_file:
         logger.info(f"Using primary video source (network stream): {PRIMARY_VIDEO_SOURCE}")
    else:
        logger.info(f"Using primary video source (local file): {PRIMARY_VIDEO_SOURCE}")


    # Initialize VideoStream
    logger.info(f"Initializing video stream from: {video_source_to_use}")
    video_stream = VideoStream(stream_url=video_source_to_use, buffer_size=60, stream_name="main_feed")

    # Initialize PersonDetector
    logger.info("Initializing PersonDetector...")
    try:
        person_detector = PersonDetector(model_path='yolov8n.pt') # yolov8n.pt is small and good for CPU
    except Exception as e:
        logger.error(f"Failed to initialize PersonDetector: {e}. Exiting.")
        return

    # Connect to stream
    if not video_stream.connect():
        logger.error("Failed to connect to video stream. Exiting.")
        return

    logger.info("Successfully connected to video stream and initialized detector.")

    frames_processed = 0
    start_time = time.time()

    try:
        while True:
            # Check for processing limits
            if frames_processed >= MAX_FRAMES_TO_PROCESS:
                logger.info(f"Reached maximum frames to process ({MAX_FRAMES_TO_PROCESS}).")
                break
            if time.time() - start_time >= PROCESSING_LOOP_DURATION_SECONDS:
                logger.info(f"Reached maximum processing time ({PROCESSING_LOOP_DURATION_SECONDS} seconds).")
                break

            frame = video_stream.read_frame(timeout=1.0) # Read frame with 1s timeout

            if frame is None:
                if not video_stream.is_opened() and video_stream.buffer.empty():
                    logger.info("Video stream seems to have ended or buffer is empty and stream closed.")
                    break
                logger.debug("No frame currently available from stream, continuing...")
                time.sleep(0.1) # Wait a bit if no frame
                continue

            frames_processed += 1

            # Perform person detection
            detected_persons = person_detector.detect_persons(frame, confidence_threshold=0.4)

            if detected_persons:
                logger.info(f"Frame {frames_processed}: Detected {len(detected_persons)} persons.")
                # For example, log details of first detected person
                # person1 = detected_persons[0]
                # logger.debug(f"  Person 1: BBox={person1['bbox']}, Confidence={person1['confidence']:.2f}")
            else:
                logger.info(f"Frame {frames_processed}: No persons detected.")

            # Optional: Display frame (requires OpenCV GUI support, may not work in all environments)
            # cv2.imshow('Live Feed with Detections', frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     logger.info("Quit signal received from OpenCV window.")
            #     break

            # Small delay to simulate other processing or to control processing speed
            # time.sleep(0.05)

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Shutting down.")
    except Exception as e:
        logger.error(f"An unexpected error occurred during processing: {e}", exc_info=True)
    finally:
        logger.info("Stopping video stream...")
        video_stream.stop()
        # if cv2 is imported and windows were opened:
        # try:
        #   if 'cv2' in globals() and cv2.getWindowProperty('Live Feed with Detections', 0) >= 0: # Check if window exists
        #     cv2.destroyAllWindows()
        # except Exception: pass
        logger.info("Main application finished.")

if __name__ == '__main__':
    main()
