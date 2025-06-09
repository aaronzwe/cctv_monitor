import cv2
import logging
import queue
import threading
import time
import os # For checking file existence

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class VideoStream:
    def __init__(self, stream_url, buffer_size=128, stream_name="stream"):
        self.stream_url = stream_url
        self.stream_name = stream_name # Added for better logging
        self.cap = None
        self.is_running = False
        self.buffer = queue.Queue(maxsize=buffer_size)
        self.reader_thread = None
        self.buffer_size = buffer_size

    def _reader(self):
        logging.info(f"Reader thread started for {self.stream_name} ({self.stream_url}).")
        while self.is_running and self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                logging.warning(f"Failed to read frame from {self.stream_name} ({self.stream_url}). Stopping reader.")
                break

            try:
                self.buffer.put(frame, block=True, timeout=0.5) # Wait 0.5s if buffer is full
            except queue.Full:
                logging.warning(f"Buffer full for {self.stream_name} ({self.stream_url}). Frame dropped.")
                time.sleep(0.01) # Brief pause

        logging.info(f"Reader thread for {self.stream_name} ({self.stream_url}) finished.")

    def connect(self):
        try:
            self.cap = cv2.VideoCapture(self.stream_url)
            if not self.cap.isOpened():
                logging.error(f"Error opening video stream: {self.stream_name} ({self.stream_url})")
                self.cap = None
                return False

            self.is_running = True
            self.reader_thread = threading.Thread(target=self._reader, daemon=True, name=f"Reader_{self.stream_name}")
            self.reader_thread.start()
            logging.info(f"Successfully connected to {self.stream_name} ({self.stream_url}) and started reader thread.")
            return True
        except Exception as e:
            logging.error(f"Exception while connecting to {self.stream_name} ({self.stream_url}): {e}")
            self.cap = None
            return False

    def read_frame(self, timeout=1):
        if not self.is_running and self.buffer.empty():
            return None

        try:
            return self.buffer.get(block=True, timeout=timeout)
        except queue.Empty:
            return None

    def stop(self):
        logging.info(f"Stopping stream: {self.stream_name} ({self.stream_url})")
        self.is_running = False

        if self.reader_thread is not None and self.reader_thread.is_alive():
            self.reader_thread.join(timeout=2)
            if self.reader_thread.is_alive():
                logging.warning(f"Reader thread for {self.stream_name} ({self.stream_url}) did not terminate gracefully.")

        if self.cap is not None:
            self.cap.release()
            logging.info(f"Released video capture for {self.stream_name} ({self.stream_url})")

        while not self.buffer.empty():
            try:
                self.buffer.get_nowait()
            except queue.Empty:
                break
        logging.info(f"Buffer cleared for {self.stream_name} ({self.stream_url}).")
        self.cap = None
        self.reader_thread = None

    def is_opened(self):
        return self.is_running and self.cap is not None and self.cap.isOpened()

if __name__ == '__main__':
    # Example Usage for Multi-Threading with Concurrent Camera Feeds

    # For testing, we'll use local video files. Ensure these files exist or modify paths.
    # It's good practice to use different video files if possible to simulate different streams.
    # If only one test video is available, we can reuse it.

    # video_file_path1 = "test_video1.mp4" # Needs to exist
    # video_file_path2 = "test_video2.mp4" # Needs to exist
    # video_file_path3 = "test_video.mp4" # General fallback

    # Let's assume a single 'test_video.mp4' for simplicity in this example.
    # In a real scenario, these would be unique RTSP/RTMP URLs or video file paths.
    test_video_file = "test_video.mp4"

    # Check if the test video file exists. If not, this example won't run properly.
    # The subtask environment should ideally provide such a file for testing.
    if not os.path.exists(test_video_file):
        logging.warning(f"Test video file '{test_video_file}' not found. Multi-stream example may not function as expected.")
        # Attempt to create a dummy video if opencv can write it - this is often problematic in restricted envs
        # For now, we'll just log a warning.
        # try:
        #     logging.info(f"Attempting to create dummy video: {test_video_file}")
        #     fourcc = cv2.VideoWriter_fourcc(*'XVID') # or 'mp4v'
        #     out_dummy = cv2.VideoWriter(test_video_file, fourcc, 20.0, (640,480))
        #     for _ in range(60): # 3 seconds of video
        #         frame = np.zeros((480,640,3), dtype=np.uint8)
        #         cv2.putText(frame, f"Frame {_}", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        #         out_dummy.write(frame)
        #     out_dummy.release()
        #     logging.info(f"Dummy video '{test_video_file}' created.")
        # except Exception as e:
        #     logging.error(f"Failed to create dummy video '{test_video_file}': {e}. Please ensure a video file is available.")
        #     # Exit if no video file, as the rest of the example depends on it.
        #     exit()


    stream_urls = {
        "camera1": test_video_file,
        "camera2": test_video_file, # Using the same file for simplicity
        # "camera3": "rtsp://another.stream.url" # Example for a real stream
    }

    streams = {}
    processing_threads = []

    # Connect to all streams
    for name, url in stream_urls.items():
        if not os.path.exists(url) and not (url.startswith("rtsp://") or url.startswith("rtmp://")):
            logging.warning(f"Video source {url} for {name} not found and not a network stream. Skipping.")
            continue
        stream = VideoStream(url, buffer_size=60, stream_name=name)
        if stream.connect():
            streams[name] = stream
            logging.info(f"Successfully initiated connection for {name}.")
        else:
            logging.error(f"Failed to connect to {name} at {url}.")

    if not streams:
        logging.error("No streams were successfully connected. Exiting multi-stream example.")
        exit()

    # Main loop to process frames from streams
    # This part would typically be more complex, e.g., feeding frames to a detection model
    run_duration = 10  # seconds
    start_time = time.time()
    frames_processed_count = {name: 0 for name in streams.keys()}

    logging.info(f"Starting to process frames from {len(streams)} streams for {run_duration} seconds...")

    try:
        while time.time() - start_time < run_duration:
            for name, stream_instance in streams.items():
                if not stream_instance.is_opened() and stream_instance.buffer.empty():
                    logging.info(f"Stream {name} seems closed and buffer is empty. Skipping.")
                    continue

                frame = stream_instance.read_frame(timeout=0.01) # Short timeout for non-blocking feel
                if frame is not None:
                    frames_processed_count[name] += 1
                    # In a real application, you would do something with the frame here
                    # For example, pass it to a processing queue or a detection model
                    if frames_processed_count[name] % 30 == 0: # Log every 30 frames per stream
                        logging.info(f"Processed frame {frames_processed_count[name]} from {name}, buffer: {stream_instance.buffer.qsize()}")
                # else:
                #     logging.debug(f"No frame from {name} currently, buffer: {stream_instance.buffer.qsize()}")

            time.sleep(0.01) # Overall loop processing delay

    except KeyboardInterrupt:
        logging.info("Keyboard interrupt received. Stopping streams.")
    finally:
        logging.info("Stopping all streams...")
        for name, stream_instance in streams.items():
            stream_instance.stop()

        for name, count in frames_processed_count.items():
            logging.info(f"Total frames processed from {name}: {count}")

    logging.info("Multi-stream example finished.")
