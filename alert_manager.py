import logging
import time
import datetime # For formatting timestamp

# Configure logging for the module if needed, or rely on main's config
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(module)s - %(levelname)s - %(message)s')

class Alert:
    '''A simple class to structure alert data.'''
    def __init__(self, event_type, confidence=1.0, person_id=None, camera_id=None, details=None, timestamp=None):
        self.timestamp = timestamp if timestamp is not None else time.time()
        self.event_type = event_type
        self.confidence = confidence
        self.person_id = person_id
        self.camera_id = camera_id
        self.details = details if details is not None else {}

    def __str__(self):
        ts_formatted = datetime.datetime.fromtimestamp(self.timestamp).strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        return (f"ALERT @ {ts_formatted} | Event: {self.event_type} | "
                f"Confidence: {self.confidence:.2f} | "
                f"Person ID: {self.person_id if self.person_id is not None else 'N/A'} | "
                f"Camera ID: {self.camera_id if self.camera_id is not None else 'N/A'} | "
                f"Details: {self.details}")

class AlertManager:
    def __init__(self, max_history_size=1000):
        '''
        Initializes the AlertManager.
        :param max_history_size: Maximum number of alerts to keep in history.
        '''
        self.alerts_history = []
        self.max_history_size = max_history_size
        # Using a dedicated logger for alerts might be good for routing them differently later
        self.alert_logger = logging.getLogger('AlertManager')
        # Ensure alert_logger has a handler if not configured globally or by main.
        # For simplicity, assume main's basicConfig covers it.
        # If running this file standalone, a basicConfig in __main__ will apply.
        logging.info("AlertManager initialized.")


    def generate_alert(self, event_type, confidence=1.0, person_id=None,
                       camera_id=None, details=None, timestamp=None):
        '''
        Generates, logs, and stores an alert.
        :param event_type: Type of the event (e.g., 'SMOKING_GESTURE', 'CLOTHING_CHANGE').
        :param confidence: Confidence of the detection (0.0 to 1.0).
        :param person_id: ID of the person involved, if applicable.
        :param camera_id: ID of the camera that caught the event.
        :param details: Additional dictionary with event-specific details.
        :param timestamp: Timestamp of the event. Defaults to current time if None.
        '''
        alert = Alert(
            event_type=event_type,
            confidence=confidence,
            person_id=person_id,
            camera_id=camera_id,
            details=details,
            timestamp=timestamp
        )

        # Log the alert using a specific logger level (e.g., WARNING or ERROR)
        self.alert_logger.warning(str(alert)) # Using .warning for visibility in logs

        # Store in history
        if self.max_history_size > 0: # Only manage history if size is positive
            if len(self.alerts_history) >= self.max_history_size:
                self.alerts_history.pop(0) # Remove the oldest alert to maintain size
            self.alerts_history.append(alert)

        return alert # Return the created alert object

    def get_recent_alerts(self, count=10):
        '''
        Retrieves the most recent 'count' alerts.
        :param count: Number of recent alerts to retrieve.
        :return: List of Alert objects.
        '''
        if count <= 0:
            return []
        return self.alerts_history[-count:]

    def get_all_alerts(self):
        '''Retrieves all alerts currently in history.'''
        return list(self.alerts_history)


if __name__ == '__main__':
    # Configure basic logging for the example if this file is run directly
    # This ensures that the AlertManager's logger output is visible.
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    # Set AlertManager's logger to also show WARNING level for its specific logs if not already captured by root.
    logging.getLogger('AlertManager').setLevel(logging.WARNING)


    logging.info("Starting AlertManager example usage...")
    alert_manager = AlertManager(max_history_size=5)

    # Generate some sample alerts
    alert1_details = {'hand': 'left', 'gesture_duration_s': 2.5}
    alert1 = alert_manager.generate_alert(
        event_type="SMOKING_GESTURE_DETECTED",
        confidence=0.85,
        person_id="person_001",
        camera_id="cam_01",
        details=alert1_details
    )
    time.sleep(0.01) # Ensure timestamps are slightly different for ordering
    alert2_details = {'previous_features_id': 'feat_abc', 'current_features_id': 'feat_xyz'}
    alert2 = alert_manager.generate_alert(
        event_type="CLOTHING_CHANGE_DETECTED",
        confidence=0.92,
        person_id="person_002",
        camera_id="cam_02",
        details=alert2_details
    )
    time.sleep(0.01)
    alert_manager.generate_alert("LOITERING_DETECTED_PLACEHOLDER", person_id="person_003", camera_id="cam_01", confidence=0.7)
    time.sleep(0.01)
    alert_manager.generate_alert("CROWD_FORMATION_PLACEHOLDER", camera_id="cam_01", details={'num_persons': 7})
    time.sleep(0.01)
    alert_manager.generate_alert("HIGH_MOTION_DETECTED", camera_id="cam_03", details={'avg_flow_magnitude': 25.5})

    logging.info(f"Total alerts in history before overflow: {len(alert_manager.alerts_history)}")
    # Test max history size
    alert6 = alert_manager.generate_alert("TEST_ALERT_OVERFLOW", camera_id="cam_04")

    logging.info(f"Total alerts in history after potential overflow: {len(alert_manager.alerts_history)}")
    assert len(alert_manager.alerts_history) <= alert_manager.max_history_size, "Alert history exceeded max size"

    if alert_manager.max_history_size > 0 and len(alert_manager.alerts_history) > 0:
        # If max_history_size is 5, alert1 should have been popped.
        # The first alert in history should now be alert2 if overflow occurred as expected.
        first_alert_in_history = alert_manager.alerts_history[0]
        if alert_manager.max_history_size == 5: # Specific check for this test case
             assert first_alert_in_history.event_type == alert2.event_type,                 f"Oldest alert not removed correctly. Expected {alert2.event_type}, got {first_alert_in_history.event_type}"

    logging.info("\nRecent Alerts (last 3):")
    recent = alert_manager.get_recent_alerts(count=3)
    for alert_obj in recent:
        logging.info(f"  - {str(alert_obj)}")
    assert len(recent) <= 3

    logging.info("\nAll Alerts in History:")
    all_alerts = alert_manager.get_all_alerts()
    for alert_obj in all_alerts:
        logging.info(f"  - {str(alert_obj)}")

    if alert_manager.max_history_size > 0 and len(all_alerts) > 0:
        assert all_alerts[-1].event_type == "TEST_ALERT_OVERFLOW"

    # Test with max_history_size = 0 (no history)
    logging.info("\nTesting with max_history_size = 0")
    alert_manager_no_history = AlertManager(max_history_size=0)
    alert_manager_no_history.generate_alert("NO_HISTORY_TEST", confidence=1.0)
    assert len(alert_manager_no_history.alerts_history) == 0, "History should be empty if max_history_size is 0"
    logging.info("Test with max_history_size = 0 successful.")

    logging.info("AlertManager example usage finished.")
