class FatigueDetector:
    def __init__(self):
        # Settings (Thresholds)
        self.EYE_CLOSED_LIMIT = 10 # Consecutive frames
        self.PITCH_THRESHOLD = -20  # Degrees (Head tilted down)
        self.YAWN_THRESHOLD = 0.7   # Probability

        self.blink_count = 0
        self.last_eye_state = 1 # 1 = Open, 0 = Closed
        
        # History
        self.eye_counter = 0
        self.is_alarm_active = False

    def check_fatigue(self, eye_state, yawn_prob, pitch, yaw):
        alarm_trigger = False
        warning_trigger = False # Added a warning state
        message = "Status: Alert"

        # 1. Update Eye Counter
        if eye_state == 0: 
            self.eye_counter += 1
        else:
            self.eye_counter = 0

        # 2. PRIORITY LOGIC (Check the most dangerous first)
        if self.eye_counter >= self.EYE_CLOSED_LIMIT:
            alarm_trigger = True
            message = "ALARM: EYES CLOSED!"
        
        elif pitch < self.PITCH_THRESHOLD:
            alarm_trigger = True
            message = "ALARM: POSTURE SLOUCH!"

        elif yawn_prob > 0.5: # Lowered threshold to 50% for testing
            warning_trigger = True
            message = "WARNING: YAWN DETECTED"

        return alarm_trigger, warning_trigger, message