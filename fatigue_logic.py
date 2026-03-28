class FatigueDetector:
    def __init__(self):
        # Weighted importance for each indicator
        self.W_EYE = 0.5    # PERCLOS is the strongest indicator
        self.W_YAWN = 0.3
        self.W_POSE = 0.2
        
        self.last_eye_state = 1 # Assume Open
    
    def get_fusion_status(self, perclos, yawn_prob, pitch):
        # 1. Normalize Posture (Score of 1.0 if slouching < -15 degrees)
        # We lowered this from -20 to -15 to be more sensitive
        pose_score = min(1.0, max(0.0, abs(pitch) / 15.0)) if pitch < -5 else 0.0
        
        # 2. Calculate Weighted Fatigue Score
        total_score = (perclos * self.W_EYE) + (yawn_prob * self.W_YAWN) + (pose_score * self.W_POSE)
        
        # 3. LOWERED THRESHOLDS for easier triggering during Demo
        # If score > 0.45 -> ALARM
        # If score > 0.20 -> WARNING
        if total_score > 0.65: 
            return total_score, True, False, "ALARM: CRITICAL", (0, 0, 255)
        elif total_score > 0.30: 
            return total_score, False, True, "WARNING: DROWSY", (0, 165, 255)
        
        return total_score, False, False, "STATUS: ALERT", (0, 255, 0)
