class FatigueDetector:
    def __init__(self):
        # Balanced weights for temporal indicators
        self.W_EYE = 0.5    # PERCLOS (Temporal)
        self.W_YAWN = 0.25   # PYAWN (Temporal)
        self.W_POSE = 0.25   # PPOSE (Temporal)
    
    def get_fusion_status(self, perclos, t_yawn, t_pose):
        # Fusion is now based on 3 stabilized temporal scores
        total_score = (perclos * self.W_EYE) + (t_yawn * self.W_YAWN) + (t_pose * self.W_POSE)
        
        # Consistent Thresholds
        if total_score > 0.50: # Sustained critical fatigue across indicators
            return total_score, "ALARM: CRITICAL", (0, 0, 255)
        elif total_score > 0.20: # Sustained mild fatigue
            return total_score, "WARNING: DROWSY", (0, 165, 255)
        
        return total_score, "STATUS: ALERT", (0, 255, 0)