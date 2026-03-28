from collections import deque
import numpy as np

class PoseTemporalTracker:
    def __init__(self, window_size=150):
        self.pitches = deque(maxlen=window_size)

    def update(self, pitch):
        self.pitches.append(pitch)

    def compute_score(self):
        if not self.pitches: return 0.0
        # Detect Sustained Downward Tilt: % of time head is below -15 degrees
        return np.mean([1 if p < -15 else 0 for p in self.pitches])