from collections import deque
import numpy as np

# Requirement 1 & 2: Temporal Modeling Classes
class YawnTemporalTracker:
    def __init__(self, window_size=150): # ~5-6 seconds at 25-30fps
        self.probs = deque(maxlen=window_size)

    def update(self, prob):
        self.probs.append(prob)

    def compute_score(self):
        if not self.probs: return 0.0
        # Calculate Frequency: % of frames in window where probability > 0.5
        # This converts frame-level 'guesses' into a stable 'state'
        return np.mean([1 if p > 0.5 else 0 for p in self.probs])