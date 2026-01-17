from collections import deque
import numpy as np

class EyeTemporalTracker:
    def __init__(self, window_size=30):
        self.window_size = window_size
        self.eye_states = deque(maxlen=window_size)

    def update(self, eye_state):
        """
        eye_state: 0 (OPEN) or 1 (CLOSED)
        """
        self.eye_states.append(eye_state)

    def compute_perclos(self):
        if len(self.eye_states) == 0:
            return 0.0
        return np.mean(self.eye_states)

    def fatigue_level(self):
        perclos = self.compute_perclos()

        if perclos < 0.2:
            return "ALERT", perclos
        elif perclos < 0.4:
            return "MILD FATIGUE", perclos
        else:
            return "HIGH FATIGUE", perclos
