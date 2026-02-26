from eye_temporal_logic import EyeTemporalTracker

tracker = EyeTemporalTracker(window_size=10)

# Simulated eye states
# 0 = OPEN, 1 = CLOSED
test_sequence = [0, 0, 1, 1, 1, 1, 1, 0, 1, 1]

for i, state in enumerate(test_sequence):
    tracker.update(state)
    level, perclos = tracker.fatigue_level()
    print(f"Frame {i+1}: Eye={state}, PERCLOS={perclos:.2f}, State={level}")
