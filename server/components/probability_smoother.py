
import numpy as np
from collections import deque
from server.dtos import FrameDecision

class ProbabilitySmoother:
    """
    Component 4.6: ProbabilitySmoother (Hardened Phase 3)
    
    Responsibility:
    - Median Aggregation.
    - Fixed window count (not float timestamps).
    - MUST IGNORE frames not in state == "speech"
    """
    # config.FRAME_MS = 20ms
    # 300ms max latency = 15 frames
    WINDOW_SIZE_FRAMES = 15
    
    def __init__(self, window_size: int = WINDOW_SIZE_FRAMES):
        self.window_size = window_size
        self.buffer = deque(maxlen=window_size)

    def add_frame(self, decision: FrameDecision) -> dict:
        """
        Input: FrameDecision
        Output: Smoothed Probs (only if speech, else empty)
        """
        # 1. Add (ONLY IF SPEECH)
        # Spec: "MUST IGNORE frames not in state == speech"
        if decision.state == "speech":
            self.buffer.append(decision.probs)
            
        # 2. Output
        if decision.state != "speech":
            return None
            
        if not self.buffer:
             return {}

        # Aggregate
        all_keys = set()
        for p in self.buffer:
            all_keys.update(p.keys())
            
        smoothed = {}
        for k in all_keys:
            vals = [p.get(k, 0.0) for p in self.buffer]
            med = np.median(vals)
            if med > 0.001:
                smoothed[k] = med
                
        return smoothed
