
import time
import logging
from server.dtos import WindowExecutionPlan
import config

logger = logging.getLogger("SlidingWindowProcessor")

class SlidingWindowProcessor:
    """
    Component 4.3: SlidingWindowProcessor (Hardened Phase 3)
    
    Responsibility:
    - Owns Window Progression.
    - Decides execution mode based on Lag.
    
    Rules:
    - Time ALWAYS advances.
    - Lag > Threshold (Samples) -> exec_mode="skip"
    - Pure sample math. No seconds.
    """
    
    # Lag Threshold in Samples
    # 1.0 second * SR
    LAG_THRESHOLD_SAMPLES = int(3.0 * config.SAMPLE_RATE)
    
    def __init__(self):
        self.chunk_samples = config.CHUNK_FRAMES * int(config.SAMPLE_RATE * config.FRAME_MS)
        self.step_samples = int(self.chunk_samples * 0.5) # 50% Overlap
        self.next_window_start = 0

    def get_pending_windows(self, buffer_head: int, ingest_head: int) -> list[WindowExecutionPlan]:
        """
        Generate execution plans up to buffer_head.
        """
        plans = []
        
        # While we can form a full window within available data
        while self.next_window_start + self.chunk_samples <= buffer_head:
            start = self.next_window_start
            end = start + self.chunk_samples
            
            # Lag Detection (Pure Sample Math)
            # buffer_head is "Latest Ingested Sample".
            # end is "Window End Sample".
            # Lag = Head - WindowEnd
            # lag_samples = buffer_head - end
            lag_samples = ingest_head - end
            
            # Fast Mode Decision
            mode = "infer"
            if lag_samples > self.LAG_THRESHOLD_SAMPLES:
                mode = "skip"
                # logger.warning(f"Fast Mode: Skipping Window (Lag {lag_samples} samples)")
            
            plans.append(WindowExecutionPlan(start, end, mode))
            
            self.next_window_start += self.step_samples
            
        return plans
