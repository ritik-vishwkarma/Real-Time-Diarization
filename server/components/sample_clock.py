
import numpy as np
import logging
import threading
from typing import Optional, Tuple
from server.dtos import AudioFrame
from server import config

logger = logging.getLogger("SampleClockBuffer")

class DiscontinuityError(Exception):
    pass

class SampleClockBuffer:
    """
    Component 4.2: SampleClockBuffer (Hardened Phase 4)
    
    Responsibility:
    - Bounded ring buffer indexed by sample clock.
    - Thread-safe access.
    - Strict Discontinuity Handling (Raise/Halt).
    - Expiration handling (No zero-padding).
    """
    
    # 50ms at SR
    MAX_GAP_SAMPLES = int(0.05 * config.SAMPLE_RATE)
    
    def __init__(self, capacity_seconds: int = 60, sample_rate: int = config.SAMPLE_RATE):
        if capacity_seconds <= 0:
            raise ValueError("Capacity must be positive")
            
        self.sample_rate = sample_rate
        self.capacity = int(capacity_seconds * sample_rate)
        
        # Buffer Storage
        self.buffer = np.zeros(self.capacity, dtype=np.float32)
        
        # Pointers (Absolute Sample Index)
        self.head = 0 
        
        # Thread Safety
        self._lock = threading.Lock()
        
        # Initial State Flag
        self.initialized = False
        
        logger.info(f"Initialized SampleClockBuffer (Thread-Safe) | Capacity: {capacity_seconds}s")

    @property
    def tail(self) -> int:
        """Oldest available sample."""
        with self._lock:
            return max(0, self.head - self.capacity)

    @property
    def current_head(self) -> int:
        with self._lock:
            return self.head

    def push(self, frame: AudioFrame):
        """
        Push new frame. Thread-safe.
        Strict Continuity Check.
        """
        if frame.num_samples == 0:
            return

        with self._lock:
            if not self.initialized:
                # First frame determines clock start
                self.head = frame.start_sample
                self.initialized = True
                logger.info(f"First frame: Aligned head to {self.head}")
            else:
                # Strict Continuity Check
                expected = self.head
                actual = frame.start_sample
                gap = actual - expected
                
                if gap != 0:
                    # Gap Handling
                    # "Gap > 50ms -> emit DISCONNECT and stop ingestion" (Inferred from prompt)
                    # "Raise RuntimeError" or similar.
                    if abs(gap) > self.MAX_GAP_SAMPLES:
                         # Heavy Violation
                         msg = f"CRITICAL: Discontinuity {gap} samples (> {self.MAX_GAP_SAMPLES}). Session Compromised."
                         logger.critical(msg)
                         raise DiscontinuityError(msg)
                    elif gap > 0:
                         # Small Gap <= 50ms. Inject Silence.
                         # Logic: We must fill buffer from expected to actual with zeros.
                         # Then process frame.
                         # logger.warning(f"Injecting Silence for Gap: {gap} samples")
                         self._inject_silence(expected, gap)
                         # head is now == actual
                    else:
                        # Negative Gap (Overlap/Duplicate? Or Re-ordering?)
                        # "Silent realignment is forbidden."
                        # If we receive old data, we ignore it? Or raise?
                        # Sample indices are monotonic. If simple overlap, we might skip.
                        # But strict rule: "Frames MUST be sample-contiguous".
                        # Raising error is safer.
                        msg = f"Negative Gap Detected (Overlap/Reorder): {gap} samples."
                        logger.error(msg)
                        raise DiscontinuityError(msg)

            # Write Frame (Ring Wrap)
            self._write_chunk(frame.samples)

    def _inject_silence(self, start_idx, count):
        zeros = np.zeros(count, dtype=np.float32)
        self._write_chunk(zeros)

    def _write_chunk(self, samples: np.ndarray):
        count = len(samples)
        start_idx = self.head % self.capacity
        space_end = self.capacity - start_idx
        
        if count <= space_end:
             self.buffer[start_idx : start_idx + count] = samples
        else:
             part1 = space_end
             part2 = count - space_end
             self.buffer[start_idx : ] = samples[:part1]
             self.buffer[0 : part2] = samples[part1:]
             
        self.head += count

    def pop_window(self, start_sample: int, end_sample: int) -> Optional[np.ndarray]:
        """
        Extract window.
        Returns None if ANY part is expired/future.
        """
        if start_sample >= end_sample:
            return np.array([], dtype=np.float32)

        with self._lock:
            buffer_head = self.head
            buffer_tail = max(0, self.head - self.capacity)
            
            # Check availability
            # 1. Lookahead check (Requesting future)
            if end_sample > buffer_head:
                return None
                
            # 2. Expiration Check (Requesting past)
            if start_sample < buffer_tail:
                return None
                
            # Data is strictly available
            num_samples = end_sample - start_sample
            output = np.zeros(num_samples, dtype=np.float32)
            
            start_idx = start_sample % self.capacity
            space_end = self.capacity - start_idx
            
            if num_samples <= space_end:
                output[:] = self.buffer[start_idx : start_idx + num_samples]
            else:
                part1 = space_end
                part2 = num_samples - space_end
                output[:part1] = self.buffer[start_idx:]
                output[part1:] = self.buffer[:part2]
                
            return output

    def get_lookback(self, start_sample: int, end_sample: int) -> Optional[np.ndarray]:
        """Alias for pop_window, used by IdentityService."""
        return self.pop_window(start_sample, end_sample)
