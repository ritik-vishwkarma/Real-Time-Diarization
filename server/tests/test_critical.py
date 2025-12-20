
import unittest
import numpy as np
import sys
import os
import threading
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from server.components.sample_clock import SampleClockBuffer, DiscontinuityError
from server.components.overlap_detector import OverlapDetector
from server.dtos import AudioFrame
from server import config

class TestCriticalFixes(unittest.TestCase):

    def test_sample_clock_strict_discontinuity(self):
        """Fix 1: Raise error on large gap, inject on small gap."""
        buf = SampleClockBuffer(capacity_seconds=1, sample_rate=100)
        
        # Initial
        f1 = AudioFrame(np.zeros(10, dtype=np.float32), 0, 10)
        buf.push(f1)
        self.assertEqual(buf.head, 10)
        
        # Small Gap (2 samples) -> Inject
        f2 = AudioFrame(np.zeros(10, dtype=np.float32), 12, 10) # Expected 10, Got 12
        buf.push(f2)
        self.assertEqual(buf.head, 22) # 10 + 2 + 10
        
        # Large Gap (> 5ms at 100hz is >0.5 samples, but threshold is 50ms = 5 samples)
        # Let's verify buffer MAX_GAP
        max_gap = buf.MAX_GAP_SAMPLES # 5 samples
        
        f3 = AudioFrame(np.zeros(10, dtype=np.float32), 22 + max_gap + 2, 10) # Gap 7
        with self.assertRaises(DiscontinuityError):
             buf.push(f3)

    def test_overlap_caching(self):
        """Fix 2: Never emit empty speakers if is_overlap=True."""
        detector = OverlapDetector()
        detector.min_speakers = 2
        detector.threshold = 0.5
        detector.in_overlap_mode = True # Force mode
        detector.last_overlap_speakers = ["spk_A", "spk_B"] # Cache exists
        
        # Input with only 1 speaker (should trigger cache use)
        probs = {"spk_A": 0.9} 
        res = detector.process(probs)
        
        # Logic: If only 1 speaker but in overlap mode, we are in "hold" state?
        # Detector counts 1, so is_instant_overlap = False.
        # Hysteresis checks clean frames.
        # Since it's clean frame #1 (less than clear windows 2), we stay in overlap mode.
        # So we MUST emit cached speakers.
        
        self.assertTrue(res.is_overlap)
        self.assertEqual(set(res.speakers), {"spk_A", "spk_B"})
        
    def test_overlap_no_cache_fallback(self):
        """Verify fallback if cache somehow empty (should use candidates)."""
        detector = OverlapDetector()
        detector.in_overlap_mode = True
        detector.last_overlap_speakers = []
        
        probs = {"spk_A": 0.9}
        res = detector.process(probs)
        
        self.assertTrue(res.is_overlap)
        self.assertEqual(res.speakers, ["spk_A"]) # Fallback to candidates

if __name__ == '__main__':
    unittest.main()
