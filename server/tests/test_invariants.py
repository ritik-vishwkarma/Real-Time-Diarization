
import unittest
import numpy as np
import sys
import os
import threading
import time

# Add root directory (parent of server) to path so 'server.types' works
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from server.components.sample_clock import SampleClockBuffer
from server.dtos import AudioFrame
from server import config

class TestHardenedInvariants(unittest.TestCase):

    def test_thread_safety(self):
        """Verify threaded push/pop doesn't crash or corrupt."""
        buf = SampleClockBuffer(capacity_seconds=1, sample_rate=100)
        
        stop = False
        def writer():
            idx = 0
            while not stop:
                f = AudioFrame(np.zeros(10, dtype=np.float32), idx, 10)
                buf.push(f)
                idx += 10
                time.sleep(0.001)
                
        def reader():
            read_count = 0
            while not stop:
                # Try reading random windows
                curr = buf.current_head
                if curr > 20:
                     w = buf.pop_window(curr-20, curr-10)
                     if w is not None: read_count += 1
                time.sleep(0.001)
                
        t1 = threading.Thread(target=writer)
        t2 = threading.Thread(target=reader)
        
        t1.start()
        t2.start()
        
        time.sleep(0.5)
        stop = True
        t1.join()
        t2.join()
        
        self.assertGreater(buf.head, 0)
        
    def test_expiration_returns_none(self):
        """Expired data must return None (Unknown state implication)."""
        buf = SampleClockBuffer(capacity_seconds=0.1, sample_rate=100) # 10 samples cap
        
        # Push 20 samples (Overwriting first 10)
        f1 = AudioFrame(np.zeros(20, dtype=np.float32), 0, 20)
        buf.push(f1)
        
        # Request 0-10 (Expired)
        res = buf.pop_window(0, 10)
        self.assertIsNone(res)
        
        # Request 10-20 (Valid)
        res2 = buf.pop_window(10, 20)
        self.assertIsNotNone(res2)

if __name__ == '__main__':
    unittest.main()
