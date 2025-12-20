import numpy as np
import threading
import logging
from typing import List, Tuple, Optional
import config

logger = logging.getLogger("EmbeddingCache")

class EmbeddingCache:
    """
    Component 4.10: EmbeddingCache (Hardened)
    
    Responsibility:
    - Session-local caching of speaker embeddings.
    - Speaker-aware: Prevents averaging embeddings from different people.
    - Deterministic: Only containment logic (s >= start and e <= end).
    - TTL Enforced: Old entries expire based on sample clock.
    """
    def __init__(self, max_entries=5000):
        # Entry format: (start_sample, end_sample, speaker_id, embedding)
        self.cache: List[Tuple[int, int, str, np.ndarray]] = []
        self.max_entries = max_entries
        self.lock = threading.Lock()

    def add(self, start: int, end: int, speaker_id: str, embedding: np.ndarray):
        with self.lock:
            self.cache.append((start, end, speaker_id, embedding))
            if len(self.cache) > self.max_entries:
                self.cache.pop(0)

    def query(self, start_sample: int, end_sample: int, speaker_id: str, current_sample: int) -> List[np.ndarray]:
        """
        Return embeddings for a specific speaker that are fully contained 
        within the segment and have not expired based on TTL.
        """
        max_age_samples = int(config.EMBEDDING_CACHE_TTL_SEC * config.SAMPLE_RATE) # TTL Enforcement
        
        with self.lock:
            results = []
            for s, e, sid, emb in self.cache:
                # 1. Speaker ID Match
                if sid != speaker_id:
                    continue
                # 2. Deterministic Containment (Fix 6)
                if not (s >= start_sample and e <= end_sample):
                    continue
                # 3. TTL Check (Fix 4)
                if current_sample - e > max_age_samples:
                    continue
                
                results.append(emb)
            return results

    def clear(self):
        with self.lock:
            self.cache.clear()
            logger.info("Embedding cache cleared.") # Fix 5