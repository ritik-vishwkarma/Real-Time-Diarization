import numpy as np
import logging
from typing import Dict, Optional
from server import config

logger = logging.getLogger("ContinuityCache")

class SpeakerContinuityCache:
    """
    Component 4.6: Speaker Continuity Cache (Hardened)
    
    Responsibility:
    - Session-local speaker ID name resolution.
    - Prevents fragmentation by mapping transient Sortformer slot IDs 
      back to previously seen IDs based on embedding similarity.
    - Mental Model: A name-resolver, not a decision-maker.
    """
    def __init__(self):
        # map: known_id -> {"centroid": np.ndarray, "last_seen_sample": int, "count": int}
        self.speaker_map = {}
        # map: sortformer_slot_id -> resolved_id (Valid for current continuous speech)
        self.slot_map = {} 

    def process(self, probs: Dict[str, float], embedding: Optional[np.ndarray], 
                current_sample: int, is_overlap: bool, is_continuity: bool) -> Dict[str, float]:
        
        # 1. Prune expired speakers
        max_gap = config.CACHE_MAX_GAP_SECONDS * config.SAMPLE_RATE
        expired = [sid for sid, meta in self.speaker_map.items() 
                   if current_sample - meta["last_seen_sample"] > max_gap]
        for sid in expired:
            del self.speaker_map[sid]
            # Also clean slot map if resolved_id is expired
            self.slot_map = {k: v for k, v in self.slot_map.items() if v != sid}

        if not probs:
            # self.slot_map.clear() # Reset slots on silence
            return {}

        # 2. Name Resolution Logic
        final_probs = {}
        
        # Case A: Clean Single Speaker -> Update Centroids & Map
        if not is_continuity and not is_overlap and embedding is not None and len(probs) == 1:
            slot_id = list(probs.keys())[0]
            prob = probs[slot_id]

            if slot_id in self.slot_map:
                resolved_id = self.slot_map[slot_id]
                final_probs[resolved_id] = prob
                return final_probs  

            if prob >= config.START_THRESHOLD:
                best_match = None
                best_sim = -1.0
                
                # Check cosine similarity against cached centroids
                for cached_id, meta in self.speaker_map.items():
                    sim = np.dot(embedding, meta["centroid"])
                    if sim > best_sim:
                        best_sim = sim
                        best_match = cached_id

                resolved_id = None
                # Sub-Case A1: Match Found
                if best_match and best_sim >= config.CACHE_SIMILARITY_THRESHOLD:
                    resolved_id = best_match
                    logger.info(f"Resolved: {slot_id} -> {best_match} (sim={best_sim:.2f})")
                
                # Sub-Case A2: No Match -> Register Slot (First Appearance)
                else:
                    resolved_id = slot_id
                    # resolved_id = f"person_{slot_id}_{current_sample}" 
                    logger.info(f"🆕 New Speaker Detected: {slot_id} -> {resolved_id}")
                
                # Update State
                self._update_speaker(resolved_id, embedding, current_sample)
                self.slot_map[slot_id] = resolved_id # Update active slot mapping
                final_probs[resolved_id] = prob
                return final_probs

        # Case B: Overlap or Ambiguous -> Use Slot Map (Best Effort)
        # We rely on previous single-speaker frames to have populated self.slot_map
        for slot_id, prob in probs.items():
            if slot_id in self.slot_map:
                resolved_id = self.slot_map[slot_id]
                # During overlap, we do NOT update centroids/timestamps
                # We just apply the name
                final_probs[resolved_id] = prob
            else:
                # Fallback: If we really don't know who this slot is, we pass raw slot.
                # Ideally this shouldn't happen deep in a session if they spoke alone before.
                # But for safety, we pass raw slot to avoid dropping audio.
                # Downstream might see "0" temporarily.
                final_probs[slot_id] = prob
        
        return final_probs

    def _update_speaker(self, sid: str, embedding: np.ndarray, sample: int):
        """Internal helper to update centroids and temporal authority."""
        if sid not in self.speaker_map:
            self.speaker_map[sid] = {"centroid": embedding, "last_seen_sample": sample, "count": 1}
        else:
            meta = self.speaker_map[sid]
            cnt = meta["count"]
            # Weighted moving average for centroid stability
            meta["centroid"] = (meta["centroid"] * cnt + embedding) / (cnt + 1)
            norm = np.linalg.norm(meta["centroid"])
            if norm > 1e-9:
                meta["centroid"] /= norm
            
            meta["last_seen_sample"] = sample
            meta["count"] += 1