import logging
from dataclasses import dataclass
from typing import Dict, List, Optional
import config

logger = logging.getLogger("OverlapDetector")


@dataclass
class OverlapResult:
    is_overlap: bool
    speakers: List[str]


class OverlapDetector:
    """
    Component 4.7: OverlapDetector (Robust Real-Time)

    Responsibility:
    - Detect multi-speaker activity with high precision.
    - Suppress transient noise (Hysteresis).
    - Limit to Top-2 speakers (Human conversation constraint).
    - Safe state management (No runtime errors).
    """

    def __init__(self):
        self.threshold = config.OVERLAP_THRESHOLD
        self.min_speakers = config.MIN_OVERLAP_SPEAKERS
        
        # Hysteresis Configuration
        self.persist_required = config.OVERLAP_PERSISTENCE
        self.clear_required = config.OVERLAP_PERSISTENCE

        # State
        self._overlap_counter = 0     # Counts consecutive overlap frames
        self._clean_counter = 0       # Counts consecutive single-speaker frames
        self._in_overlap_mode = False # Current state
        self._cached_speakers: List[str] = [] # Last confirmed overlap set

    def process(self, probs: Dict[str, float]) -> OverlapResult:
        """
        Input: smoothed probabilities
        Output: OverlapResult (is_overlap, speakers)
        """
        # 1. Candidate Selection
        # Filter by threshold and sort by probability (descending)
        candidates = [k for k, v in probs.items() if v >= self.threshold]
        candidates.sort(key=lambda k: probs[k], reverse=True)

        # 2. Instant Overlap Check
        # We strictly limit to Top-2 distinct speakers for intelligibility
        active_count = len(candidates)
        instant_overlap = active_count >= self.min_speakers

        if instant_overlap:
            # Overlap Signal Detected
            self._overlap_counter += 1
            self._clean_counter = 0

            # Enter Overlap Mode if Persistence Met
            if self._overlap_counter >= self.persist_required:
                self._in_overlap_mode = True
                # Cache the Top-2 speakers
                self._cached_speakers = candidates[:2]

        else:
            # Single Speaker / Silence
            self._clean_counter += 1
            self._overlap_counter = 0

            # Exit Overlap Mode if Clear Persistence Met
            if self._clean_counter >= self.clear_required:
                self._in_overlap_mode = False
                self._cached_speakers = []

        # 3. Output Construction
        if self._in_overlap_mode:
            # SAFETY: If cache is empty (should not happen if logic flows), 
            # fallback to current candidates or fail gracefully to standard speech.
            if not self._cached_speakers:
                if instant_overlap:
                    return OverlapResult(is_overlap=True, speakers=candidates[:2])
                else:
                    # Weird state: in_overlap but no cache and no instant -> Force Exit
                    self._in_overlap_mode = False
                    return OverlapResult(is_overlap=False, speakers=[])
            
            return OverlapResult(
                is_overlap=True,
                speakers=self._cached_speakers
            )

        return OverlapResult(
            is_overlap=False,
            speakers=[]
        )


# import logging
# from dataclasses import dataclass, field

# from typing import List, Dict
# from server import config

# logger = logging.getLogger("OverlapDetector")

# @dataclass
# class OverlapResult:
#     is_overlap: bool
#     speakers: List[str] = field(default_factory=list)

# class OverlapDetector:
#     """
#     Component 4.7: OverlapDetector (Hardened Phase 4)
    
#     Responsibility:
#     - Detect multi-speaker activity.
#     - Hysteresis Logic.
#     - PREVENT EMPTY OVERLAP: Cache last valid set.
#     """
    
#     OVERLAP_CLEAR_WINDOWS = 2
#     MIN_PERSISTENCE = 2
    
#     def __init__(self):
#         # State
#         self.in_overlap_mode = False
#         self.consecutive_overlap_frames = 0
#         self.consecutive_clean_frames = 0
        
#         # Caching (Critial Fix 2)
#         self.last_overlap_speakers: List[str] = []
        
#         self.threshold = config.OVERLAP_THRESHOLD 
#         self.min_speakers = config.MIN_OVERLAP_SPEAKERS 

#     def process(self, active_probs: Dict[str, float]) -> OverlapResult:
#         """
#         Input: Smoothed probabilities.
#         Returns: OverlapResult status.
#         CACHE RULE: If in_overlap_mode is True, speakers list MUST NOT be empty.
#         """
#         # 1. Count Active Speakers
#         candidates = [k for k, v in active_probs.items() if v > self.threshold]
#         count = len(candidates)
        
#         is_instant_overlap = count >= self.min_speakers
        
#         # 2. Hysteresis Logic
#         if is_instant_overlap:
#             self.consecutive_overlap_frames += 1
#             self.consecutive_clean_frames = 0
            
#             # Turn ON if persistence met
#             if self.consecutive_overlap_frames >= self.MIN_PERSISTENCE:
#                 self.in_overlap_mode = True
                
#         else:
#             self.consecutive_clean_frames += 1
#             self.consecutive_overlap_frames = 0
            
#             # Turn OFF if clear persistence met
#             if self.consecutive_clean_frames >= self.OVERLAP_CLEAR_WINDOWS:
#                 self.in_overlap_mode = False
#                 self.last_overlap_speakers = [] # Reset cache on clean exit

#         # 3. Cache Logic (The Fix)
#         final_speakers = []
        
#         if self.in_overlap_mode:
#             # If we have instant candidates (>=2), use them and update cache
#             if is_instant_overlap:
#                 final_speakers = candidates
#                 self.last_overlap_speakers = candidates # Update Cache
#             else:
#                 # We are in hold state (hysteresis gave us overlap=True but currently < 2 speakers)
#                 # # We MUST emit cached speakers
#                 # if not self.last_overlap_speakers:
#                 #     # Rare edge case: triggered overlap mode but cache empty?
#                 #     # Should be impossible if MIN_PERSISTENCE >= 1.
#                 #     # Fallback to candidates (even if 1) to avoid total blank?
#                 #     # Or stick to invariant "Must not emit < 2"?
#                 #     # If we emit 1 speaker with is_overlap=True, StateMachine might freak out or treat as overlap of 1? 
#                 #     # StateMachine uses `overlap.speakers` if is_overlap=True.
#                 #     # Best effort: use candidates if > 0, else cache.
#                 #     if candidates:
#                 #         final_speakers = candidates 
#                 #     else:
#                 #         final_speakers = [] # Nothing we can do
                
#                 if self.in_overlap_mode:
#                     if not self.last_overlap_speakers:
#                         raise RuntimeError("Invariant violated: overlap with empty speaker cache")
#                     final_speakers = self.last_overlap_speakers
    
#                 else:
#                     final_speakers = self.last_overlap_speakers

#             # Safety check: If we return empty speakers but is_overlap=True, 
#             # StateMachine will produce empty target -> Silence.
#             # This effectively ends the segment. 
#             # Ideally we want to prevent that if we are holding state.
            
#         return OverlapResult(
#             is_overlap=self.in_overlap_mode,
#             speakers=final_speakers
#         )
