import logging
from typing import Set, Dict, List, Optional
from server.dtos import FrameDecision
from infrastructure.event_bus import EventBus
from components.overlap_detector import OverlapResult
import config

logger = logging.getLogger("SpeakerStateMachine")

class SpeakerStateMachine:
    """
    Component 4.8: SpeakerStateMachine (Hardened Phase 4)
    """
    
    def __init__(self, event_bus: EventBus):
        self.bus = event_bus
        self.committed_speakers: Set[str] = set()
        self.committed_start_sample = 0
        self.segment_confidences: Dict[str, List[float]] = {}
        
        self.pending_speakers: Set[str] = set()
        self.pending_since_sample = 0
        
        # BOOTSTRAP STATE
        self.in_bootstrap = True
        self.bootstrap_speakers: Set[str] = set()
        self.bootstrap_since_sample: int = 0

    def process(self, decision: FrameDecision, overlap: OverlapResult):
        current_sample = decision.end_sample
        
        # 1. Determine Thresholds (Dynamic)
        min_duration = (
            config.BOOTSTRAP_MIN_DURATION 
            if self.in_bootstrap 
            else config.MIN_SEGMENT_DURATION
        )

        # 2. Extract Hypothesis 
        # Even in bootstrap, we DO NOT accept low-prob inputs.
        speech_target = set()
        if decision.state == "speech":
            if overlap.is_overlap:
                speech_target = set(overlap.speakers)
            elif decision.probs:
                best = max(decision.probs, key=decision.probs.get)
                prob = decision.probs[best]
                # During bootstrap, use START_THRESHOLD strictly
                thresh = (
                    config.CONTINUE_THRESHOLD
                    if best in self.committed_speakers
                    else config.START_THRESHOLD
                )
                if prob > thresh:
                    speech_target = {best}

        # ===== BOOTSTRAP PHASE =====
        if self.in_bootstrap:
            if decision.state == "uncertain":
                return  

            # Case A: Continuity (Backpressure handling)
            # If we are skipping, and have a hypothesis, assume it holds.
            if getattr(decision, 'is_continuity', False):
                if self.bootstrap_speakers:
                    # ADVANCE TIMER
                    duration = (current_sample - self.bootstrap_since_sample) / config.SAMPLE_RATE
                    if duration >= min_duration:
                        self._commit_bootstrap()
                return

            # Case B: Real Evidence
            if speech_target:
                if not self.bootstrap_speakers:
                    # New Hypothesis
                    self.bootstrap_speakers = speech_target
                    self.bootstrap_since_sample = current_sample
                elif speech_target == self.bootstrap_speakers:
                    # Sustained Hypothesis
                    duration = (current_sample - self.bootstrap_since_sample) / config.SAMPLE_RATE
                    if duration >= min_duration:
                        self._commit_bootstrap()
                else:
                    # Contradiction -> Reset
                    self.bootstrap_speakers = speech_target
                    self.bootstrap_since_sample = current_sample
            
            elif decision.state == "silence":
                # Silence breaks bootstrap
                self.bootstrap_speakers = set()
            
            return # Block normal logic

        if decision.state == "uncertain":
            if self.committed_speakers:
                self._accumulate_frame_evidence(decision)
            return

        # ===== NORMAL PHASE =====
        
        new_target = set()
        
        # 1. Determine Target
        if getattr(decision, 'is_continuity', False):
            # Strict Rule: Hold Committed State
            new_target = self.committed_speakers
        else:
            if decision.state == "unknown":
                return # Freeze
            elif decision.state == "silence":
                new_target = set()
            elif decision.state == "speech":
                new_target = speech_target
        
        # 2. Debounce
        if new_target != self.pending_speakers:
            self.pending_speakers = new_target
            self.pending_since_sample = current_sample
        else:
            duration = (current_sample - self.pending_since_sample) / config.SAMPLE_RATE
            if duration >= min_duration:
                if self.pending_speakers != self.committed_speakers:
                    self._transition(current_sample, self.pending_speakers)

        # 3. Evidence Accumulation (Real Frames Only)
        if not getattr(decision, 'is_continuity', False):
            if decision.state == "speech" and self.committed_speakers:
                 for spk in self.committed_speakers:
                     if spk in decision.probs:
                         if spk not in self.segment_confidences:
                             self.segment_confidences[spk] = []
                         self.segment_confidences[spk].append(decision.probs[spk])

    def _commit_bootstrap(self):
        """Helper to exit bootstrap and enter normal mode"""
        logger.info(f"BOOTSTRAP: Committing {self.bootstrap_speakers}")
        self.committed_speakers = self.bootstrap_speakers
        self.committed_start_sample = self.bootstrap_since_sample
        self.bootstrap_speakers = set()
        self.in_bootstrap = False

    def _transition(self, end_sample: int, new_speakers: Set[str]):
        if self.committed_speakers:
            start_sample = self.committed_start_sample
            if end_sample > start_sample:
                 self._emit_segment(start_sample, end_sample, self.committed_speakers)
        
        self.committed_speakers = new_speakers
        self.committed_start_sample = end_sample
        self.segment_confidences = {}

    def _emit_segment(self, start_sample: int, end_sample: int, speakers):
        final_conf = {}
        for spk, vals in self.segment_confidences.items():
            if vals: final_conf[spk] = float(sum(vals)/len(vals))
        
        start_sec = start_sample / config.SAMPLE_RATE
        end_sec = end_sample / config.SAMPLE_RATE
        primary = max(final_conf, key=final_conf.get) if final_conf else (list(speakers)[0] if speakers else "unknown")
            
        event = {
            "type": "segment",
            "start_sec": float(f"{start_sec:.3f}"),
            "end_sec": float(f"{end_sec:.3f}"),
            "start_sample": start_sample,
            "end_sample": end_sample,
            "speakers": sorted(list(speakers)),
            "is_final": True,
            "meta": {
                "confidence": final_conf,
                "primary_speaker": primary
            }
        }
        
        # ADD THIS LOGGING (helps you see emissions)
        duration = end_sec - start_sec
        logger.info(f"📤 EMITTING SEGMENT: [{start_sec:.2f}s → {end_sec:.2f}s] "
                    f"duration={duration:.2f}s, speakers={event['speakers']}, "
                    f"confidence={final_conf}")

        self.bus.publish(event)

    def finish(self, end_sample: int):
        # ISSUE 5 FIX: Do not emit if we are still bootstrapping
        if self.in_bootstrap:
            return 
        self._transition(end_sample, set())

    def _accumulate_frame_evidence(self, decision: FrameDecision):
        """Helper for evidence accumulation shared by speech and uncertain states"""
        if not getattr(decision, 'is_continuity', False):
            if self.committed_speakers:
                for spk in self.committed_speakers:
                    if spk in decision.probs:
                        if spk not in self.segment_confidences:
                            self.segment_confidences[spk] = []
                        self.segment_confidences[spk].append(decision.probs[spk])



# import logging
# from typing import Set, Dict, List, Optional
# from server.dtos import FrameDecision
# from infrastructure.event_bus import EventBus
# from components.overlap_detector import OverlapResult
# import config

# logger = logging.getLogger("SpeakerStateMachine")

# class SpeakerStateMachine:
#     """
#     Component 4.8: SpeakerStateMachine (Hardened Phase 4)
    
#     Responsibility:
#     - FrameDecision -> Diarization Segments.
#     - Emit events with Sample Indices.
#     - Freeze on Unknown.
#     - Bootstrap initial speakers under backpressure.
#     """
    
#     MIN_DURATION = 0.30
    
#     def __init__(self, event_bus: EventBus):
#         self.bus = event_bus
        
#         # Committed Segment State
#         self.committed_speakers: Set[str] = set()
#         self.committed_start_sample = 0
#         self.segment_confidences: Dict[str, List[float]] = {}
        
#         # Pending State (Debouncing)
#         self.pending_speakers: Set[str] = set()
#         self.pending_since_sample = 0
        
#         # BOOTSTRAP state (New Architecture for Lag Handling)
#         self.bootstrap_speakers: Set[str] = set()
#         self.bootstrap_since_sample: int = 0

#     def process(self, decision: FrameDecision, overlap: OverlapResult):
#         current_sample = decision.end_sample
        
#         # 1. Determine Speech Hypothesis
#         speech_target = set()
#         if decision.state == "speech":
#             if overlap.is_overlap:
#                 speech_target = set(overlap.speakers)
#             elif decision.probs:
#                 best = max(decision.probs, key=decision.probs.get)
#                 prob = decision.probs[best]
#                 thresh = (
#                     config.CONTINUE_THRESHOLD
#                     if best in self.committed_speakers
#                     else config.START_THRESHOLD
#                 )
#                 if prob > thresh:
#                     speech_target = {best}

#         # 2. BOOTSTRAP PHASE (Handle First Commit under Lag)
#         # If we have nothing committed yet, use lighter rules to get started.
#         if not self.committed_speakers and not self.pending_speakers:
#             if decision.state == "speech" and speech_target:
#                 # Start Bootstrap
#                 if not self.bootstrap_speakers:
#                     self.bootstrap_speakers = speech_target
#                     self.bootstrap_since_sample = current_sample
#                     return
#                 else:
#                     # Continue Bootstrap
#                     if speech_target == self.bootstrap_speakers:
#                         duration = (current_sample - self.bootstrap_since_sample) / config.SAMPLE_RATE
#                         if duration >= self.MIN_DURATION:
#                             # Commit!
#                             self.committed_speakers = self.bootstrap_speakers
#                             self.committed_start_sample = self.bootstrap_since_sample
#                             self.bootstrap_speakers = set()
#                             self.bootstrap_since_sample = 0
#                         return
#                     else:
#                         # Contradiction -> Reset
#                         self.bootstrap_speakers = speech_target
#                         self.bootstrap_since_sample = current_sample
#                         return

#             # Special Rule: Allow "Continuity" (Skip) frames to advance Bootstrap
#             elif getattr(decision, 'is_continuity', False) and self.bootstrap_speakers:
#                 duration = (current_sample - self.bootstrap_since_sample) / config.SAMPLE_RATE
#                 if duration >= self.MIN_DURATION:
#                     self.committed_speakers = self.bootstrap_speakers
#                     self.committed_start_sample = self.bootstrap_since_sample
#                     self.bootstrap_speakers = set()
#                     self.bootstrap_since_sample = 0
#                 return

#             elif decision.state == "silence":
#                 self.bootstrap_speakers = set()
#                 return

#         # If bootstrapping is active, don't run standard logic yet
#         if self.bootstrap_speakers:
#             return

#         # 3. STANDARD LOGIC (Committed & Pending)
        
#         new_target = set()
        
#         # Continuity Frame (Skip): Hold Committed State Strictly
#         if getattr(decision, 'is_continuity', False):
#             new_target = self.committed_speakers
        
#         # Normal Frame
#         else:
#             if decision.state == "unknown":
#                 return # Freeze logic (don't change anything)
#             elif decision.state == "silence":
#                 new_target = set() # Empty set means silence
#             elif decision.state == "speech":
#                 new_target = speech_target
        
#         # Debounce Logic
#         if new_target != self.pending_speakers:
#             # State change detected -> Reset pending timer
#             self.pending_speakers = new_target
#             self.pending_since_sample = current_sample
#         else:
#             # State is stable -> Check Duration
#             duration_samples = current_sample - self.pending_since_sample
#             duration_sec = duration_samples / config.SAMPLE_RATE
            
#             if duration_sec >= self.MIN_DURATION:
#                 if self.pending_speakers != self.committed_speakers:
#                     self._transition(current_sample, self.pending_speakers)

#         # 4. Evidence Accumulation
#         # Only accumulate confidence on REAL speech frames, not Continuity/Skips
#         if not getattr(decision, 'is_continuity', False):
#             if decision.state == "speech" and self.committed_speakers:
#                  for spk in self.committed_speakers:
#                      if spk in decision.probs:
#                          if spk not in self.segment_confidences:
#                              self.segment_confidences[spk] = []
#                          self.segment_confidences[spk].append(decision.probs[spk])

#     def _transition(self, end_sample: int, new_speakers: Set[str]):
#         # Close old segment
#         if self.committed_speakers:
#             start_sample = self.committed_start_sample
#             # Guard against zero-length segments
#             if end_sample > start_sample:
#                  self._emit_segment(start_sample, end_sample, self.committed_speakers)
                 
#         # Start new segment
#         self.committed_speakers = new_speakers
#         self.committed_start_sample = end_sample
#         self.segment_confidences = {}

#     def _emit_segment(self, start_sample: int, end_sample: int, speakers):
#         final_conf = {}
#         for spk, vals in self.segment_confidences.items():
#             if vals: final_conf[spk] = float(sum(vals)/len(vals))
        
#         start_sec = start_sample / config.SAMPLE_RATE
#         end_sec = end_sample / config.SAMPLE_RATE
        
#         primary = max(final_conf, key=final_conf.get) if final_conf else (list(speakers)[0] if speakers else "unknown")
            
#         event = {
#             "type": "segment",
#             "start_sec": float(f"{start_sec:.3f}"),
#             "end_sec": float(f"{end_sec:.3f}"),
#             "start_sample": start_sample,
#             "end_sample": end_sample,
#             "speakers": sorted(list(speakers)),
#             "is_final": True,
#             "meta": {
#                 "confidence": final_conf,
#                 "primary_speaker": primary
#             }
#         }
#         self.bus.publish(event)

#     def finish(self, end_sample: int):
#         self._transition(end_sample, set())
        
# import logging
# from typing import Set, Dict, List, Optional
# from server.dtos import FrameDecision
# from infrastructure.event_bus import EventBus
# from components.overlap_detector import OverlapResult
# import config

# logger = logging.getLogger("SpeakerStateMachine")

# class SpeakerStateMachine:
#     """
#     Component 4.8: SpeakerStateMachine (Hardened Phase 4)
    
#     Responsibility:
#     - FrameDecision -> Diarization Segments.
#     - Emit events with Sample Indices (Fix 4).
#     - Freeze on Unknown.
#     """
    
#     MIN_DURATION = 0.30
    
#     def __init__(self, event_bus: EventBus):
#         self.bus = event_bus
        
#         # Committed Segment State
#         self.committed_speakers: Set[str] = set()
#         self.committed_start_sample = 0 # Int
#         self.segment_confidences: Dict[str, List[float]] = {}
        
#         # Pending State (Debouncing)
#         self.pending_speakers: Set[str] = set()
#         self.pending_since_sample = 0 # Int
        
#         self.last_emit_end_sample = 0
        
#         # Fix Bug 2: Track valid state for time-bounded degradation
#         self.last_valid_state_sample = 0

#     def process(self, decision: FrameDecision, overlap: OverlapResult):
#         """
#         Ingest decision.
#         Backpressure Update: Handle 'is_continuity' frames.
#         """
#         current_sample = decision.end_sample
#         new_target = set()
        
#         # 0. strict Continuity Handling
#         if decision.is_continuity:
#             # Strict Rule: Persist committed state ONLY. Ignore pending shifts.
#             new_target = self.committed_speakers
            
#             # Accumulate Confidence (Decayed)
#             if self.committed_speakers and decision.probs:
#                 for spk in self.committed_speakers:
#                     if spk in decision.probs:
#                         if spk not in self.segment_confidences:
#                             self.segment_confidences[spk] = []
#                         self.segment_confidences[spk].append(decision.probs[spk])
            
#             # Debounce Logic: 
#             # We do NOT update pending_speakers or pending_since_sample.
#             # We treat this as "holding" existing state.
            
#             # However, we must ensure we don't accidentally "close" the segment 
#             # if the logic below thinks target != committed.
#             # actually logic below checks pending != committed.
#             # If we don't update pending, it stays whatever it was.
#             # But wait, if pending was trying to transition to something else, 
#             # and we get continuity (which forces committed), 
#             # should we reset pending to committed?
#             # User Rule: "Continuity frames may only preserve committed states."
#             # "If last_decision is pending, continuity must not advance it."
#             # So we should probably NOT update pending state at all, 
#             # OR we force pending = committed to "cancel" the pending transition?
#             # "Pending duration never reaching MIN_DURATION" was a problem.
#             # If we just skip updating pending, 'current_sample' advances, 
#             # so `duration = current - pending_since` INCREASES.
#             # This would allow the pending state to commit!
#             # BUT the user said: "Continuity frames may only preserve committed states".
#             # This implies we should NOT allow a pending state to commit during continuity.
#             # So if we have a pending change, continuity should probably cancel it or freeze time for it?
#             # Prompt: "Pending duration never reaching MIN_DURATION... This failure is logically guaranteed... 
#             # Change 2: Timeline advances... Pending timers to advance... MIN_DURATION to be reached... Proper transitions to occur"
#             # WAIT. 
#             # The prompt says: "This allows: ... MIN_DURATION to be reached ... Proper transitions to occur".
#             # This implies continuity SHOULD allow pending states to commit?
#             # BUT Problem 2 says: "Continuity frames may only preserve committed states, not pending states."
#             # "If last_decision is pending, continuity must not advance it."
#             # "Only committed speaker sets are eligible for continuity preservation."
            
#             # Contradiction?
#             # Change 2 says: "Timeline still advances... Pending timers to advance".
#             # Problem 2 says: "If last_decision is pending, continuity must not advance it".
#             # The "not advance it" probably means "Do not create NEW pending states" or "Do not assume pending state is truth".
#             # But if we were *already* pending, and we get skip, do we hold the pending state?
#             # "If last_decision is pending, continuity must not advance it" likely means: 
#             # if we were pending X, and we skip, we cannot confirm X.
#             # So we should probably fall back to committed.
#             # Let's read "Confusion 2" carefully: "If last_decision is pending, continuity must not advance it... Only committed speaker sets are eligible."
#             # This suggests we should NOT let pending states confirm during skip.
#             # So in continuity mode: Target = Committed.
#             # If Target == Committed, and previously Pending was X != Committed.
#             # Then New Target (Committed) != Previous Pending (X).
#             # So Debounce Logic will see change! -> pending_speakers = Committed.
#             # So we effectively CANCEL the pending transition.
            
#             # This seems safer and aligns with "Phantom speech" risk.
#             # I will set new_target = committed_speakers.
#             # And let the standard debounce logic below run, which will likely reset pending to committed.
#             pass

#         else:
#             # Normal Logic
#             # 1. Determine Target State
#             if decision.state == "unknown":
#                  # Fallback (Legacy/Error) -> Hold vs Silence?
#                  # If we get true unknown (not continuity), we use legacy Degradation logic?
#                  # Or just freeze? 
#                  # Let's keep legacy time-bound logic for safety (if bot emits unknown on error).
#                 if self.last_valid_state_sample == 0:
#                       self.last_valid_state_sample = current_sample
                 
#                 #  dur = (current_sample - self.last_valid_state_sample) / config.SAMPLE_RATE
#                 #  if dur > 1.0:
#                 #       new_target = set()
#                 #  else:
#                 #       return # Freeze
#                 if decision.state == "unknown":
#                     return # Freeze
                
#             else:
#                  self.last_valid_state_sample = current_sample
                 
#                  if decision.state == "silence":
#                      new_target = set()
#                  elif decision.state == "speech":
#                      if overlap.is_overlap:
#                          new_target = set(overlap.speakers)
#                      else:
#                           if decision.probs:
#                               best_spk = max(decision.probs, key=decision.probs.get)
#                               prob = decision.probs[best_spk]
#                               is_continuing = best_spk in self.committed_speakers
#                               thresh = config.CONTINUE_THRESHOLD if is_continuing else config.START_THRESHOLD
#                               if prob > thresh:
#                                   new_target = {best_spk}
        
#         # 2. Debounce
#         # If is_continuity=True, we manually override new_target to committed earlier? 
#         # Actually I can just set new_target in the block above.
        
#         if decision.is_continuity:
#              new_target = self.committed_speakers
             
#         # Standard Debounce Implementation
#         if new_target != self.pending_speakers:
#             self.pending_speakers = new_target
#             self.pending_since_sample = current_sample
#         else:
#             # Stable
#             duration_samples = current_sample - self.pending_since_sample
#             duration_sec = duration_samples / config.SAMPLE_RATE
            
#             if duration_sec >= self.MIN_DURATION:
#                 if self.pending_speakers != self.committed_speakers:
#                     self._transition(current_sample, self.pending_speakers)

#         # 3. Accumulate Confidence (Normal frames only here, continuity handled above?)
#         # Actually, if we merge logic:
#         # If continuity, we already accumulated.
#         # If normal, we accumulate here.
#         if not decision.is_continuity:
#             is_speech_now = (decision.state == "speech")
#             if is_speech_now and self.committed_speakers:
#                  for spk in self.committed_speakers:
#                      if spk in decision.probs:
#                          if spk not in self.segment_confidences:
#                              self.segment_confidences[spk] = []
#                          self.segment_confidences[spk].append(decision.probs[spk])

#     def _transition(self, end_sample: int, new_speakers: Set[str]):
#         # Close old
#         if self.committed_speakers:
#             start_sample = self.committed_start_sample
#             if end_sample > start_sample:
#                  self._emit_segment(start_sample, end_sample, self.committed_speakers)
                 
#         # Start new
#         self.committed_speakers = new_speakers
#         self.committed_start_sample = end_sample
#         self.segment_confidences = {}
#         self.last_emit_end_sample = end_sample

#     def _emit_segment(self, start_sample: int, end_sample: int, speakers):
#         final_conf = {}
#         for spk, vals in self.segment_confidences.items():
#             if vals: final_conf[spk] = float(sum(vals)/len(vals))
        
#         # Calculate seconds for UI consumption (Logging/Client), 
#         # but keep strict samples for IdentityService.
#         start_sec = start_sample / config.SAMPLE_RATE
#         end_sec = end_sample / config.SAMPLE_RATE
        
#         # Determine primary speaker deterministically
#         if final_conf:
#             primary = max(final_conf, key=final_conf.get)
#         else:
#             # Fallback: deterministic ordering
#             primary = sorted(speakers)[0]
            
#         event = {
#             "type": "segment",
#             "start_sec": float(f"{start_sec:.3f}"), # Fix: Demote from authoritative
#             "end_sec": float(f"{end_sec:.3f}"),     # Fix: Demote from authoritative
#             "start_sample": start_sample, # Fix 4: Authoritative
#             "end_sample": end_sample,     # Fix 4: Authoritative
#             "speakers": sorted(list(speakers)),
#             "is_final": True,
#             "meta": {
#                 "confidence": final_conf,
#                 "primary_speaker": primary
#             }
#         }
#         self.bus.publish(event)

#     # def finish(self, end_time_sec: float):
#     #      # Convert to sample for final transition?
#     #      # Bot calls finish with float time.
#     #      end_sample = int(end_time_sec * config.SAMPLE_RATE)
#     #      self._transition(end_sample, set())
    
#     def finish(self, end_sample: int):
#         self._transition(end_sample, set())

