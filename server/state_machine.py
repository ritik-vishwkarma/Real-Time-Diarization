import time

class SpeakerStateMachine:
    def __init__(self, silence_thresh=0.6):
        self.current_speakers = set()
        self.start_time = 0.0
        self.last_active_time = 0.0
        self.silence_thresh = silence_thresh # Seconds of silence to trigger "Stop"
        
    def process_step(self, active_indices, current_stream_time, active_probs=None, vad_confidence=1.0, overlap_detected=False):
        """
        Input: 
            active_indices: List[int] e.g. [0, 1]
            current_stream_time: float
            active_probs: dict {speaker_idx: probability} e.g. {0: 0.9, 1: 0.8}
            vad_confidence: float (0.0 - 1.0)
            overlap_detected: bool (True if robust overlap logic triggered)
        Output: A list of event Dicts if state changed, else empty list.
        """
        # Convert indices to consistent string format for frontend safely
        # E.g. 0 -> "spk_0"
        new_speakers = set([f"spk_{i}" for i in active_indices])
        
        # Build confidence map: {"spk_0": 0.9}
        confidence = {}
        if active_probs:
            for i, prob in active_probs.items():
                confidence[f"spk_{i}"] = prob
                
        events = []

        # 1. Active Speech Detected
        if new_speakers:
            self.last_active_time = current_stream_time
            
            # Change in speakers (Switch or Overlap Change)
            if new_speakers != self.current_speakers:
                # Close previous segment if it existed
                if self.current_speakers:
                    duration = current_stream_time - self.start_time
                    if duration > 0.1:
                        # Determine end type based on PREVIOUS state
                        # If previous state had > 1 speaker, it was an overlap
                        prev_was_overlap = len(self.current_speakers) > 1
                        events.append({
                            "version": "1.0",
                            "type": "overlap_end" if prev_was_overlap else "segment_end",
                            "speakers": list(self.current_speakers),
                            "start": self.start_time,
                            "end": current_stream_time,
                            "duration": duration
                        })
                
                # Start new segment
                is_overlap = len(new_speakers) > 1 or overlap_detected
                events.append({
                    "version": "1.0",
                    "type": "overlap_start" if is_overlap else "segment_start",
                    "speakers": sorted(list(new_speakers)),
                    "start": current_stream_time,
                    "confidence": confidence,
                    "vad_prob": vad_confidence,
                    "is_overlap": is_overlap
                })
                
                self.current_speakers = new_speakers
                self.start_time = current_stream_time

        # 2. Silence (No Active Speakers)
        else:
            if self.current_speakers:
                # Check if silence has exceeded threshold
                time_since_active = current_stream_time - self.last_active_time
                if time_since_active > self.silence_thresh:
                    # End the segment
                    duration = self.last_active_time - self.start_time
                    if duration > 0.1:
                        events.append({
                            "version": "1.0",
                            "type": "segment_end",
                            "speakers": sorted(list(self.current_speakers)),
                            "start": self.start_time,
                            "end": self.last_active_time, # End at last known activity
                            "duration": duration
                        })
                    self.current_speakers = set()
            
        return events

    def finish(self, end_time):
        """
        Force close any active segment at session end.
        """
        events = []
        if self.current_speakers:
            events.append({
                "version": "1.0",
                "type": "segment_end",
                "speakers": sorted(list(self.current_speakers)),
                "start": self.start_time,
                "end": end_time, 
                "duration": end_time - self.start_time
            })
            self.current_speakers = set()
        return events
