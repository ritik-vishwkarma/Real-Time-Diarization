import logging
import numpy as np
import torch
from collections import deque
from diarization_engine import DiarizationEngine
from state_machine import SpeakerStateMachine
from vad_engine import NeMoVADGate

logger = logging.getLogger("StreamingDiarizer")

class StreamingDiarizer:
    """
    Unified class for Real-time Diarization.
    Wraps Model Engine + State Machine + VAD.
    """
    def __init__(self, engine=None, silence_thresh=0.6, energy_thresh=0.005, use_nemo_vad=True):
        if engine:
            self.engine = engine
        else:
            self.engine = DiarizationEngine() # This loads the model (heavy)
            
        self.state_machine = SpeakerStateMachine(silence_thresh=silence_thresh)
        self.chunk_samples = self.engine.chunk_samples
        self.device = self.engine.device
        self.energy_thresh = energy_thresh
        
        self.use_nemo_vad = use_nemo_vad
        self.vad_model = None
        if self.use_nemo_vad:
            try:
                self.vad_model = NeMoVADGate() # Default marblenet
            except Exception as e:
                logger.warning(f"Could not load NeMo VAD: {e}. Falling back to Energy VAD.")
                self.use_nemo_vad = False

        # Smoothing Buffer (Store last 5 sets of probabilities)
        # 8 chunks * 0.08s = 0.64s window (matches "0.3-0.5s" requirement)
        self.prob_history = deque(maxlen=8)

    def _get_smoothed_probs(self, current_probs):
        """
        current_probs: dict {speaker_idx: prob}
        Returns: dict {speaker_idx: smoothed_prob}
        """
        self.prob_history.append(current_probs)
        
        # Aggregate all keys seen in history
        all_speakers = set()
        for probs in self.prob_history:
            all_speakers.update(probs.keys())
            
        smoothed = {}
        for spk in all_speakers:
            # Collect values for this speaker (0.0 if missing in a frame)
            vals = [p.get(spk, 0.0) for p in self.prob_history]
            # Median Filter
            smoothed[spk] = float(np.median(vals))
            
        return smoothed

    def process_chunk(self, audio_chunk, current_time, vad_is_speech=None):
        """
        Process a single chunk of audio and update state.
        
        Args:
            audio_chunk (np.array): Float32 audio chunk of size `chunk_samples`
            current_time (float): Current audio stream time in seconds
            vad_is_speech (bool, optional): External VAD decision. If None, uses internal VAD.
            
        Returns:
            List[dict]: List of event directories
        """
        try:
            # 0. VAD Check
            is_speech = False
            speech_prob = 0.0
            
            if vad_is_speech is not None:
                is_speech = vad_is_speech
                speech_prob = 1.0 if is_speech else 0.0
            elif self.use_nemo_vad and self.vad_model:
                # Use NeMo VAD
                is_speech, speech_prob = self.vad_model.is_speech(audio_chunk)
            else:
                # Internal Energy based VAD
                energy = np.mean(audio_chunk**2)
                is_speech = energy > self.energy_thresh
                speech_prob = float(min(energy * 100, 1.0)) # Rough heuristic
            
            # 1. Initialize variables for both speech/silence paths
            active_indices = []
            active_probs = {}
            is_overlap = False

            if is_speech:
                # 1. Inference
                # Now returns tuple: (indices, probs)
                _, raw_probs = self.engine.process_chunk(audio_chunk)
                
                # 2. Smooth Probabilities
                active_probs = self._get_smoothed_probs(raw_probs)
                
                # DEBUG: Log if we have ANY signal
                max_p = max(active_probs.values()) if active_probs else 0.0
                if max_p > 0.0: # Log EVERYTHING that survives the engine
                     logger.info(f"🔍 Diarizer Probs: { {k: f'{v:.2f}' for k,v in active_probs.items()} }")

                # 3. Decision Logic
                # Primary Speakers: Moderate signal (> 0.45) - lowered from 0.6
                final_active_indices = [k for k,v in active_probs.items() if v > 0.45]
                
                # Overlap Candidates: Weak signal (> 0.35) - lowered from 0.4
                # If we have >= 2 speakers above 0.4, it's an overlap.
                overlap_candidates = [k for k,v in active_probs.items() if v > 0.35]
                is_overlap = len(overlap_candidates) >= 2
                
                if is_overlap:
                    final_active_indices = overlap_candidates
                    active_indices = final_active_indices
                else:
                    active_indices = final_active_indices
            else:
                 # Flush history (add empty dict) to degrade confidence naturally over time
                 self.prob_history.append({})

            # 3. Update State Machine
            # Corrected Order: active_indices, current_time
            events = self.state_machine.process_step(active_indices, current_time, active_probs, vad_confidence=speech_prob, overlap_detected=is_overlap)
            
            return events
        except Exception as e:
            logger.error(f"Error in process_chunk: {e}")
            return []

    def close(self, end_time):
        """
        Finalize the session, closing any open segments.
        """
        return self.state_machine.finish(end_time)
