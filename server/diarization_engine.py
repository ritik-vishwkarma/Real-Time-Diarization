import torch
import numpy as np
from nemo.collections.asr.models import SortformerEncLabelModel

import logging

logger = logging.getLogger(__name__)

class DiarizationEngine:
    def __init__(self, model_name="nvidia/diar_streaming_sortformer_4spk-v2.1", device=None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Loading {model_name} on {self.device}...")
        self.model = SortformerEncLabelModel.from_pretrained(model_name).to(self.device).eval()
        
        # Configuration
        self.sample_rate = 16000
        self.frame_ms = 0.08
        self.samples_per_frame = int(self.sample_rate * self.frame_ms)
        self.chunk_frames = 4     # 0.32s
        self.context_frames = 22  # 1.76s (Total ~2s window)
        
        # Apply Config to Model
        self.model.sortformer_modules.chunk_len = self.chunk_frames
        self.model.sortformer_modules.chunk_right_context = self.context_frames
        
        # Buffer Setup
        self.chunk_samples = self.chunk_frames * self.samples_per_frame
        self.buffer_samples = (self.chunk_frames + self.context_frames) * self.samples_per_frame
        self.rolling_buffer = np.zeros(self.buffer_samples, dtype=np.float32)

    def process_chunk(self, audio_chunk):
        """
        Input: audio_chunk (numpy array, float32) - should be size of self.chunk_samples
        Output: list of active speaker IDs (e.g. [0, 2])
        """
        # 1. Update Buffer (Shift left, append new)
        self.rolling_buffer = np.roll(self.rolling_buffer, -len(audio_chunk))
        self.rolling_buffer[-len(audio_chunk):] = audio_chunk
        
        # 2. Prepare Inputs
        input_tensor = torch.tensor(self.rolling_buffer).unsqueeze(0).to(self.device)
        input_length = torch.tensor([input_tensor.shape[1]]).to(self.device)

        # 2.5 Dynamic Gain Normalization (Fix for low volume inputs)
        # Sortformer is sensitive to amplitude. If signal is weak but present (VAD passed upstream), boost it.
        max_val = torch.max(torch.abs(input_tensor))
        if max_val > 0.0001 and max_val < 0.1:
            # Boost to target peak of 0.5 (safe headroom)
            gain = 0.5 / max_val
            input_tensor = input_tensor * gain
            # logger.debug(f"🔊 Boosted Input Signal by {gain:.2f}x (Peak: {max_val:.4f} -> 0.5)")

        # 3. Inference
        with torch.no_grad():
            preds = self.model(input_tensor, input_length)
            if isinstance(preds, tuple) or isinstance(preds, list):
                probs = preds[0]
            else:
                probs = preds

        # 4. Decode
        # Probs shape: [Batch, Time, Speakers] (e.g., [1, 13, 4] for 6 chunk + 7 context)
        # We only want to classify the *new* part of the audio (the chunk), 
        # not the context which was already classified in previous step.
        # The chunk is at the END of the buffer/input.
        
        # input_length is the total frames (e.g. 13)
        # we care about the last `self.chunk_frames` frames.
        
        total_frames = probs.shape[1]
        relevant_probs = probs[:, total_frames - self.chunk_frames :, :] # [1, chunk_frames, S]
        
        # Calculate average probability per speaker over the chunk
        avg_probs = relevant_probs.mean(dim=1)[0] # [Speakers]
        
        # Determine "Active" speakers based on a lower threshold (0.4) used for Overlap check
        # But we return probabilities for anyone showing ANY signal (> 0.01) for smoothing
        candidate_indices = (avg_probs > 0.01).nonzero().flatten().tolist()
        
        # Extract probabilities for candidates
        speaker_probs = {i: float(avg_probs[i]) for i in candidate_indices}
        
        # Legacy: Return 'active' indices for callers who don't want to parse probs
        # We use a standard threshold 0.5 here, but StreamingDiarizer will recalculate.
        active_indices = [i for i, p in speaker_probs.items() if p > 0.5]
        
        return active_indices, speaker_probs
