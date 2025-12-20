import torch
import logging
import numpy as np
import os
from nemo.collections.asr.models import EncDecClassificationModel
import config

logger = logging.getLogger("VADEngine")

class NeMoVADGate:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load the Multilingual MarbleNet (Lightweight CNN)
        model_name = config.VAD_MODEL
        logger.info(f"Loading NeMo VAD Gate: {model_name} on {self.device}...")
        
        # Using EncDecClassificationModel as marblenet is a classifier
        try:
            self.model = EncDecClassificationModel.from_pretrained(model_name=model_name).to(self.device)
            self.model.eval()
        except Exception as e:
            logger.error(f"Failed to load VAD model: {e}")
            raise e

    def is_speech(self, audio_chunk, threshold=0.5):
        """
        Input: Numpy array or Torch Tensor (samples) at 16kHz. 
               Expected duration 0.02s - 0.1s.
        Output: is_speech (bool), prob (float)
        """
        # Ensure tensor on device
        if isinstance(audio_chunk, np.ndarray):
            audio_tensor = torch.from_numpy(audio_chunk).to(self.device)
        else:
            audio_tensor = audio_chunk.to(self.device)
             
        # MarbleNet expects [Batch, Time]
        if audio_tensor.ndim == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
            
        # Input length tensor
        input_len = torch.tensor([audio_tensor.shape[1]]).to(self.device)
        
        with torch.no_grad():
            # 1. Forward
            logits = self.model(input_signal=audio_tensor, input_signal_length=input_len)
            
            # 2. Softmax
            probs = torch.softmax(logits, dim=-1)
            
            # Class 1 = Speech
            speech_prob = probs[0, 1].item()
            
            return speech_prob > threshold, speech_prob

class EnergyVAD:
    """
    Lightweight Energy-based VAD for fast Gating and Normalization Masking.
    """
    def __init__(self, energy_thresh=0.005, frame_size=0.02, sample_rate=16000):
        self.energy_thresh = energy_thresh
        self.frame_len = int(frame_size * sample_rate)
        
    def get_speech_mask(self, audio_chunk):
        """
        Returns boolean mask of same length as audio_chunk.
        True = Speech (Energy > Thresh), False = Noise
        """
        # 1. Squared Energy
        sq = audio_chunk ** 2
        
        # 2. Block processing for efficiency (simulating frame-wise)
        # We reshape to [NumFrames, FrameLen] to compute frame energies
        # Truncate to multiple of frame_len
        n_samples = len(audio_chunk)
        n_frames = n_samples // self.frame_len
        
        if n_frames == 0:
            return np.zeros_like(audio_chunk, dtype=bool)
            
        trunc_len = n_frames * self.frame_len
        frames = sq[:trunc_len].reshape(n_frames, self.frame_len)
        
        # Mean Energy per frame
        frame_energies = np.mean(frames, axis=1)
        
        # Threshold
        frame_mask = frame_energies > (self.energy_thresh ** 2)
        
        # Expand back to sample mask
        # Repeat each frame boolean 'frame_len' times
        mask_trunc = np.repeat(frame_mask, self.frame_len)
        
        # Pad remainder with False (usually silence at end)
        full_mask = np.zeros(n_samples, dtype=bool)
        full_mask[:trunc_len] = mask_trunc
        
        return full_mask

    def is_speech(self, audio_chunk):
        """
        Global decision for the chunk.
        True if > 20% of frames are active? Or Median? 
        Let's say if > 10% is active.
        """
        mask = self.get_speech_mask(audio_chunk)
        if np.sum(mask) == 0: return False, 0.0
        
        active_ratio = np.mean(mask)
        # Prob = Active Ratio scaled? 
        # Just binary for now. 
        is_active = active_ratio > 0.1
        return is_active, float(active_ratio)
