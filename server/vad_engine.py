import torch
import logging
import numpy as np
import os
from nemo.collections.asr.models import EncDecClassificationModel

logger = logging.getLogger("VADEngine")

class NeMoVADGate:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load the Multilingual MarbleNet (Lightweight CNN)
        model_name = "vad_multilingual_marblenet"
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
