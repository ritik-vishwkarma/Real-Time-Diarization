
import logging
import torch
import numpy as np
from typing import Optional, Dict
from server.components.inference_controller import InferenceController
from server.dtos import DiarizationFrameOutput

logger = logging.getLogger("StreamingDiarizer")

class StreamingDiarizer:
    """
    Component 4.5: StreamingDiarizer (Hardened)
    
    Responsibility:
    - Pure Inference ONLY.
    - Input: ConditionedAudio.
    - Output: DiarizationFrameOutput.
    
    NO side effects, NO state machine, NO smoothing.
    """
    def __init__(self):
        # We wrap the InferenceController which wraps the Engine
        self.controller = InferenceController() 

    def process_window(self, cond_audio) -> DiarizationFrameOutput:
        """
        Run inference.
        Returns:
            DiarizationFrameOutput
        """
        try:
            # Pass both normalized and raw
            # cond_audio has .normalized and .raw
            result = self.controller.process(
                norm_audio=cond_audio.normalized,
                raw_audio=cond_audio.raw,
                fast_mode=False
            )
            return result
            
        except Exception as e:
            logger.error(f"Diarizer Inference Failed: {e}")
            return DiarizationFrameOutput(probs={}, status="error")
