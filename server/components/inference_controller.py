
from server.diarization_engine import DiarizationEngine
from server.components.embedding_engine import EmbeddingEngine
from server.dtos import DiarizationFrameOutput
import logging

logger = logging.getLogger("InferenceController")

class InferenceController:
    """
    Manages Model Inference (Sortformer + TitaNet).
    Handles 'Fast Mode' (skipping inference under load).
    """
    def __init__(self, engine=None):
        if engine:
            self.engine = engine
        else:
            self.engine = DiarizationEngine() # Heavy Load
            
        # Lifecycle Management: Initialize once, not per frame
        self.embedding_engine = EmbeddingEngine() 

    @property
    def chunk_samples(self):
        return self.engine.chunk_samples

    def process(self, norm_audio, raw_audio=None, fast_mode=False) -> DiarizationFrameOutput:
        """
        Run inference on audio chunk.
        Args:
            norm_audio: Normalized Audio (for Sortformer).
            raw_audio: Raw Audio (for TitaNet). Optional.
            fast_mode: If True, skip model and return status="skip".
        Returns:
            DiarizationFrameOutput
        """
        if fast_mode:
            # Skip heavy inference
            return DiarizationFrameOutput(probs={}, status="skip")
            
        try:
            _, raw_probs = self.engine.process_chunk(norm_audio)
            
            if not raw_probs:
                return DiarizationFrameOutput(probs={}, status="empty")

            # Optimization: Only run heavy TitaNet if it's a single speaker
            embedding = None
            is_overlap = len(raw_probs) > 1
            
            if not is_overlap and raw_audio is not None:
                # Run embedding on raw chunk
                embedding = self.embedding_engine.extract_embedding(raw_audio)

            return DiarizationFrameOutput(
                probs=raw_probs, 
                embedding=embedding, 
                status="success"
            )
        except Exception as e:
            import traceback
            logger.error(f"Inference Failed: {e}\n{traceback.format_exc()}")
            return DiarizationFrameOutput(probs={}, status="error")
