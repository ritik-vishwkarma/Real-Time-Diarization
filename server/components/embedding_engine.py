import logging
import torch
import numpy as np
from threading import Lock
from server import config

try:
    import nemo.collections.asr as nemo_asr
except ImportError:
    nemo_asr = None

logger = logging.getLogger("EmbeddingEngine")

class EmbeddingEngine:
    """
    Component 5.1: EmbeddingEngine (Phase 5A)
    
    Responsibility:
    - Wrap TitaNet Model.
    - Accept raw float32 audio (NO Amplitude Normalization).
    - Return L2 Normalized Embeddings.
    - Thread-safe Inference.
    """

    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lock = Lock()
        self.model = None
        
        try:
            logger.info(f"Loading TitaNet model: {config.EMBEDDING_MODEL} on {self.device}")
            # Suppress NeMo logging
            logging.getLogger("nemo_logger").setLevel(logging.ERROR)
            
            self.model = nemo_asr.models.EncDecSpeakerLabelModel.from_pretrained(
                model_name=config.EMBEDDING_MODEL
            )
            self.model.to(self.device)
            self.model.eval()
            self.model.freeze()
            logger.info("TitaNet loaded successfully.")
            
        except Exception as e:
            logger.error(f"Failed to load TitaNet: {e}")
            self.model = None

    def extract_embedding(self, audio_chunk: np.ndarray) -> np.ndarray:
        """
        Extract embedding from raw audio.
        Input: 1D numpy array (float32).
        Output: 1D numpy array (normalized embedding) or None.
        
        Strict Contract:
        - Input is assumed raw (no amplitude scaling applied here).
        - Output is L2 normalized.
        """
        if self.model is None:
            return None
            
        if len(audio_chunk) < int(config.MIN_AUDIO_FOR_EMBEDDING * config.SAMPLE_RATE):
            return None

        with self.lock:
            try:
                # Convert to tensor
                wav_tensor = torch.from_numpy(audio_chunk).float().to(self.device)
                input_signal = wav_tensor.unsqueeze(0) # [1, T]
                input_length = torch.tensor([wav_tensor.shape[0]]).to(self.device)
                
                # Inference
                with torch.no_grad():
                    # TitaNet forward returns (logits, embeddings)
                    # We need to extract the embedding from the intermediate layer usually
                    # NeMo's verify_speaker usually handles this, but let's use the forward pass.
                    # Or use `forward_for_export` if available?
                    # Safer: input_signal, input_signal_length -> embs
                    _, embs = self.model.forward(input_signal=input_signal, input_signal_length=input_length)
                    
                    # embs shape: [1, D]
                    emb_np = embs.squeeze(0).cpu().numpy()
                    
                    # L2 Normalize
                    norm = np.linalg.norm(emb_np)
                    if norm > 1e-9:
                        emb_np = emb_np / norm
                        
                    return emb_np
                    
            except Exception as e:
                logger.error(f"Embedding Extraction Failed: {e}")
                return None
