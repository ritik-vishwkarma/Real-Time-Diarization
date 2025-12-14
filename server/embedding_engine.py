import torch
import logging
import numpy as np
from nemo.collections.asr.models import EncDecSpeakerLabelModel

logger = logging.getLogger("EmbeddingEngine")

class SpeakerEmbedding:
    def __init__(self, model_name="nvidia/speakerverification_en_titanet_large", device=None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Loading Embedding Model: {model_name} on {self.device}...")
        
        try:
            self.model = EncDecSpeakerLabelModel.from_pretrained(model_name).to(self.device).eval()
        except Exception as e:
            logger.error(f"Failed to load TitaNet: {e}")
            raise e

    def get_embedding(self, audio_chunk_16k):
        """
        Extract 192-dim embedding from an audio chunk.
        Args:
            audio_chunk_16k (np.array): 1D float32 array @ 16kHz. 
                                        Should be at least 0.8s long usually.
        Returns:
            np.array: (192,) Normalized embedding vector.
        """
        # 1. Preprocess
        # audio_chunk needs to be a tensor [1, Length]
        if len(audio_chunk_16k) < 160: # minimal check
            return None
            
        with torch.no_grad():
            tensor = torch.from_numpy(audio_chunk_16k).float().to(self.device)
            if tensor.ndim == 1:
                tensor = tensor.unsqueeze(0)
            
            input_length = torch.tensor([tensor.shape[1]]).to(self.device)
            
            # 2. Extract
            # TitaNet returns (embedding, logits) usually, or just embedding depending on mode.
            # forward() -> (logits, embeddings)
            _, embs = self.model(input_signal=tensor, input_signal_length=input_length)
            
            # embs shape: [Batch, 192]
            emb_vector = embs[0].cpu().numpy()
            
            # 3. L2 Normalize (Critical for Cosine Similarity / FAISS IP)
            norm = np.linalg.norm(emb_vector)
            if norm > 1e-6:
                emb_vector = emb_vector / norm
                
            return emb_vector
