import asyncio
import logging
import numpy as np
from typing import Dict

import config
from infrastructure.event_bus import EventBus
from components.sample_clock import SampleClockBuffer
from components.embedding_engine import EmbeddingEngine

logger = logging.getLogger("IdentityService")


class IdentityService:
    """
    Annotation-only identity verification.

    Guarantees:
    - Uses sample indices only
    - Never mutates diarization
    - Never blocks pipeline
    """

    def __init__(self, bus: EventBus, clock_buffer: SampleClockBuffer, ingestion, bot):
        self.bus = bus
        self.ingestion = ingestion
        self.bot = bot
        self.clock = clock_buffer
        self.queue = self.bus.subscribe("IdentityService")

        self.embedding_engine = EmbeddingEngine()

        # Injected before run; immutable during session
        self.centroids: Dict[str, np.ndarray] = {}

        self._running = False

    async def run(self):
        self._running = True
        logger.info("IdentityService started.")

        while self._running:
            event = await self.queue.get()
            if event is None:
                break

            if event.get("type") != "segment" or not event.get("is_final"):
                continue

            await self._handle_segment(event)

        logger.info("IdentityService stopped.")

    async def _handle_segment(self, event: dict):
        start = event.get("start_sample")
        end = event.get("end_sample")
        speakers = event.get("speakers", [])
        speaker_id = event.get("meta", {}).get("primary_speaker")

        # Strict guards
        if start is None or end is None:
            return

        if len(speakers) != 1:
            # Identity never runs on overlap or silence
            return

            
        # Get current sample from ingestion to enforce TTL
        current_sample = self.ingestion.total_ingested_samples

        # Query the cache for embeddings belonging strictly to this speaker in this range
        cached_embeddings = self.bot.embedding_cache.query(
            start, end, speaker_id, current_sample
        )

        if cached_embeddings:
            logger.info(f"♻️ Cache Hit: Reusing {len(cached_embeddings)} embeddings for {speaker_id}")
            # Deterministic Aggregation: Mean pooling of cached windows
            embedding = np.mean(cached_embeddings, axis=0)
            embedding /= np.linalg.norm(embedding) # Safe Normalization
        else:
            # Fallback: Standard extraction if cache misses or data is too old
            audio = self.clock.get_lookback(start, end)

            # Guard against insufficient audio for a new extraction
            min_samples = int(config.MIN_AUDIO_FOR_EMBEDDING * config.SAMPLE_RATE)
            if audio is None or len(audio) < min_samples:
                self._emit_annotation(
                    start, end, speaker_id,
                    status="unknown",
                    similarity=None,
                    reason="insufficient_audio"
                )
                return

        logger.debug(f"⚠️ Cache Miss: Extracting embedding for {speaker_id} from {start} to {end}")
        embedding = self.embedding_engine.extract_embedding(audio)
        if embedding is None:
            self._emit_annotation(
                start, end, speaker_id,
                status="unknown",
                similarity=None,
                reason="embedding_failure"
            )
            return

        # IDENTITY VERIFICATION (Similarity Check)
        centroid = self.centroids.get(speaker_id)
        if centroid is None:
            self._emit_annotation(
                start, end, speaker_id,
                status="unknown",
                similarity=None,
                reason="no_centroid"
            )
            return

        # Perform Cosine Similarity check against the speaker's known centroid
        similarity = float(np.dot(embedding, centroid))
        if similarity >= config.SIMILARITY_THRESHOLD:
            self._emit_annotation(
                start, end, speaker_id,
                status="verified",
                similarity=similarity,
                reason=None
            )
        else:
            self._emit_annotation(
                start, end, speaker_id,
                status="unverified",
                similarity=similarity,
                reason="low_similarity"
            )

    def _emit_annotation(
        self,
        start: int,
        end: int,
        speaker: str,
        status: str,
        similarity: float | None,
        reason: str | None
    ):
        annotation = {
            "type": "speaker_identity",
            "segment_start_sample": start,
            "segment_end_sample": end,
            "speaker": speaker,
            "status": status,
            "similarity": similarity,
            "reason": reason,
        }
        self.bus.publish(annotation)


# import asyncio
# import logging

# from typing import Dict
# from infrastructure.event_bus import EventBus
# # from services.audio_ingestion import AudioIngestionService
# from components.sample_clock import SampleClockBuffer
# import config

# from components.embedding_engine import EmbeddingEngine
# import numpy as np

# logger = logging.getLogger("IdentityService")

# class IdentityService:
#     """
#     Component 4.9: IdentityService (Hardened Phase 4)
#     Phase 5A: Annotation-Only Verification (Stateless)
    
#     Responsibility:
#     - Consume Segments.
#     - Extract Embeddings (Lookback).
#     - Verify against Injected Centroids (Read-Only).
#     - STRICT: Uses sample indices from Event. NO Seconds Conversion.
#     """
    
#     # def __init__(self, event_bus: EventBus, ingestion_service: AudioIngestionService):
#     def __init__(self, bus: EventBus, clock_buffer: SampleClockBuffer):
#         self.bus = bus
#         # self.ingestion = ingestion_service
#         self.clock_buffer = clock_buffer
#         self.queue = self.bus.subscribe("IdentityService")
        
#         # Phase 5A: Identity Engine
#         self.embedding_engine = EmbeddingEngine()
        
#         # Injected before run; immutable during session.
#         self.centroids = Dict[str, np.ndarray] = {}
#         self.running = False
        
        

#     async def run(self):
#         self.running = True
#         logger.info("IdentityService Started (Sample Authority)")
        
#         while self.running:
#             event = await self.queue.get()
#             if event is None: break
            
#             if event.get("type") != "segment" and event.get("is_final"):
#                 continue
            
#             await self._handle_segment(event)
            
#         logger.info("IdentityService Stopped")

#     async def _handle_segment(self, event):
#         try:
#             # Fix 4: Use Authoritative Sample Indices
#             start_sample = event.get("start_sample")
#             end_sample = event.get("end_sample")
#             speakers = event.get("speakers", [])
            
#             if start_sample is None or end_sample is None:
#                 logger.error("IdentityService: Missing sample indices in event.")
#                 return
            
#             if len(speakers) != 1:
#                  # Identity never runs on overlap or silence
#                 return

#             # Lookup Audio
#             if self.clock_buffer:
#                 # Direct sample access
#                 audio = self.clock_buffer.get_lookback(start_sample, end_sample)
                
#                 if audio is not None and len(audio) > 0:
                     
#                      # Check Ambiguity (Strict Rule: Identity works on single-speaker segments only)
#                      if len(speakers) != 1:
#                          # logger.warning(f"IdentityService: Skipping overlap/empty segment {start_sample}-{end_sample} speakers={speakers}")
#                          return
                     
#                      speaker_id = speakers[0]
                     
#                      # 1. Compute Embedding
#                      embedding = self.embedding_engine.extract_embedding(audio)
                     
#                      status = "unknown"
#                      reason = "no_centroid"
#                      similarity = None
                     
#                      if embedding is None:
#                          reason = "insufficient_audio" # or model_failure
#                          if len(audio) < int(config.MIN_AUDIO_FOR_EMBEDDING * config.SAMPLE_RATE):
#                               reason = "insufficient_audio"
#                          else:
#                               reason = "model_failure"
#                      else:
#                         # 2. Compare against Centroid (Read-Only)
#                         # Centroids must be injected. If missing, we cannot verify.
#                         centroid = self.centroids.get(speaker_id)
                        
#                         if centroid is not None:
#                             # Cosine Similarity (Both are L2 normalized)
#                             similarity = float(np.dot(embedding, centroid))
                            
#                             if similarity >= config.SIMILARITY_THRESHOLD:
#                                 status = "verified"
#                                 reason = None # Verified needs no reason? Or explicit "high_similarity"?
#                                 # Schema doesn't strictly forbid null reason for verified, but plan implied stricter.
#                                 # Plan: "unverified is ONLY valid with reason = low_similarity". "unknown for no_centroid..."
#                                 # Let's assume reason is null or "threshold_met" for verified. Let's send null.
#                             else:
#                                 status = "unverified"
#                                 reason = "low_similarity"
#                         else:
#                             status = "unknown"
#                             reason = "no_centroid"

#                      # Emit Annotation Event (Adhering to Schema Layer 2 + Phase 5A Strict)
#                      annotation = {
#                          "type": "speaker_identity",
#                          "segment_start_sample": start_sample,
#                          "segment_end_sample": end_sample,
#                          "speaker": speaker_id,
#                          "status": status,
#                          "similarity": similarity,
#                          "reason": reason
#                      }
                     
#                      # STRICT RULE: No aggregation, No centroid updates.
#                      self.bus.publish(annotation)
#                 else:
#                      logger.warning(f"IdentityService: Audio expired/missing for {start_sample}-{end_sample}")
                     
#         except Exception as e:
#             logger.error(f"Identity Handling Error: {e}")
