import asyncio
import numpy as np
import torch
import logging
import json
import time
import io
import torchaudio.transforms as T
from collections import deque
from livekit import agents, rtc
from livekit.agents import JobContext, WorkerOptions, cli
from streaming_diarizer import StreamingDiarizer
from vad_engine import NeMoVADGate

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DiarizationBot")

import soundfile as sf
import os
import datetime
from storage.minio_client import MinioUploader

# --- GLOBAL ENGINES (Singleton) ---
_embedding_model = None # TitaNet
_speaker_db = None # FAISS DB

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from embedding_engine import SpeakerEmbedding
        # Initialize on CPU or CUDA (TitaNet is ~500MB)
        logger.info("Initializing Shared Speaker Embedding Model...")
        _embedding_model = SpeakerEmbedding()
    return _embedding_model

def get_speaker_db():
    global _speaker_db
    if _speaker_db is None:
        from speaker_db import SpeakerProfileDB
        logger.info("Initializing Shared Speaker DB...")
        _speaker_db = SpeakerProfileDB()
    return _speaker_db

class DiarizationAgent:
    def __init__(self, room: rtc.Room):
        self.room = room
        # Create a fresh Diarizer (Own Model + Own State) for every session.
        self.diarizer = StreamingDiarizer(use_nemo_vad=False) 
        
        # Load Shared Engines
        self.embedding_model = get_embedding_model()
        self.speaker_db = get_speaker_db()
        self.session_labels = {} # { "spk_0": "Alice" }
        
        self.uploader = MinioUploader()
        self.vad_gate = NeMoVADGate() # Lightweight Gate
        
        # Audio Processing State
        # Dynamic Resamplers: { source_rate: resampler_obj }
        self.resamplers = {} 
        self.audio_accum = np.array([], dtype=np.float32)
        self.vad_buffer = np.array([], dtype=np.float32)
        
        # --- NEW: Ring Buffer for Re-ID (Store last 5 seconds) ---
        # LiveKit chunks are usually 10ms (160 samples) or 20ms? 
        # Actually in run_audio_loop we get varying sizes but usually small.
        # Let's say we store raw chunks. 
        # 5 seconds at 10ms/chunk = 500 chunks. Safe buffer 1000.
        self.audio_ring_buffer = deque(maxlen=2000) 
        
        self.processed_samples = 0
        self.total_dropped_samples = 0
        
        # Session Recording
        self.session_id = f"{room.name}_{int(time.time())}"
        self.temp_wav_path = f"session_{self.session_id}.wav"
        self.sf_file = sf.SoundFile(self.temp_wav_path, mode='w', samplerate=16000, channels=1)
        self.events_log = []

    async def run_audio_loop(self, audio_stream: rtc.AudioStream):
        logger.info(f"🔴 LiveKit Audio Stream Started. Recording to {self.temp_wav_path}")
        loop = asyncio.get_running_loop()
        
        try:
            async for event in audio_stream:
                # 1. Convert LiveKit Frame (Int16 48k) to Float32 Tensor
                frame = event.frame
                data = np.frombuffer(frame.data, dtype=np.int16).astype(np.float32) / 32768.0
                
                if len(data) == 0:
                    continue
    
                tensor = torch.from_numpy(data)
                
                # 2. Resample (Dynamic) via CPU (Save GPU for Model)
                source_rate = frame.sample_rate
                if source_rate != 16000:
                    if source_rate not in self.resamplers:
                        logger.info(f"Creating Resampler: {source_rate} -> 16000")
                        self.resamplers[source_rate] = T.Resample(source_rate, 16000)
                    
                    tensor_16k = self.resamplers[source_rate](tensor)
                else:
                    tensor_16k = tensor
                
                chunk_np_16k = tensor_16k.numpy()
                
                # 3. Write RAW audio to recording (ASYNC IO - Non-Blocking)
                # We offload the synchronous disk write to a thread executor
                await loop.run_in_executor(None, self.sf_file.write, chunk_np_16k)
                
                # 3.5 Update Ring Buffer (For Re-ID lookback)
                self.audio_ring_buffer.append(chunk_np_16k)
                
                # 4. VAD Gating Strategy
                # Accumulate small chunks for VAD check (0.1s = 1600 samples)
                self.vad_buffer = np.concatenate((self.vad_buffer, chunk_np_16k))
                
                if len(self.vad_buffer) >= 1600:
                    # Take exactly 1600 samples
                    gate_chunk = self.vad_buffer[:1600]
                    self.vad_buffer = self.vad_buffer[1600:] # Keep remainder
                    
                    is_speech, prob = self.vad_gate.is_speech(gate_chunk, threshold=0.2)
                    
                    if is_speech:
                         # PASS: Speech detected, add to main buffer
                         if prob < 0.5:
                             logger.debug(f"✅ VAD Pass (Weak): {prob:.4f}")
                         self.audio_accum = np.concatenate((self.audio_accum, gate_chunk))
                    else:
                         # BLOCK: Log why we blocked it if prob was borderline
                         if prob > 0.05:
                            logger.debug(f"🛑 VAD Blocked: Prob {prob:.4f} < 0.2")
                         
                         # Silence. Feed Zeros.
                         zeros = np.zeros_like(gate_chunk)
                         self.audio_accum = np.concatenate((self.audio_accum, zeros))
                
                # 5. Diarization Pipeline (Main Buffer)
                # If buffer exceeds ~1.5s (approx 3 chunks)
                CHUNK_SAMPLES = self.diarizer.chunk_samples # ~24000 for 1.5s
                
                if len(self.audio_accum) >= CHUNK_SAMPLES:
                    # --- BACKPRESSURE CHECK ---
                    MAX_BUFFER = 3 * CHUNK_SAMPLES
                    if len(self.audio_accum) > MAX_BUFFER:
                        overflow = len(self.audio_accum) - MAX_BUFFER
                        if overflow > 1600: 
                             logger.warning(f"⚠️ Backpressure: Dropping {overflow/16000:.2f}s.")
                        self.processed_samples += overflow
                        self.total_dropped_samples += overflow
                        self.audio_accum = self.audio_accum[overflow:]
                    
                    # Process exactly one chunk
                    if len(self.audio_accum) >= CHUNK_SAMPLES:
                        process_chunk = self.audio_accum[:CHUNK_SAMPLES]
                        self.audio_accum = self.audio_accum[CHUNK_SAMPLES:] 
                        
                        # Optimization: Skip Inference if pure silence (zeros)
                        if np.max(np.abs(process_chunk)) < 0.0001:
                            pass
                        else:
                            current_time = (self.processed_samples + CHUNK_SAMPLES) / 16000.0
                            await self.inference_step(process_chunk, current_time)
                        
                        # Note: We need to increment time even if skipped? 
                        # Ah, processed_samples is incremented below. Correct.
                        
                        # --- TIER 5: Real-time Identification ---
                        # Check if we have a clean segment to identify
                        sm = self.diarizer.state_machine
                        # Only ID if single speaker to avoid pollution
                        if len(sm.current_speakers) == 1:
                            spk = list(sm.current_speakers)[0]
                            # Only identify if unknown in this session
                            if spk not in self.session_labels:
                                duration = current_time - sm.start_time
                                if duration > 1.2: # Wait for 1.2s of clean speech
                                    # Trigger Async Re-ID
                                    # Mark as "pending" to avoid spam
                                    self.session_labels[spk] = "pending"
                                    asyncio.create_task(self.identify_speaker_task(spk, current_time))
                                    
                        self.processed_samples += CHUNK_SAMPLES
                    
        except asyncio.CancelledError:
            logger.info("Audio loop cancelled")
        except Exception as e:
            logger.error(f"Error in audio loop: {e}")
        finally:
            logger.info("🛑 Cleaning up session...")
            
            # FINALIZE State Machine (Close last segment)
            try:
                final_time = self.processed_samples / 16000.0
                final_events = self.diarizer.close(final_time)
                if final_events:
                    self.events_log.extend(final_events)
                    logger.info(f"✅ Closed {len(final_events)} final segments.")
            except Exception as e:
                logger.error(f"Error closing diarizer: {e}")
                
            self.sf_file.close()

            # Metric Summary
            dropped_sec = self.total_dropped_samples / 16000.0
            logger.info(f"📊 Session Metrics: Dropped {dropped_sec:.2f}s due to backpressure.")
            
            # Upload to MinIO
            logger.info(f"Uploading session {self.session_id} to Storage...")
            success = self.uploader.upload_file(f"sessions/{self.session_id}/full.wav", self.temp_wav_path, "audio/wav")
            
            # Upload Metadata (Structured)
            logger.info(f"💾 Saving Metadata with {len(self.events_log)} events.")
            metadata_struct = {
                "session_id": self.session_id,
                "sample_rate": 16000,
                "events": self.events_log
            }
            metadata_json = json.dumps(metadata_struct, indent=2)
            self.uploader.upload_bytes(f"sessions/{self.session_id}/metadata.json", metadata_json.encode('utf-8'), "application/json")
            
            # Clean local file ONLY if upload succeeded
            if success:
                if os.path.exists(self.temp_wav_path):
                    os.remove(self.temp_wav_path)
                logger.info("✅ Session Uploaded & Cleanup Complete.")
                
                # Log Presigned URL for convenience
                url = self.uploader.get_presigned_url(f"sessions/{self.session_id}/full.wav")
                if url:
                    logger.info(f"🔗 Download URL: {url}")
            else:
                logger.error(f"❌ Upload failed. Keeping local file: {self.temp_wav_path}")

    async def identify_speaker_task(self, spk_id, end_time):
        """
        Extracts embedding from MEMORY Ring Buffer.
        Zero Disk I/O. Safe from race conditions.
        """
        try:
            # 1. Reconstruct recent audio from Ring Buffer
            if not self.audio_ring_buffer:
                return

            # Combine all chunks currently in memory
            # This is fast: < 5ms for 5 seconds of audio
            full_buffer = np.concatenate(list(self.audio_ring_buffer))
            
            # We want the LAST 2 seconds (32000 samples)
            # TitaNet needs > 0.5s
            if len(full_buffer) < 8000:
                logger.debug(f"Not enough audio in buffer for Re-ID of {spk_id}")
                del self.session_labels[spk_id] 
                return
                
            # Take last 2.5s to be safe
            data = full_buffer[-40000:] 
            
            if data is not None and len(data) > 8000: # at least 0.5s
                     # 2. Extract
                     emb = self.embedding_model.get_embedding(data)
                     if emb is not None:
                         # --- TIER 8: Active Verification (Re-ID) ---
                         # Keep track of what this session speaker sounds like.
                         if not hasattr(self, "session_centroids"):
                             self.session_centroids = {} # { "spk_0": {"vector": np.array, "count": N} }
                             
                         # Normalize embedding
                         emb_norm = emb / np.linalg.norm(emb)
                         
                         verified_label = None
                         drift_detected = False
                         
                         # CASE A: We already labeled this speaker (e.g. spk_0 -> "Alice")
                         if spk_id in self.session_labels and self.session_labels[spk_id] not in ["unknown", "pending"]:
                             current_label = self.session_labels[spk_id]
                             
                             # Verify against Session Centroid
                             if spk_id in self.session_centroids:
                                 centroid = self.session_centroids[spk_id]["vector"]
                                 sim = np.dot(emb_norm, centroid)
                                 
                                 if sim < 0.6: # Drift Threshold
                                     logger.warning(f"⚠️ Drift Detected for {spk_id} ({current_label}). Sim: {sim:.2f}. Invalidating Label.")
                                     del self.session_labels[spk_id] # Force re-id
                                     drift_detected = True
                                 else:
                                     # Strong Match - Update Centroid
                                     # Running Avg
                                     N = self.session_centroids[spk_id]["count"]
                                     new_cent = (centroid * N + emb_norm) / (N + 1)
                                     new_cent = new_cent / np.linalg.norm(new_cent)
                                     self.session_centroids[spk_id] = {"vector": new_cent, "count": N + 1}
                                     verified_label = current_label
                                     score = float(sim) # Use internal similarity as score
                             else:
                                 # Should not happen if logic is correct, but init if missing
                                 self.session_centroids[spk_id] = {"vector": emb_norm, "count": 1}
                                 verified_label = current_label

                         # CASE B: Not Labeled or Drifted -> Identify against DB
                         if verified_label is None:
                             # 3. Identify
                             label, score = self.speaker_db.identify(emb)
                             
                             if label:
                                 self.session_labels[spk_id] = label
                                 # Init/Reset Session Centroid
                                 self.session_centroids[spk_id] = {"vector": emb_norm, "count": 1}
                             else:
                                 self.session_labels[spk_id] = "unknown"
                             
                             verified_label = label

                         # --- TIER 7: Upload Snippet (Keep existing logic) ---
                         snippet_path = f"sessions/{self.session_id}/segments/{spk_id}.wav"
                         if not hasattr(self, "uploaded_snippets"):
                             self.uploaded_snippets = set()
                             
                         if (spk_id not in self.uploaded_snippets) or drift_detected:
                             try:
                                buffer = io.BytesIO()
                                sf.write(buffer, data, 16000, format='WAV', subtype='PCM_16')
                                buffer.seek(0)
                                self.uploader.upload_bytes(snippet_path, buffer.read(), "audio/wav")
                                self.uploaded_snippets.add(spk_id)
                                
                                emb_list = emb.flatten().tolist()
                                vec_path = f"sessions/{self.session_id}/segments/{spk_id}.json"
                                vec_json = json.dumps({"vector": emb_list})
                                self.uploader.upload_bytes(vec_path, vec_json.encode('utf-8'), "application/json")
                                
                                logger.info(f"📤 Uploaded snippet/vector for {spk_id} (Drift: {drift_detected})")
                             except Exception as exc:
                                logger.error(f"Snippet/Vector upload failed: {exc}")

                         if verified_label:
                             logger.info(f"🔍 Identified {spk_id} as {verified_label} (Score: {score:.4f})")
                             
                             event = {
                                 "type": "speaker_identity",
                                 "spk_id": spk_id,
                                 "label": verified_label,
                                 "score": score
                             }
                             self.events_log.append(event)
                             if self.room.local_participant:
                                 await self.room.local_participant.publish_data(
                                     payload=json.dumps(event),
                                     topic="diarization"
                                 )
                         else:
                             logger.info(f"UNKNOWN Speaker {spk_id} (Score: {score:.4f})")



        except Exception as e:
            logger.error(f"Identity Task Error: {e}")
            if spk_id in self.session_labels:
                del self.session_labels[spk_id] # Retry next time
            

    async def inference_step(self, audio_chunk, current_time):
        # 1. Pipeline
        # PASS vad_is_speech=True to skip internal VAD (Double VAD Fix)
        events = self.diarizer.process_chunk(audio_chunk, current_time, vad_is_speech=True)
        
        # 2. Publish Events (and Log)
        if events:
            for event in events:
                self.events_log.append(event) # Store for metadata
                payload = json.dumps(event)
                try:
                    if self.room.local_participant:
                        await self.room.local_participant.publish_data(
                            payload=payload,
                            topic="diarization"
                        )
                        logger.info(f"📡 Emit: {event['type']} {event.get('speakers', [])}")
                except Exception as e:
                    logger.error(f"Error publishing: {e}")

async def entrypoint(ctx: JobContext):
    # 1. Connect to Room
    await ctx.connect()
    logger.info(f"Joined Room: {ctx.room.name}")
    
    agent = DiarizationAgent(ctx.room)

    # 2. Subscribe to the first audio track we see
    @ctx.room.on("track_subscribed")
    def on_track_subscribed(track: rtc.Track, publication: rtc.TrackPublication, participant: rtc.RemoteParticipant):
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            logger.info(f"Subscribed to audio from {participant.identity}")
            # Start the audio loop
            audio_stream = rtc.AudioStream(track)
            asyncio.create_task(agent.run_audio_loop(audio_stream))

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    # Run the worker
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
