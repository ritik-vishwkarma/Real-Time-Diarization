
import logging
import wave
import os
import shutil
from typing import Optional
import numpy as np
import config

# Optional MinIO
try:
    from storage.minio_client import MinioUploader
except ImportError:
    MinioUploader = None

logger = logging.getLogger("AudioRecorder")

class AudioRecorderService:
    """
    Component 4.6: AudioRecorderService (New, Strict Separation)

    Responsibility:
    - Record raw, environment audio (16kHz Mono).
    - Produce one continuous WAV using incremental writes.
    - Upload to MinIO on finalize.
    - NO dependency on diarization decisions.
    
    Hard Rules:
    - Input: float32, normalized to [-1, 1], writes int16.
    - Time Authority: Accepts chunks in order.
    - Output: Local /tmp WAV -> MinIO.
    """
    
    def __init__(self, session_id: str, sample_rate: int = config.SAMPLE_RATE, output_dir: str = "."):
        self.session_id = session_id
        self.sample_rate = sample_rate
        self.output_dir = output_dir
        self.temp_filename = f"session_{self.session_id}_raw.wav"
        self.temp_path = os.path.join(self.output_dir, self.temp_filename)
        
        self.uploader = MinioUploader() if MinioUploader else None
        self.wav_file: Optional[wave.Wave_write] = None
        self.is_recording = False
        self.total_samples_written = 0

    def start(self):
        """Open WAV file for writing."""
        try:
            self.wav_file = wave.open(self.temp_path, "wb")
            self.wav_file.setnchannels(1)
            self.wav_file.setsampwidth(2) # 16-bit
            self.wav_file.setframerate(self.sample_rate)
            self.is_recording = True
            logger.info(f"Recording started: {self.temp_path}")
        except Exception as e:
            logger.error(f"Failed to start recording: {e}")
            self.is_recording = False

    def write(self, audio_chunk: np.ndarray, start_sample: int):
        """
        Write audio chunk to WAV. 
        Args:
           audio_chunk: float32 array [-1.0, 1.0]
           start_sample: (Unused for file structure, simple append, assumes continuity from Ingestion)
                         (Could be used for sanity check against total_samples_written)
        """
        if not self.is_recording or self.wav_file is None:
            return

        try:
            # 1. Float32 -> Int16
            # Clip safe
            audio_clipped = np.clip(audio_chunk, -1.0, 1.0)
            # Scale
            audio_int16 = (audio_clipped * 32767).astype(np.int16)
            
            # 2. Write
            self.wav_file.writeframes(audio_int16.tobytes())
            self.total_samples_written += len(audio_chunk)
            
        except Exception as e:
            logger.error(f"Write failed: {e}")

    async def finalize_and_upload(self):
        """
        Close file and upload to MinIO.
        """
        if self.wav_file:
            try:
                self.wav_file.close()
                self.is_recording = False
                logger.info(f"Recording finalized. Total Samples: {self.total_samples_written}")
            except Exception as e:
                logger.error(f"Error closing WAV: {e}")

        # Upload
        if self.uploader and os.path.exists(self.temp_path):
            remote_path = f"sessions/{self.session_id}/raw_audio.wav"
            logger.info(f"Uploading recording to {remote_path}...")
            
            success = self.uploader.upload_file(
                object_name=remote_path,
                file_path=self.temp_path
            )
            
            if success:
                logger.info("Recording Upload Successful.")
                # Optional: Delete local file? 
                # Keep for debug or delete? Let's keep for now or strictly follow "Optionally delete".
                # User said "Optionally delete local file". Let's delete to save space.
                try:
                    os.remove(self.temp_path)
                    logger.info("Local recording deleted.")
                except Exception as e:
                    logger.warning(f"Could not delete local recording: {e}")
            else:
                logger.error("Recording Upload Failed.")
        else:
            if not self.uploader:
                logger.warning("No MinIO uploader, recording remains local.")
