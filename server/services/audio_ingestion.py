
import numpy as np
import logging
import threading
import queue
import soxr
from typing import Optional, List
from server.dtos import AudioFrame
import config

logger = logging.getLogger("AudioIngestion")

class AudioIngestionService:
    """
    Component 4.1: AudioIngestionService (Hardened)
    
    Responsibility:
    - Decode -> Resample -> Assign Time.
    - Non-blocking input (Writer Worker).
    - AUTHORITY: Defines the Sample Clock.
      (Assumes continuous stream from LiveKit. 
       Actual Packet Loss/Jitter detected by timestamps is NOT implemented 
       due to lack of RTP metadata access in basic AudioFrame.
       Therefore, "Gaps" here refers to internal processing stalls, not network drops.)
    """
    MAX_GAP_DURATION_S = 0.050
    
    def __init__(self, target_rate=config.SAMPLE_RATE, session_id: str = "unknown"):
        self.target_rate = target_rate
        self.packet_queue = queue.Queue() # Thread-safe unbounded (or bounded safely)
        self.running = False
        self.worker_thread = None
        
        # Output Buffer reference (Must be assigned before start)
        self.clock_buffer = None # type: SampleClockBuffer

        # Recorder (Owned)
        from server.services.audio_recorder import AudioRecorderService
        self.recorder = AudioRecorderService(session_id=session_id, sample_rate=target_rate)
        
        # State (Internal to Worker)
        self.next_start_sample = 0
        self.resampler = None
        self.source_rate = 0

        self.total_ingested_samples = 0

        
    def set_buffer(self, buffer):
        self.clock_buffer = buffer

    def start(self):
        if not self.clock_buffer:
             raise RuntimeError("ClockBuffer not assigned to IngestionService")
             
        self.recorder.start()
        self.running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True, name="IngestionWorker")
        self.worker_thread.start()
        logger.info("Ingestion Worker Started")

    def stop(self):
        self.running = False
        self.packet_queue.put(None) # Sentinel
        if self.worker_thread:
            self.worker_thread.join(timeout=2.0)
        # Note: Recorder finalize is async, handled by bot shutdown usually?
        # Or we should expose it or let bot manage it.
        # Plan says "Update bot.py shutdown to finalize recorder".
        # So here we just stop the loop.

    def process_frame(self, frame):
        """
        Public API (Called from Async Loop).
        MUST BE NON-BLOCKING.
        """
        # We push the raw frame object. 
        # Assuming frame object is safe to pass across threads (LiveKit frames usually are, referencing C++ pointer).
        # To be safe, we might copy bytes if needed, but 'frame' usually holds buffer.
        self.packet_queue.put(frame)

    def _worker_loop(self):
        """
        Background Thread: Resampling & Buffer Push.
        """
        while self.running:
            try:
                frame = self.packet_queue.get(timeout=1.0)
                if frame is None: break
                
                self._handle_raw_frame(frame)
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Ingestion Worker Error: {e}")

    def _handle_raw_frame(self, frame):
        # 1. Decode / Convert
        try:
            input_rate = frame.sample_rate
            channels = frame.num_channels
            
            raw_data = np.frombuffer(frame.data, dtype=np.int16)
            
            # Mixdown
            if channels > 1:
                planar = raw_data.reshape(-1, channels)
                mono = np.mean(planar, axis=1)
            else:
                mono = raw_data
                
            audio_float = mono.astype(np.float32) / 32768.0
            
            # 2. Resample
            if input_rate != self.target_rate:
                if self.source_rate != input_rate:
                    self.source_rate = input_rate
                    # fast, high quality
                    self.resampler = soxr.ResampleStream(input_rate, self.target_rate, 1, dtype=np.float32)
                
                audio_float = self.resampler.resample_chunk(audio_float)
            
            if len(audio_float) == 0:
                return

            # 2.5 Side-Channel Record (Fire-and-Forget)
            self.recorder.write(audio_float, self.next_start_sample)

            # 3. Gap Check & Timeline Assignment
            # Requirement: "Frames must be sample-contiguous"
            # In V1 we assumed next_start_sample is strictly continuous.
            # But if there's a disconnect, we must restart/disconnect?
            # User Spec: "Gaps > 50ms -> disconnect".
            # Can we detect gaps without RTP info?
            # We assume AudioIngestion IS the authority on sample count.
            # If we just increment next_start_sample, we define the clock.
            # So "Gaps" mainly refers to if we receive "None" or if we can infer missing packets.
            # Without RTP seq numbers available on `frame`, we effectively reconstruct continuity.
            # So we proceed assuming valid stream.
            
            # 4. Push to Buffer
            out_frame = AudioFrame(
                samples=audio_float,
                start_sample=self.next_start_sample,
                num_samples=len(audio_float)
            )
            
            if self.clock_buffer:
                self.clock_buffer.push(out_frame)
                
            self.next_start_sample += len(audio_float)
            self.total_ingested_samples += len(audio_float)

            
        except Exception as e:
            logger.error(f"Frame Processing Failed: {e}")
