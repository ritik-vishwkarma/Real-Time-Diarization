import asyncio
import logging
import time
import sys
import os
from functools import partial

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from livekit import agents, rtc
from livekit.agents import JobContext, WorkerOptions, cli
from dotenv import load_dotenv

# Types
from server.dtos import FrameDecision, WindowExecutionPlan, DiarizationFrameOutput

# Infrastructure
from infrastructure.event_bus import EventBus
from services.audio_ingestion import AudioIngestionService
from services.identity_service import IdentityService
from services.storage_service import StorageService

# Components
from components.sample_clock import SampleClockBuffer
from components.vad_gate import VADGate
from components.sliding_window import SlidingWindowProcessor
from components.probability_smoother import ProbabilitySmoother
from components.overlap_detector import OverlapDetector
from components.speaker_continuity_cache import SpeakerContinuityCache
from streaming_diarizer import StreamingDiarizer
from state_machine import SpeakerStateMachine
from components.embedding_engine import EmbeddingEngine
from components.embedding_cache import EmbeddingCache
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DiarizationBot")

# Architecture Constants
MAX_CONTINUITY_SEC = 10.0 # Allow long lags to be bridged

class DiarizationAgent:
    def __init__(self, room: rtc.Room):
        self.room = room
        self.session_id = f"{room.name}_{int(time.time())}"
        
        self.event_bus = EventBus()
        self.clock_buffer = SampleClockBuffer()
        
        self.ingestion = AudioIngestionService(session_id=self.session_id)
        self.ingestion.set_buffer(self.clock_buffer)
        
        self.window_processor = SlidingWindowProcessor()
        self.vad_gate = VADGate()
        self.diarizer = StreamingDiarizer()
        self.smoother = ProbabilitySmoother()
        self.overlap = OverlapDetector()
        self.cache = SpeakerContinuityCache()
        self.embedding_engine = EmbeddingEngine()
        self.embedding_cache = EmbeddingCache(max_entries=config.EMBEDDING_CACHE_MAX)
        self.state_machine = SpeakerStateMachine(self.event_bus)
        
        self.identity = IdentityService(bus=self.event_bus, ingestion=self.ingestion, bot=self, clock_buffer=self.clock_buffer)
        # self.identity.clock_buffer = self.clock_buffer
        self.storage = StorageService(self.event_bus, self.session_id)
        
        self.tasks = []
        self._shutdown_complete = False
        self.last_known_probs = {}

        self.stats = {
            "windows_processed": 0,
            "windows_skipped": 0,
            "segments_emitted": 0, 
            "decisions_speech": 0,
            "decisions_uncertain": 0,
        }
        
        self.consecutive_skips = 0

    async def start(self):
        self.ingestion.start()
        
        self.tasks.append(asyncio.create_task(self.identity.run()))
        self.tasks.append(asyncio.create_task(self.storage.run()))
        self.tasks.append(asyncio.create_task(self.emit_loop()))
        self.tasks.append(asyncio.create_task(self.processing_loop()))
        self.tasks.append(asyncio.create_task(self.metrics_loop()))
        
        logger.info(f"Session {self.session_id} Started.")

    async def run_audio_source(self, stream: rtc.AudioStream):
        try:
            async for event in stream:
                self.ingestion.process_frame(event.frame)
        finally:
            await self.shutdown()

    async def processing_loop(self):
        logger.info("Processing Loop Started")
        while not self._shutdown_complete:
            try:
                await asyncio.to_thread(self.tick_pipeline)
                await asyncio.sleep(0.01)
            except Exception as e:
                logger.error(f"Processing Loop Error: {e}")
                await asyncio.sleep(0.1)

    def tick_pipeline(self):
        current_head = self.clock_buffer.current_head
        plans = self.window_processor.get_pending_windows(
            buffer_head=self.clock_buffer.current_head,
            ingest_head=self.ingestion.total_ingested_samples
        )

        for plan in plans:
            self.execute_plan(plan)

    def execute_plan(self, plan: WindowExecutionPlan):
        decision = None
        self.stats["windows_processed"] += 1
        
        # --- A. SKIP MODE (Lag Handling) ---
        if plan.exec_mode == "skip":
            self.stats["windows_skipped"] += 1
            self.consecutive_skips += 1
            
            window_duration_samples = plan.end_sample - plan.start_sample
            skipped_duration_sec = (self.consecutive_skips * window_duration_samples) / config.SAMPLE_RATE
            
            if skipped_duration_sec > MAX_CONTINUITY_SEC:
                # Expired -> Hard Silence (Reset)
                decision = FrameDecision(end_sample=plan.end_sample, state="silence", is_continuity=False)
            else:
                # Continuity -> Unknown but bridgeable
                # ISSUE 4 FIX: Probs MUST be empty. No invented decay here.
                decision = FrameDecision(
                    end_sample=plan.end_sample,
                    state="unknown",
                    probs={}, 
                    is_continuity=True
                )
        
        # --- B. INFERENCE MODE ---
        else:
            self.consecutive_skips = 0
            audio = self.clock_buffer.pop_window(plan.start_sample, plan.end_sample)
            
            if audio is None:
                decision = FrameDecision(end_sample=plan.end_sample, state="silence")
            else:
                is_speech, cond_audio = self.vad_gate.process(audio)
                if not is_speech:
                    decision = FrameDecision(end_sample=plan.end_sample, state="silence")
                else:
                    # 1. Inference (Diarization + Embedding)
                    # Returns DiarizationFrameOutput
                    frame_output = self.diarizer.process_window(cond_audio)
                    
                    if frame_output.status == "error":
                         decision = FrameDecision(end_sample=plan.end_sample, state="unknown")
                    elif frame_output.status == "empty":
                         decision = FrameDecision(end_sample=plan.end_sample, state="silence")
                    elif frame_output.status == "skip":
                         # Should not happen since we filtered exec_mode="skip" above, 
                         # but if fast_mode triggered internally:
                         decision = FrameDecision(
                            end_sample=plan.end_sample,
                            state="unknown",
                            probs={}, 
                            is_continuity=True
                         )
                    else:
                        # Success (probs + optional embedding)
                        raw_probs = frame_output.probs
                        embedding = frame_output.embedding
                        
                        # 2. CONTINUITY CACHE (Rewrite IDs)
                        # Pipeline Order: Diarizer -> Cache -> Smoother
                        is_overlap_raw = len(raw_probs) > 1
                        
                        stable_probs = self.cache.process(
                            probs=raw_probs,
                            embedding=embedding,
                            current_sample=plan.end_sample,
                            is_overlap=is_overlap_raw,
                            is_continuity=False
                        )

                        # max_probs = max(raw_probs.values() if raw_probs else 0.0)
                        max_probs = max(raw_probs.values()) if raw_probs else 0.0

                        is_uncertain = (
                            max_probs < config.START_THRESHOLD and
                            max_probs >= (config.START_THRESHOLD - config.UNCERTAINTY_MARGIN)
                        )

                        
                        if (
                            not self.state_machine.in_bootstrap
                            and not is_uncertain 
                            and embedding is not None 
                            and len(stable_probs) == 1
                        ):
                            resolved_id = list(stable_probs.keys())[0] # The final post-continuity ID
                            self.embedding_cache.add(
                                start=plan.start_sample,
                                end=plan.end_sample,
                                speaker_id=resolved_id,
                                embedding=embedding
                            )

                        if is_uncertain:
                            decision = FrameDecision(
                                end_sample=plan.end_sample,
                                state="uncertain",
                                probs=stable_probs # Pass for evidence accumulation
                            )
                            self.stats["decisions_uncertain"] += 1
                        else:
                            decision = FrameDecision(
                                end_sample=plan.end_sample,
                                state="speech",
                                probs=stable_probs
                            )
                            self.stats["decisions_speech"] += 1
                        
                        

                        
                        # # 3. Decision Construction
                        # decision = FrameDecision(
                        #     end_sample=plan.end_sample,
                        #     state="speech",
                        #     probs=stable_probs
                        # )

        # --- C. WIRING ---
        current_probs = {}

        if decision.is_continuity:
            current_probs = {}

        elif decision.state == "speech" and decision.probs:
            if self.state_machine.in_bootstrap:
                # Bootstrap: Bypass Smoother Lag
                self.smoother.add_frame(decision) # Warmup
                current_probs = decision.probs
            else:
                # Normal: Smoother (Stabilized IDs)
                # Smoother now receives "session_X" IDs, so it accumulates valid history
                smoothed = self.smoother.add_frame(decision)
                decision.probs = smoothed
                current_probs = smoothed
                self.last_known_probs = smoothed

        elif decision.state in ("silence", "unknown"):
            current_probs = {}

        # 4. Overlap (On Stabilized IDs)
        overlap_input = {}
        if not self.state_machine.in_bootstrap and current_probs:
            overlap_input = current_probs
            
        overlap_res = self.overlap.process(overlap_input)
        
        # 5. State Machine
        self.state_machine.process(decision, overlap_res)

    async def emit_loop(self):
        queue = self.event_bus.subscribe("all")
        logger.info("Event emitter started, listening on 'all' topic")
        
        while True:
            ev = await queue.get()
            if ev is None: 
                logger.info("Event emitter shutting down")
                break
            
            # Log what we receive
            if ev.get('type') == 'segment':
                self.stats['segments_emitted'] += 1
                logger.info(f"✅ Received segment event: [{ev.get('start_sec'):.2f}s → {ev.get('end_sec'):.2f}s]")

            if self.room.local_participant:
                 import json
                 await self.room.local_participant.publish_data(payload=json.dumps(ev), topic="diarization")

    async def shutdown(self):
        if self._shutdown_complete: return
        self._shutdown_complete = True
        logger.info("Shutting down...")

        for task in self.tasks:
            task.cancel()
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
        
        self.ingestion.stop()
        await self.ingestion.recorder.finalize_and_upload()
        
        final_sample = self.clock_buffer.current_head
        self.state_machine.finish(final_sample)
        await self.event_bus.shutdown()
        await self.storage.finalize()
        self.embedding_cache.clear()
        
    async def metrics_loop(self):
        while not self._shutdown_complete:
            await asyncio.sleep(5.0)
            logger.info(f"STATS: {self.stats}")

async def request_fnc(ctx: JobContext):
    await ctx.connect()
    agent = DiarizationAgent(ctx.room)
    await agent.start()
    ctx.add_shutdown_callback(agent.shutdown)
    @ctx.room.on("track_subscribed")
    def on_track(track, pub, part):
        if track.kind == rtc.TrackKind.KIND_AUDIO:
             asyncio.create_task(agent.run_audio_source(rtc.AudioStream(track)))

if __name__ == "__main__":
    load_dotenv()
    cli.run_app(WorkerOptions(entrypoint_fnc=request_fnc))
