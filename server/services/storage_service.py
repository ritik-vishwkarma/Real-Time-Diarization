
import logging
import asyncio
import json
import os
import config
from infrastructure.event_bus import EventBus

# Optional MinIO
try:
    from storage.minio_client import MinioUploader
except ImportError:
    MinioUploader = None

logger = logging.getLogger("StorageService")

class StorageService:
    """
    Component 3.11: StorageService
    
    Responsibility:
    - Persist session artifacts AFTER completion.
    - Barrier Sequence: STOP -> FLUSH -> WAIT -> UPLOAD.
    
    Failure Rule:
    - Storage failures must not affect session state.
    """
    def __init__(self, event_bus: EventBus, session_id: str, output_dir: str = "."):
        self.bus = event_bus
        self.session_id = session_id
        self.output_dir = output_dir
        self.queue = self.bus.subscribe("StorageService")
        self.events_log = []
        self.running = False
        
        self.uploader = MinioUploader() if MinioUploader else None

    async def run(self):
        """Consumer Loop"""
        self.running = True
        logger.info("StorageService Started.")
        
        while self.running:
            try:
                event = await self.queue.get()
                if event is None: # Shutdown Signal
                    break
                
                # Append to log
                self.events_log.append(event)
                
            except Exception as e:
                logger.error(f"Storage Loop Error: {e}")

    async def finalize(self):
        """
        Barrier Sequence / Shutdown Logic.
        Should be called after run() loop finishes.
        """
        logger.info("Finalizing Storage...")
        
        try:
            # 1. Create Metadata Payload
            metadata = {
                "schema_version": "1.0",
                "session_id": self.session_id,
                "events": self.events_log,
                "stats": {
                    "total_events": len(self.events_log)
                }
            }
            
            # 2. Save Local
            filename = f"{self.session_id}_metadata.json"
            local_path = os.path.join(self.output_dir, filename)
            
            with open(local_path, "w") as f:
                json.dump(metadata, f, indent=2)
                
            logger.info(f"Saved Metadata: {local_path}")
            
            # 3. Upload (Best Effort)
            if self.uploader:
                success = self.uploader.upload_file(
                    object_name=f"sessions/{self.session_id}/metadata.json", 
                    file_path=local_path
                )
                if success:
                    logger.info("Upload Successful.")
                else:
                    logger.error("Upload Failed (Check MinIO logs).")
                     
        except Exception as e:
             logger.error(f"Storage Finalize Failed: {e}")
             # "Storage failures must not affect session state" -> We catch and return.
