
import asyncio
import logging
from typing import Dict, Set

logger = logging.getLogger("EventBus")

class EventBus:
    """
    Component 3: EventBus (Hardened Phase 4)
    
    Responsibility:
    - Pub/Sub.
    - BOUNDED Queues (Fix 5).
    - Drop-Oldest behavior on overflow.
    - Prevents Memory Leaks.
    """
    
    MAX_QUEUE_SIZE = 1000 # Reasonable buffer for segments
    
    def __init__(self):
        self.subscribers: Dict[str, Set[asyncio.Queue]] = {}
        self.active = True

    def subscribe(self, consumer_name: str) -> asyncio.Queue:
        """
        Returns a bounded queue for the consumer.
        """
        q = asyncio.Queue(maxsize=self.MAX_QUEUE_SIZE)
        if "all" not in self.subscribers:
            self.subscribers["all"] = set()
        self.subscribers["all"].add(q)
        logger.info(f"Subscriber connected: {consumer_name}")
        return q

    def publish(self, event: dict):
        """
        Non-blocking publish.
        If queue full: Drop Oldest (get_nowait) then Put.
        """
        if not self.active: return
        
        # We publish to 'all' topic for now (Simple Bus)
        if "all" in self.subscribers:
            for q in self.subscribers["all"]:
                try:
                    if q.full():
                         # Drop Oldest
                         try:
                             _ = q.get_nowait()
                             # logger.warning("EventBus: Dropped event due to full queue.")
                         except asyncio.QueueEmpty:
                             pass
                    
                    q.put_nowait(event)
                except asyncio.QueueFull:
                     # Should catch race condition
                     pass

    async def shutdown(self):
        self.active = False
        if "all" in self.subscribers:
            for q in self.subscribers["all"]:
                # await q.put(None) # Poison Pill
                while not q.empty():
                    q.get_nowait()
                await q.put(None)

        self.subscribers.clear()
        logger.info("EventBus shutdown complete.")
