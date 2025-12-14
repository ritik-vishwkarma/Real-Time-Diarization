import asyncio
import os
import argparse
from livekit import rtc
from livekit.api import AccessToken, VideoGrants
from bot import DiarizationAgent
from dotenv import load_dotenv
import logging

# Configure Logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DebugBot")

load_dotenv()

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--room", type=str, default="my-room", help="Room name to connect to")
    args = parser.parse_args()

    ROOM_NAME = args.room
    API_KEY = os.getenv("LIVEKIT_API_KEY")
    API_SECRET = os.getenv("LIVEKIT_API_SECRET")
    URL = os.getenv("LIVEKIT_URL")

    if not API_KEY or not API_SECRET or not URL:
        logger.error(f"Missing Keys. URL: {URL}, KEY: {API_KEY}")
        return

    logger.info(f"Connecting to {URL} room: {ROOM_NAME}")

    # 1. Create Token for BOT
    # We give it permission to join, subscribe (to hear users), and publish (to send data events)
    grant = VideoGrants(room_join=True, room=ROOM_NAME, can_subscribe=True, can_publish=True, can_publish_data=True)
    token = AccessToken(API_KEY, API_SECRET).with_grants(grant).with_identity("bot-diarizer").to_jwt()

    # 2. Connect
    room = rtc.Room()
    try:
        await room.connect(URL, token)
        logger.info(f"✅ Bot Connected to {ROOM_NAME}")
    except Exception as e:
        logger.error(f"Failed to connect: {e}")
        return

    # 3. Start Agent Logic
    agent = DiarizationAgent(room)
    
    # helper to start loop
    def start_processing(track, participant):
        logger.info(f"🎤 Starting diarization for {participant.identity}")
        stream = rtc.AudioStream(track)
        asyncio.create_task(agent.run_audio_loop(stream))

    # Subscribe to existing tracks
    @room.on("track_subscribed")
    def on_track_subscribed(track, publication, participant):
        if track.kind == rtc.TrackKind.KIND_AUDIO:
             logger.info(f"Event: track_subscribed from {participant.identity}")
             start_processing(track, participant)

    # Handle tracks already in room (if bot joins late)
    logger.info("Checking for existing participants...")
    for p_id, p in room.remote_participants.items():
        for t_id, t_pub in p.track_publications.items():
            if t_pub.track and t_pub.kind == rtc.TrackKind.KIND_AUDIO:
                 logger.info(f"Found existing audio from {p.identity}")
                 start_processing(t_pub.track, p)

    # Keep alive until Ctrl+C
    try:
        # Wait forever
        await asyncio.Future() 
    except asyncio.CancelledError:
        await room.disconnect()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
