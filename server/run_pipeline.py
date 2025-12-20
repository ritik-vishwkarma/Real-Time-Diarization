import argparse
import logging
from pathlib import Path
from tools.reconstructor import SessionReconstructor
from tools.finalizer import SessionFinalizer
from config import SAMPLE_RATE
from typing import Optional, Tuple

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [Pipeline] %(message)s"
)
logger = logging.getLogger("Pipeline")

# Optional MinIO
try:
    from storage.minio_client import MinioUploader
except ImportError:
    MinioUploader = None

def fetch_from_minio(session_id: str, dest_dir: Path) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Attempt to download raw_audio.wav and metadata.json from MinIO.
    """
    if not MinioUploader: 
        return None, None
        
    uploader = MinioUploader()
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    audio_path = dest_dir / "raw_audio.wav"
    meta_path = dest_dir / "metadata.json"
    
    # Download Audio
    try:
        if not audio_path.exists():
            logger.info(f"Downloading Audio for {session_id} from MinIO...")
            uploader.client.fget_object(uploader.bucket, f"sessions/{session_id}/raw_audio.wav", str(audio_path))
    except Exception as e:
        logger.warning(f"Audio download failed: {e}")

    # Download Metadata
    try:
        if not meta_path.exists():
            logger.info(f"Downloading Metadata for {session_id} from MinIO...")
            uploader.client.fget_object(uploader.bucket, f"sessions/{session_id}/metadata.json", str(meta_path))
    except Exception as e:
        logger.warning(f"Metadata download failed: {e}")
        
    return (audio_path if audio_path.exists() else None, 
            meta_path if meta_path.exists() else None)

def run_pipeline(session_id: str, base_dir: str = "sessions", snapshot_path: str = None):
    """
    Automates Phase 5B (Reconstruction) and Phase 5C (Finalization).
    """
    logger.info(f"=== Starting Pipeline for Session: {session_id} ===")

    # Path Discovery Logic
    raw_audio = None
    metadata = None

    # 0. Prep Work Dir
    session_work_dir = Path(base_dir) / session_id
    session_work_dir.mkdir(parents=True, exist_ok=True)

    # 1. Check Standard Structure
    if (session_work_dir / "raw_audio.wav").exists():
        raw_audio = session_work_dir / "raw_audio.wav"
    if (session_work_dir / "metadata.json").exists():
        metadata = session_work_dir / "metadata.json"
            
    # 2. Check Flat Structure (Fallback)
    if not metadata:
        flat_meta = Path(base_dir) / f"{session_id}_metadata.json"
        if flat_meta.exists():
            metadata = flat_meta
            
    if not raw_audio:
        candidates = [
            Path(base_dir) / f"session_{session_id}_raw.wav",
            Path(base_dir) / f"session_{session_id}.wav",
            Path(base_dir) / f"{session_id}.wav"
        ]
        for c in candidates:
            if c.exists():
                raw_audio = c
                break

    # 3. Cloud Fetch (Phase 7A)
    if not raw_audio or not metadata:
        logger.info("Local files missing. Attempting Cloud Fetch...")
        c_audio, c_meta = fetch_from_minio(session_id, session_work_dir)
        if c_audio: raw_audio = c_audio
        if c_meta: metadata = c_meta

    # 4. Validation
    if not raw_audio or not metadata:
        logger.error(f"Missing Files for Session '{session_id}':")
        logger.error(f"  - Audio Found: {raw_audio}")
        logger.error(f"  - Metadata Found: {metadata}")
        logger.error("Please ensure files exist locally or in MinIO.")
        return

    reconst_dir = session_work_dir / "reconstruction"
    final_dir = session_work_dir / "final"

    # Phase 5B: Reconstruction
    try:
        logger.info(f"--- Phase 5B: Offline Reconstruction ---")
        reconstructor = SessionReconstructor(
            str(raw_audio), 
            str(metadata), 
            str(reconst_dir)
        )
        reconstructor.run()
    except Exception as e:
        logger.error(f"Phase 5B Failed: {e}")
        return

    # Phase 5C: Finalization
    try:
        logger.info("--- Phase 5C: Identity-Aware Finalization ---")
        finalizer = SessionFinalizer(
            str(reconst_dir),
            str(metadata),
            str(final_dir),
            snapshot_path=snapshot_path, # Phase 6
            sample_rate=SAMPLE_RATE
        )
        finalizer.run()
    except Exception as e:
        logger.error(f"Phase 5C Failed: {e}")
        return

    logger.info(f"=== Pipeline Complete. Outputs in {final_dir} ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Offline Diarization Pipeline")
    parser.add_argument("session_id", help="Session ID (folder name in sessions/)")
    parser.add_argument("--dir", default="sessions", help="Base sessions directory")
    parser.add_argument("--snapshot", default=None, help="Path to Identity Snapshot (Phase 6)")
    
    args = parser.parse_args()
    run_pipeline(args.session_id, args.dir, args.snapshot)
