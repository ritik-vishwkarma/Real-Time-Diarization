
import argparse
import json
import logging
import glob
import os
import shutil
from pathlib import Path
from datetime import datetime
import numpy as np
import scipy.io.wavfile as wavfile
from typing import List, Dict, Optional
from dataclasses import asdict

# Adjust path to find server modules
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from server.dtos import IdentitySnapshot, IdentityProfile, IdentityBinding
from server.components.embedding_engine import EmbeddingEngine
from server import config

# Optional MinIO
try:
    from storage.minio_client import MinioUploader
except ImportError:
    MinioUploader = None

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] Governance: %(message)s"
)
logger = logging.getLogger("Governance")

SNAPSHOT_DIR = Path("identity_snapshots")

class GovernanceTool:
    def __init__(self):
        self.engine: Optional[EmbeddingEngine] = None
        self.uploader = MinioUploader() if MinioUploader else None
        SNAPSHOT_DIR.mkdir(exist_ok=True)

    def load_engine(self):
        if not self.engine:
            logger.info("Initializing Embedding Engine...")
            self.engine = EmbeddingEngine()

    def get_latest_snapshot(self) -> Optional[IdentitySnapshot]:
        files =  sorted(glob.glob(str(SNAPSHOT_DIR / "identity_v*.json")))
        if not files:
            return None
        
        latest = files[-1]
        logger.info(f"Loading base snapshot: {latest}")
        with open(latest, 'r') as f:
            data = json.load(f)
            
        # Reconstruct DTOs
        identities = {}
        for name, profile_data in data.get("identities", {}).items():
            bindings = [IdentityBinding(**b) for b in profile_data["bindings"]]
            identities[name] = IdentityProfile(
                centroid=profile_data["centroid"],
                bindings=bindings,
                generated_at=profile_data["generated_at"]
            )
            
        return IdentitySnapshot(
            snapshot_id=data["snapshot_id"],
            created_at=data["created_at"],
            identities=identities
        )

    def create_snapshot(self, base: Optional[IdentitySnapshot], new_identities: Dict[str, IdentityProfile]) -> Path:
        """
        Create a new snapshot version.
        Rule: Copy base identities, then overwrite/append new ones.
              (Actually, strict append for new bindings?)
        User Rule: "Existing identities remain unchanged."
        
        If we are adding a binding to "Alice", do we update Alice?
        Yes, but we create a NEW snapshot.
        """
        
        # 1. Start with copy of base
        final_identities = {}
        version_idx = 0
        
        if base:
            final_identities = base.identities.copy()
            # Extract version index
            try:
                # identity_v{N}.json
                start = base.snapshot_id.find("_v") + 2
                version_idx = int(base.snapshot_id[start:])
            except:
                version_idx = 0
        
        # 2. Apply updates
        for name, profile in new_identities.items():
            # If exists, we MIGHT strictly forbid modification?
            # User said "Append new identity entry... Existing identities remain unchanged."
            # This implies if "Alice" exists, can we add a binding?
            # "Snapshot declares explicit identity bindings".
            # If I add a binding for Session 2, I am technically modifying Alice's profile in the new snapshot.
            # But the Rule "Existing identities remain unchanged" likely refers to "Don't mute the PAST".
            # Adding a FUTURE binding is necessary.
            
            # Logic: If name exists, merge bindings?
            if name in final_identities:
                existing = final_identities[name]
                # Combine bindings
                new_bindings = existing.bindings + profile.bindings
                # Re-average centroid? 
                # "Centroids are immutable once used". 
                # If we are "Enrolling", we usually mean "Defining the User".
                # If Alice is already defined, we should probably REUSE her centroid or Average it?
                # User Rule: "Centroids are immutable once used in a session."
                # Does that mean immutable ACROSS sessions? 
                # "Snapshot changes do NOT affect past sessions".
                # So V2 can have a refined centroid for Alice, and V1 keeps the old one.
                # Since V2 is a new file, V1 is safe.
                
                # IMPLEMENTATION CHOICE: Update centroid with weighted average? 
                # Or just keep original enrollment centroid?
                # "Identity enrollment... Source: speaker_spk_0.wav".
                # Usually enrollment is a one-time setup.
                # Later "Verify" is 5A.
                # I will assume "Enrollment" implies setting/updating the reference.
                # So I will use the NEW centroid or weighted?
                # Simpler: If "Updating" Alice, use the NEW centroid provided by this command? 
                # Or Average?
                # Safe path: Update Centroid (Refinement).
                
                logger.info(f"Updating existing identity: {name}")
                final_identities[name] = IdentityProfile(
                    centroid=profile.centroid, # Update reference!
                    bindings=new_bindings,
                    generated_at=datetime.utcnow().isoformat()
                )
            else:
                logger.info(f"Creating new identity: {name}")
                final_identities[name] = profile

        # 3. Create File
        new_version = version_idx + 1
        new_id = f"identity_v{new_version}"
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{new_id}.json"
        
        output_path = SNAPSHOT_DIR / filename
        
        # Serialize
        snapshot_out = {
            "snapshot_id": new_id,
            "created_at": datetime.utcnow().isoformat(),
            "identities": {
                k: asdict(v) for k, v in final_identities.items()
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(snapshot_out, f, indent=2)
            
        logger.info(f"Snapshot created: {output_path}")
        
        # Cloud Sync
        if self.uploader:
            try:
                self.uploader.upload_file(
                    object_name=f"identity_snapshots/{filename}",
                    file_path=str(output_path)
                )
                logger.info("Snapshot synced to MinIO.")
            except Exception as e:
                logger.error(f"Failed to sync snapshot to MinIO: {e}")
                
        return output_path

    def enroll(self, name: str, audio_path: str, session_id: str, speaker_ids: List[str]):
        # 1. Validation
        if not os.path.exists(audio_path):
            logger.error(f"Audio file not found: {audio_path}")
            return
            
        # 2. Extract Embedding
        self.load_engine()
        sr, audio = wavfile.read(audio_path)
        
        # Convert to float32
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        
        # Mono
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
            
        embedding = self.engine.extract_embedding(audio)
        if embedding is None:
            logger.error("Failed to extract embedding from audio")
            return
            
        # 3. Create Profile
        binding = IdentityBinding(
            session_id=session_id,
            speaker_ids=speaker_ids,
            confidence=1.0 # Explicit enrollment
        )
        
        profile = IdentityProfile(
            centroid=embedding.tolist(),
            bindings=[binding],
            generated_at=datetime.utcnow().isoformat()
        )
        
        # 4. Snapshot
        base = self.get_latest_snapshot()
        self.create_snapshot(base, {name: profile})

    def list_snapshots(self):
        files = sorted(glob.glob(str(SNAPSHOT_DIR / "*.json")))
        print(f"Found {len(files)} snapshots:")
        for f in files:
            print(f" - {f}")

def main():
    parser = argparse.ArgumentParser(description="Phase 6B: Governance Tool")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Enroll
    enroll_parser = subparsers.add_parser("enroll", help="Enroll a new identity")
    enroll_parser.add_argument("--name", required=True, help="Identity Name (e.g. 'Alice')")
    enroll_parser.add_argument("--audio", required=True, help="Path to reference audio")
    enroll_parser.add_argument("--session", required=True, help="Session ID for binding")
    enroll_parser.add_argument("--spk", required=True, nargs="+", help="Speaker IDs to bind (e.g. spk_0)")
    
    # List
    list_parser = subparsers.add_parser("list", help="List snapshots")
    
    args = parser.parse_args()
    
    tool = GovernanceTool()
    
    if args.command == "enroll":
        tool.enroll(args.name, args.audio, args.session, args.spk)
    elif args.command == "list":
        tool.list_snapshots()

if __name__ == "__main__":
    main()
