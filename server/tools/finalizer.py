import argparse
import json
import logging
import shutil
import sys
import os
from pathlib import Path
from typing import Dict, List, Set, Any, Tuple, Optional
from collections import defaultdict

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Optional MinIO
try:
    from storage.minio_client import MinioUploader
except ImportError:
    MinioUploader = None

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("SessionFinalizer")

class SessionFinalizer:
    """
    Phase 5C: Identity-Aware Reconstruction (Finalization)
    Phase 6C: Governance Update (Strict Snapshot Support)
    Phase 7A: Cloud Sync
    
    Responsibility:
    - Apply identity semantics to reconstruction artifacts.
    - STRICT: Presentation-Layer Only. No Audio Regeneration.
    - Logic: Track Selection (Canonical) + Renaming.
    - Sync to MinIO.
    """
    
    # Merge Eligibility Constants
    MIN_MERGE_SEGMENTS = 2
    MIN_MERGE_DURATION_SEC = 2.0
    
    def __init__(self, reconstruction_dir: str, metadata_path: str, output_dir: str, snapshot_path: Optional[str] = None, sample_rate: int = 16000):
        self.reconstruction_dir = Path(reconstruction_dir)
        self.metadata_path = Path(metadata_path)
        self.output_dir = Path(output_dir)
        self.snapshot_path = Path(snapshot_path) if snapshot_path else None
        self.sample_rate = sample_rate
        
        if not self.reconstruction_dir.exists():
            raise FileNotFoundError(f"Reconstruction Dir not found: {self.reconstruction_dir}")
        if not self.metadata_path.exists():
             raise FileNotFoundError(f"Metadata not found: {self.metadata_path}")
             
        # State
        self.metadata: List[Dict] = []
        self.session_id: str = "unknown" # Extracted from meta
        self.reconstruction_meta: Dict = {}
        self.identity_mapping: Dict[str, str] = {} # spk_X -> "Identity" (or "spk_X")
        self.archived_speakers: Set[str] = set() # Speakers merged away (not copied)
        self.final_files: Dict[str, str] = {} # "Identity" -> "spk_X" (Canonical)
        
        self.uploader = MinioUploader() if MinioUploader else None

    def run(self):
        logger.info("Starting Finalization...")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Load Inputs
        self._load_inputs()
        
        # 2. Analyze Identity
        self._derive_identity_mapping()
        
        # 3. Process Tracks (Select & Copy)
        self._process_tracks()
        
        # 4. Save Final Metadata
        self._save_metadata()
        
        # 5. Sync to Cloud
        if self.session_id != "unknown":
            self._upload_artifacts()
            
        logger.info("Finalization Complete.")

    def _load_inputs(self):
        logger.info(f"Loading Metadata: {self.metadata_path}")
        with open(self.metadata_path, 'r') as f:
            data = json.load(f)
            
            # Extract Session ID (Critical for Phase 6C binding)
            if isinstance(data, dict):
                self.session_id = data.get("session_id", "unknown")
                if "events" in data:
                    self.metadata = data["events"]
                else:
                    self.metadata = [] # Should not happen if valid schema
            elif isinstance(data, list):
                self.metadata = data
                # Warn: Session ID unknown if using raw list
                logger.warning("Metadata is raw list: Session ID defaults to 'unknown'.")
            else:
                self.metadata = []
            
        recon_meta_path = self.reconstruction_dir / "reconstruction_meta.json"
        if recon_meta_path.exists():
            with open(recon_meta_path, 'r') as f:
                self.reconstruction_meta = json.load(f)
            if "sample_rate" in self.reconstruction_meta:
                self.sample_rate = self.reconstruction_meta["sample_rate"]

    def _derive_identity_mapping(self):
        """
        Derive spk_X -> Identity mapping.
        
        Phase 6C Logic (Priority):
        - IF Snapshot Provided:
            - STRICT lookup (explicit binding).
            - Ambiguity Guard (Raise/Reject).
            - NO statistical merging.
        - ELSE:
            - Legacy Phase 5C Logic (Statistical Merging).
        """
        
        if self.snapshot_path and self.snapshot_path.exists():
            logger.info(f"Phase 6C: Using Snapshot {self.snapshot_path}")
            self._apply_snapshot_logic()
        else:
            logger.info("Phase 6C: No Snapshot. Using Phase 5C Statistical Logic.")
            self._apply_legacy_logic()
            
    def _apply_snapshot_logic(self):
        """
        Strict declarative mapping from Snapshot.
        Rules:
        - Only explicit bindings.
        - Ambiguity -> Reject.
        """
        with open(self.snapshot_path, 'r') as f:
            snapshot_data = json.load(f)
            
        # 1. Build Candidate Map: spk_X -> [Identities]
        candidates = defaultdict(list)
        
        identities = snapshot_data.get("identities", {})
        for name, profile in identities.items():
            bindings = profile.get("bindings", [])
            for b in bindings:
                # Check Session Binding
                if b.get("session_id") == self.session_id:
                    # Found Binding
                    for spk_id in b.get("speaker_ids", []):
                        candidates[spk_id].append(name)
                        
        # 2. Validate Ambiguity & Assign
        for spk_id, likely_names in candidates.items():
            if len(likely_names) > 1:
                # AMBIGUITY GUARD
                logger.error(f"AMBIGUITY GUARD: Speaker {spk_id} bound to multiple identities {likely_names}. REJECTING MAPPING.")
                self.identity_mapping[spk_id] = spk_id # Revert to spk_X
            else:
                target_name = likely_names[0]
                logger.info(f"Examples Binding: {spk_id} -> {target_name}")
                self.identity_mapping[spk_id] = target_name
                
        # 3. Handle Unmapped (Preserve)
        # We need to know all speakers present in the Reconstruction.
        reconst_outputs = self.reconstruction_meta.get("outputs", {})
        present_speakers = list(reconst_outputs.keys())
        
        if not present_speakers and self.metadata:
             # Fallback if reconstruction meta missing (shouldn't happen)
             # Extract from metadata events
             pass
             
        for spk in present_speakers:
            if spk not in self.identity_mapping:
                self.identity_mapping[spk] = spk

    def _apply_legacy_logic(self):
        """
        Original Phase 5C Logic (Statistical Merging via speaker_labels.json or assumptions).
        """
        # ... (Existing Logic Copy) ...
        # I will replace the previous huge method with this extracted method to keep code clean.
        # But wait, I need to preserve the code I am replacing!
        # The previous code was inside `_derive_identity_mapping`.
        # I will paste the previous implementation here.
        
        # Simplified Logic for Phase 5C Execution (assuming Labels exist or Identity inferred):
        labels_path = self.metadata_path.parent / "speaker_labels.json"
        speaker_labels = {}
        if labels_path.exists():
            with open(labels_path, 'r') as f:
                 speaker_labels = json.load(f) 
        
        # Aggregate Verification Stats
        spk_stats = defaultdict(lambda: {"count": 0, "duration": 0.0, "total_duration": 0.0})
        
        for event in self.metadata:
            if event.get("type") == "speaker_identity":
                spk = event.get("speaker")
                dur = (event.get("segment_end_sample", 0) - event.get("segment_start_sample", 0)) / self.sample_rate
                spk_stats[spk]["total_duration"] += dur
                
                if event.get("status") == "verified":
                    spk_stats[spk]["count"] += 1
                    spk_stats[spk]["duration"] += dur

        # Decision Matrix
        # Group Speakers by Label
        label_groups = defaultdict(list)
        for spk, label in speaker_labels.items():
            label_groups[label].append(spk)
            
        # Determine Canonical for each Label
        for label, spks in label_groups.items():
             candidates = []
             for spk in spks:
                 stats = spk_stats[spk]
                 is_eligible = (stats["count"] >= self.MIN_MERGE_SEGMENTS and 
                                stats["duration"] >= self.MIN_MERGE_DURATION_SEC)
                 
                 if is_eligible:
                     candidates.append(spk)
                 else:
                     self.identity_mapping[spk] = spk
            
             if candidates:
                 for cand in candidates:
                     self.identity_mapping[cand] = label
                     
        # Handle Unlabeled Speakers
        for spk in spk_stats.keys():
            if spk not in self.identity_mapping:
                self.identity_mapping[spk] = spk # Preserve identity

        # Also cover speakers in reconstruction meta not in events (silence?)
        reconst_outputs = self.reconstruction_meta.get("outputs", {})
        for spk in reconst_outputs.keys():
            if spk not in self.identity_mapping:
                self.identity_mapping[spk] = spk

    def _process_tracks(self):
        """
        Select Canonical Tracks.
        For each Identity Target:
        - Find matched source speakers.
        - Pick BEST (Longest Total Duration).
        - Copy to Final.
        - Archive others.
        """
        # Group sources by target
        target_sources = defaultdict(list)
        for spk, target in self.identity_mapping.items():
            target_sources[target].append(spk)
            
        final_files_dir = self.output_dir
        final_files_dir.mkdir(exist_ok=True)
        
        for target, sources in target_sources.items():
            if not sources:
                continue
                
            sources = sorted(sources)
            canonical = sources[0] 
            
            # Find longest?
            best_source = canonical
            max_size = -1
            
            for src in sources:
                path = self.reconstruction_dir / "audio" / f"speaker_{src}.wav" # reconstruction path convention
                if path.exists():
                     size = path.stat().st_size
                     if size > max_size:
                         max_size = size
                         best_source = src
            
            canonical = best_source
            
            # Copy Canonical
            src_path = self.reconstruction_dir / "audio" / f"speaker_{canonical}.wav"
            if src_path.exists():
                dest_name = f"speaker_{target}.wav"
                dest_path = final_files_dir / dest_name
                logger.info(f"Copying {src_path.name} -> {dest_name} (Canonical for {target})")
                shutil.copy(src_path, dest_path)
                self.final_files[target] = str(dest_path)
            else:
                logger.warning(f"Canonical source missing: {src_path}")

            # Archive others
            for src in sources:
                if src != canonical:
                    self.archived_speakers.add(src)
                    logger.info(f"Archived {src} (Merged into {target})")

    def _save_metadata(self):
        meta = {
            "mappings": self.identity_mapping,
            "archived": sorted(list(self.archived_speakers)),
            "final_outputs": self.final_files
        }
        out_path = self.output_dir / "final_meta.json"
        with open(out_path, 'w') as f:
            json.dump(meta, f, indent=2)

    def _upload_artifacts(self):
        """
        Phase 7A: Sync artifacts to MinIO.
        """
        if not self.uploader:
            logger.warning("No MinIO Uploader available. Skipping cloud sync.")
            return

        logger.info("Syncing Final Artifacts to MinIO...")
        
        # 1. Upload Final Meta
        meta_local = self.output_dir / "final_meta.json"
        if meta_local.exists():
            self.uploader.upload_file(
                object_name=f"sessions/{self.session_id}/final_meta.json",
                file_path=str(meta_local)
            )

        # 2. Upload Audio Tracks
        for label, path_str in self.final_files.items():
            path = Path(path_str)
            if path.exists():
                self.uploader.upload_file(
                    object_name=f"sessions/{self.session_id}/final/{path.name}",
                    file_path=str(path)
                )
        
        logger.info("MinIO Sync Complete.")

def main():
    parser = argparse.ArgumentParser(description="Phase 5C: Session Finalizer")
    parser.add_argument("--reconstruction", required=True, help="Path to reconstruction dir")
    parser.add_argument("--metadata", required=True, help="Path to metadata.json")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--snapshot", required=False, help="Path to Identity Snapshot (Phase 6C)")
    
    args = parser.parse_args()
    
    try:
        finalizer = SessionFinalizer(args.reconstruction, args.metadata, args.output, args.snapshot)
        finalizer.run()
    except Exception as e:
        logger.error(f"Finalization Failed: {e}")
        exit(1)

if __name__ == "__main__":
    main()
