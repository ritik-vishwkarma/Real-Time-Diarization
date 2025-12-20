import argparse
import json
import logging
import numpy as np
import scipy.io.wavfile as wavfile
from pathlib import Path
from typing import Dict, List, Set, Any

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("SessionReconstructor")

class SessionReconstructor:
    """
    Phase 5B: Offline Reconstruction (Identity-Agnostic)
    
    Responsibility:
    - Mechanically reconstruct per-speaker audio streams.
    - STRICT: Copy-Paste logic only. No inference. No smoothing.
    - Input: raw_audio.wav, metadata.json
    - Output: per-speaker WAVs (byte-exact deterministic)
    """

    def __init__(self, raw_audio_path: str, metadata_path: str, output_dir: str):
        self.raw_audio_path = Path(raw_audio_path)
        self.metadata_path = Path(metadata_path)
        self.output_dir = Path(output_dir)
        
        if not self.raw_audio_path.exists():
            raise FileNotFoundError(f"Raw Audio not found: {self.raw_audio_path}")
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {self.metadata_path}")

        self.sample_rate = 0
        self.raw_audio: np.ndarray = None
        self.total_samples = 0
        self.metadata: List[Dict[str, Any]] = []
        self.speakers: List[str] = [] # Sorted list
        self.speaker_buffers: Dict[str, np.ndarray] = {}

    def run(self):
        logger.info("Starting Reconstruction...")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Initialize (Load & Allocate)
        self._load_resources()
        self._allocate_buffers()

        # 2. Process Segments (Strict Order)
        self._process_segments()

        # 3. Output
        self._save_outputs()
        logger.info("Reconstruction Complete.")

    def _load_resources(self):
        # Load Audio (Expect float32 or int16, convert/keep as is? 
        # Plan says "Exact Copy". We should respect source dtype if possible, 
        # but internal buffer is usually float32 for simplicity in numpy. 
        # Let's read as is.)
        logger.info(f"Loading Raw Audio: {self.raw_audio_path}")
        self.sample_rate, self.raw_audio = wavfile.read(str(self.raw_audio_path))
        
        # Ensure 1D
        if len(self.raw_audio.shape) > 1:
            self.raw_audio = self.raw_audio.mean(axis=1) # Flatten logic if stereo? 
            # Architecture says "Single microphone stream". Should be mono.
        
        self.total_samples = len(self.raw_audio)
        logger.info(f"Audio Loaded: {self.total_samples} samples @ {self.sample_rate}Hz")

        # Load Metadata
        logger.info(f"Loading Metadata: {self.metadata_path}")
        with open(self.metadata_path, 'r') as f:
            data = json.load(f)
            
        if isinstance(data, dict) and "events" in data:
            self.metadata = data["events"]
        elif isinstance(data, list):
            self.metadata = data
        else:
            logger.warning("Metadata format unknown (expected dict with 'events' or list). using raw.")
            self.metadata = []

    def _allocate_buffers(self):
        # 1. Discover Speakers (Pre-allocation Rule)
        unique_speakers = set()
        for event in self.metadata:
            if event.get("type") == "segment" and event.get("is_final"):
                unique_speakers.update(event.get("speakers", []))
        
        self.speakers = sorted(list(unique_speakers))
        logger.info(f"Discovered Speakers: {self.speakers}")

        # 2. Allocate Zero-Filled Buffers
        logger.info(f"Allocating Buffers ({len(self.speakers)} x {self.total_samples} samples)...")
        dtype = self.raw_audio.dtype
        for spk in self.speakers:
            self.speaker_buffers[spk] = np.zeros(self.total_samples, dtype=dtype)

    def _process_segments(self):
        """
        Strict Segment Processing:
        - Filter: type="segment" AND is_final=True
        - Sort: start_sample ASC, end_sample ASC
        - Action: Copy slice to ALL active speakers.
        """
        # 1. Filter and Extract
        segments = []
        for event in self.metadata:
            if event.get("type") == "segment" and event.get("is_final", False):
                segments.append(event)

        # 2. Strict Sort
        # Sort key: (start, end)
        segments.sort(key=lambda x: (x.get("start_sample", 0), x.get("end_sample", 0)))
        
        logger.info(f"Processing {len(segments)} segments...")

        for seg in segments:
            start = seg.get("start_sample")
            end = seg.get("end_sample")
            active_speakers = seg.get("speakers", [])

            # Validation
            if start is None or end is None:
                continue
            
            # Bounds Check (Safe Degrade: Clip to audio limits)
            start = max(0, min(start, self.total_samples))
            end = max(0, min(end, self.total_samples))

            if start >= end:
                continue

            # Extract Slice (Read-Only)
            audio_slice = self.raw_audio[start:end]

            # Copy to Active Speakers
            for spk in active_speakers:
                if spk in self.speaker_buffers:
                    # Overwrite or Add?
                    # Reconstruction implies "Only one source of truth per moment".
                    # If overlaps exist in metadata, they map to the same time.
                    # We simply write the data.
                    self.speaker_buffers[spk][start:end] = audio_slice

    def _save_outputs(self):
        reconst_meta = {
            "source": str(self.raw_audio_path),
            "total_samples": self.total_samples,
            "sample_rate": self.sample_rate,
            "speakers": self.speakers,
            "outputs": {}
        }

        output_files_dir = self.output_dir / "audio"
        output_files_dir.mkdir(exist_ok=True)

        for spk in self.speakers:
            fname = f"speaker_{spk}.wav"
            out_path = output_files_dir / fname
            
            # Write WAV
            wavfile.write(str(out_path), self.sample_rate, self.speaker_buffers[spk])
            
            reconst_meta["outputs"][spk] = str(out_path)
            logger.info(f"Wrote: {out_path}")

        # Save Meta
        meta_out = self.output_dir / "reconstruction_meta.json"
        with open(meta_out, 'w') as f:
            json.dump(reconst_meta, f, indent=2)
        logger.info(f"Saved Reconstruction Metadata: {meta_out}")

def main():
    parser = argparse.ArgumentParser(description="Phase 5B: Offline Reconstruction")
    parser.add_argument("--audio", required=True, help="Path to raw_audio.wav")
    parser.add_argument("--metadata", required=True, help="Path to metadata.json")
    parser.add_argument("--output", required=True, help="Output directory")
    
    args = parser.parse_args()
    
    try:
        reconstructor = SessionReconstructor(args.audio, args.metadata, args.output)
        reconstructor.run()
    except Exception as e:
        logger.error(f"Reconstruction Failed: {e}")
        exit(1)

if __name__ == "__main__":
    main()
