
from dataclasses import dataclass, field
from typing import Dict, Optional, Literal
import numpy as np

@dataclass(frozen=True)
class AudioFrame:
    """
    Immutable audio payload from AudioIngestion.
    """
    samples: np.ndarray # float32, 1D
    start_sample: int
    num_samples: int
    
    def __post_init__(self):
        if self.samples.ndim != 1:
            raise ValueError(f"AudioFrame samples must be 1D, got {self.samples.shape}")
        if len(self.samples) != self.num_samples:
            raise ValueError(f"sample count mismatch: len={len(self.samples)} vs declared={self.num_samples}")

@dataclass
class WindowExecutionPlan:
    """
    Instructions from SlidingWindowProcessor.
    """
    start_sample: int
    end_sample: int
    exec_mode: Literal["infer", "skip"]

@dataclass
class FrameDecision:
    """
    Authoritative decision for a TIMESTAMP (Sample Index).
    Strict Phase 3: No Seconds.
    """
    end_sample: int # Authoritative timestamp (Sample Index)
    state: Literal["speech", "silence", "unknown", "uncertain"]
    probs: Dict[str, float] = field(default_factory=dict) # Only if state="speech"
    is_continuity: bool = False # Fix: Backpressure Architecture (Flag for decay/skip)

@dataclass
class IdentityBinding:
    """
    Phase 6: Explicit binding of an identity to a specific session.
    """
    session_id: str
    speaker_ids: list[str]
    confidence: float = 1.0

@dataclass
class IdentityProfile:
    """
    Phase 6: Identity Definition.
    """
    centroid: list[float] # Serializable list
    bindings: list[IdentityBinding]
    generated_at: str # ISO8601

@dataclass
class IdentitySnapshot:
    """
    Phase 6: Immutable Identity State.
    """
    snapshot_id: str
    created_at: str
    identities: Dict[str, IdentityProfile]

@dataclass
class DiarizationFrameOutput:
    """
    Rich output from StreamingDiarizer.
    Fixes Problem 2: Disambiguates silence vs error.
    """
    probs: Dict[str, float]
    embedding: Optional[np.ndarray] = None
    # 'success' = inferred, 'empty' = silence, 'skip' = fast mode, 'error' = crash
    status: Literal["success", "empty", "skip", "error"] = "success"
