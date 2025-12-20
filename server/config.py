# --- Diarization Engine Settings ---
SAMPLE_RATE = 16000
FRAME_MS = 0.08 # 80ms
# CHUNK_FRAMES = 4     # 0.32s
# CONTEXT_FRAMES = 22  # 1.76s (Total ~2.08s window)
CHUNK_FRAMES = 12    # ~0.96s (Optimized for GTX 1650: reduced frequency)
CONTEXT_FRAMES = 32  # ~2.56s (Maintains context ratio)

# --- Threshold Hysteresis (Speech Stability) ---
# To START a segment, probability must be high (Avoids False Positives)
START_THRESHOLD = 0.50  # 0.58 was intially

# To CONTINUE a segment, probability can drop slightly (Prevents Fragmentation)
CONTINUE_THRESHOLD = 0.35 # 0.45 was intially

# To END a segment (or consider it silence/noise), it must drop below this
END_THRESHOLD = 0.30 # 0.40 was intially

# --- Overlap Detection ---
# If multiple speakers valid > OVERLAP_THRESHOLD, trigger overlap
OVERLAP_THRESHOLD = 0.40
MIN_OVERLAP_SPEAKERS = 2

# --- Model Paths ---
# Can be overridden by env vars if needed
DIARIZATION_MODEL = "nvidia/diar_streaming_sortformer_4spk-v2.1"
EMBEDDING_MODEL = "nvidia/speakerverification_en_titanet_large"
VAD_MODEL = "vad_multilingual_marblenet"

# --- Smoothing ---
SMOOTHING_WINDOW = 8 # Number of historical chunks to median filter

# --- Identification & Verification ---
SIMILARITY_THRESHOLD = 0.70 # Frozen Phase 5A (Do not tune)
SPEAKER_DRIFT_THRESHOLD = 0.6
MIN_SPEECH_DURATION_FOR_ID = 1.2
REID_WINDOW_DURATION = 2.0
MIN_AUDIO_FOR_EMBEDDING = 0.5
REID_RING_BUFFER_SIZE = 2000 # ~20-30s of chunks

RETRY_UNKNOWN_INTERVAL = 3.0 # Retry identifying "unknown" speakers every 3 seconds

# --- Audio Processing ---
ENERGY_AMBIGUITY_LOW = -45
ENERGY_AMBIGUITY_HIGH = -38  
ENERGY_THRESH_DBFS = -38 # Hardened Design Threshold
GAIN_MIN_AMP = 0.0001
GAIN_MAX_AMP = 0.1
GAIN_TARGET = 0.5
OVERLAP_PERSISTENCE = 2 # Required consecutive windows to confirm overlap

# Bootstrap Tuning (Easier start)
BOOTSTRAP_MIN_DURATION = 0.15  # 150ms to start
MIN_SEGMENT_DURATION = 0.30

# --- Continuity Cache ---
CACHE_SIMILARITY_THRESHOLD = 0.7 # 0.65
CACHE_MAX_GAP_SECONDS = 15.0

# --- Uncertainty Modeling ---
UNCERTAINTY_MARGIN = 0.10 

# --- Embedding Reuse Cache ---
EMBEDDING_CACHE_MAX = 5000           # Total windows (Fix 1)
EMBEDDING_CACHE_TTL_SEC = 20.0       # Expire after 20s (Fix 4)
EMBEDDING_CACHE_AGG_MODE = "mean"    # Deterministic aggregation