import torch
import numpy as np
import sounddevice as sd
import queue
import sys
import tqdm
from nemo.collections.asr.models import SortformerEncLabelModel
import time

# --- 0. SILENCE PROGRESS BARS (The Fix) ---
# We monkey-patch tqdm to force 'disable=True' globally.
# This stops the "Streaming Steps: 100%..." logs.
original_init = tqdm.tqdm.__init__
def silenced_init(self, *args, **kwargs):
    kwargs['disable'] = True
    original_init(self, *args, **kwargs)
tqdm.tqdm.__init__ = silenced_init

# --- 1. SETUP ---
MODEL_NAME = "nvidia/diar_streaming_sortformer_4spk-v2.1"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Loading {MODEL_NAME} on {DEVICE}...")
diar_model = SortformerEncLabelModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()

# --- 2. CONFIGURATION (Integers) ---
SAMPLE_RATE = 16000
FRAME_MS = 0.08 
SAMPLES_PER_FRAME = int(SAMPLE_RATE * FRAME_MS)

# Latency Setup
CHUNK_FRAMES = 6     # 0.48s
CONTEXT_FRAMES = 7   # 0.56s

# Apply Config
diar_model.sortformer_modules.chunk_len = CHUNK_FRAMES
diar_model.sortformer_modules.chunk_right_context = CONTEXT_FRAMES

# Calculate Buffers
chunk_samples = CHUNK_FRAMES * SAMPLES_PER_FRAME
buffer_samples = (CHUNK_FRAMES + CONTEXT_FRAMES) * SAMPLES_PER_FRAME

# --- 3. LIVE AUDIO LOOP ---
audio_queue = queue.Queue()

def callback(indata, frames, time, status):
    if status: print(status, file=sys.stderr)
    audio_queue.put(indata.copy())

rolling_buffer = np.zeros(buffer_samples, dtype=np.float32)

# print(f"\n=== LISTENING (Latency: {(CHUNK_FRAMES+CONTEXT_FRAMES)*FRAME_MS:.2f}s) ===")
# print("Speak now...")

last_speaker = None
silence_counter = 0

print(f"\n=== LISTENING (Latency: {(CHUNK_FRAMES+CONTEXT_FRAMES)*FRAME_MS:.2f}s) ===")
print("Try playing a YouTube interview to see Spk 1 vs Spk 2 switching!\n")

try:
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, 
                        blocksize=chunk_samples, callback=callback):
        while True:
            # 1. Get Audio
            new_data = audio_queue.get().flatten().astype(np.float32)
            
            # 2. Update Buffer
            rolling_buffer = np.roll(rolling_buffer, -chunk_samples)
            rolling_buffer[-chunk_samples:] = new_data
            
            # 3. Prepare Inputs
            input_tensor = torch.tensor(rolling_buffer).unsqueeze(0).to(DEVICE)
            input_length = torch.tensor([input_tensor.shape[1]]).to(DEVICE)

            # 4. INFERENCE
            with torch.no_grad():
                # Call model with positional args
                preds = diar_model(input_tensor, input_length)
                # print(f"Predictions output: {preds}")

                # Unpack tuple if necessary
                if isinstance(preds, tuple) or isinstance(preds, list):
                    probs = preds[0]
                else:
                    probs = preds
                    # print(probs.min().item(), probs.max().item())

            # 5. DECODE
            # Probs shape: [Batch, Time, Speakers]
            # Threshold > 0.55
            active_mask = (probs > 0.55).float()
            
            # Check for ANY activity in this chunk
            speaker_activity = torch.max(active_mask, dim=1)[0][0]
            
            active_indices = (speaker_activity > 0).nonzero().flatten().tolist()
            
            # if active_indices:
            #     spk_labels = [f"Spk {i+1}" for i in active_indices]
            #     # Using \r to overwrite line can look cleaner, or just print
            #     print(f"🔴 Speaking: {', '.join(spk_labels)}")
            # else:
            #     # Silence
            #     print(".", end="", flush=True)
            
            # 6. SMART PRINTING LOGIC
            if active_indices:
                # Reset silence since someone is talking
                silence_counter = 0
                
                # Create a string ID for the current speaker(s)
                # e.g., "Speaker 1" or "Speaker 1 + Speaker 2"
                current_speaker = " + ".join([f"Speaker {i+1}" for i in active_indices])
                
                # Only print if the speaker CHANGED (or if we are resuming from silence)
                if current_speaker != last_speaker:
                    print(f"\n🔴 [Active] {current_speaker} started speaking...")
                    last_speaker = current_speaker
                else:
                    # If it's the same person continuing to speak, just print a small indicator
                    # so you know it's still alive, but without spamming new lines.
                    print(">", end="", flush=True)
            else:
                # Silence
                silence_counter += 1
                if silence_counter > 5 and last_speaker is not None:
                    # If silent for a few chunks (~2.5s), reset state
                    # This allows "Speaker 1" to appear again as a fresh turn later
                    last_speaker = None
                    print(".", end="", flush=True)
                else:
                    print(".", end="", flush=True)

except KeyboardInterrupt:
    print("\nStopped.")
