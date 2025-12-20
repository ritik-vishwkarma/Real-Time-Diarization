import numpy as np
import logging
from typing import Tuple
import config
from vad_engine import NeMoVADGate

logger = logging.getLogger("VADGate")


class ConditionedAudio:
    """
    Immutable conditioned audio passed downstream.
    """
    def __init__(self, raw: np.ndarray, normalized: np.ndarray, energy_db: float):
        self.raw = raw
        self.normalized = normalized
        self.energy_db = energy_db


class VADGate:
    """
    SINGLE speech authority.

    Rules:
    - Energy gate is primary
    - Neural gate only inside ambiguity band
    - Normalize exactly once
    - Hysteresis over windows (not samples)
    """

    HYSTERESIS_WINDOWS = 2

    def __init__(self):
        self.energy_thresh = config.ENERGY_THRESH_DBFS
        self.amb_low = getattr(config, "ENERGY_AMBIGUITY_LOW", -45)
        self.amb_high = getattr(config, "ENERGY_AMBIGUITY_HIGH", self.energy_thresh)

        try:
            self.neural = NeMoVADGate()
        except Exception:
            self.neural = None
            logger.warning("Neural VAD unavailable. Ambiguity → silence.")

        # Hysteresis state
        self._speech_count = 0
        self._silence_count = 0
        self._is_speech = False

    def process(self, audio: np.ndarray) -> Tuple[bool, ConditionedAudio]:
        """
        Input: raw float32 audio
        Output: (is_speech, ConditionedAudio)
        """

        # --- Energy ---
        rms = np.sqrt(np.mean(audio ** 2) + 1e-9)
        energy_db = 20 * np.log10(rms + 1e-9)

        # --- Normalize ONCE ---
        peak = np.max(np.abs(audio))
        if peak > config.GAIN_MIN_AMP:
            gain = min(config.GAIN_TARGET / peak, config.GAIN_MAX_AMP)
        else:
            gain = 1.0

        norm = np.clip(audio * gain, -1.0, 1.0)

        conditioned = ConditionedAudio(
            raw=audio,
            normalized=norm,
            energy_db=energy_db
        )

        # --- Raw Decision ---
        is_speech_raw = False

        if energy_db >= self.energy_thresh:
            is_speech_raw = True

        elif self.amb_low < energy_db < self.amb_high:
            if self.neural:
                try:
                    is_sp, _ = self.neural.is_speech(norm)
                    is_speech_raw = bool(is_sp)
                except Exception as e:
                    logger.error(f"Neural VAD failure: {e}")
                    is_speech_raw = False
            else:
                is_speech_raw = False

        else:
            is_speech_raw = False

        # --- Hysteresis ---
        if is_speech_raw:
            self._speech_count += 1
            self._silence_count = 0
            if self._speech_count >= self.HYSTERESIS_WINDOWS:
                self._is_speech = True
        else:
            self._silence_count += 1
            self._speech_count = 0
            if self._silence_count >= self.HYSTERESIS_WINDOWS:
                self._is_speech = False

        return self._is_speech, conditioned


# import numpy as np
# import logging
# import torch
# import config
# from vad_engine import NeMoVADGate
# from typing import Tuple

# logger = logging.getLogger("VADGate")

# class ConditionedAudio:
#     """Helper for passing safe audio + energy"""
#     def __init__(self, raw: np.ndarray, energy: float, normalized: np.ndarray):
#         self.raw = raw
#         self.energy_db = energy
#         self.normalized = normalized

# class VADGate:
#     """
#     Component 4.4: VADGate (Hardened Phase 4)
    
#     Responsibility:
#     - Energy VAD (Primary)
#     - Ambiguity Confirmation (MarbleNet) use strict API.
#     - Sole normalization authority (Peak Norm to 0.5).
#     - Hysteresis.
#     """
    
#     DB_THRESHOLD = config.ENERGY_AMBIGUITY_HIGH # -38
#     HYSTERESIS_WINDOWS = 2
    
#     def __init__(self):
#         try:
#              self.neural_vad = NeMoVADGate() 
#         except:
#              self.neural_vad = None
#              logger.warning("NeMo VAD not loaded. Ambiguity check fallback.")
             
#         # State
#         self.speech_windows_consecutive = 0
#         self.silence_windows_consecutive = 0
#         self.is_speech_state = False

#     def process(self, audio_chunk: np.ndarray) -> Tuple[bool, ConditionedAudio]:
#         """
#         Returns: (is_speech_final, ConditionedAudio)
#         """
#         # 1. Calc Energy (RMS)
#         rms = np.sqrt(np.mean(audio_chunk**2) + 1e-9)
#         db = 20 * np.log10(rms + 1e-9)
        
#         # 2. Peak Normalization (Hardened Phase 4)
#         # "Normalize exactly once"
#         peak = np.max(np.abs(audio_chunk))
#         gain = 1.0
        
#         if peak > config.GAIN_MIN_AMP:
#             gain = config.GAIN_TARGET / peak
        
#         # Apply Gain using float32 precision
#         norm_audio = audio_chunk * gain
        
#         # Clamp [-1.0, 1.0]
#         norm_audio = np.clip(norm_audio, -1.0, 1.0)
        
#         conditioned = ConditionedAudio(audio_chunk, db, norm_audio)

#         # 3. Decision Logic (Strict)
#         # Rule: "Energy VAD OR (Ambiguity AND Neural)"?
#         # Spec 4.4: "Energy VAD ... Ambiguity band [0.05, 0.15] ... MarbleNet confirmation ONLY inside ambiguity"
#         # We translate ratio band to dB band approx (-38 to -45).
        
#         is_speech_raw = False
        
#         if db >= self.DB_THRESHOLD:
#             # Clear Speech
#             is_speech_raw = True
#         elif config.ENERGY_AMBIGUITY_LOW < db < config.ENERGY_AMBIGUITY_HIGH:
#             # Ambiguity Band -> Confirm with Neural
#             if self.neural_vad:
#                 # Fix 3: Proper Unpacking
#                 # API: is_speech returns (bool, float) usually?
#                 # User says: "is_speech() returns (bool, prob)"
#                 try:
#                     is_sp, prob = self.neural_vad.is_speech(norm_audio)
#                     if is_sp: # Trust boolean decision from engine
#                          is_speech_raw = True
#                 except Exception as e:
#                     # Fallback if API mismatch
#                      logger.error(f"VAD API Error: {e}")
#             else:
#                 # No Neural VAD -> Conservative (Silence)
#                 is_speech_raw = False
#         else:
#             # Clear Silence
#             is_speech_raw = False

#         # 4. Hysteresis
#         final_decision = self.is_speech_state
        
#         if is_speech_raw:
#             self.speech_windows_consecutive += 1
#             self.silence_windows_consecutive = 0
#             if self.speech_windows_consecutive >= self.HYSTERESIS_WINDOWS:
#                 final_decision = True
#         else:
#             self.silence_windows_consecutive += 1
#             self.speech_windows_consecutive = 0
#             if self.silence_windows_consecutive >= self.HYSTERESIS_WINDOWS:
#                 final_decision = False
                
#         self.is_speech_state = final_decision
        
#         return final_decision, conditioned
