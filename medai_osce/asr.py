# medai_osce/asr.py
"""
asr.py – streaming and offline transcription using faster_whisper.

Enhancements (drop-in, backward compatible):
- Quantization & precision control via ASR_COMPUTE_TYPE (float16, int8_float16, int8, float32).
- VAD (voice activity detection) enabled by default to skip silence + tunable VAD params.
- Cross‑chunk context: condition_on_previous_text + initial_prompt tail.
- Configurable beam size via ASR_BEAM_SIZE (default 5).
- Low-latency chunk handling used by offline PCM transcription.
- Multi-model cache keyed by (size, compute_type, device, threads, workers).
- Language pinning via ASR_LANG (e.g., "en") for stability/perf.
- CPU threads/workers knobs via ASR_CPU_THREADS / ASR_NUM_WORKERS.

Environment variables (optional):
  ASR_MODEL_SIZE          default 'small'
  ASR_COMPUTE_TYPE        default 'float16' on CUDA else 'float32'
                          (also accepts 'int8_float16' on GPU, 'int8' on CPU)
  ASR_BEAM_SIZE           default '5'
  ASR_VAD                 default '1' (enable VAD)
  ASR_CONDITION_ON_PREV   default '1' (enable cross‑chunk conditioning)
  ASR_CPU_THREADS         default '0'  (CTranslate2 default)
  ASR_NUM_WORKERS         default '1'
  ASR_VAD_MIN_SIL_MS      default '400'
  ASR_VAD_SPEECH_PAD      default '200'
  ASR_LANG                default None (pin e.g. 'en' to speed up/steady output)
"""

from __future__ import annotations

import io
import os
from typing import Iterable, List, Tuple, Optional, Dict

import numpy as np
import soundfile as sf
import torch

from faster_whisper import WhisperModel  # type: ignore

# --------------------------------------------------------------------------- #
# Configuration via environment variables
# --------------------------------------------------------------------------- #

def _bool_env(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")

ASR_MODEL_SIZE = os.getenv("ASR_MODEL_SIZE", "small")
ASR_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ASR_COMPUTE_TYPE = os.getenv("ASR_COMPUTE_TYPE") or ("float16" if ASR_DEVICE == "cuda" else "float32")
ASR_BEAM_SIZE = int(os.getenv("ASR_BEAM_SIZE", "5"))
ASR_VAD = _bool_env("ASR_VAD", True)
ASR_CONDITION_ON_PREV = _bool_env("ASR_CONDITION_ON_PREV", True)
ASR_CPU_THREADS = int(os.getenv("ASR_CPU_THREADS", "0") or "0")
ASR_NUM_WORKERS = int(os.getenv("ASR_NUM_WORKERS", "1"))
ASR_VAD_MIN_SIL_MS = int(os.getenv("ASR_VAD_MIN_SIL_MS", "400"))
ASR_VAD_SPEECH_PAD = int(os.getenv("ASR_VAD_SPEECH_PAD", "200"))
ASR_LANG = os.getenv("ASR_LANG")  # e.g., "en"

# Assumed sample rate for overlap math when chunking PCM; ingestion emits 16 kHz.
_ASSUMED_SR = 16000

# Global model cache keyed by (size, compute_type, device, cpu_threads, num_workers)
_model_cache: Dict[Tuple[str, str, str, int, int], WhisperModel] = {}


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #

def _load_model(size: Optional[str] = None, compute_type: Optional[str] = None) -> WhisperModel:
    """
    Lazily load a Whisper model instance and cache it. Multiple models can co-exist,
    keyed by (size, compute_type, device, cpu_threads, num_workers).
    """
    _size = size or ASR_MODEL_SIZE
    _ctype = compute_type or ASR_COMPUTE_TYPE
    key = (_size, _ctype, ASR_DEVICE, ASR_CPU_THREADS, ASR_NUM_WORKERS)
    if key not in _model_cache:
        cpu_threads = max(0, ASR_CPU_THREADS)     # ensure int; 0 lets CT2 decide
        num_workers = max(1, ASR_NUM_WORKERS)     # ensure >=1
        _model_cache[key] = WhisperModel(
            _size,
            device=ASR_DEVICE,
            compute_type=_ctype,
            cpu_threads=cpu_threads,              # never None
            num_workers=num_workers,
        )
    return _model_cache[key]


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #

def _decode_wav_bytes(buf: bytes) -> Tuple[np.ndarray, int]:
    """Decode a WAV byte buffer into a numpy array and return (audio, sample_rate)."""
    with io.BytesIO(buf) as f:
        audio, sr = sf.read(f, dtype="float32")
    return audio, sr


def _vad_params() -> dict:
    return {
        "min_silence_duration_ms": ASR_VAD_MIN_SIL_MS,
        "speech_pad_ms": ASR_VAD_SPEECH_PAD,
    }


# --------------------------------------------------------------------------- #
# Offline transcription helpers
# --------------------------------------------------------------------------- #

def transcribe_wav_bytes(buf: bytes, language: Optional[str] = None) -> str:
    """Transcribe a complete WAV byte string."""
    audio, _ = _decode_wav_bytes(buf)
    model = _load_model()
    segments, _ = model.transcribe(
        audio,
        language=language or ASR_LANG,
        vad_filter=ASR_VAD,
        vad_parameters=_vad_params(),
        beam_size=ASR_BEAM_SIZE,
    )
    return " ".join(seg.text for seg in segments)


# NOTE: We use the same chunked logic for offline PCM to avoid re-spawns.
class AsrStreamer:
    """
    Stateful chunk‑by‑chunk transcriber that maintains cross‑chunk context.

    Use with ingestion.audio_pcm_stream(...) which yields float32 mono arrays.
    """
    def __init__(
        self,
        language: Optional[str] = None,
        tail_words: int = 200,
        model_size: Optional[str] = None,
        compute_type: Optional[str] = None,
    ) -> None:
        self.model = _load_model(size=model_size, compute_type=compute_type)
        self.language = language or ASR_LANG
        self.tail_words = tail_words
        self._prev_tail: str = ""

    def reset_context(self) -> None:
        """Clear the cross‑chunk textual context."""
        self._prev_tail = ""

    def transcribe_chunk(self, audio_f32_mono: np.ndarray) -> str:
        """
        Transcribe a single float32 mono PCM array (range [-1, 1]).
        """
        segments, _ = self.model.transcribe(
            audio_f32_mono,
            language=self.language,
            vad_filter=ASR_VAD,
            vad_parameters=_vad_params(),
            condition_on_previous_text=ASR_CONDITION_ON_PREV,
            initial_prompt=(self._prev_tail or None),
            beam_size=ASR_BEAM_SIZE,
        )
        text = " ".join(s.text for s in segments)
        if text:
            words = (self._prev_tail + " " + text).split()
            self._prev_tail = " ".join(words[-self.tail_words :])
        return text

    def transcribe_iter(self, pcm_chunks: Iterable[np.ndarray], overlap: float = 0.25) -> str:
        """
        Transcribe an iterable of float32 PCM chunks with small overlap between chunks.
        """
        full: List[str] = []
        tail_samples = int(overlap * _ASSUMED_SR)
        prev: Optional[np.ndarray] = None

        for arr in pcm_chunks:
            audio = arr
            if prev is not None and tail_samples > 0 and len(prev) >= tail_samples:
                audio = np.concatenate([prev[-tail_samples:], arr], axis=0)
            full.append(self.transcribe_chunk(audio))
            prev = arr
        return " ".join(full)


def transcribe_pcm_stream(
    pcm_chunks: Iterable[np.ndarray],
    overlap: float = 0.25,
    language: Optional[str] = None,
    model_size: Optional[str] = None,
    compute_type: Optional[str] = None,
) -> str:
    """
    Transcribe float32 PCM chunks with cross‑chunk context.
    Used by the offline pipeline for speed (single ffmpeg process).
    """
    streamer = AsrStreamer(language=language, model_size=model_size, compute_type=compute_type)
    return streamer.transcribe_iter(pcm_chunks, overlap=overlap)


# --------------------------------------------------------------------------- #
# (Streaming path previously lived here for online updates)
# It is intentionally not used by the offline pipeline and therefore omitted.
# --------------------------------------------------------------------------- #
