"""
asr.py – streaming and offline transcription using faster_whisper.

This module wraps the faster_whisper WhisperModel to provide two
convenient functions:

• transcribe_wav_bytes(buf): transcribe a complete WAV byte string.
• transcribe_stream(wav_chunks): transcribe a generator of WAV chunks
  produced by ingestion.audio_stream().

The Whisper model is loaded once and reused.  GPU and half-precision
support is enabled if available.
"""

from __future__ import annotations

import io
import os
from typing import Iterable, List, Tuple

import numpy as np
import soundfile as sf
import torch

from faster_whisper import WhisperModel  # type: ignore

# Configuration via environment variables
ASR_MODEL_SIZE = os.getenv("ASR_MODEL_SIZE", "small")
ASR_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ASR_COMPUTE_TYPE = "float16" if ASR_DEVICE == "cuda" else "float32"

# Global model cache
_model: WhisperModel | None = None


def _load_model() -> WhisperModel:
    """
    Lazily load the Whisper model and cache it for future calls.
    """
    global _model
    if _model is None:
        _model = WhisperModel(
            ASR_MODEL_SIZE,
            device=ASR_DEVICE,
            compute_type=ASR_COMPUTE_TYPE,
        )
    return _model


def _decode_wav_bytes(buf: bytes) -> Tuple[np.ndarray, int]:
    """
    Decode a WAV byte buffer into a numpy array and return (audio, sample_rate).
    """
    with io.BytesIO(buf) as f:
        audio, sr = sf.read(f, dtype="float32")
    return audio, sr


def transcribe_wav_bytes(buf: bytes) -> str:
    """
    Transcribe a complete WAV byte string.

    Parameters
    ----------
    buf:
        Bytes containing WAV-encoded audio.

    Returns
    -------
    str
        The transcription as a single string.
    """
    audio, _ = _decode_wav_bytes(buf)
    model = _load_model()
    # Do not pass sample_rate to faster_whisper; the model infers it
    segments, _ = model.transcribe(audio)
    return " ".join(seg.text for seg in segments)


def transcribe_stream(wav_chunks: Iterable[bytes], overlap: float = 0.5) -> str:
    """
    Transcribe a sequence of contiguous WAV chunks yielded by ingestion.audio_stream().

    Each chunk is decoded and transcribed.  To avoid missing words at
    boundaries, the last ``overlap`` seconds of the previous chunk are
    prepended to the current chunk before transcription.

    Parameters
    ----------
    wav_chunks:
        Iterable of WAV byte strings in sequence.
    overlap:
        Seconds of audio to overlap between chunks.  Defaults to 0.5s.

    Returns
    -------
    str
        The concatenated transcription of all chunks.
    """
    model = _load_model()
    full_text: List[str] = []
    last_audio: np.ndarray | None = None

    # Assumed sample rate based on ffmpeg conversion (16kHz)
    assumed_sr = 16000

    for chunk in wav_chunks:
        audio, _ = _decode_wav_bytes(chunk)
        if last_audio is not None:
            n_samples = int(overlap * assumed_sr)
            audio = np.concatenate((last_audio[-n_samples:], audio))
        # Call transcribe without sample_rate; faster_whisper handles resampling
        segments, _ = model.transcribe(audio)
        full_text.extend(seg.text for seg in segments)
        last_audio = audio

    return " ".join(full_text)
