"""
ingestion.py – load session videos, sample frames, and stream audio.

This refactored module avoids temporary audio files and exposes an audio_stream
generator that yields small WAV chunks directly from ffmpeg.  It still
discovers video files and samples frames for non‑verbal analysis.
"""

from __future__ import annotations

import io
import os
from typing import Generator, List, Tuple

import ffmpeg  # type: ignore
import numpy as np
from PIL import Image

try:
    import cv2  # Efficient video decoding
except ImportError:
    cv2 = None  # Fall back to PIL if unavailable

VIDEO_EXTS = (".mp4", ".m4v", ".mov", ".mkv")


def discover_videos(session_dir: str, exts: Tuple[str, ...] = VIDEO_EXTS) -> List[str]:
    """Return absolute paths to all video files in ``session_dir`` with allowed extensions."""
    return [
        os.path.join(session_dir, f)
        for f in os.listdir(session_dir)
        if f.lower().endswith(exts)
    ]


def sample_frames(video_path: str, num_frames: int = 6) -> List[Image.Image]:
    """
    Sample ``num_frames`` frames uniformly across a video.  Uses OpenCV for speed.
    """
    if cv2 is None:
        raise RuntimeError("OpenCV is required for frame sampling but is not installed.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open {video_path}")

    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if count <= 0:
        cap.release()
        return []

    indices = np.linspace(0, count - 1, num_frames, dtype=int)
    images: List[Image.Image] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            images.append(Image.fromarray(frame_rgb))
    cap.release()
    return images


def audio_stream(
    video_path: str, chunk_duration: float = 10.0, sample_rate: int = 16000
) -> Generator[bytes, None, None]:
    """
    Yield contiguous WAV byte chunks from the video's audio track.

    Each yielded chunk contains up to ``chunk_duration`` seconds of mono audio
    at ``sample_rate`` Hz.  A single ffmpeg process is spawned per chunk.
    """
    try:
        info = ffmpeg.probe(video_path)
    except ffmpeg.Error as e:
        raise RuntimeError(f"Could not probe {video_path}: {e.stderr}") from e

    duration = float(info["format"]["duration"])
    t = 0.0
    while t < duration:
        # Seek to ``t`` seconds and extract ``chunk_duration`` seconds of audio
        process = (
            ffmpeg.input(video_path, ss=t, t=chunk_duration)
            .output("pipe:", format="wav", ac=1, ar=str(sample_rate))
            .run_async(pipe_stdout=True, pipe_stderr=True)
        )
        out, _ = process.communicate()
        process.kill()
        if not out:
            break
        yield out
        t += chunk_duration


def load_entire_audio(video_path: str, sample_rate: int = 16000) -> bytes:
    """
    Convenience function to read the entire audio track into memory as WAV bytes.
    Uses ``audio_stream`` with an effectively infinite chunk size.
    """
    return b"".join(audio_stream(video_path, chunk_duration=1e9, sample_rate=sample_rate))
