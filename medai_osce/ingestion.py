# medai_osce/ingestion.py
"""
ingestion.py – load session videos, sample frames, and stream audio.

This module discovers session videos, samples frames for non‑verbal analysis,
and provides two audio streaming paths:

1) audio_pcm_stream(...)  → single, long‑lived ffmpeg process that outputs
   contiguous float32 PCM chunks (low overhead; used by offline pipeline).
2) audio_stream(...)      → legacy path that re‑spawns ffmpeg per chunk and
   outputs WAV bytes (kept for backward compatibility).

Enhancements:
- multi_audio_pcm_stream([...]) to chain multiple videos seamlessly.
- Optional scene‑change gating inside sample_frames_window() (opt‑in via env):
    SCENE_CHANGE_ENABLE=1
    SCENE_CHANGE_DIFF_THR=6.0 (mean absolute per‑pixel threshold in [0..255])
- FFmpeg fallback for frame extraction when OpenCV isn't installed or
  USE_FFMPEG_SAMPLING=1.
"""

from __future__ import annotations

import io
import os
from typing import Generator, List, Tuple, Optional

import ffmpeg  # type: ignore
import numpy as np
from PIL import Image

try:
    import cv2  # Efficient video decoding (optional)
except ImportError:
    cv2 = None  # Optional: we can fall back to FFmpeg image2pipe

VIDEO_EXTS = (".mp4", ".m4v", ".mov", ".mkv")


def _bool_env(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")

SCENE_CHANGE_ENABLE = _bool_env("SCENE_CHANGE_ENABLE", False)
SCENE_CHANGE_DIFF_THR = float(os.getenv("SCENE_CHANGE_DIFF_THR", "6.0"))
USE_FFMPEG_SAMPLING = _bool_env("USE_FFMPEG_SAMPLING", False)


# --------------------------------------------------------------------------- #
# Video discovery
# --------------------------------------------------------------------------- #

def discover_videos(session_dir: str, exts: Tuple[str, ...] = VIDEO_EXTS) -> List[str]:
    """Return absolute paths to all video files in ``session_dir`` with allowed extensions."""
    paths = [
        os.path.join(session_dir, f)
        for f in os.listdir(session_dir)
        if f.lower().endswith(exts)
    ]
    # Deterministic order helps reproducibility
    return sorted(paths)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _significant_change(prev_rgb: Optional[np.ndarray], cur_rgb: np.ndarray, thr: float) -> bool:
    """Cheap per-pixel mean absolute difference (RGB)."""
    if prev_rgb is None:
        return True
    prev = prev_rgb.astype(np.int16)
    cur = cur_rgb.astype(np.int16)
    mad = float(np.mean(np.abs(cur - prev)))
    return mad > thr


def _probe_video_duration(video_path: str) -> float:
    """Return video duration in seconds using ffmpeg.probe (fallback: 0.0)."""
    try:
        info = ffmpeg.probe(video_path)
        dur = float(info.get("format", {}).get("duration", 0.0) or 0.0)
        if dur > 0:
            return dur
        for s in info.get("streams", []):
            if s.get("codec_type") == "video" and "duration" in s:
                return float(s["duration"])
    except Exception:
        pass
    return 0.0


def _extract_frame_ffmpeg(video_path: str, t: float) -> Optional[Image.Image]:
    """Extract a single frame at timestamp t (seconds) via FFmpeg image2pipe."""
    try:
        out, _ = (
            ffmpeg
            .input(video_path, ss=max(0.0, float(t)))
            .output("pipe:", vframes=1, format="image2", vcodec="png")
            .global_args("-nostdin")
            .global_args("-loglevel", "error")
            .run(capture_stdout=True, capture_stderr=True)
        )
        if not out:
            return None
        return Image.open(io.BytesIO(out)).convert("RGB")
    except Exception:
        return None


def _sample_frames_ffmpeg(video_path: str, num_frames: int = 6) -> List[Image.Image]:
    """Uniformly sample num_frames frames using FFmpeg at timestamped seeks."""
    if num_frames <= 0:
        return []
    duration = _probe_video_duration(video_path)
    if duration <= 0.0:
        times = [i * 0.5 for i in range(num_frames)]
    else:
        times = np.linspace(0, max(0.0, duration - 0.05), num_frames, dtype=float).tolist()

    images: List[Image.Image] = []
    prev_np: Optional[np.ndarray] = None
    for t in times:
        img = _extract_frame_ffmpeg(video_path, t)
        if img is None:
            continue
        rgb = np.array(img)
        if not SCENE_CHANGE_ENABLE or _significant_change(prev_np, rgb, SCENE_CHANGE_DIFF_THR):
            images.append(img)
            prev_np = rgb
        if len(images) >= num_frames:
            break
    return images[:num_frames]


def _sample_frames_window_ffmpeg(video_path: str, t0: float, t1: float, num_frames: int = 2) -> List[Image.Image]:
    """Uniformly sample num_frames frames within [t0, t1) via FFmpeg seeks."""
    if num_frames <= 0 or t1 <= t0:
        return []
    times = np.linspace(float(t0), float(t1 - 1e-2), max(num_frames, 2), dtype=float).tolist()

    images: List[Image.Image] = []
    prev_np: Optional[np.ndarray] = None
    for t in times:
        img = _extract_frame_ffmpeg(video_path, t)
        if img is None:
            continue
        rgb = np.array(img)
        if not SCENE_CHANGE_ENABLE or _significant_change(prev_np, rgb, SCENE_CHANGE_DIFF_THR):
            images.append(img)
            prev_np = rgb
        if len(images) >= num_frames:
            break
    return images[:num_frames]


# --------------------------------------------------------------------------- #
# Frame sampling (OpenCV preferred, FFmpeg fallback)
# --------------------------------------------------------------------------- #

def sample_frames(video_path: str, num_frames: int = 6) -> List[Image.Image]:
    """
    Sample ``num_frames`` frames uniformly across a video.

    Priority:
      1) OpenCV (if available and not forced off)
      2) FFmpeg image2pipe fallback (no OpenCV required)
    """
    if (cv2 is None) or USE_FFMPEG_SAMPLING:
        return _sample_frames_ffmpeg(video_path, num_frames=num_frames)

    # --- OpenCV path ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return _sample_frames_ffmpeg(video_path, num_frames=num_frames)

    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if count <= 0 or num_frames <= 0:
        cap.release()
        return []

    indices = np.linspace(0, count - 1, num_frames, dtype=int)
    images: List[Image.Image] = []
    prev_rgb_np: Optional[np.ndarray] = None

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if not SCENE_CHANGE_ENABLE or _significant_change(prev_rgb_np, frame_rgb, SCENE_CHANGE_DIFF_THR):
            images.append(Image.fromarray(frame_rgb))
            prev_rgb_np = frame_rgb
        if len(images) >= num_frames:
            break

    cap.release()
    if not images:
        return _sample_frames_ffmpeg(video_path, num_frames=num_frames)
    return images[:num_frames]


def sample_frames_window(video_path: str, t0: float, t1: float, num_frames: int = 2) -> List[Image.Image]:
    """
    Streaming helper retained for compatibility, not used by offline path.
    """
    if (cv2 is None) or USE_FFMPEG_SAMPLING:
        return _sample_frames_window_ffmpeg(video_path, t0, t1, num_frames=num_frames)

    if num_frames <= 0 or t1 <= t0:
        return []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return _sample_frames_window_ffmpeg(video_path, t0, t1, num_frames=num_frames)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    start_f = max(0, int(t0 * fps))
    end_f = max(start_f + 1, int(t1 * fps))
    raw_indices = np.linspace(start_f, end_f - 1, max(num_frames, 2), dtype=int)

    images: List[Image.Image] = []
    prev_rgb_np: Optional[np.ndarray] = None

    for idx in raw_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if not SCENE_CHANGE_ENABLE or _significant_change(prev_rgb_np, frame_rgb, SCENE_CHANGE_DIFF_THR):
            images.append(Image.fromarray(frame_rgb))
            prev_rgb_np = frame_rgb
        if len(images) >= num_frames:
            break

    cap.release()

    if not images:
        return _sample_frames_window_ffmpeg(video_path, t0, t1, num_frames=num_frames)
    return images[:num_frames]


# --------------------------------------------------------------------------- #
# Audio streaming (recommended for offline): single process → float32 PCM chunks
# --------------------------------------------------------------------------- #

def audio_pcm_stream(
    video_path: str,
    chunk_duration: float = 10.0,
    sample_rate: int = 16000,
) -> Generator[np.ndarray, None, None]:
    """
    Yield contiguous **float32 mono PCM** arrays from a single long‑lived ffmpeg pipe.
    Used by the offline pipeline to avoid re-spawning ffmpeg or WAV repacking.
    """
    if chunk_duration <= 0:
        raise ValueError("chunk_duration must be > 0")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be > 0")

    process = (
        ffmpeg
        .input(video_path)
        .output("pipe:", format="s16le", ac=1, ar=str(sample_rate))
        .global_args("-nostdin")
        .global_args("-loglevel", "error")
        .run_async(pipe_stdout=True, pipe_stderr=True)
    )

    bytes_per_sample = 2  # s16le
    chunk_bytes = int(round(chunk_duration * sample_rate)) * bytes_per_sample
    if chunk_bytes <= 0:
        chunk_bytes = bytes_per_sample

    buf = b""
    try:
        while True:
            need = chunk_bytes - len(buf)
            data = process.stdout.read(need)
            if not data:
                if buf:
                    arr = np.frombuffer(buf, dtype=np.int16).astype(np.float32) / 32768.0
                    yield arr
                break
            buf += data
            if len(buf) >= chunk_bytes:
                out, buf = buf[:chunk_bytes], buf[chunk_bytes:]
                arr = np.frombuffer(out, dtype=np.int16).astype(np.float32) / 32768.0
                yield arr
    finally:
        try:
            if process.stdout:
                process.stdout.close()
        except Exception:
            pass
        try:
            process.wait()
        except Exception:
            pass


def multi_audio_pcm_stream(
    video_paths: List[str],
    chunk_duration: float = 10.0,
    sample_rate: int = 16000,
) -> Generator[np.ndarray, None, None]:
    """
    Chain multiple videos into a single PCM stream (useful when a session has several files).
    """
    for p in video_paths:
        yield from audio_pcm_stream(p, chunk_duration=chunk_duration, sample_rate=sample_rate)


# --------------------------------------------------------------------------- #
# Legacy WAV chunk path (kept for compatibility; not used offline)
# --------------------------------------------------------------------------- #

def audio_stream(
    video_path: str, chunk_duration: float = 10.0, sample_rate: int = 16000
) -> Generator[bytes, None, None]:
    """
    Yield contiguous WAV byte chunks from the video's audio track (legacy).
    """
    try:
        info = ffmpeg.probe(video_path)
    except ffmpeg.Error as e:
        raise RuntimeError(f"Could not probe {video_path}: {e.stderr}") from e

    duration = float(info["format"]["duration"])
    t = 0.0
    while t < duration:
        process = (
            ffmpeg.input(video_path, ss=t, t=chunk_duration)
            .output("pipe:", format="wav", ac=1, ar=str(sample_rate))
            .global_args("-nostdin")
            .global_args("-loglevel", "error")
            .run_async(pipe_stdout=True, pipe_stderr=True)
        )
        out, _ = process.communicate()
        process.kill()
        if not out:
            break
        yield out
        t += chunk_duration


def load_entire_audio(video_path: str, sample_rate: int = 16000) -> bytes:
    """Read entire audio track into WAV bytes (legacy)."""
    return b"".join(audio_stream(video_path, chunk_duration=1e9, sample_rate=sample_rate))
