"""Media ingestion utilities for the simplified OSCE pipeline.

This module contains functions to extract audio from video files and to
sample image frames at a fixed interval.  It relies on the `ffmpeg`
binary (via subprocess for frame extraction and ffmpeg-python for audio
extraction).  Ensure you have `ffmpeg` installed on your system.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import List, Sequence

import ffmpeg  # type: ignore
from PIL import Image
import numpy as np


def extract_audio(video_paths: Sequence[str], out_dir: str | None = None) -> Path:
    """Extract a mono 16 kHz WAV file from the first available video.

    Parameters
    ----------
    video_paths:
        A sequence of video file paths.  The first path that exists on
        disk will be used for audio extraction.  If a separate room
        microphone recording is available (e.g. `room_audio.wav`), you
        should pass it as the only element in `video_paths`.
    out_dir:
        Directory where the extracted WAV file will be written.  If
        ``None``, a temporary directory is created.

    Returns
    -------
    Path
        The path to the extracted WAV file.

    Raises
    ------
    FileNotFoundError:
        If none of the provided paths exist.
    RuntimeError:
        If ffmpeg fails to extract the audio.
    """
    if not video_paths:
        raise ValueError("video_paths must be non-empty")
    src: Path | None = None
    for vp in video_paths:
        p = Path(vp)
        if p.exists():
            src = p
            break
    if src is None:
        raise FileNotFoundError(f"None of the provided video paths exist: {video_paths}")

    out_dir = out_dir or tempfile.mkdtemp(prefix="medai_audio_")
    out_path = Path(out_dir) / (src.stem + ".wav")

    try:
        (
            ffmpeg
            .input(str(src))
            .output(str(out_path), ac=1, ar=16000, format="wav")
            .run(quiet=True, overwrite_output=True)
        )
    except ffmpeg.Error as e:
        raise RuntimeError(f"ffmpeg failed: {e.stderr}") from e

    return out_path


def sample_frames(video_path: str, fps: float = 0.5, max_frames: int | None = None) -> List[Image.Image]:
    """Sample frames from a video at a given rate and return PIL Images.

    Parameters
    ----------
    video_path:
        Path to a video file from which to extract frames.
    fps:
        Number of frames per second to sample.  A value of 0.5 means one
        frame every two seconds.  The default of 0.5 yields a small
        number of frames for long videos and is suitable for scoring
        behavioural rubrics.
    max_frames:
        Optional cap on the total number of frames returned.  If set
        to ``None``, all sampled frames are returned.

    Returns
    -------
    List[Image.Image]
        A list of PIL Image objects in RGB mode.

    Raises
    ------
    FileNotFoundError:
        If the video file does not exist.
    RuntimeError:
        If ffmpeg or ffprobe fails.
    """
    video = Path(video_path)
    if not video.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    # Construct the ffmpeg command.  Place -vframes right after the input.
    args = ["ffmpeg", "-i", str(video)]
    if max_frames is not None:
        args += ["-vframes", str(max_frames)]
    args += [
        "-vf", f"fps={fps}",
        "-f", "image2pipe",
        "-pix_fmt", "rgb24",
        "-vcodec", "rawvideo",
        "-"
    ]

    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Determine frame dimensions using ffprobe
    try:
        probe = ffmpeg.probe(str(video))
        streams = probe.get("streams", [])
        vs = next((s for s in streams if s.get("codec_type") == "video"), None)
        if vs is None:
            raise RuntimeError("No video stream found")
        width = int(vs["width"])
        height = int(vs["height"])
    except ffmpeg.Error as e:
        proc.kill()
        raise RuntimeError(f"ffprobe failed: {e.stderr}") from e

    frames: List[Image.Image] = []
    frame_size = width * height * 3  # for RGB24
    count = 0

    while True:
        if max_frames is not None and count >= max_frames:
            break
        raw = proc.stdout.read(frame_size)
        if not raw:
            break
        arr = np.frombuffer(raw, np.uint8).reshape((height, width, 3))
        frames.append(Image.fromarray(arr, "RGB"))
        count += 1

    stderr = proc.stderr.read()
    proc.stdout.close()
    proc.stderr.close()
    proc.wait()

    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg returned error code {proc.returncode}: {stderr.decode()}")

    return frames
