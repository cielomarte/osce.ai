#!/usr/bin/env python3
"""Command‑line entry point to grade an OSCE session.

This script orchestrates the simplified OSCE pipeline: it loads the
session videos, extracts audio, runs ASR to obtain a transcript,
optionally samples frames and generates vision captions, and then
scores the encounter against one or more rubrics.  Results are
printed to stdout and optionally written to a JSON file.

Example usage:

```bash
python medai_osce/grade_session.py \
    --session /path/to/session_2025_07_23 \
    --rubric communication_engagement_v1 \
    --rubric professionalism_behavior_v1 \
    --use-vision \
    --out report.json
```
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List

from . import ingestion, asr, vision, scoring


def _discover_videos(session_dir: Path) -> List[str]:
    """Return a list of video file paths in the session directory.

    Any files with extensions .mp4, .m4v, .mov or .mkv are considered
    videos.  The list is sorted alphabetically.
    """
    exts = {".mp4", ".m4v", ".mov", ".mkv"}
    vids = [str(p) for p in session_dir.iterdir() if p.suffix.lower() in exts]
    vids.sort()
    return vids


def main() -> None:
    parser = argparse.ArgumentParser(description="Grade an OSCE session against selected rubrics")
    parser.add_argument("--session", required=True, type=Path, help="Path to the session folder containing video files")
    parser.add_argument("--rubric", action="append", help="Rubric name to apply (can be used multiple times)")
    parser.add_argument("--use-vision", action="store_true", help="Enable vision captioning (requires implementation in vision.py)")
    parser.add_argument("--model", default="gpt-4o", help="OpenAI model name (default: gpt-4o)")
    parser.add_argument("--out", type=Path, help="Write the result to this JSON file")
    parser.add_argument("--asr-model-size", default="base", help="Whisper ASR model size (tiny/base/small/medium/large)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    session_dir: Path = args.session
    if not session_dir.is_dir():
        raise NotADirectoryError(f"Session directory not found: {session_dir}")
    video_paths = _discover_videos(session_dir)
    if not video_paths:
        raise FileNotFoundError(f"No video files found in {session_dir}")
    logging.info("Found %d video(s) in session", len(video_paths))

    # Extract audio from the first available source.  If a standalone
    # room_audio.wav file exists, use that exclusively.
    room_audio = session_dir / "room_audio.wav"
    audio_srcs = [str(room_audio)] if room_audio.exists() else video_paths
    audio_path = ingestion.extract_audio(audio_srcs)
    logging.info("Extracted audio to %s", audio_path)

    # Transcribe using Whisper
    transcript = asr.transcribe_audio(str(audio_path), model_size=args.asr_model_size)
    logging.info("Transcript length: %d characters", len(transcript))

    # Sample frames and caption them if requested
    vision_captions: List[str] = []
    if args.use_vision:
        try:
            # Use the first video for vision captions
            frames = ingestion.sample_frames(video_paths[0], fps=0.5, max_frames=60)
            vision_captions = vision.caption_frames(frames)
            logging.info("Generated %d vision captions", len(vision_captions))
        except NotImplementedError:
            logging.warning("Vision captioning is not implemented; continuing without vision cues")
            vision_captions = []

    # Determine rubrics to evaluate.  If none specified, evaluate all rubrics
    rubrics = args.rubric or [p.stem for p in (Path(__file__).resolve().parent / "rubrics").iterdir() if p.suffix == ".json"]

    results = []
    for rubric_name in rubrics:
        logging.info("Scoring rubric %s", rubric_name)
        try:
            result = scoring.score_rubric(rubric_name, transcript, vision_captions, model=args.model)
        except FileNotFoundError as exc:
            logging.error("Rubric not found: %s", exc)
            continue
        results.append(result.to_dict())

    # Print result JSON
    final = {"session": session_dir.name, "results": results}
    print(json.dumps(final, indent=2, ensure_ascii=False))
    if args.out:
        args.out.write_text(json.dumps(final, indent=2, ensure_ascii=False))
        logging.info("Wrote report to %s", args.out)


if __name__ == "__main__":
    main()
