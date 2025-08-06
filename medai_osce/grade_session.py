"""
grade_session.py – entry point for grading an OSCE session.

This script orchestrates the ingestion of video files, audio transcription,
vision captioning (optional), and scoring of rubrics.  It leverages the
refactored streaming ingestion and concurrent scoring modules to improve
performance.

Usage example:
    python -m medai_osce.grade_session \
        --session /path/to/session_dir \
        --rubric abdomen \
        --rubric cardiac_vascular \
        --use-vision \
        --out report.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from typing import List

from . import ingestion, asr, vision, scoring

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grade a single OSCE session.")
    parser.add_argument("--session", required=True, help="Path to the session directory containing video files.")
    parser.add_argument(
        "--rubric",
        action="append",
        required=True,
        help="Name of a rubric to score (e.g. 'abdomen').  Specify multiple times for multiple rubrics.",
    )
    parser.add_argument(
        "--use-vision",
        action="store_true",
        help="If set, sample frames and generate vision captions for non-verbal cues.",
    )
    parser.add_argument(
        "--model",
        default="gpt-3.5-turbo",
        help="OpenAI ChatCompletion model to use for scoring.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Path to write JSON results.  If omitted, results are printed to stdout.",
    )
    parser.add_argument(
        "--asr-model-size",
        default="small",
        help="Size of the faster-whisper ASR model (tiny, base, small, medium, large).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Discover videos
    videos = ingestion.discover_videos(args.session)
    if not videos:
        raise FileNotFoundError(f"No video files found in {args.session}")
    logger.info("Found %d video(s) in session", len(videos))
    first_video = videos[0]

    # Optionally sample frames and caption them
    vision_captions: List[str] = []
    if args.use_vision:
        frames = ingestion.sample_frames(first_video, num_frames=6)
        logger.info("Sampled %d frame(s) for vision", len(frames))
        vision_captions = vision.caption_frames(frames)
        logger.info("Generated %d vision caption(s)", len(vision_captions))

    # Stream audio and transcribe incrementally
    audio_chunks = ingestion.audio_stream(first_video, chunk_duration=10.0, sample_rate=16000)
    transcript = asr.transcribe_stream(audio_chunks)
    logger.info("Transcript length: %d characters", len(transcript))

    # Score all rubrics concurrently
    logger.info("Scoring %d rubric(s)", len(args.rubric))
    results = scoring.score_rubrics_concurrently(args.rubric, transcript, vision_captions, model=args.model)

    # Output results
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info("Results written to %s", args.out)
    else:
        print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
