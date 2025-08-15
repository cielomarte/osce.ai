# medai_osce/grade_session.py
"""
grade_session.py – entry point for grading an OSCE session (OFFLINE ONLY in this build).

Supported mode:
- Default (non-streaming): process full audio → single scoring pass

Streaming support remains in comments for easy re-enable (search for "STREAMING SUPPORT").
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
    parser = argparse.ArgumentParser(description="Grade a single OSCE session (offline).")
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

    # ----------------------- STREAMING SUPPORT (commented) -----------------------
    # parser.add_argument("--stream", action="store_true", help="(Disabled) Stream the session with partial updates.")
    # parser.add_argument("--update-interval", type=float, default=20.0)
    # parser.add_argument("--min-chars-delta", type=int, default=1200)
    # parser.add_argument("--asr-online-model-size", default=None)
    # parser.add_argument("--use-memory-store", action="store_true")
    # parser.add_argument("--recent-window-seconds", type=float, default=90.0)
    # ----------------------------------------------------------------------------

    # LLM engine and concurrency
    parser.add_argument(
        "--engine-base-url",
        default=None,
        help="OpenAI-compatible base URL (e.g., http://localhost:8000/v1 for vLLM).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=int(os.getenv("LLM_MAX_CONCURRENCY", "4")),
        help="Max concurrent rubric requests to the LLM.",
    )
    parser.add_argument(
        "--model",
        default="gpt-3.5-turbo",
        help="OpenAI ChatCompletion model to use for scoring.",
    )

    # Caching
    parser.add_argument(
        "--cache-dir",
        default=os.getenv("MEDAI_CACHE_DIR", os.path.expanduser("~/.medai_osce_cache")),
        help="Base directory for local caches (LLM, vision).",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable on-disk caching of LLM responses.",
    )

    # ASR config
    parser.add_argument(
        "--asr-model-size",
        default="small",
        help="Size of the faster-whisper ASR model (tiny, base, small, medium, large).",
    )
    parser.add_argument(
        "--asr-compute-type",
        default=None,
        help="Override ASR compute_type (e.g., float16, int8_float16, int8, float32).",
    )

    parser.add_argument(
        "--out",
        default=None,
        help="Path to write JSON results.  If omitted, results are printed to stdout.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Validate session path early
    if not os.path.isdir(args.session):
        raise FileNotFoundError(f"Session directory not found: {args.session}")

    # Discover videos
    videos = ingestion.discover_videos(args.session)
    if not videos:
        raise FileNotFoundError(f"No video files found in {args.session}")
    logger.info("Found %d video(s) in session", len(videos))
    first_video = videos[0]

    # Wire env for downstream modules
    os.environ["ASR_MODEL_SIZE"] = args.asr_model_size
    if args.asr_compute_type:
        os.environ["ASR_COMPUTE_TYPE"] = args.asr_compute_type
    if args.engine_base_url:
        os.environ["OPENAI_BASE_URL"] = args.engine_base_url
    if args.no_cache:
        os.environ["LLM_CACHE_DISABLE"] = "1"
    if args.cache_dir:
        llm_dir = os.path.join(args.cache_dir, "llm")
        vision_dir = os.path.join(args.cache_dir, "vision")
        os.environ.setdefault("LLM_CACHE_DIR", llm_dir)
        os.environ.setdefault("VISION_CACHE_DIR", vision_dir)
        os.makedirs(llm_dir, exist_ok=True)
        os.makedirs(vision_dir, exist_ok=True)

    # ----------------------- STREAMING SUPPORT (commented) -----------------------
    # if getattr(args, "stream", False):
    #     logger.warning("Streaming mode is currently disabled in this build; running offline instead.")
    # ---------------------------------------------------------------------------

    # ----- Non-streaming path (original behavior) -----
    vision_captions: List[str] = []
    if args.use_vision:
        try:
            frames = ingestion.sample_frames(first_video, num_frames=6)
            logger.info("Sampled %d frame(s) for vision", len(frames))
            vision_captions = vision.caption_frames(frames)
            logger.info("Generated %d vision caption(s)", len(vision_captions))
        except Exception as e:
            logger.warning("Vision sampling disabled due to error: %s", e)
            vision_captions = []

    # Single PCM pipeline across all videos (fast offline ASR)
    audio_chunks = (
        ingestion.multi_audio_pcm_stream(videos, chunk_duration=10.0, sample_rate=16000)
        if len(videos) > 1 else
        ingestion.audio_pcm_stream(first_video, chunk_duration=10.0, sample_rate=16000)
    )
    transcript = asr.transcribe_pcm_stream(audio_chunks)
    logger.info("Transcript length: %d characters", len(transcript))

    # Score all rubrics concurrently
    logger.info("Scoring %d rubric(s)", len(args.rubric))
    results = scoring.score_rubrics_concurrently(
        args.rubric, transcript, vision_captions, model=args.model, max_concurrency=args.concurrency
    )

    _emit_results(results, args.out)


# ----------------------- STREAMING SUPPORT (commented) -----------------------
# def _run_streaming_session(video_path: str, args: argparse.Namespace) -> None:
#     """Streaming mode is disabled in this build; see README to re-enable."""
#     raise NotImplementedError("Streaming mode is currently disabled.")
# --------------------------------------------------------------------------- #


def _emit_results(results: List[dict], out_path: str | None) -> None:
    if out_path:
        dirn = os.path.dirname(out_path)
        if dirn:
            os.makedirs(dirn, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info("Results written to %s", out_path)
    else:
        print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
