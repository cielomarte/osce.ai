# medai_osce/vision.py
"""
Vision model helper for the simplified OSCE pipeline.

Hardening:
- Lazy, defensive imports for torch/transformers to avoid import-time crashes
  caused by optional deps (e.g., SciPy) in the user's environment.
- On failure to load BLIP, return cached captions (if any); otherwise [].
- Logs a concise warning but NEVER aborts the run.

Signature remains: caption_frames(frames: List[PIL.Image.Image]) -> List[str]
"""

from __future__ import annotations

import io
import os
import hashlib
import logging
from pathlib import Path
from typing import Any, List, Optional

from PIL import Image

logger = logging.getLogger(__name__)

# Global singletons (created lazily)
_processor: Optional[Any] = None
_model: Optional[Any] = None

# Config
MODEL_NAME = os.getenv("BLIP_MODEL_NAME", "Salesforce/blip-image-captioning-base")
DEFAULT_PROMPT = os.getenv(
    "BLIP_BEHAVIOUR_PROMPT",
    (
        "Observe the medical student and simulated patient in this clinical exam and "
        "describe their non‑verbal behaviour in detail. Note whether the student washes "
        "or sanitises their hands, introduces themselves clearly, maintains appropriate "
        "eye contact, shows empathy through posture and gestures, and respects the "
        "patient's personal space."
    ),
)
MAX_NEW_TOKENS = int(os.getenv("BLIP_MAX_NEW_TOKENS", "30"))
BATCH_SIZE = int(os.getenv("BLIP_BATCH_SIZE", "8"))

def _vision_cache_dir() -> Path:
    d = Path(os.getenv("VISION_CACHE_DIR", "~/.medai_osce_cache/vision")).expanduser()
    d.mkdir(parents=True, exist_ok=True)
    return d

def _hash_image(img: Image.Image) -> str:
    """Stable content hash for an image (perceptual if available, else SHA256)."""
    try:
        import imagehash  # type: ignore
        return f"phash_{str(imagehash.phash(img))}"
    except Exception:
        buf = io.BytesIO()
        img.convert("RGB").resize((64, 64)).save(buf, format="PNG", optimize=True)
        return "sha256_" + hashlib.sha256(buf.getvalue()).hexdigest()

def _load_blip() -> tuple[Any, Any]:
    """
    Lazily load BLIP processor and model.
    NEVER raises import-time exceptions beyond this function; callers must handle.
    """
    global _processor, _model
    if _processor is not None and _model is not None:
        return _processor, _model

    try:
        import torch  # lazy import
        try:
            # Import only when needed; avoid module-level side effects
            from transformers import BlipProcessor, BlipForConditionalGeneration  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Transformers BLIP import failed; likely due to optional dependency issues "
                "(e.g., SciPy/NumPy mismatch)."
            ) from e

        # Pick device (prefer CUDA; try MPS on Apple; else CPU)
        if torch.cuda.is_available():
            device = torch.device("cuda")
            dtype = torch.float16
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = torch.device("mps")
            dtype = torch.float16
        else:
            device = torch.device("cpu")
            dtype = torch.float32

        processor = BlipProcessor.from_pretrained(MODEL_NAME)
        model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME, torch_dtype=dtype)
        model.to(device)
        model.eval()

        _processor, _model = processor, model
        return processor, model

    except Exception as e:
        # Do NOT crash the pipeline. Caller will handle fallback to cache-only.
        raise RuntimeError(f"BLIP unavailable: {e!s}") from e

def _generate_batch_captions(
    images: List[Image.Image],
    processor: Any,
    model: Any,
    prompt: Optional[str],
) -> List[str]:
    """Generate captions for a batch of images."""
    if not images:
        return []

    # Import torch locally to avoid global import
    import torch

    device = next(model.parameters()).device

    if prompt:
        inputs = processor(
            images=images,
            text=[prompt] * len(images),
            return_tensors="pt",
            padding=True,
        ).to(device)
    else:
        inputs = processor(images=images, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)

    return [processor.decode(ids, skip_special_tokens=True).strip() for ids in out_ids]

def caption_frames(frames: List[Image.Image]) -> List[str]:
    """
    Generate behaviour‑focused captions for a list of frames.
    Uses an on‑disk cache keyed by perceptual hash to avoid recomputation.
    On any BLIP load failure, returns cached captions (if any) and logs a warning.
    """
    if not frames:
        return []

    cache_dir = _vision_cache_dir()
    cached: List[str] = []
    to_process: List[Image.Image] = []
    keys: List[str] = []

    # Cache probe
    for img in frames:
        k = _hash_image(img)
        p = cache_dir / f"{k}.txt"
        if p.exists():
            try:
                cached.append(p.read_text(encoding="utf-8").strip())
                continue
            except Exception:
                pass
        keys.append(k)
        to_process.append(img)

    # If nothing to process or BLIP cannot be loaded, return what we have
    if not to_process:
        return cached

    try:
        processor, model = _load_blip()
    except Exception as e:
        logger.warning("Vision: BLIP unavailable; returning cached captions only. Details: %s", e)
        return cached  # degrade gracefully

    captions: List[str] = cached[:]
    # Generate only for cache misses
    for i in range(0, len(to_process), BATCH_SIZE):
        batch = to_process[i : i + BATCH_SIZE]
        try:
            batch_captions = _generate_batch_captions(batch, processor, model, DEFAULT_PROMPT)
            captions.extend(batch_captions)
            # Persist to cache
            for k, cap in zip(keys[i : i + BATCH_SIZE], batch_captions):
                try:
                    (cache_dir / f"{k}.txt").write_text(cap.strip() + "\n", encoding="utf-8")
                except Exception:
                    pass
        except Exception as e:
            logger.warning("Vision: BLIP captioning failed for a batch; skipping. Details: %s", e)
            continue

    return captions
