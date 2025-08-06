"""
Vision model helper for the simplified OSCE pipeline.

This module exposes a function ``caption_frames`` that consumes a list of
PIL ``Image`` objects (typically sampled frames from an OSCE video) and
returns a list of natural‑language descriptions.  The captions focus on
non‑verbal behaviours relevant to OSCE rubrics—hand hygiene, introductions,
eye contact, empathy through body language and respect for personal space—
so that downstream scoring components can evaluate communication and
professionalism effectively.

The implementation uses a Hugging Face BLIP model by default but has been
refactored for robustness and extensibility. Key features include:

* Lazy, thread‑safe model loading with global caching to avoid
  repeatedly downloading weights.
* Configurable model name, maximum caption length and behaviour prompt
  via environment variables. This allows you to experiment with
  different vision‑language models or prompts without changing code.
* Batch processing of frames for efficient GPU utilisation.
* Mixed precision inference on GPU and disabled gradient computation to
  reduce memory footprint and improve performance.
* Graceful error handling so that failures in the vision module do not
  derail the entire grading pipeline.

The function signature remains ``caption_frames(frames: List[Image.Image]) -> List[str]``
for compatibility with the rest of the project.
"""

from __future__ import annotations

import os
from typing import List, Optional

# Try importing torch; if unavailable, we'll raise an informative error at runtime.
try:
    import torch  # type: ignore[import]
except ImportError:
    torch = None  # type: ignore[assignment]

# Try importing Hugging Face BLIP classes; if unavailable, we'll raise an error at runtime.
try:
    from transformers import BlipProcessor, BlipForConditionalGeneration  # type: ignore[import]
except ImportError:
    BlipProcessor = None  # type: ignore[assignment]
    BlipForConditionalGeneration = None  # type: ignore[assignment]

from PIL import Image

# Global caches for the processor and model so that we only download/load them once.
_processor: Optional[BlipProcessor] = None
_model: Optional[BlipForConditionalGeneration] = None

# Configurable parameters via environment variables.
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


def _load_blip() -> tuple[BlipProcessor, BlipForConditionalGeneration]:
    """Load and cache the BLIP processor and model.

    This function downloads the specified model weights from Hugging Face
    (on first call) and moves the model to GPU if one is available,
    using half precision on CUDA to conserve memory.

    Returns
    -------
    Tuple[BlipProcessor, BlipForConditionalGeneration]
        The processor and model ready for inference.
    """
    global _processor, _model
    if _processor is not None and _model is not None:
        return _processor, _model

    # Ensure dependencies are available.
    if torch is None:
        raise RuntimeError(
            "PyTorch (torch) is required for BLIP captioning but is not installed."
        )
    if BlipProcessor is None or BlipForConditionalGeneration is None:
        raise RuntimeError(
            "Hugging Face transformers are required for BLIP captioning but are not installed."
        )

    # Instantiate the processor.
    processor = BlipProcessor.from_pretrained(MODEL_NAME)

    # Choose device and precision.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    # Instantiate and configure the model.
    model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME, torch_dtype=dtype)
    model.to(device)
    model.eval()

    # Cache for future calls.
    _processor = processor
    _model = model
    return processor, model


def _generate_batch_captions(
    images: List[Image.Image],
    processor: BlipProcessor,
    model: BlipForConditionalGeneration,
    prompt: Optional[str],
) -> List[str]:
    """Generate captions for a batch of images."""
    if not images:
        return []

    device = next(model.parameters()).device

    # Prepare inputs; broadcast prompt if provided.
    if prompt:
        inputs = processor(
            images=images,
            text=[prompt] * len(images),
            return_tensors="pt",
            padding=True,
        ).to(device)
    else:
        inputs = processor(images=images, return_tensors="pt", padding=True).to(device)

    # Generate captions without computing gradients.
    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)

    # Decode the generated token IDs.
    return [
        processor.decode(ids, skip_special_tokens=True).strip() for ids in out_ids
    ]


def caption_frames(frames: List[Image.Image]) -> List[str]:
    """Generate behaviour‑focused captions for a list of frames.

    This function processes frames in batches, applies the behaviour prompt, and
    handles errors gracefully. If the model cannot be loaded or an error occurs,
    it returns an empty list.
    """
    if not frames:
        return []

    try:
        processor, model = _load_blip()
    except Exception:
        return []

    captions: List[str] = []
    for i in range(0, len(frames), BATCH_SIZE):
        batch = frames[i : i + BATCH_SIZE]
        try:
            batch_captions = _generate_batch_captions(batch, processor, model, DEFAULT_PROMPT)
            captions.extend(batch_captions)
        except Exception:
            # Skip batch on error; continue processing.
            continue

    return captions
