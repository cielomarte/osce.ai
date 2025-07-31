"""Vision model helper for the simplified OSCE pipeline.

This module contains a placeholder function for captioning a list of
PIL Images using a vision‑language model (VLM).  Rubrics such as
*Professionalism & Behaviour* evaluate non‑verbal cues like hand
hygiene and respectful demeanour.  To score such rubrics you should
provide short textual descriptions of the observed behaviour.  A VLM
can generate these descriptions from sampled video frames.

The implementation below demonstrates how you might integrate a
Hugging Face model such as BLIP‑2 or Qwen‑VL.  It returns a list of
strings (one caption per frame) which are later concatenated and
included in the prompt sent to the LLM.  Feel free to replace this
with your own vision model or skip the vision step entirely.

Note: To run the example, you need to install the relevant libraries
such as `transformers` and `torch`, and download the model weights.
You can comment out the body of ``caption_frames`` to bypass vision
cues altogether.
"""

# from __future__ import annotations
#
# from typing import List
#
# from PIL import Image
#
#
# def caption_frames(frames: List[Image.Image]) -> List[str]:
#     """Generate captions for a list of PIL Images.
#
#     Parameters
#     ----------
#     frames:
#         A list of PIL Images sampled from the session video.  See
#         ``ingestion.sample_frames`` for obtaining these frames.
#
#     Returns
#     -------
#     List[str]
#         A list of strings.  Each string should describe the content of
#         the corresponding image (e.g. "doctor washes hands" or
#         "student maintains eye contact").
#
#     Raises
#     ------
#     NotImplementedError
#         If the function has not been implemented.  Replace the body
#         of this function with your own captioning model.
#     """
#     # TODO: implement vision captioning.  Below is a stub using a
#     # hypothetical Hugging Face model.  Uncomment and adjust as needed.
#     #
#     # from transformers import AutoProcessor, AutoModelForVision2Seq
#     # import torch
#     # processor = AutoProcessor.from_pretrained("model/name")
#     # model = AutoModelForVision2Seq.from_pretrained("model/name")
#     # captions = []
#     # for img in frames:
#     #     inputs = processor(images=img, return_tensors="pt").to(model.device)
#     #     outputs = model.generate(**inputs, max_new_tokens=20)
#     #     caption = processor.decode(outputs[0], skip_special_tokens=True)
#     #     captions.append(caption.strip())
#     # return captions
#     raise NotImplementedError(
#         "caption_frames is not implemented.  Provide your own VLM or return an empty list."
#     )

#BLIP IMPLEMENTATION

from __future__ import annotations
import torch
from typing import List
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# Global variables to avoid reloading the model on every call
_processor = None
_model = None

def _load_blip():
    global _processor, _model
    if _processor is None or _model is None:
        _processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        _model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _model.to(device)
    return _processor, _model

def caption_frames(frames: List[Image.Image]) -> List[str]:
    """
    Generate behaviour-focused captions for a list of PIL images using BLIP.
    Each caption will describe what the people are doing or how they are interacting.
    """
    processor, model = _load_blip()
    device = next(model.parameters()).device
    captions: List[str] = []

    # Use a behaviour‑oriented prompt to encourage BLIP to describe non-verbal cues.
    for img in frames:
        # Unconditional captioning; we can also add a text prompt like "people interacting" if needed
        inputs = processor(images=img, return_tensors="pt").to(device)
        # Generate a caption (max_new_tokens controls length)
        out_ids = model.generate(**inputs, max_new_tokens=30)
        caption = processor.decode(out_ids[0], skip_special_tokens=True)
        captions.append(caption.strip())

    return captions
