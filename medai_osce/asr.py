"""ASR wrapper for the simplified OSCE pipeline.

This module wraps a pre‑trained Whisper model using the
`faster-whisper` library.  It exposes a single function
`transcribe_audio` which accepts a WAV file path and returns a text
transcription.  You can replace the underlying implementation with any
other automatic speech recognition model by editing this file.

To use the Whisper model you must install `faster-whisper` and have
FFmpeg installed on your system.

```bash
pip install faster-whisper
```

Example usage:

```python
from medai_osce.asr import transcribe_audio

text = transcribe_audio("/tmp/session.wav", model_size="medium")
print(text)
```
"""

from __future__ import annotations

from typing import Optional

from faster_whisper import WhisperModel  # type: ignore


def transcribe_audio(
    audio_path: str,
    model_size: str = "base",
    device: str | None = None,
    language: Optional[str] = None,
) -> str:
    """Transcribe a WAV file to text using Whisper.

    Parameters
    ----------
    audio_path:
        Path to a WAV file.  The file should be mono and sampled at
        16 kHz; see ``ingestion.extract_audio`` for producing such a
        file from an MP4.
    model_size:
        Size of the Whisper checkpoint to load.  Available sizes
        include ``tiny``, ``base``, ``small``, ``medium`` and ``large``.
        Larger models are more accurate but consume more memory.
    device:
        Optional device specifier (e.g. ``"cuda"`` or ``"cpu"``).
        If ``None``, the model will automatically choose GPU if
        available, otherwise CPU.
    language:
        Optional language code (e.g. ``"en"``).  When ``None``, the
        model will detect the language automatically.  Providing the
        language can improve accuracy and speed.

    Returns
    -------
    str
        The concatenated transcript of the audio file.
    """
    # Instantiate the model.  It is recommended to cache the model
    # outside of this function in production.  For simplicity we load
    # it here; the overhead is a few seconds on first call.
    model = WhisperModel(model_size, device=device or "auto")
    segments, _info = model.transcribe(audio_path, beam_size=5, language=language)
    transcript_parts = [seg.text.strip() for seg in segments if seg.text]
    return " ".join(transcript_parts)
