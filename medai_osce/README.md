# MedAI‑OSCE — Simplified OSCE Grading Pipeline

This project provides a **minimal, single‑process pipeline** for grading
Objective Structured Clinical Examination (OSCE) sessions.  It distills
the core functionality of the original MedAI prototype down to just the
components you need: media ingestion, speech transcription, optional
vision captioning and rubric‑based scoring.  There are no Docker
containers or micro‑services—everything runs in a single Python
process for ease of development and experimentation.

## 📂 Directory Structure

```
medai_osce/
├── rubrics/                # JSON rubrics defining checklist items
├── prompts/                # Prompt templates used when calling the LLM
├── ingestion.py            # Functions to extract audio and sample frames
├── asr.py                  # Wrapper around an ASR model (e.g. Whisper)
├── vision.py               # Optional VLM helper to caption frames
├── scoring.py              # Build prompts, call LLM, parse responses
├── grade_session.py        # Command‑line tool to process an OSCE session
└── tests/
    └── test_rubrics.py     # Simple rubric schema validation
```

## 🛠️  Installation

1. **Clone this repository** (or copy the `medai_osce` folder into your
   project).
2. Create a Python environment with Python ≥ 3.10.  You can use
   `virtualenv`, `conda` or `pipx`.  Activate the environment and
   install the required packages:

   ```bash
   pip install faster-whisper ffmpeg-python openai pillow numpy
   ```

   * `faster-whisper` provides fast automatic speech recognition (ASR).
   * `ffmpeg-python` is a Python wrapper around `ffmpeg` for audio and
     frame extraction.  You must have the `ffmpeg` binary installed on
     your system.
   * `openai` is used for the scoring step when calling an LLM via the
     OpenAI API.  You can replace it with a different client if you
     prefer another model.

3. Download or copy your OSCE videos into a **session folder**.  A
   session folder may contain one or more camera files (`camA.mp4`,
   `camB.mp4`, …) and optionally a separate room audio recording
   (`room_audio.wav`).  For example:

   ```
   /path/to/session_2025_07_23/
     ├─ camA.mp4
     ├─ camB.mp4
     └─ room_audio.wav  # optional
   ```

4. Choose the rubrics you want to apply.  The `rubrics/` folder
   contains four examples: *Communication & Engagement*,
   *Professionalism & Behaviour*, *Clinical Reasoning & Organisation* and
   *History‑Taking – Acute Chest Pain*.  You may add new rubrics by
   following the same JSON schema (see `tests/test_rubrics.py`).

5. **Run the grading script**:

   ```bash
   python medai_osce/grade_session.py \
       --session /path/to/session_2025_07_23 \
       --rubric communication_engagement_v1 \
       --rubric professionalism_behavior_v1 \
       --out report.json
   ```

   The script will extract audio, run the ASR model, optionally sample
   frames for vision cues and call an LLM to score each rubric.  The
   results are printed to stdout and optionally written to `report.json`.

## 🧠  How it works

1. **Ingestion** – `ingestion.py` provides utilities to extract a
   single mono WAV file from your videos and optionally down‑sample and
   synchronise multiple audio sources.  It also offers a function to
   sample video frames at a fixed rate for vision models.

2. **Transcription** – `asr.py` wraps the `faster-whisper` model.  It
   loads a Whisper checkpoint and transcribes the extracted audio,
   returning plain text.  You can swap in another ASR library if you
   prefer.

3. **Vision (optional)** – `vision.py` contains a placeholder function
   demonstrating how to caption sampled frames using a vision‑language
   model such as BLIP‑2 or Qwen‑VL.  Implementing this requires
   installing the appropriate model; otherwise, you can skip vision
   entirely for audio‑only rubrics.

4. **Scoring** – `scoring.py` reads a rubric and its corresponding
   prompt template, builds a prompt by injecting the transcript and
   optional vision captions, and calls the OpenAI API.  The LLM returns
   a JSON object containing the total score, per‑item scores,
   explanation and evidence trace.  The script validates the response
   and returns a structured result.

5. **Orchestration** – `grade_session.py` glues these pieces together.
   It loops over the selected rubrics, calls the ingestion and
   modelling functions, and prints or saves the final grade report.

## ⚠️  Notes

* Large video files may take several minutes to process.  If memory is
  limited, adjust the chunk size in `ingestion.extract_audio()`.
* The OpenAI API requires an API key.  Set the `OPENAI_API_KEY`
  environment variable before running `grade_session.py` or modify
  `scoring.py` to load the key from a file.
* Vision captioning is optional.  Many rubrics can be scored using
  audio alone.  For rubrics that evaluate non‑verbal behaviour,
  implementing `vision.caption_frames()` is recommended.
