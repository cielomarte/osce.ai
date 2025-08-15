OSCE.ai — Streaming, rubric‑based grading for OSCE videos

OSCE.ai grades Objective Structured Clinical Examination (OSCE) sessions from video by combining:

Ingestion: video discovery, audio streaming (FFmpeg), frame sampling (OpenCV or FFmpeg fallback).

ASR: low‑latency transcription with faster-whisper (CTranslate2 backend).

Vision (optional): BLIP captions for non‑verbal cues, with on‑disk caching & de‑dup.

MemoryStore (optional): bounded, rubric‑aware summaries to keep LLM context small.

Scoring: rubric prompts to an OpenAI‑compatible LLM (e.g., vLLM), JSON outputs.

CLI: offline single‑pass or streaming with partial updates.

The design maintains accuracy while reducing latency by: (1) reusing prefixes for vLLM prefix caching, (2) bounding context growth with MemoryStore, (3) ASR speedups (quantization, threads, workers), and (4) deduplicating vision work.

Repository layout
medai_osce
├── asr.py                  # streaming/offline ASR with faster-whisper (CTranslate2)
├── grade_session.py        # CLI entrypoint for grading one session (offline/streaming)
├── ingestion.py            # video discovery, frame sampling (OpenCV or FFmpeg), audio PCM streaming
├── scoring.py              # rubric loading, prompt building, LLM calls, caching, structured outputs
├── memory.py               # (optional) bounded MemoryStore for summaries/evidence
├── vision.py               # BLIP captioning + phash-based on-disk cache
├── prompts/                # rubric prompt templates (per rubric)
├── rubrics/                # rubric JSONs (title + checklist)
├── tests/                  # minimal sanity tests
└── README.md               # this file

How it works (end‑to‑end)

Offline (single pass)

ingestion.discover_videos() finds all videos in a session folder.

(Optional) ingestion.sample_frames() samples frames uniformly (OpenCV or FFmpeg fallback).

ingestion.audio_pcm_stream() streams mono PCM via one long‑lived FFmpeg process.

asr.transcribe_pcm_stream() transcribes with faster‑whisper, using overlap + cross‑chunk context.

scoring.score_rubrics_concurrently() loads each rubric, builds prompts, and calls the LLM.

Results are emitted as a JSON list (one object per rubric).

Streaming (partial updates)

ingestion.audio_pcm_stream() yields 10‑s PCM chunks continuously.

asr.AsrStreamer.transcribe_chunk() produces rolling transcript with cross‑chunk textual context.

(Optional) ingestion.sample_frames_window() samples frames for each chunk; vision.caption_frames() captions them (cached).

(Optional) memory.MemoryStore.update_from_chunk() ingests each ASR chunk and keeps:

a recent window (last ~90s) and a bounded long summary (~2k chars),

deduped vision summary,

(optionally) per‑rubric evidence.

Periodically (e.g., every 20 s), the LLM re‑scores:

Legacy path: full partial transcript + all vision captions, or

Memory path: preamble‑first prompts for vLLM prefix caching with only the memory snapshot + recent window (much smaller tokens).

After the stream ends, a final pass is produced and written to JSON.

Requirements

Python: 3.11+

FFmpeg: must be in PATH

macOS (Homebrew): brew install ffmpeg

Ubuntu/Debian: sudo apt-get install ffmpeg

ASR: faster-whisper, soundfile, numpy, ffmpeg-python

LLM client: openai (for OpenAI or OpenAI‑compatible servers like vLLM)

Vision (optional): torch, transformers, Pillow, (optional) imagehash

OpenCV (optional): opencv-python — not required; the system falls back to FFmpeg frame sampling when missing or when USE_FFMPEG_SAMPLING=1.

Installation
python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install faster-whisper soundfile numpy pillow ffmpeg-python openai

# Vision (optional; recommended on GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121  # choose your CUDA
pip install transformers imagehash

# Optional: OpenCV for faster/random access frame sampling
pip install opencv-python


macOS M‑series: you can run on CPU or Metal for torch (install from PyPI). For ASR on CPU, quantization + threads help.

Configuration (Environment Variables)
LLM (client‑side)
Variable	Default	Purpose
OPENAI_API_KEY	"EMPTY"	API key (ignored by some local servers)
OPENAI_BASE_URL	unset	Point to vLLM or any OpenAI‑compatible endpoint, e.g. http://localhost:8000/v1
LLM_MAX_TOKENS	1024	Per‑request max tokens for completion
LLM_TEMPERATURE	0.2	Set 0.0 for deterministic grading
LLM_MAX_RETRIES	4	Attempts with structured/fallbacks
LLM_MAX_CONCURRENCY	4	Concurrent rubric requests
LLM_CACHE_DIR	~/.medai_osce_cache/llm	On‑disk response cache
LLM_CACHE_TTL_SECONDS	2592000	30 days
LLM_CACHE_DISABLE	unset	1 disables disk cache
LLM_USE_JSON_SCHEMA	1	Prefer schema‑validated JSON responses (falls back automatically)
LLM_USE_PREAMBLE	0	Enable preamble‑first prompts (prefix caching)
LLM_SEED	unset	If server supports, sets deterministic seed
ASR
Variable	Default	Purpose
ASR_MODEL_SIZE	small	tiny, base, small, medium, large
ASR_COMPUTE_TYPE	float16 (GPU) / float32 (CPU)	Supports int8_float16 (GPU), int8 (CPU)
ASR_BEAM_SIZE	5	Beam size
ASR_VAD	1	Voice activity detection enabled
ASR_CONDITION_ON_PREV	1	Cross‑chunk textual conditioning
ASR_CPU_THREADS	0	CTranslate2 intra threads (0 lets CT2 decide)
ASR_NUM_WORKERS	1	Decoder workers
ASR_VAD_MIN_SIL_MS	400	VAD min silence (ms)
ASR_VAD_SPEECH_PAD	200	VAD speech pad (ms)
ASR_LANG	unset	Pin language (e.g., en) for speed/stability
Vision
Variable	Default	Purpose
BLIP_MODEL_NAME	Salesforce/blip-image-captioning-base	BLIP model
BLIP_MAX_NEW_TOKENS	30	Caption length
BLIP_BATCH_SIZE	8	Captioning batch size
BLIP_BEHAVIOUR_PROMPT	built‑in	Behavior‑focused prompt
VISION_CACHE_DIR	~/.medai_osce_cache/vision	On‑disk caption cache
SCENE_CHANGE_ENABLE	0	Gate captions by scene change
SCENE_CHANGE_DIFF_THR	6.0	Mean absolute per‑pixel threshold
USE_FFMPEG_SAMPLING	0	Force FFmpeg (no OpenCV) for frame sampling
Memory (optional)
Variable	Default	Purpose
MEM_RECENT_WINDOW_SECONDS	90	Length of recent window
MEM_SUMMARY_MAX_CHARS	2000	Bounded long summary size
MEM_SUMMARIZE_EVERY_CHUNKS	3	Re‑summarize cadence
MEM_USE_LLM_SUMMARY	0	Use LLM to summarize (else truncate)
MEM_LLM_MODEL	gpt-3.5-turbo	Model for memory summarization
MEM_LLM_MAX_TOKENS	256	Tokens for memory summarization
MEM_MIN_CHARS_DELTA	800	Gate for “enough new content”
Running the grader
Offline (single pass)

Simple, accurate baseline. Recommends GPU for speed, but CPU works.

# If using a local vLLM server:
# export OPENAI_BASE_URL=http://localhost:8000/v1
# export OPENAI_API_KEY=EMPTY

# CPU tuning example:
export ASR_CPU_THREADS=4
export ASR_NUM_WORKERS=1

python -m medai_osce.grade_session \
  --session "/path/to/session_dir" \
  --rubric HEENT_cranial_nerves \
  --rubric MSK \
  --rubric abdomen \
  --rubric cardiac_vascular \
  --rubric clinical_reasoning_organization_v1 \
  --rubric communication_engagement_v1 \
  --rubric history_taking_chest_pain_v1 \
  --rubric neuro \
  --rubric professionalism_behavior_v1 \
  --rubric thorax_lungs \
  --use-vision \
  --concurrency 6 \
  --out report_all.json


If you don’t have OpenCV installed, either install it (pip install opencv-python) or set USE_FFMPEG_SAMPLING=1 to force the built‑in FFmpeg fallback for frame sampling.

Streaming (partial updates) with MemoryStore + preamble (faster)

Lower token usage and faster updates by using bounded memory and prefix-cached prompts.

# vLLM (optional)
# export OPENAI_BASE_URL=http://localhost:8000/v1
# export OPENAI_API_KEY=EMPTY

# Enable preamble & scene gating
export LLM_USE_PREAMBLE=1
export SCENE_CHANGE_ENABLE=1
export USE_FFMPEG_SAMPLING=1  # optional, avoids OpenCV dependency

# CPU ASR tuning
export ASR_CPU_THREADS=4
export ASR_NUM_WORKERS=1

python -m medai_osce.grade_session \
  --session "/path/to/session_dir" \
  --rubric HEENT_cranial_nerves \
  --rubric MSK \
  --rubric abdomen \
  --rubric cardiac_vascular \
  --rubric clinical_reasoning_organization_v1 \
  --rubric communication_engagement_v1 \
  --rubric history_taking_chest_pain_v1 \
  --rubric neuro \
  --rubric professionalism_behavior_v1 \
  --rubric thorax_lungs \
  --stream \
  --update-interval 20 \
  --min-chars-delta 1200 \
  --use-memory-store \
  --recent-window-seconds 90 \
  --asr-online-model-size tiny \
  --concurrency 6 \
  --out report_all.json


Notes

The final pass can re‑ASR with a larger model (--asr-model-size small) for accuracy; online updates use --asr-online-model-size tiny for speed.

If LLM_USE_PREAMBLE=1 and --use-memory-store are set, scoring prompts will use the preamble‑first format to maximize vLLM prefix caching gains.

Output format

The CLI prints or writes a JSON array; each element corresponds to a rubric:

[
  {
    "title": "Abdomen Exam",
    "score": 78,
    "scores": {
      "Introduces self and obtains consent": 1,
      "Washes or sanitizes hands": 0,
      "Inspects abdomen": 1,
      "...": 1
    },
    "explanation": "Student completed most inspection and palpation steps but missed hand hygiene.",
    "trace": "e:00:30 introduces self; e:03:45 palpation; miss: hand hygiene"
  },
  { "... next rubric ..." }
]


scores is a map from checklist item → 0/1. score is recomputed locally as % of items hit.

Extending or editing rubrics

Create rubrics/<name>.json:

{
  "title": "Cardiac & Vascular Exam",
  "checklist": [
    "Introduces self and obtains consent",
    "Auscultates at four standard areas",
    "Assesses peripheral pulses",
    "..."
  ]
}


Create prompts/<name>_prompt.txt.

The template may use {{rubric}}, {{transcript}}, {{vision}}.

If LLM_USE_PREAMBLE=1 (recommended with MemoryStore), treat the file as a suffix—the preamble already includes memory and recent transcript.

Run with --rubric <name>.

Performance tuning
ASR

GPU: ASR_COMPUTE_TYPE=int8_float16 often gives a great speed/quality trade‑off.

CPU: ASR_COMPUTE_TYPE=int8, set ASR_CPU_THREADS to physical cores, ASR_NUM_WORKERS=1–2.

Pin language for stability/speed: ASR_LANG=en.

Adjust VAD to skip silence: ASR_VAD_MIN_SIL_MS=400–800.

LLM

Prefer LLM_TEMPERATURE=0.0 for deterministic scoring.

Set LLM_USE_JSON_SCHEMA=1 (default) for structured outputs; auto‑fallback is built‑in.

Keep LLM_MAX_CONCURRENCY=4–8 (tune based on your server throughput).

Use preamble‑first prompts (LLM_USE_PREAMBLE=1) with MemoryStore for lower tokens and better prefix caching.

Vision

Enable scene gating to reduce redundant captioning: SCENE_CHANGE_ENABLE=1.

Tune BLIP_BATCH_SIZE based on GPU memory.

Troubleshooting

OpenCV not installed

Either pip install opencv-python or set USE_FFMPEG_SAMPLING=1 to force FFmpeg sampling (no OpenCV needed).

FFmpeg not found / frame sampling fails

Ensure ffmpeg is installed and in PATH. The app logs a warning and continues without vision if --use-vision fails.

CTranslate2 __init__ incompatible arguments

Ensure ASR_CPU_THREADS is an integer (not None). The provided asr.py guarantees an int; otherwise set export ASR_CPU_THREADS=4.

LLM JSON parse errors

LLM_USE_JSON_SCHEMA=1 (default) yields structured outputs; if your server doesn’t support it, the code auto‑fallbacks to guided/compact JSON.

Slow updates

Increase --update-interval, set --min-chars-delta, enable MemoryStore + preamble to reduce token usage.

GPU OOM (vision)

Lower BLIP_BATCH_SIZE and/or BLIP_MAX_NEW_TOKENS; consider CPU offload (slower).

Testing

Run basic tests:

pytest -q


Suggested additional tests:

JSON parsing round‑trip in scoring.parse_llm_response.

ASR smoke test with a small WAV.

Vision cache hit (same image twice).

MemoryStore summarization (bounded size, recent window).

Security & privacy

Caches are local by default: ~/.medai_osce_cache/llm and ~/.medai_osce_cache/vision.

When using a remote LLM backend, ensure compliance with your data policies.

For de‑identification needs, pre‑process audio/video upstream.

Example: run a quick all‑rubrics test

Offline with vision, FFmpeg sampling fallback:

export USE_FFMPEG_SAMPLING=1
export ASR_CPU_THREADS=4
export ASR_NUM_WORKERS=1

python -m medai_osce.grade_session \
  --session "/path/to/session_dir" \
  --rubric HEENT_cranial_nerves \
  --rubric MSK \
  --rubric abdomen \
  --rubric cardiac_vascular \
  --rubric clinical_reasoning_organization_v1 \
  --rubric communication_engagement_v1 \
  --rubric history_taking_chest_pain_v1 \
  --rubric neuro \
  --rubric professionalism_behavior_v1 \
  --rubric thorax_lungs \
  --use-vision \
  --concurrency 6 \
  --out report_all.json


Streaming with MemoryStore + preamble:

export LLM_USE_PREAMBLE=1
export USE_FFMPEG_SAMPLING=1
export SCENE_CHANGE_ENABLE=1
export ASR_CPU_THREADS=4
export ASR_NUM_WORKERS=1

python -m medai_osce.grade_session \
  --session "/path/to/session_dir" \
  --rubric HEENT_cranial_nerves \
  --rubric MSK \
  --rubric abdomen \
  --rubric cardiac_vascular \
  --rubric clinical_reasoning_organization_v1 \
  --rubric communication_engagement_v1 \
  --rubric history_taking_chest_pain_v1 \
  --rubric neuro \
  --rubric professionalism_behavior_v1 \
  --rubric thorax_lungs \
  --stream \
  --update-interval 20 \
  --min-chars-delta 1200 \
  --use-memory-store \
  --recent-window-seconds 90 \
  --asr-online-model-size tiny \
  --concurrency 6 \
  --out report_all.json

Notes on reproducibility

Set LLM_TEMPERATURE=0.0.

If your backend supports it, set LLM_SEED (e.g., LLM_SEED=42).

Keep model versions fixed (ASR, vision, LLM) for consistent results.

Known limitations

Scoring fidelity depends on rubric clarity; ambiguous items can cause LLM variability if temperature/seed are not fixed.

Vision captions (BLIP) provide coarse cues; they are not action recognition.

Latency is bounded by LLM throughput; MemoryStore + preamble significantly reduces tokens but does not eliminate network/compute time.

Maintainer checklist (quick)

Validate FFmpeg in PATH.

(Optional) Install OpenCV or set USE_FFMPEG_SAMPLING=1.

GPU: set ASR_COMPUTE_TYPE=int8_float16, adjust BLIP_BATCH_SIZE.

vLLM: set OPENAI_BASE_URL, enable server features (prefix caching, JSON schema if supported).

Use LLM_USE_PREAMBLE=1 + --use-memory-store in streaming for best throughput.