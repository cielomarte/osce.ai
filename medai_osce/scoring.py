# medai_osce/scoring.py
"""
scoring.py – load rubrics, build prompts, call the OpenAI API (v1.0+) and parse results.

Key capabilities:
- Lazy OpenAI client creation; respects OPENAI_BASE_URL (e.g., vLLM).
- Structured outputs via response_format JSON schema (opt-in, with fallback).
- Guided JSON fallback and final compact-hint fallback.
- File-based response caching (disable via LLM_CACHE_DISABLE=1).
- Concurrency limiting with asyncio.Semaphore.
- Local schema validation/fixing and score recomputation for consistency.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import openai

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Data structures and rubric loading
# --------------------------------------------------------------------------- #

@dataclass
class Rubric:
    """Data structure to hold a rubric and its associated prompt template."""
    name: str
    title: str
    checklist: List[str]
    prompt_template: str


_PACKAGE_DIR = Path(__file__).resolve().parent


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_rubric(rubric_name: str) -> Rubric:
    """
    Load a rubric definition and its prompt template from disk.

    Requires a JSON file in 'rubrics/' and a prompt file named
    '<rubric_name>_prompt.txt' in 'prompts/'.
    """
    rubric_path = _PACKAGE_DIR / "rubrics" / f"{rubric_name}.json"
    prompt_path = _PACKAGE_DIR / "prompts" / f"{rubric_name}_prompt.txt"

    rubric_data = _load_json(rubric_path)
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt_template = f.read()

    return Rubric(
        name=rubric_name,
        title=rubric_data.get("title", rubric_name),
        checklist=rubric_data.get("checklist", []),
        prompt_template=prompt_template,
    )


# --------------------------------------------------------------------------- #
# Prompt construction
# --------------------------------------------------------------------------- #

def build_prompt(rubric: Rubric, transcript: str, vision_captions: Iterable[str]) -> str:
    """
    Fill the rubric's prompt template with the actual rubric JSON, transcript,
    and vision cues.  The template may contain placeholders {{rubric}},
    {{transcript}} and {{vision}}.
    """
    rubric_json = {
        "title": rubric.title,
        "checklist": rubric.checklist,
    }
    vision_text = "\n".join(vision_captions)
    prompt = rubric.prompt_template.replace("{{rubric}}", json.dumps(rubric_json, ensure_ascii=False))
    prompt = prompt.replace("{{transcript}}", transcript)
    prompt = prompt.replace("{{vision}}", vision_text)
    return prompt


# --------------------------------------------------------------------------- #
# Parsing LLM responses
# --------------------------------------------------------------------------- #

def _extract_json(text: str) -> Optional[str]:
    """Return the first complete JSON object found in ``text`` by counting braces."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i, ch in enumerate(text[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def parse_llm_response(text: str) -> dict:
    """
    Extract and parse the first JSON object from an LLM response.

    Raises ValueError if no complete JSON object is found or parsing fails.
    """
    if not text:
        raise ValueError("Empty response from LLM.")
    if text.lstrip().startswith("{") and text.rstrip().endswith("}"):
        try:
            return json.loads(text)
        except Exception:
            pass
    json_str = _extract_json(text)
    if json_str is None:
        raise ValueError(f"Could not locate JSON object in LLM response: {text}")
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON returned by LLM: {e}\nResponse text: {text}") from e


# --------------------------------------------------------------------------- #
# LLM client, caching, retries
# --------------------------------------------------------------------------- #

def _make_clients() -> tuple[openai.OpenAI, openai.AsyncOpenAI]:
    """
    Construct OpenAI and AsyncOpenAI clients on demand. Honors OPENAI_BASE_URL.
    """
    api_key = os.getenv("OPENAI_API_KEY", "EMPTY")
    base_url = os.getenv("OPENAI_BASE_URL")
    kwargs: Dict[str, Any] = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return openai.OpenAI(**kwargs), openai.AsyncOpenAI(**kwargs)


def _llm_cache_dir() -> Path:
    d = Path(os.getenv("LLM_CACHE_DIR", "~/.medai_osce_cache/llm")).expanduser()
    d.mkdir(parents=True, exist_ok=True)
    return d


def _llm_cache_get(key: str) -> Optional[str]:
    if os.getenv("LLM_CACHE_DISABLE") == "1":
        return None
    path = _llm_cache_dir() / f"{key}.json"
    if not path.exists():
        return None
    try:
        ttl = int(os.getenv("LLM_CACHE_TTL_SECONDS", str(30 * 24 * 3600)))
        if ttl > 0:
            if time.time() - path.stat().st_mtime > ttl:
                return None
        return path.read_text(encoding="utf-8")
    except Exception:
        return None


def _llm_cache_put(key: str, value: str) -> None:
    if os.getenv("LLM_CACHE_DISABLE") == "1":
        return
    try:
        (_llm_cache_dir() / f"{key}.json").write_text(value, encoding="utf-8")
    except Exception:
        pass


def _hash_prompt(model: str, base_url: Optional[str], prompt: str) -> str:
    import hashlib
    h = hashlib.blake2s()
    h.update((base_url or "").encode("utf-8") + b"\n")
    h.update(model.encode("utf-8") + b"\n")
    h.update(prompt.encode("utf-8"))
    return h.hexdigest()


def _guided_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "score": {"type": "integer"},
            "scores": {"type": "object"},
            "explanation": {"type": "string"},
            "trace": {"type": "string"}
        },
        "required": ["title", "score", "scores", "explanation", "trace"]
    }


def _fallback_compact_hint() -> str:
    return (
        "Respond AGAIN with a SINGLE minified JSON object only. "
        "Keys: title, score, scores, explanation, trace. "
        "Do not include markdown or any extra text. "
        "Keep 'trace' ≤ 180 characters total and 'explanation' ≤ 60 words."
    )


def _recompute_score(scores_obj: Dict[str, int]) -> int:
    items = list(scores_obj.values())
    if not items:
        return 0
    total = sum(1 if int(v) else 0 for v in items)
    return round(100 * total / len(items))


def _validate_and_fix(result: dict, rubric: Rubric) -> dict:
    """Ensure the object matches our expected structure and recompute score locally."""
    if "scores" not in result or not isinstance(result["scores"], dict):
        result["scores"] = {}
    fixed_scores: Dict[str, int] = {}
    for item in rubric.checklist:
        v = result["scores"].get(item, 0)
        fixed_scores[item] = 1 if int(v) == 1 else 0
    result["scores"] = fixed_scores

    result["score"] = _recompute_score(fixed_scores)

    if "title" not in result or not isinstance(result["title"], str):
        result["title"] = rubric.title
    for k in ("explanation", "trace"):
        if k not in result or not isinstance(result[k], str):
            result[k] = ""
    return result


# Tunables
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1024"))
DEFAULT_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "4"))
DEFAULT_CONCURRENCY = int(os.getenv("LLM_MAX_CONCURRENCY", "4"))
USE_JSON_SCHEMA = os.getenv("LLM_USE_JSON_SCHEMA", "1") == "1"
LLM_SEED = os.getenv("LLM_SEED")  # optional; e.g., "42"


# --------------------------------------------------------------------------- #
# LLM scoring functions
# --------------------------------------------------------------------------- #

async def _score_prompt_async(prompt: str, model: str, rubric: Optional[Rubric] = None) -> dict:
    """
    Asynchronously send a single rubric prompt to the OpenAI API and parse the JSON response.
    Includes: caching, structured JSON (opt-in), guided JSON fallback, and retry/compact fallback.
    """
    _sync, _async = _make_clients()

    cache_key = _hash_prompt(model, os.getenv("OPENAI_BASE_URL"), prompt)
    cached = _llm_cache_get(cache_key)
    if cached:
        try:
            obj = json.loads(cached)
            if rubric is not None:
                obj = _validate_and_fix(obj, rubric)
            return obj
        except Exception:
            pass

    async def _call(use_schema: bool, use_guided: bool, max_tokens: int, add_hint: bool = False) -> dict:
        messages = [{"role": "system", "content": prompt}]
        if add_hint:
            messages.append({"role": "user", "content": _fallback_compact_hint()})

        kwargs: Dict[str, Any] = dict(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=DEFAULT_TEMPERATURE,
        )

        # seed (if backend supports it)
        extra_body: Dict[str, Any] = {}
        if LLM_SEED is not None:
            try:
                extra_body["seed"] = int(LLM_SEED)
            except Exception:
                pass
        if extra_body:
            kwargs["extra_body"] = extra_body

        # Try response_format JSON Schema first (if enabled)
        if use_schema and USE_JSON_SCHEMA:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "rubric_result", "schema": _guided_schema()}
            }
        elif use_guided:
            kwargs["extra_body"] = {**kwargs.get("extra_body", {}), "guided_json": _guided_schema()}

        try:
            response = await _async.chat.completions.create(**kwargs)
        except Exception:
            # Retry once without schema/guided if server rejects
            kwargs.pop("response_format", None)
            if "extra_body" in kwargs and "guided_json" in kwargs["extra_body"]:
                eb = dict(kwargs["extra_body"])
                eb.pop("guided_json", None)
                if eb:
                    kwargs["extra_body"] = eb
                else:
                    kwargs.pop("extra_body", None)
            response = await _async.chat.completions.create(**kwargs)

        raw = response.choices[0].message.content or ""
        return parse_llm_response(raw)

    delay = 0.5
    err: Optional[Exception] = None
    for attempt in range(MAX_RETRIES):
        try:
            if attempt == 0:
                obj = await _call(use_schema=True, use_guided=False, max_tokens=LLM_MAX_TOKENS)
            elif attempt == 1:
                obj = await _call(use_schema=False, use_guided=True, max_tokens=LLM_MAX_TOKENS)
            else:
                bigger = min(2048, LLM_MAX_TOKENS * 2)
                obj = await _call(use_schema=False, use_guided=False, max_tokens=bigger, add_hint=True)
            if rubric is not None:
                obj = _validate_and_fix(obj, rubric)
            _llm_cache_put(cache_key, json.dumps(obj, ensure_ascii=False))
            return obj
        except Exception as e:
            err = e
            await asyncio.sleep(delay)
            delay = min(2.0, delay * 2)
            continue

    assert err is not None
    raise err


async def _score_all_async(prompts: List[str], model: str, rubrics: Optional[List[Rubric]] = None) -> List[dict]:
    """Score multiple prompts concurrently using a semaphore to cap concurrency."""
    max_conc = int(os.getenv("LLM_MAX_CONCURRENCY", str(DEFAULT_CONCURRENCY)))
    sem = asyncio.Semaphore(max_conc)

    async def _run(i: int, p: str) -> dict:
        async with sem:
            r = rubrics[i] if rubrics and i < len(rubrics) else None
            return await _score_prompt_async(p, model, rubric=r)

    tasks = [_run(i, p) for i, p in enumerate(prompts)]
    return await asyncio.gather(*tasks)


def score_rubric(
    rubric_name: str,
    transcript: str,
    vision: Iterable[str],
    model: str = "gpt-3.5-turbo",
) -> dict:
    """Score a single rubric synchronously."""
    rubric = load_rubric(rubric_name)
    prompt = build_prompt(rubric, transcript, vision)

    async def _one() -> dict:
        return await _score_prompt_async(prompt, model, rubric=rubric)

    return asyncio.run(_one())


def score_rubrics_concurrently(
    rubric_names: Iterable[str],
    transcript: str,
    vision: Iterable[str],
    model: str = "gpt-3.5-turbo",
    max_concurrency: Optional[int] = None,
) -> List[dict]:
    """
    Score multiple rubrics concurrently. Results are in the same order as rubric_names.
    """
    rubrics = [load_rubric(name) for name in rubric_names]
    prompts = [build_prompt(r, transcript, vision) for r in rubrics]
    if max_concurrency is not None:
        os.environ["LLM_MAX_CONCURRENCY"] = str(max_concurrency)
    return asyncio.run(_score_all_async(prompts, model, rubrics=rubrics))
