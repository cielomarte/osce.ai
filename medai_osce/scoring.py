"""
scoring.py – load rubrics, build prompts, call the OpenAI API (v1.0+) and parse results.

This module provides synchronous and asynchronous functions to score OSCE rubrics.
It works with openai>=1.0.0 by using the AsyncOpenAI/OpenAI client classes.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

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
    """
    Return the first complete JSON object found in ``text`` by counting braces.
    """
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

    Raises ValueError if no JSON object is found or if parsing fails.
    """
    json_str = _extract_json(text)
    if json_str is None:
        raise ValueError(f"Could not locate JSON object in LLM response: {text}")
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON returned by LLM: {e}\nResponse text: {text}") from e


# --------------------------------------------------------------------------- #
# LLM client setup for openai>=1.0.0
# --------------------------------------------------------------------------- #

# Initialise synchronous and asynchronous OpenAI clients
_openai_api_key = os.getenv("OPENAI_API_KEY")
if not _openai_api_key:
    raise EnvironmentError("OPENAI_API_KEY environment variable is required for scoring.")

_async_client = openai.AsyncOpenAI(api_key=_openai_api_key)
_sync_client = openai.OpenAI(api_key=_openai_api_key)


# --------------------------------------------------------------------------- #
# LLM scoring functions
# --------------------------------------------------------------------------- #

async def _score_prompt_async(prompt: str, model: str) -> dict:
    """
    Asynchronously send a single rubric prompt to the OpenAI API and parse the JSON response.
    """
    logger.debug("Sending prompt to OpenAI model %s", model)
    response = await _async_client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": prompt}],
        max_tokens=512,
        temperature=0.2,
    )
    raw = response.choices[0].message.content
    logger.debug("Received response: %s", raw)
    return parse_llm_response(raw)


async def _score_all_async(prompts: List[str], model: str) -> List[dict]:
    """
    Helper to score multiple prompts concurrently using asyncio.
    """
    tasks = [_score_prompt_async(p, model) for p in prompts]
    return await asyncio.gather(*tasks)


def score_rubric(
    rubric_name: str,
    transcript: str,
    vision: Iterable[str],
    model: str = "gpt-3.5-turbo",
) -> dict:
    """
    Score a single rubric synchronously.

    Useful when only one rubric is being evaluated.  Loads the rubric,
    builds the prompt, sends it to the model, and returns the parsed result.
    """
    rubric = load_rubric(rubric_name)
    prompt = build_prompt(rubric, transcript, vision)
    # Use the synchronous client for a single request
    response = _sync_client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": prompt}],
        max_tokens=512,
        temperature=0.2,
    )
    raw = response.choices[0].message.content
    return parse_llm_response(raw)


def score_rubrics_concurrently(
    rubric_names: Iterable[str],
    transcript: str,
    vision: Iterable[str],
    model: str = "gpt-3.5-turbo",
) -> List[dict]:
    """
    Score multiple rubrics concurrently using asyncio.

    Each rubric is loaded and its prompt constructed before sending.  The
    results are returned in the same order as ``rubric_names``.
    """
    rubrics = [load_rubric(name) for name in rubric_names]
    prompts = [build_prompt(r, transcript, vision) for r in rubrics]
    return asyncio.run(_score_all_async(prompts, model))
