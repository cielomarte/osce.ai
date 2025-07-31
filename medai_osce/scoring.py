"""Scoring utilities for the simplified OSCE pipeline.

This module reads rubric JSON and prompt templates, builds a prompt by
injecting the transcript and optional vision captions, and calls an
LLM (via the OpenAI API) to obtain structured scores.  The LLM is
expected to return a single‑line JSON object containing a total
`score`, a per‑item `scores` dictionary, an `explanation` and a
`trace`.  See the prompt templates under ``medai_osce/prompts`` for
examples.

The default implementation uses the OpenAI chat completion API via
the ``openai`` Python package.  To use it you must set the
``OPENAI_API_KEY`` environment variable.  You can substitute any
other LLM by modifying the ``call_llm`` function.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import openai  # type: ignore


@dataclass
class ScoringResult:
    rubric_title: str
    score: float
    scores: Dict[str, int]
    explanation: str
    trace: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.rubric_title,
            "score": self.score,
            "scores": self.scores,
            "explanation": self.explanation,
            "trace": self.trace,
        }


def load_rubric(rubric_name: str) -> Dict[str, Any]:
    """Load a rubric JSON by name.

    Parameters
    ----------
    rubric_name:
        The filename stem of the rubric (e.g. ``"communication_engagement_v1"``).

    Returns
    -------
    dict
        The parsed rubric.  Raises ``FileNotFoundError`` if the
        corresponding file does not exist.
    """
    path = Path(__file__).resolve().parent / "rubrics" / f"{rubric_name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Rubric not found: {path}")
    return json.loads(path.read_text())


def load_prompt_template(rubric_name: str) -> str:
    """Load the prompt template associated with a rubric.

    Prompt files are stored under ``medai_osce/prompts`` with the same
    name as the rubric (e.g. ``communication_engagement_v1_prompt.txt``).
    """
    path = Path(__file__).resolve().parent / "prompts" / f"{rubric_name}_prompt.txt"
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    return path.read_text()


def build_prompt(template: str, rubric: Dict[str, Any], transcript: str, vision: Optional[str] = None) -> str:
    """Inject transcript, vision and rubric into the prompt template.

    This function replaces the curly‑brace placeholders ``{{transcript}}``,
    ``{{vision}}`` and ``{{rubric}}`` in the template.  If ``vision`` is
    ``None``, the placeholder is replaced with an empty string.
    """
    return (
        template
        .replace("{{transcript}}", transcript.strip())
        .replace("{{vision}}", (vision or "").strip())
        .replace("{{rubric}}", json.dumps(rubric, ensure_ascii=False))
    )


def call_llm(prompt: str, model: str = "gpt-4o", max_tokens: int = 512) -> str:
    """Call the OpenAI chat completion API with a prompt.

    This function supports openai-python >= 1.0 by using the client‑based
    interface.  It converts the annotated prompt template into the
    ``messages`` format expected by ``chat.completions.create``.

    Parameters
    ----------
    prompt:
        Full prompt text containing `[ROLE: role]` markers.  If no
        markers are present, the entire prompt is sent as a single user
        message.
    model:
        The OpenAI model identifier (e.g. ``gpt-4o`` or
        ``gpt-3.5-turbo``).
    max_tokens:
        Maximum number of tokens to generate in the response.

    Returns
    -------
    str
        The assistant message content returned by the model.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY must be set to call the OpenAI API")

    # Convert the prompt into a list of chat messages.  See the prompt
    # templates for the `[ROLE: system]` annotation syntax.  If the
    # template contains no role markers, everything is treated as a
    # single user message.
    messages: List[Dict[str, str]] = []
    current_role: Optional[str] = None
    current_content: List[str] = []
    for line in prompt.splitlines():
        if line.startswith("[ROLE: ") and line.endswith("]"):
            # Flush the accumulated content under the previous role
            if current_role is not None:
                messages.append({
                    "role": current_role.lower(),
                    "content": "\n".join(current_content).strip(),
                })
                current_content = []
            # Extract the role name (e.g. "system" or "user")
            current_role = line[len("[ROLE: ") : -1]
        else:
            current_content.append(line)
    # Append the final segment
    if current_role is not None:
        messages.append({
            "role": current_role.lower(),
            "content": "\n".join(current_content).strip(),
        })
    else:
        # Fall back to a single user message if no role markers were found
        messages.append({"role": "user", "content": prompt})

    # Use the new client-based API introduced in openai-python >= 1.0
    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.0,
    )
    # Extract and return the assistant response content
    return response.choices[0].message.content


def parse_llm_response(text: str) -> ScoringResult:
    """Parse the JSON returned by the LLM into a ScoringResult.

    This function strips any leading or trailing code fences or extra
    characters before attempting to decode JSON.  It validates that
    required fields are present and types are as expected.
    """
    cleaned = text.strip()
    # Remove common markdown fences
    if cleaned.startswith("```json"):
        cleaned = cleaned[len("```json"):]
    if cleaned.startswith("```"):
        cleaned = cleaned[len("```"):]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    # Find the first and last curly braces
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or start >= end:
        raise ValueError(f"Could not locate JSON object in LLM response: {text}")
    try:
        payload = json.loads(cleaned[start:end+1])
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse JSON from LLM response: {cleaned}") from exc
    # Validate payload
    title = payload.get("title")
    score = payload.get("score")
    scores = payload.get("scores")
    explanation = payload.get("explanation")
    trace = payload.get("trace")
    if not isinstance(title, str) or not isinstance(explanation, str) or not isinstance(trace, str):
        raise ValueError(f"Invalid response fields: {payload}")
    if not isinstance(score, (int, float)):
        raise ValueError(f"Score must be numeric: {payload}")
    if not isinstance(scores, dict):
        raise ValueError(f"scores must be a dict: {payload}")
    # Cast per‑item scores to ints
    norm_scores: Dict[str, int] = {}
    for k, v in scores.items():
        norm_scores[str(k)] = int(v)
    return ScoringResult(
        rubric_title=title,
        score=float(score),
        scores=norm_scores,
        explanation=explanation,
        trace=trace,
    )


def score_rubric(
    rubric_name: str,
    transcript: str,
    vision_captions: Optional[List[str]] = None,
    model: str = "gpt-4o",
) -> ScoringResult:
    """Score a transcript against a rubric.

    Parameters
    ----------
    rubric_name:
        Name of the rubric (file stem without extension).
    transcript:
        Full transcript of the encounter.
    vision_captions:
        Optional list of vision captions.  When provided, the list will
        be concatenated with semicolons and inserted into the prompt.
    model:
        Which OpenAI model to use.  Defaults to ``gpt-4o``.

    Returns
    -------
    ScoringResult
        The parsed result containing total and per‑item scores.
    """
    rubric = load_rubric(rubric_name)
    template = load_prompt_template(rubric_name)
    vision_text = "; ".join(vision_captions or [])
    prompt = build_prompt(template, rubric, transcript, vision_text)
    raw = call_llm(prompt, model=model)
    return parse_llm_response(raw)
