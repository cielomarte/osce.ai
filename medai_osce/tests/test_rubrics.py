"""Simple tests to validate rubric JSON files.

Run these tests with pytest to ensure that new rubrics conform to the
expected schema (title: str, checklist: list of strings, no extra
properties).  Keeping this test ensures that rubric files remain
machine‑parsable.
"""

import json
from pathlib import Path

import pytest
import jsonschema

SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "checklist": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
        },
    },
    "required": ["title", "checklist"],
    "additionalProperties": False,
}


def test_rubrics_valid() -> None:
    rubrics_dir = Path(__file__).resolve().parent.parent / "rubrics"
    for path in rubrics_dir.glob("*.json"):
        data = json.loads(path.read_text())
        try:
            jsonschema.validate(data, SCHEMA)
        except jsonschema.ValidationError as exc:
            pytest.fail(f"Rubric {path} is invalid: {exc.message}")
