# medai_osce/memory.py
"""
Lightweight MemoryStore to bound context growth and enable prefix-cached grading.

Design goals:
- Keep a rolling recent window and a compact long-term summary (bounded chars).
- Optionally summarize via a small LLM (opt-in) or fall back to truncation.
- Track whether meaningful changes occurred to justify re-scoring (versioning).

Environment (optional):
  MEM_RECENT_WINDOW_SECONDS  default '90'
  MEM_SUMMARY_MAX_CHARS      default '2000'
  MEM_SUMMARIZE_EVERY_CHUNKS default '3'
  MEM_USE_LLM_SUMMARY        default '0'
  MEM_LLM_MODEL              default 'gpt-3.5-turbo'
  MEM_LLM_MAX_TOKENS         default '256'
  MEM_MIN_CHARS_DELTA        default '800'
"""

from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Tuple, Optional

try:
    import openai
except Exception:
    openai = None  # optional dependency

MEM_RECENT_WINDOW_SECONDS = float(os.getenv("MEM_RECENT_WINDOW_SECONDS", "90"))
MEM_SUMMARY_MAX_CHARS = int(os.getenv("MEM_SUMMARY_MAX_CHARS", "2000"))
MEM_SUMMARIZE_EVERY_CHUNKS = int(os.getenv("MEM_SUMMARIZE_EVERY_CHUNKS", "3"))
MEM_USE_LLM_SUMMARY = os.getenv("MEM_USE_LLM_SUMMARY", "0") == "1"
MEM_LLM_MODEL = os.getenv("MEM_LLM_MODEL", "gpt-3.5-turbo")
MEM_LLM_MAX_TOKENS = int(os.getenv("MEM_LLM_MAX_TOKENS", "256"))
MEM_MIN_CHARS_DELTA = int(os.getenv("MEM_MIN_CHARS_DELTA", "800"))


@dataclass
class MemoryStore:
    recent_window_seconds: float = MEM_RECENT_WINDOW_SECONDS
    long_summary: str = ""
    vision_summary: str = ""
    recent_window: str = ""
    version: int = 0  # increment when content changes in a way that may affect scoring

    # Internals
    _chunks: Deque[Tuple[str, float, float]] = field(default_factory=deque)  # (text, t0, t1)
    _chunk_count: int = 0
    _last_summary_chunk: int = 0
    _total_chars: int = 0
    _last_total_chars: int = 0
    _seen_vision: set = field(default_factory=set)

    def update_from_chunk(self, text: str, t0: float, t1: float) -> bool:
        """
        Ingest a transcript chunk for [t0, t1). Returns True if we believe
        a re-score might be warranted (version bump or enough new content).
        """
        if text:
            self._chunks.append((text, t0, t1))
            self._chunk_count += 1
            self._total_chars += len(text)
            self.version += 1  # conservative: count any non-empty chunk as a change

        # Trim recent window by time
        while self._chunks and (t1 - self._chunks[0][1]) > self.recent_window_seconds:
            self._chunks.popleft()

        # Rebuild recent window
        self.recent_window = " ".join(c[0] for c in self._chunks)

        # Maintain a bounded long_summary
        if self._chunk_count - self._last_summary_chunk >= MEM_SUMMARIZE_EVERY_CHUNKS:
            self._refresh_summary()
            self._last_summary_chunk = self._chunk_count

        # Decide if enough has changed (chars threshold)
        changed = (self._total_chars - self._last_total_chars) >= MEM_MIN_CHARS_DELTA or bool(text)
        if changed:
            self._last_total_chars = self._total_chars
        return changed

    def update_vision(self, captions: List[str]) -> None:
        if not captions:
            return
        # Deduplicate while preserving order
        new = []
        for c in captions:
            c2 = c.strip()
            if c2 and c2 not in self._seen_vision:
                self._seen_vision.add(c2)
                new.append(c2)
        if new:
            joined = (self.vision_summary + "\n" + "\n".join(new)).strip()
            # keep a bounded number of lines ~80-100 words
            lines = joined.splitlines()
            if len(lines) > 20:
                lines = lines[-20:]
            self.vision_summary = "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _refresh_summary(self) -> None:
        """
        Keep long_summary bounded. Optionally use a small LLM to condense;
        otherwise, fall back to heuristic truncation.
        """
        candidate = (self.long_summary + " " + self.recent_window).strip()
        if len(candidate) <= MEM_SUMMARY_MAX_CHARS:
            self.long_summary = candidate
            return

        if MEM_USE_LLM_SUMMARY and openai is not None:
            try:
                client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY", "EMPTY"),
                                       base_url=os.getenv("OPENAI_BASE_URL") or None)
                prompt = (
                    "Condense the following OSCE transcript into a compact, factual summary "
                    f"≤ {MEM_SUMMARY_MAX_CHARS} characters, preserving key clinical actions and findings.\n"
                    "=== TRANSCRIPT ===\n" + candidate
                )
                resp = client.chat.completions.create(
                    model=MEM_LLM_MODEL,
                    messages=[{"role": "system", "content": prompt}],
                    max_tokens=MEM_LLM_MAX_TOKENS,
                    temperature=0.0,
                )
                text = (resp.choices[0].message.content or "").strip()
                self.long_summary = text[:MEM_SUMMARY_MAX_CHARS]
                return
            except Exception:
                pass

        # Fallback: simple compression – keep head and tail
        head = candidate[: MEM_SUMMARY_MAX_CHARS // 2]
        tail = candidate[- MEM_SUMMARY_MAX_CHARS // 2 :]
        self.long_summary = (head + " ... " + tail)[:MEM_SUMMARY_MAX_CHARS]
