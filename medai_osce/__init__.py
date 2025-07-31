"""Simplified OSCE grading package.

Importing this package makes the submodules ``ingestion``, ``asr``,
``vision`` and ``scoring`` available.  You can also run
``medai_osce/grade_session.py`` as a CLI entry point to process
sessions.
"""

__all__ = ["ingestion", "asr", "vision", "scoring"]
