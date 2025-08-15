"""Simplified OSCE grading package.

Importing this package makes the submodules ``ingestion``, ``asr``,
``vision`` and ``scoring`` available.  You can also run
``medai_osce/grade_session.py`` as a CLI entry point to process
sessions.
"""
# medai_osce/__init__.py
__all__ = ["asr", "ingestion", "scoring", "vision"]
__version__ = "0.3.0-offline"
