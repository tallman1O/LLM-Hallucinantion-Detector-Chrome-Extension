from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path
from typing import Final


def _add_local_venv_site_packages() -> None:
    project_root = Path(__file__).resolve().parent
    candidates = sorted((project_root / "venv" / "lib").glob("python*/site-packages"))

    for candidate in candidates:
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)


def _run_command(args: list[str]) -> None:
    subprocess.check_call(args)


def _ensure_spacy() -> None:
    try:
        importlib.import_module("spacy")
    except ModuleNotFoundError:
        _run_command([sys.executable, "-m", "pip", "install", "spacy"])


def _ensure_model() -> None:
    try:
        importlib.import_module("en_core_web_sm")
    except ModuleNotFoundError:
        _run_command([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])

_add_local_venv_site_packages()
_ensure_spacy()
_ensure_model()

import spacy

MIN_LENGTH: Final[int] = 10
MAX_LENGTH: Final[int] = 300
BLOCKED_FIRST_WORDS: Final[set[str]] = {
    "this",
    "it",
    "they",
    "these",
    "that",
    "however",
    "therefore",
    "thus",
    "also",
    "additionally",
    "furthermore",
    "as",
    "which",
    "who",
}
ALLOWED_ENTITY_LABELS: Final[set[str]] = {
    "PERSON",
    "ORG",
    "GPE",
    "DATE",
    "CARDINAL",
    "PRODUCT",
    "EVENT",
    "LAW",
    "FAC",
    "NORP",
    "WORK_OF_ART",
}
ALLOWED_POS_TAGS: Final[set[str]] = {"PROPN", "NUM"}

_nlp = spacy.load("en_core_web_sm", exclude=["parser", "lemmatizer", "textcat"])
if "sentencizer" not in _nlp.pipe_names:
    _nlp.add_pipe("sentencizer")


def _first_word(sentence: str) -> str:
    return sentence.strip().split()[0].lower() if sentence.strip().split() else ""


def _has_entity_or_allowed_pos(span) -> bool:
    has_entity = any(ent.label_ in ALLOWED_ENTITY_LABELS for ent in span.ents)
    has_allowed_pos = any(token.pos_ in ALLOWED_POS_TAGS for token in span)
    return has_entity or has_allowed_pos


def extract_verifiable_claims_debug(text: str) -> dict:
    doc = _nlp(text)
    passed: list[str] = []
    skipped: list[str] = []

    for sentence in doc.sents:
        content = sentence.text.strip()
        if not content:
            continue

        if len(content) < MIN_LENGTH or len(content) > MAX_LENGTH:
            skipped.append(content)
            continue

        if _first_word(content) in BLOCKED_FIRST_WORDS:
            skipped.append(content)
            continue

        if content.endswith("?"):
            skipped.append(content)
            continue

        if not _has_entity_or_allowed_pos(sentence):
            skipped.append(content)
            continue

        passed.append(content)

    return {"passed": passed, "skipped": skipped}


def extract_verifiable_claims(text: str) -> list[str]:
    return extract_verifiable_claims_debug(text)["passed"]
