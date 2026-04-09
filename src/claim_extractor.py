from __future__ import annotations

import re
from typing import Final

import spacy

MIN_LENGTH: Final[int] = 10
MAX_LENGTH: Final[int] = 300
SKIP_PREFIXES: Final[tuple[str, ...]] = (
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
)
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


def _starts_with_skip_prefix(sentence: str) -> bool:
    lowered = sentence.lstrip().lower()
    return lowered.startswith(SKIP_PREFIXES)


def _has_entity_or_tag(span) -> bool:
    has_entity = any(ent.label_ in ALLOWED_ENTITY_LABELS for ent in span.ents)
    has_pos_tag = any(token.pos_ in ALLOWED_POS_TAGS for token in span)
    return has_entity or has_pos_tag


def _normalize_marker(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def extract_action_phrase(text: str) -> str:
    doc = _nlp(text)
    actions: list[str] = []

    for token in doc:
        if token.pos_ not in {"VERB", "AUX"}:
            continue

        normalized = _normalize_marker(token.text)
        if not normalized:
            continue

        if not actions or actions[-1] != normalized:
            actions.append(normalized)

    return " ".join(actions[:3])


def extract_entity_markers(text: str) -> list[dict]:
    doc = _nlp(text)
    markers: dict[str, dict] = {}
    entity_token_indexes = set()

    for ent in doc.ents:
        if ent.label_ not in ALLOWED_ENTITY_LABELS:
            continue

        normalized = _normalize_marker(ent.text)
        if not normalized:
            continue

        markers[normalized] = {
            "text": ent.text,
            "normalized": normalized,
            "label": ent.label_,
            "penalizable": (
                ent.label_ in {"DATE", "CARDINAL", "PRODUCT", "EVENT", "LAW", "WORK_OF_ART"}
                or any(char.isdigit() for char in ent.text)
                or "-" in ent.text
            ),
        }
        entity_token_indexes.update(range(ent.start, ent.end))

    for token in doc:
        if token.i in entity_token_indexes or token.pos_ not in ALLOWED_POS_TAGS:
            continue

        normalized = _normalize_marker(token.text)
        if not normalized:
            continue

        markers[normalized] = {
            "text": token.text,
            "normalized": normalized,
            "label": token.pos_,
            "penalizable": token.pos_ == "NUM" or any(char.isdigit() for char in token.text) or "-" in token.text,
        }

    return list(markers.values())


def extract_verifiable_claims(text: str) -> tuple[list[str], list[str]]:
    doc = _nlp(text)
    claims: list[str] = []
    skipped: list[str] = []

    for sentence in doc.sents:
        content = sentence.text.strip()

        if not content:
            continue

        if len(content) < MIN_LENGTH or len(content) > MAX_LENGTH:
            skipped.append(content)
            continue

        if content.endswith("?"):
            skipped.append(content)
            continue

        if _starts_with_skip_prefix(content):
            skipped.append(content)
            continue

        if not _has_entity_or_tag(sentence):
            skipped.append(content)
            continue

        claims.append(content)

    return claims, skipped
