"""`rbtr-eval extract` and `rbtr-eval merge-dataset` subcommands.

`extract` walks one repo, samples docstring-derived queries,
writes a per-repo JSONL.  `merge-dataset` concatenates per-repo
JSONLs into one dataset.

This module is the *only* place rbtr-eval imports from `rbtr` —
it uses `rbtr.index.treesitter.extract_doc_spans` for docstring
identification because reimplementing it would mean rolling our
own per-language comment parsing (the same smell we exorcised
during Phase 5).  Measurement-side tooling reusing the indexer's
own docstring view is the cleanest available answer.

Stubbed for Phase P1 — implementation lands in P2 / P3.
"""

from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel, Field

# ── Query projection ───────────────────────────────────────────────────────────
#
# `first_sentence` turns a raw docstring (comment markers and
# all) into a search-friendly query string.  We deliberately do
# *not* strip comment markers — doing so means re-implementing
# per-language comment syntax, which is the smell rbtr-eval is
# built to avoid.  The indexer already saw `///` / `"""` / `#`
# inside chunk content; the bench treats them the same way.
#
# YAGNI: if marker noise turns out to materially hurt search
# quality asymmetrically (default vs --strip-docstrings), we
# revisit.  Empirical answer beats premature normalisation.

# A sentence ends at `.`, `!`, or `?` followed by whitespace,
# or at a blank line (paragraph break).  Pure English
# punctuation, not language-specific.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n\s*\n")

# Length band for usable queries.  Shorter than this is
# usually a stub comment (`hi`, `wip`); longer than this makes
# a poor query.
_QUERY_MIN_LEN = 15
_QUERY_MAX_LEN = 200

# Leading-character noise the boilerplate check skips before
# comparing against `_REJECT_PREFIXES`.  Used only for the
# rejection decision — not applied to the returned query text.
_REJECT_LOOKAHEAD_RE = re.compile(r"^[\s/#*\-!\"'`]+")

# Boilerplate / scaffolding markers; queries starting with any
# of these (after skipping marker noise) are dropped.
_REJECT_PREFIXES = (
    "todo",
    "fixme",
    "xxx",
    "hack",
    "deprecated",
    "@param",
    "@return",
    "@returns",
    "@throws",
    "@type",
    "@see",
    "@link",
)


def first_sentence(doc_text: str) -> str | None:
    """Return the first sentence of *doc_text*, or None if unusable.

    Splits on a sentence terminator followed by whitespace, or
    on a blank line.  Truncates at `_QUERY_MAX_LEN` if no
    terminator appears in the first paragraph.  Returns `None`
    when the candidate is shorter than `_QUERY_MIN_LEN` or
    starts with a boilerplate prefix (after skipping leading
    comment-marker noise).

    Comment markers (line comments, triple-quotes, block-
    comment delimiters, `*` gutters) are *kept* in the
    returned text - see the module-level note.
    """
    cleaned = doc_text.strip()
    if not cleaned:
        return None
    parts = _SENTENCE_SPLIT_RE.split(cleaned, maxsplit=1)
    candidate = parts[0].strip() if parts else cleaned
    if len(candidate) > _QUERY_MAX_LEN:
        candidate = candidate[:_QUERY_MAX_LEN].rstrip()
    if len(candidate) < _QUERY_MIN_LEN:
        return None
    post_noise = _REJECT_LOOKAHEAD_RE.sub("", candidate).lower()
    if any(post_noise.startswith(p) for p in _REJECT_PREFIXES):
        return None
    return candidate


# ── CLI subcommand stubs (P1) ──────────────────────────────────────────────


class ExtractCmd(BaseModel):
    """Sample docstring-derived queries from one repo.

    Reads a checked-out repo at *repo-path*; writes a JSONL
    artefact with one header line plus one query per line.
    """

    slug: str = Field(description="Repo slug (used for the JSONL header).")
    repo_path: Path = Field(description="Path to the checked-out repository.")
    output: Path = Field(description="Per-repo JSONL output path.")
    seed: int = Field(0, description="Deterministic-sampling RNG seed.")
    sample_cap: int = Field(300, description="Maximum number of queries to keep per repo.")

    def cli_cmd(self) -> None:
        msg = f"rbtr-eval extract: not implemented yet (slug={self.slug})"
        raise SystemExit(msg)


class MergeDatasetCmd(BaseModel):
    """Concatenate per-repo JSONL files into one dataset."""

    inputs: Path = Field(description="Directory holding per-repo JSONL files.")
    output: Path = Field(description="Combined dataset output path.")

    def cli_cmd(self) -> None:
        msg = "rbtr-eval merge-dataset: not implemented yet"
        raise SystemExit(msg)
