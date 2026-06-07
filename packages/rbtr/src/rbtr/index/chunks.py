"""Chunking helpers and fallback for unsupported content.

Provides `make_chunk_id` (shared by all chunkers), the
prose-format detection heuristic, and the line-based fallback
chunker used when no language plugin matches.

Actual chunker implementations live in their language plugin
files under `rbtr/languages/`.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator

from rbtr.config import config
from rbtr.index.models import Chunk, ChunkKind

# ── Prose format detection ─────────────────────────────────────────

_RST_UNDERLINE = re.compile(r"^[=\-~^\"'+#*]{3,}$", re.MULTILINE)
_RST_DIRECTIVE = re.compile(r"^\.\.\s+\w+::", re.MULTILINE)
_RST_ROLE = re.compile(r":\w+:`[^`]+`")
_MD_ATX_HEADING = re.compile(r"^#{1,6}\s+", re.MULTILINE)
_MD_FENCED = re.compile(r"^```", re.MULTILINE)


def detect_prose_format(content: str) -> str | None:
    """Detect RST or Markdown from content heuristics.

    Checks the first 2KB for distinctive signals. Returns
    `"rst"`, `"markdown"`, or `None` (neither detected).
    """
    sample = content[:2048]
    rst_signals = (
        len(_RST_UNDERLINE.findall(sample))
        + len(_RST_DIRECTIVE.findall(sample))
        + len(_RST_ROLE.findall(sample))
    )
    md_signals = len(_MD_ATX_HEADING.findall(sample)) + len(_MD_FENCED.findall(sample))
    if rst_signals == 0 and md_signals == 0:
        return None
    if rst_signals > md_signals:
        return "rst"
    return "markdown"


# ── Fallback: raw line-based chunks ──────────────────────────────────


def _raw_chunks(file_path: str, blob_sha: str, content: str) -> Iterator[Chunk]:
    """Split into fixed-size line-based chunks with overlap.

    Chunk size and overlap come from `config.index`.
    """
    chunk_lines = config.chunk_lines
    overlap = config.chunk_overlap
    lines = content.split("\n")
    start = 0

    while start < len(lines):
        end = min(start + chunk_lines, len(lines))
        text = "\n".join(lines[start:end]).strip()
        if text:
            yield Chunk(
                id=make_chunk_id(file_path, blob_sha, f"chunk:{start}", start),
                blob_sha=blob_sha,
                file_path=file_path,
                kind=ChunkKind.RAW_CHUNK,
                name=f"{file_path}:{start + 1}-{end}",
                scope="",
                content=text,
                line_start=start + 1,
                line_end=end,
            )
        start += chunk_lines - overlap


def chunk_plaintext(file_path: str, blob_sha: str, content: str) -> Iterator[Chunk]:
    """Chunk plain text or unsupported file types."""
    yield from _raw_chunks(file_path, blob_sha, content)


# ── Helpers ──────────────────────────────────────────────────────────


def make_chunk_id(file_path: str, blob_sha: str, name: str, line_start: int) -> str:
    """Deterministic chunk ID from file path, blob SHA, symbol name, and line."""
    raw = f"{file_path}:{blob_sha}:{name}:{line_start}"
    return hashlib.blake2b(raw.encode(), digest_size=8).hexdigest()
