"""Chunking strategies for prose, config, and fallback content.

Markdown/RST are split by heading hierarchy.  Config files (TOML,
YAML, JSON) are split by top-level keys when possible.  Unknown or
unsupported files get fixed-size line-based chunks.
"""

from __future__ import annotations

import hashlib
import re

from rbtr_legacy.config import config
from rbtr_legacy.index.models import Chunk, ChunkKind

# ── Markdown chunking ────────────────────────────────────────────────

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)


def chunk_markdown(file_path: str, blob_sha: str, content: str) -> list[Chunk]:
    """Split Markdown by heading hierarchy, preserving scope chain."""
    chunks: list[Chunk] = []
    lines = content.split("\n")

    # Find all heading positions.
    headings: list[tuple[int, int, str]] = []  # (line_idx, level, title)
    for i, line in enumerate(lines):
        m = _HEADING_RE.match(line)
        if m:
            headings.append((i, len(m.group(1)), m.group(2).strip()))

    if not headings:
        # No headings — treat entire file as one chunk.
        return _raw_chunks(file_path, blob_sha, content)

    # Build sections from headings.
    scope_stack: list[str] = []
    for idx, (line_idx, level, title) in enumerate(headings):
        # Determine end of this section.
        end_idx = headings[idx + 1][0] if idx + 1 < len(headings) else len(lines)
        section_text = "\n".join(lines[line_idx:end_idx]).strip()

        if not section_text:
            continue

        # Maintain scope stack based on heading level.
        while len(scope_stack) >= level:
            scope_stack.pop()
        scope = " > ".join(scope_stack)
        scope_stack.append(title)

        chunks.append(
            Chunk(
                id=_chunk_id(file_path, title, line_idx),
                blob_sha=blob_sha,
                file_path=file_path,
                kind=ChunkKind.DOC_SECTION,
                name=title,
                scope=scope,
                content=section_text,
                line_start=line_idx + 1,
                line_end=end_idx,
            )
        )

    return chunks


# ── Fallback: raw line-based chunks ──────────────────────────────────


def _raw_chunks(file_path: str, blob_sha: str, content: str) -> list[Chunk]:
    """Split into fixed-size line-based chunks with overlap.

    Chunk size and overlap come from `config.index`.
    """
    chunk_lines = config.index.chunk_lines
    overlap = config.index.chunk_overlap
    lines = content.split("\n")
    chunks: list[Chunk] = []
    start = 0

    while start < len(lines):
        end = min(start + chunk_lines, len(lines))
        text = "\n".join(lines[start:end]).strip()
        if text:
            chunks.append(
                Chunk(
                    id=_chunk_id(file_path, f"chunk:{start}", start),
                    blob_sha=blob_sha,
                    file_path=file_path,
                    kind=ChunkKind.RAW_CHUNK,
                    name=f"{file_path}:{start + 1}-{end}",
                    scope="",
                    content=text,
                    line_start=start + 1,
                    line_end=end,
                )
            )
        start += chunk_lines - overlap

    return chunks


def chunk_plaintext(file_path: str, blob_sha: str, content: str) -> list[Chunk]:
    """Chunk plain text or unsupported file types."""
    return _raw_chunks(file_path, blob_sha, content)


# ── Helpers ──────────────────────────────────────────────────────────


def _chunk_id(file_path: str, name: str, line_start: int) -> str:
    raw = f"{file_path}:{name}:{line_start}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]
