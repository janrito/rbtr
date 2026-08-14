"""Chunking helpers and fallback for unsupported content.

Provides the span arithmetic every chunker shares, the prose-format
detection heuristic, the host-presence sentinel, and the line-based
fallback chunker used when no language plugin matches.

Actual chunker implementations live in their language plugin
files under `rbtr/languages/`.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from typing import TYPE_CHECKING

from rbtr.config import config
from rbtr.domain.models import Chunk, ChunkKind
from rbtr.languages.plaintext import PLAINTEXT

if TYPE_CHECKING:
    from tree_sitter import Node

# ── Spans ────────────────────────────────────────────────


def last_line(node: Node) -> int:
    """The 1-based number of the last line *node* occupies.

    Tree-sitter points are 0-based, and a node that consumes its
    trailing newline ends at column 0 of the following row, which it
    does not occupy. Adding one to the end row is therefore right for a
    node ending mid-line and one too many for a node ending at a line
    break.

    Every chunker and the query engine share this, so a span means the
    same thing in every language.
    """
    end_row, end_column = node.end_point
    if end_column == 0 and end_row > node.start_point[0]:
        return end_row
    return end_row + 1


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

    Chunk size and overlap come from `config.index`. A slice of lines
    has no name — `line_start`/`line_end` locate it, and readers label
    it from those and the path they read it at.
    """
    chunk_lines = config.chunk_lines
    overlap = config.chunk_overlap
    lines = content.split("\n")
    # A file ending in a newline splits to a final empty string, which is not a
    # line the chunk occupies. Splitting on "\n" rather than by line keeps `\r`
    # in the content of a CRLF file, so only the count needs correcting.
    if lines and lines[-1] == "":
        lines.pop()
    start = 0

    while start < len(lines):
        end = min(start + chunk_lines, len(lines))
        text = "\n".join(lines[start:end]).strip()
        if text:
            yield Chunk(
                blob_sha=blob_sha,
                file_path=file_path,
                kind=ChunkKind.RAW_CHUNK,
                name="",
                scope="",
                content=text,
                line_start=start + 1,
                line_end=end,
            )
        start += chunk_lines - overlap


def chunk_plaintext(file_path: str, blob_sha: str, content: str) -> Iterator[Chunk]:
    """Chunk plain text or unsupported file types.

    The chunks carry `PLAINTEXT`, which is the language they are in:
    that is what stops `extract_file` deciding the file produced nothing
    in its own language and appending a presence chunk to every one.
    """
    for chunk in _raw_chunks(file_path, blob_sha, content):
        yield chunk.model_copy(update={"language": PLAINTEXT})


def host_presence_chunk(file_path: str, blob_sha: str, language: str) -> Chunk:
    """A content-less chunk recording a file's host language for dedup.

    Emitted when extraction produced no chunk in the file's own language:
    an empty file (an empty `__init__.py`), or a multi-language file whose
    host contributes no content (a script-only SFC). It carries the host
    language so the blob-dedup gate records that version and skips the file
    on later builds instead of re-parsing it every time. Empty content
    never ranks in search, and it has no name for the same reason a raw
    chunk has none: nothing about it is a symbol.

    Both language fields are *language*: the chunk stands for the file
    itself, so the file's language and the chunk's own are the same one.
    That also makes it complete without `extract_file`, which is how the
    build's extraction-failure fallback can emit one directly.
    """
    return Chunk(
        blob_sha=blob_sha,
        file_path=file_path,
        kind=ChunkKind.RAW_CHUNK,
        name="",
        scope="",
        content="",
        language=language,
        file_language=language,
        line_start=1,
        line_end=1,
    )
