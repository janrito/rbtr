"""TOML language plugin.

Splits TOML by tables. Dotted keys produce hierarchical scope.

Extracted chunks::

    [project]                       → doc_section "project", scope ""
    name = "rbtr"

    [tool.ruff]                     → doc_section "ruff", scope "tool"
    line-length = 99
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

from tree_sitter import Parser

from rbtr.index.identity import make_chunk_id
from rbtr.index.models import Chunk, ChunkKind
from rbtr.languages.hookspec import LanguageRegistration, hookimpl

if TYPE_CHECKING:
    from tree_sitter import Language

# ── Chunking ─────────────────────────────────────────────────────────


def chunk_toml(file_path: str, blob_sha: str, content: str, grammar: Language) -> Iterator[Chunk]:
    """Split TOML by tables."""
    if not content.strip():
        return
    content_bytes = content.encode("utf-8")
    parser = Parser(grammar)
    tree = parser.parse(content_bytes)
    root = tree.root_node

    for child in root.children:
        if child.type == "table":
            key = ""
            for gc in child.children:
                if gc.type in ("bare_key", "dotted_key") and gc.text:
                    key = gc.text.decode("utf-8", errors="replace")
                    break
            text = (
                content_bytes[child.start_byte : child.end_byte]
                .decode("utf-8", errors="replace")
                .strip()
            )
            if text:
                parts = key.split(".")
                name = parts[-1] if parts else key
                scope = " > ".join(parts[:-1]) if len(parts) > 1 else ""
                yield Chunk(
                    id=make_chunk_id(file_path, blob_sha, key, child.start_point[0]),
                    blob_sha=blob_sha,
                    file_path=file_path,
                    kind=ChunkKind.DOC_SECTION,
                    name=name,
                    scope=scope,
                    content=text,
                    line_start=child.start_point[0] + 1,
                    line_end=child.end_point[0] + 1,
                )
    return


# ── Plugin ───────────────────────────────────────────────────────────


class TomlPlugin:
    """TOML language support — table-based chunking."""

    @hookimpl
    def rbtr_register_languages(self) -> list[LanguageRegistration]:
        return [
            LanguageRegistration(
                id="toml",
                extensions=frozenset({".toml"}),
                grammar_module="tree_sitter_toml",
                chunker=chunk_toml,
            ),
        ]
