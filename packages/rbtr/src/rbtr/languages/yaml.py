"""YAML language plugin.

Splits YAML by top-level mapping keys. Documents without a
block mapping are treated as a single chunk.

Extracted chunks::

    name: CI                        → doc_section "name", scope ""
    on: [push]                      → doc_section "on", scope ""
    jobs:                           → doc_section "jobs", scope ""
      test:
        runs-on: ubuntu-latest
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

from tree_sitter import Parser

from rbtr.index.identity import make_chunk_id
from rbtr.index.models import Chunk, ChunkKind
from rbtr.languages.hookspec import LanguageRegistration, hookimpl

if TYPE_CHECKING:
    from tree_sitter import Language, Node

# ── Chunking ─────────────────────────────────────────────────────────


def _find_mapping(node: Node) -> Node | None:
    """Walk into YAML document/stream to find the block_mapping."""
    if node.type == "block_mapping":
        return node
    for child in node.children:
        result = _find_mapping(child)
        if result is not None:
            return result
    return None


def chunk_yaml(file_path: str, blob_sha: str, content: str, grammar: Language) -> Iterator[Chunk]:
    """Split YAML by top-level mapping keys."""
    if not content.strip():
        return
    content_bytes = content.encode("utf-8")
    parser = Parser(grammar)
    tree = parser.parse(content_bytes)

    mapping = _find_mapping(tree.root_node)
    if mapping is None:
        yield Chunk(
            id=make_chunk_id(file_path, blob_sha, file_path, 0),
            blob_sha=blob_sha,
            file_path=file_path,
            kind=ChunkKind.DOC_SECTION,
            name=file_path,
            scope="",
            content=content.strip(),
            line_start=1,
            line_end=tree.root_node.end_point[0] + 1,
        )
        return

    for child in mapping.children:
        if child.type == "block_mapping_pair":
            key_node = child.child_by_field_name("key")
            key = ""
            if key_node and key_node.text:
                key = key_node.text.decode("utf-8", errors="replace")
            text = (
                content_bytes[child.start_byte : child.end_byte]
                .decode("utf-8", errors="replace")
                .strip()
            )
            if text:
                yield Chunk(
                    id=make_chunk_id(file_path, blob_sha, key, child.start_point[0]),
                    blob_sha=blob_sha,
                    file_path=file_path,
                    kind=ChunkKind.DOC_SECTION,
                    name=key,
                    scope="",
                    content=text,
                    line_start=child.start_point[0] + 1,
                    line_end=child.end_point[0] + 1,
                )
    return


# ── Plugin ───────────────────────────────────────────────────────────


class YamlPlugin:
    """YAML language support — top-level key chunking."""

    @hookimpl
    def rbtr_register_languages(self) -> list[LanguageRegistration]:
        return [
            LanguageRegistration(
                id="yaml",
                extensions=frozenset({".yaml", ".yml"}),
                grammar_module="tree_sitter_yaml",
                chunker=chunk_yaml,
            ),
        ]
