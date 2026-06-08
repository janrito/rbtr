"""HCL language plugin.

Splits HCL (HashiCorp Configuration Language) by top-level
blocks. Block names include the type and labels.

Extracted chunks::

    resource "aws_instance" "web" {
      ami = "ami-12345"             → doc_section "resource aws_instance web"
    }
    variable "region" {
      default = "us-east-1"         → doc_section "variable region"
    }
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

from tree_sitter import Parser

from rbtr.index.chunks import make_chunk_id
from rbtr.index.models import Chunk, ChunkKind
from rbtr.languages.hookspec import LanguageRegistration, hookimpl

if TYPE_CHECKING:
    from tree_sitter import Language

# ── Chunking ─────────────────────────────────────────────────────────


def chunk_hcl(file_path: str, blob_sha: str, content: str, grammar: Language) -> Iterator[Chunk]:
    """Split HCL by top-level blocks (resource, variable, output, etc.)."""
    if not content.strip():
        return
    content_bytes = content.encode("utf-8")
    parser = Parser(grammar)
    tree = parser.parse(content_bytes)
    root = tree.root_node

    # HCL: config_file → body → block*
    for child in root.children:
        if child.type == "body":
            for block in child.children:
                if block.type == "block":
                    labels: list[str] = []
                    for gc in block.children:
                        if gc.type == "identifier" and gc.text:
                            labels.append(gc.text.decode("utf-8", errors="replace"))
                        elif gc.type == "string_lit" and gc.text:
                            labels.append(gc.text.decode("utf-8", errors="replace").strip('"'))
                    name = " ".join(labels)
                    text = (
                        content_bytes[block.start_byte : block.end_byte]
                        .decode("utf-8", errors="replace")
                        .strip()
                    )
                    if text:
                        yield Chunk(
                            id=make_chunk_id(file_path, blob_sha, name, block.start_point[0]),
                            blob_sha=blob_sha,
                            file_path=file_path,
                            kind=ChunkKind.DOC_SECTION,
                            name=name,
                            scope="",
                            content=text,
                            line_start=block.start_point[0] + 1,
                            line_end=block.end_point[0] + 1,
                        )


# ── Plugin ───────────────────────────────────────────────────────────


class HclPlugin:
    """HCL language support — block-based chunking."""

    @hookimpl
    def rbtr_register_languages(self) -> list[LanguageRegistration]:
        return [
            LanguageRegistration(
                id="hcl",
                extensions=frozenset({".hcl", ".tf"}),
                grammar_module="tree_sitter_hcl",
                chunker=chunk_hcl,
            ),
        ]
