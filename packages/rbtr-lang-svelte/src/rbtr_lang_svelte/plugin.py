"""Svelte plugin, plus the single-file-component (SFC) machinery shared with Vue.

A Svelte (or Vue) component embeds code in several languages in one
file: a `<script>` block (JavaScript/TypeScript), a `<style>`
block (CSS/SCSS/Less), and a markup template. The `<script>` and
`<style>` blocks are delegated to their embedded language by the
engine's injection mechanism (`injection_query`); the chunker emits
only the markup template, as a host chunk so component markup is
searchable.

The template chunk's `language` is left blank for the orchestrator
to fill with the host language (svelte/vue); delegated chunks carry
their embedded language, set by the target's extractor.

`chunk_sfc` and `injections.scm` are shared: the `rbtr-lang-vue`
package reuses both for `.vue` files, which expose the same
`<script>`/`<style>` shape.
"""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import TYPE_CHECKING

from tree_sitter import Parser

from rbtr.domain.models import Chunk, ChunkKind, Comment, Fragment, FragmentMeta, Section, Span
from rbtr.languages.registration import LanguageRegistration, last_line, load_query

if TYPE_CHECKING:
    from collections.abc import Iterator

    from tree_sitter import Language, Node, Range


def template_chunk(
    file_path: str, blob_sha: str, content_bytes: bytes, nodes: list[Node]
) -> Chunk | None:
    """Emit the markup template as one host-language chunk, or `None`.

    Returns `None` when the component has no template markup (a script-only
    component); the orchestrator records the host language for dedup in that
    case. `language` is left blank for the orchestrator to fill (svelte/vue);
    spans the markup nodes.
    """
    if not nodes:
        return None
    first, last = nodes[0], nodes[-1]
    return Chunk(
        blob_sha=blob_sha,
        file_path=file_path,
        role=Section(kind=ChunkKind.DOC_SECTION, name=PurePosixPath(file_path).stem),
        content=content_bytes[first.start_byte : last.end_byte].decode("utf-8", errors="replace"),
        span=Span(
            start=first.start_point[0] + 1,
            end=last_line(last),
            start_byte=first.start_byte,
            end_byte=last.end_byte,
        ),
    )


def chunk_sfc(
    file_path: str,
    blob_sha: str,
    content: str,
    grammar: Language,
    ranges: list[Range] | None = None,
) -> Iterator[Chunk]:
    """Emit the SFC's markup template, any comment above it, and its blocks' tags.

    The `<script>`/`<style>` blocks' content is handled by the engine's
    injection mechanism (`_SFC_INJECTIONS`), not here; their tags are the
    component's own syntax, each a fragment, so every line reaches a chunk.
    A top-level comment is not markup, so it does not belong to the template
    chunk; it is content someone searches for, so it gets a chunk of its own.
    """
    content_bytes = content.encode("utf-8")
    parser = Parser(grammar)
    if ranges is not None:
        parser.included_ranges = ranges
    tree = parser.parse(content_bytes)
    template_nodes = [
        el
        for el in tree.root_node.children
        if el.type not in ("script_element", "style_element", "comment")
    ]
    template = template_chunk(file_path, blob_sha, content_bytes, template_nodes)
    if template is not None:
        yield template

    for node in tree.root_node.children:
        if node.type not in ("script_element", "style_element"):
            continue
        for tag in node.named_children:
            if tag.type not in ("start_tag", "end_tag"):
                continue
            yield Chunk(
                blob_sha=blob_sha,
                file_path=file_path,
                role=Fragment(meta=FragmentMeta(node_types=(tag.type,))),
                content=content_bytes[tag.start_byte : tag.end_byte].decode(
                    "utf-8", errors="replace"
                ),
                span=Span(
                    start=tag.start_point[0] + 1,
                    end=last_line(tag),
                    start_byte=tag.start_byte,
                    end_byte=tag.end_byte,
                ),
            )

    for node in tree.root_node.children:
        if node.type != "comment":
            continue
        yield Chunk(
            blob_sha=blob_sha,
            file_path=file_path,
            role=Comment(),
            content=content_bytes[node.start_byte : node.end_byte].decode(
                "utf-8", errors="replace"
            ),
            span=Span(
                start=node.start_point[0] + 1,
                end=last_line(node),
                start_byte=node.start_byte,
                end_byte=node.end_byte,
            ),
        )


svelte = LanguageRegistration(
    id="svelte",
    extensions=frozenset({".svelte"}),
    grammar_module="tree_sitter_svelte",
    injection_query=load_query(__package__, "injections"),
    extraction_serial=4,
)

svelte.chunker(chunk_sfc)
