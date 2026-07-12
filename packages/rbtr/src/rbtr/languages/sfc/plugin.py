"""Single-file-component (SFC) language plugin.

Svelte and Vue components embed code in several languages in one
file: a `<script>` block (JavaScript/TypeScript), a `<style>`
block (CSS/SCSS/Less), and a markup template. The `<script>` and
`<style>` blocks are delegated to their embedded language by the
engine's injection mechanism (`injection_query`, `_SFC_INJECTIONS`
below); the chunker emits only the markup template, as a host chunk
so component markup is searchable.

The template chunk's `language` is left blank for the orchestrator
to fill with the host language (svelte/vue); delegated chunks carry
their embedded language, set by the target's extractor.
"""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import TYPE_CHECKING

from tree_sitter import Parser

from rbtr.index.identity import make_chunk_id
from rbtr.index.models import Chunk, ChunkKind
from rbtr.languages.queries import load_query
from rbtr.languages.registration import LanguageRegistration

if TYPE_CHECKING:
    from collections.abc import Iterator

    from tree_sitter import Language, Node, Range

# Injection query shared by both SFC grammars — svelte and vue expose the
# same `script_element`/`style_element` → `raw_text` shape. Each block's
# `lang` attribute picks the embedded language; a `lang`-tagged rule outranks
# the bare fallback by priority, and the hint→id mapping is spelled out below.
_SFC_INJECTIONS = load_query(__package__, "injections")


def _template_chunk(
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
    start = min(n.start_byte for n in nodes)
    end = max(n.end_byte for n in nodes)
    text = content_bytes[start:end].decode("utf-8", errors="replace").strip()
    if not text:
        return None
    name = PurePosixPath(file_path).stem
    line_start = min(n.start_point[0] for n in nodes)
    line_end = max(n.end_point[0] for n in nodes)
    return Chunk(
        id=make_chunk_id(file_path, blob_sha, name, line_start),
        blob_sha=blob_sha,
        file_path=file_path,
        kind=ChunkKind.DOC_SECTION,
        name=name,
        scope="",
        content=text,
        line_start=line_start + 1,
        line_end=line_end + 1,
    )


def chunk_sfc(
    file_path: str,
    blob_sha: str,
    content: str,
    grammar: Language,
    ranges: list[Range] | None = None,
) -> Iterator[Chunk]:
    """Emit the SFC's markup template as a host chunk.

    The `<script>`/`<style>` blocks are handled by the engine's injection
    mechanism (`_SFC_INJECTIONS`), not here.
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
    template = _template_chunk(file_path, blob_sha, content_bytes, template_nodes)
    if template is not None:
        yield template


def _vue_grammar() -> Language:
    """Load the Vue grammar from the bundled language pack."""
    import tree_sitter_language_pack  # deferred: heavy bundled native grammars

    return tree_sitter_language_pack.get_language("vue")


svelte = LanguageRegistration(
    id="svelte",
    extensions=frozenset({".svelte"}),
    grammar_module="tree_sitter_svelte",
    injection_query=_SFC_INJECTIONS,
    language_plugin_version=1,
)
vue = LanguageRegistration(
    id="vue",
    extensions=frozenset({".vue"}),
    grammar_factory=_vue_grammar,
    injection_query=_SFC_INJECTIONS,
    language_plugin_version=1,
)

svelte.chunker(chunk_sfc)
vue.chunker(chunk_sfc)
