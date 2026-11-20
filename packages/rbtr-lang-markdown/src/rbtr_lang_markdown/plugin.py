"""Markdown language plugin: two query plugins, the grammar's block and inline halves.

`markdown` reads a document's blocks. Each headed section is a
`doc_section` named by its heading, scoped by the headings above it, and
holding its text up to its first subsection. Fenced code is extracted in
the fence's language, and raw HTML as HTML.

`markdown_inline` reads the text inside each block, which Markdown's
injections hand it: a link is an import, and an HTML tag is HTML.

Extracted chunks::

    # Title                         → doc_section "Title", scope ""
    Intro text.                       (content: heading + intro)

    ## Section A                    → doc_section "Section A", scope "Title"
    Body A, see [the guide](guide.md#setup).
                                    → import "guide.md#setup"
                                      {module: guide.md, names: setup}

    First paragraph of a document   → fragment
    with no heading.
"""

from __future__ import annotations

import re

from rbtr.domain.edge_frames import SectionNames
from rbtr.languages.registration import LanguageRegistration, QueryExtraction, load_query

markdown = LanguageRegistration(
    id="markdown",
    extensions=frozenset({".md"}),
    grammar_module="tree_sitter_markdown",
    extraction=QueryExtraction(
        query=load_query(__package__, "sections"), scope_types=frozenset({"section"})
    ),
    injection_query=load_query(__package__, "injections"),
    prose=True,
    # ATX headings and fenced code.
    content_signals=(re.compile(r"^#{1,6}\s+", re.MULTILINE), re.compile(r"^```", re.MULTILINE)),
    section_names=SectionNames.FOLDED,
    extraction_serial=7,
)

# The grammar's inline half: emphasis, code spans, links and HTML tags
# inside a block's text, which the block grammar leaves as one `inline`
# node. It has no files of its own; Markdown's injections hand it each
# block's text, and its own hand the HTML tags in it to HTML. Its links
# document what they name, and resolve as Markdown's do.
markdown_inline = LanguageRegistration(
    id="markdown_inline",
    grammar_module="tree_sitter_markdown",
    grammar_entry="inline_language",
    extraction=QueryExtraction(query=load_query(__package__, "links"), fragments=False),
    injection_query=load_query(__package__, "inline_injections"),
    import_targets=frozenset({"markdown"}),
    prose=True,
)
