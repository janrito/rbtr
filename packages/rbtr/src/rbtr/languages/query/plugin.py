"""tree-sitter query language plugin.

Indexes `.scm` files — the tree-sitter query language that rbtr's own
language plugins are written in (see the `.scm` files beside each
`plugin.py`). Third-party grammars ship `.scm` too (`tags.scm`,
`highlights.scm`, `injections.scm`), so cloned repos are covered as well.

Each top-level pattern becomes a `doc_section` chunk, named by its own
outer capture where the author gave one (`query.scm` pairs it as
`@_section_name`):

    (function_definition name: (identifier) @_fn_name) @function
        → section "function"

    ["if" "else"] @keyword
        → section "keyword"

Patterns with no outer label of their own — a predicate-wrapped injection
rule, or one anchored at a structural wrapper — are anonymous sections;
their nested captures stay full-text-searchable.
"""

from __future__ import annotations

from rbtr.languages.registration import LanguageRegistration, QueryExtraction, load_query

query = LanguageRegistration(
    id="query",
    extensions=frozenset({".scm"}),
    grammar_module="tree_sitter_query",
    extraction=QueryExtraction(
        query=load_query(__package__, "query"),
        doc_comment_node_types=frozenset({"comment"}),
    ),
    language_plugin_version=1,
)
