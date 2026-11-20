"""Code-aware tokenisation for full-text search.

Splits identifiers on camelCase, acronym, digit and underscore boundaries,
emitting both the original lowered form and its parts. This lets DuckDB's
BM25 index match queries like `agent deps` against chunks containing
`AgentDeps`.
"""

from __future__ import annotations

import polars as pl

IDENTIFIER = r"[A-Za-z_][A-Za-z0-9_]*"
"""What an identifier looks like: a letter or `_`, then letters, digits and `_`s."""

BOUNDARIES = (
    (r"([a-z])([A-Z])", "${1} ${2}"),  # camelCase: aB → a B
    (r"([A-Z])([A-Z][a-z])", "${1} ${2}"),  # acronym end: XMLParser → XML Parser
    (r"([0-9])([A-Za-z])", "${1} ${2}"),  # digit→letter: html5Doc → html5 Doc
    (r"([A-Za-z])([0-9])", "${1} ${2}"),  # letter→digit: parseHTML5 → parseHTML 5
)
"""Where an identifier splits, each as a regex and the replacement that puts a space there.

Polars' regex engine has no lookarounds, so a boundary is matched with the
characters either side of it and written back around a space.
"""


def identifier_tokens(identifier: pl.Expr) -> pl.Expr:
    """An identifier's tokens: itself lowered, then its lowered parts.

    Its parts are what is left between its boundaries and its `_`s,
    leading and trailing `_` stripped first: `XMLParser_v2` →
    `["xmlparser_v2", "xml", "parser", "v", "2"]`. A part may be empty;
    `search_tokens` drops it.
    """
    spaced = identifier.str.strip_chars("_")
    for pattern, replacement in BOUNDARIES:
        spaced = spaced.str.replace_all(pattern, replacement)
    parts = spaced.str.replace_all("_", " ", literal=True).str.to_lowercase().str.split(" ")
    return pl.concat_list(identifier.str.to_lowercase(), parts)


def search_tokens(text: pl.Expr) -> pl.Expr:
    """*text*'s tokens for full-text search, space-separated, each once, in the order they first appear.

    Each identifier in *text* gives its `identifier_tokens`; operators,
    punctuation and anything else that is not an identifier give nothing:

        `AgentDeps` → `agentdeps agent deps`
        `_deep_merge` → `_deep_merge deep merge`
        `parseHTML5Doc` → `parsehtml5doc parse html 5 doc`
        `` → ``

    The chunks table stores these for its content and names, and a query
    is tokenised the same way before BM25 matches it.
    """
    return (
        text.str.extract_all(IDENTIFIER)
        .list.unique(maintain_order=True)
        .list.eval(identifier_tokens(pl.element()).explode())
        .list.eval(pl.element().filter(pl.element() != ""))
        .list.unique(maintain_order=True)
        .list.join(" ")
    )
