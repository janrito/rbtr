"""Code-aware tokenisation for full-text search.

Splits identifiers on camelCase and underscore boundaries, emitting
both the original lowered form and individual parts.  This lets
DuckDB's BM25 index match queries like ``agent deps`` against
chunks containing ``AgentDeps``.
"""

from __future__ import annotations

import re

# Matches identifier-like tokens: starts with a letter or underscore,
# followed by letters, digits, or underscores.
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

# Splits camelCase / PascalCase boundaries and underscore separators.
# Handles runs of uppercase (e.g. XMLParser → XML, Parser).
_SPLIT_RE = re.compile(
    r"""
    (?<=[a-z])(?=[A-Z])        # camelCase: aB → a, B
    | (?<=[A-Z])(?=[A-Z][a-z]) # acronym end: XMLParser → XML, Parser
    | (?<=[0-9])(?=[A-Za-z])   # digit→letter: html5Doc → html5, Doc
    | (?<=[A-Za-z])(?=[0-9])   # letter→digit: parseHTML5 → parseHTML, 5
    | _                         # underscore separator
    """,
    re.VERBOSE,
)


def tokenise_code(text: str) -> str:
    """Tokenise *text* for code-aware full-text search.

    Extracts identifier-like tokens, splits on camelCase and
    underscore boundaries, and returns a space-separated string
    of lowered terms.  Both the original compound form and its
    parts are emitted so that either query style matches::

        >>> tokenise_code("AgentDeps")
        'agentdeps agent deps'
        >>> tokenise_code("_deep_merge")
        'deep merge'
        >>> tokenise_code("")
        ''

    Non-identifier text (operators, punctuation, string literals)
    is discarded — the FTS index doesn't need it.
    """
    if not text:
        return ""

    tokens: list[str] = []
    seen: set[str] = set()

    for match in _IDENT_RE.finditer(text):
        ident = match.group()

        # Strip leading/trailing underscores for splitting,
        # but keep the original lowered form if it differs.
        stripped = ident.strip("_")
        lowered = ident.lower()

        # Emit original compound form.
        if lowered not in seen:
            tokens.append(lowered)
            seen.add(lowered)

        # Split into parts and emit each.
        if stripped:
            parts = _SPLIT_RE.split(stripped)
            for part in parts:
                p = part.lower()
                if p and p not in seen:
                    tokens.append(p)
                    seen.add(p)

    return " ".join(tokens)
