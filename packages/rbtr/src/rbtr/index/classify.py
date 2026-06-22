"""Heuristic query classifier for expansion routing.

Classifies search queries into three processing tiers
(`CODE`, `CONCEPT`, `IDENTIFIER`) using a point-based scoring
system.  No model calls — keywords are derived from the
registered tree-sitter grammars.

Code detection uses additive scoring: syntax characters and
language keywords each contribute points, and a query is
classified as `CODE` when its total reaches the threshold.
This avoids the false positives of a single-regex approach —
a docstring with one stray parenthesis or a concept query
starting with `return` no longer triggers `CODE`.

Keywords are extracted at runtime from all registered
tree-sitter grammars via `LanguageManager`.
"""

from __future__ import annotations

import re
from functools import cache

from rbtr.index.models import QueryKind
from rbtr.languages import get_manager

_CONCEPT_MIN_WORDS = 3
_CODE_THRESHOLD = 2
_CONCEPT_MAX_PUNCT = 0.04


# Tree-sitter internal node names that leak into the anonymous
# node-kind list but are not real language keywords.
_TS_META = frozenset(
    {
        "block",
        "declaration",
        "end",
        "expr",
        "expression",
        "ident",
        "item",
        "lifetime",
        "literal",
        "parameter",
        "pat",
        "path",
        "pattern",
        "statement",
        "stmt",
        "target",
        "tt",
        "ty",
        "vis",
    }
)

# Common English words that grammars include as keywords.
# These are prepositions, conjunctions, and short verbs that
# start natural-language queries too often to be useful signals.
_ENGLISH_STOPWORDS = frozenset(
    {
        "and",
        "as",
        "do",
        "else",
        "for",
        "from",
        "get",
        "if",
        "in",
        "is",
        "new",
        "not",
        "of",
        "or",
        "return",
        "set",
        "then",
        "to",
        "try",
        "with",
    }
)


@cache
def _code_keywords() -> frozenset[str]:
    """Language keywords for code-detection scoring.

    Extracts anonymous keyword node kinds from every registered
    tree-sitter grammar, unions them, and subtracts `_TS_META`.
    Cached after the first call.
    """
    keywords: set[str] = set()
    mgr = get_manager()
    for lang_id in mgr.all_language_ids():
        grammar = mgr.load_grammar(lang_id)
        if grammar is None:
            continue
        for i in range(grammar.node_kind_count):
            name = grammar.node_kind_for_id(i)
            if name and not grammar.node_kind_is_named(i) and len(name) > 1 and name.isalpha():
                keywords.add(name)
    return frozenset(keywords - _TS_META - _ENGLISH_STOPWORDS)


# ── Syntax patterns ──────────────────────────────────────────────────

_BRACES = re.compile(r"[{}]")
_FUNC_CALL = re.compile(r"\w\(")
_OPERATORS = re.compile(r"==|!=|<=|>=|=>|->")
_SEMICOLON = re.compile(r";")
_ANGLE_OPEN = re.compile(r"<\w")
_ANGLE_CLOSE = re.compile(r"\w>")
_PARENS = re.compile(r"[()]")
_TIGHT_ASSIGN = re.compile(r"\w=\w")
_TRAILING_COLON = re.compile(r":\s*$", re.MULTILINE)


def _code_score(query: str) -> int:
    """Score how code-like *query* is.

    Higher scores mean stronger code signals.  Each syntax
    pattern and keyword match contributes independently.
    """
    score = 0

    # Strong signals (2 pts).
    if _BRACES.search(query):
        score += 2
    has_func_call = bool(_FUNC_CALL.search(query))
    if has_func_call:
        score += 2
    if _OPERATORS.search(query):
        score += 2
    if _SEMICOLON.search(query):
        score += 2

    # Angle brackets: pair = 2, lone side = 1.
    has_open = bool(_ANGLE_OPEN.search(query))
    has_close = bool(_ANGLE_CLOSE.search(query))
    if has_open and has_close:
        score += 2
    elif has_open or has_close:
        score += 1

    # Weak signals (1 pt).
    if not has_func_call and _PARENS.search(query):
        score += 1
    if _TIGHT_ASSIGN.search(query):
        score += 1
    if _TRAILING_COLON.search(query):
        score += 1

    # Keyword at start of query (1 pt).
    words = query.split()
    if words and words[0].lower() in _code_keywords():
        score += 1

    return score


def _punct_ratio(query: str) -> float:
    """Fraction of characters that are neither alphanumeric nor space."""
    if not query:
        return 0.0
    natural = sum(1 for c in query if c.isalnum() or c == " ")
    return 1.0 - natural / len(query)


def classify_query(query: str) -> QueryKind:
    """Classify a search query for expansion routing.

    Returns the processing tier:

    - `CODE` — query accumulates enough syntax and keyword
      signals (score >= threshold).  Expansion is skipped.
    - `CONCEPT` — query has 3+ words, isn't code, and has
      low punctuation density (ratio <= threshold).  Gets
      keywords + variant rephrases.  The punctuation gate
      is language-agnostic (`str.isalnum` is Unicode-aware)
      and filters structured names like breadcrumbs, CSS
      selectors, and dot-joined section paths.
    - `IDENTIFIER` — everything else.  Gets keywords only.

    The default tier is `IDENTIFIER`, not `CONCEPT` — a short
    ambiguous query like `"description"` is more likely a
    symbol name than a sentence.
    """
    q = query.strip()

    if _code_score(q) >= _CODE_THRESHOLD:
        return QueryKind.CODE

    if len(q.split()) >= _CONCEPT_MIN_WORDS and _punct_ratio(q) <= _CONCEPT_MAX_PUNCT:
        return QueryKind.CONCEPT

    return QueryKind.IDENTIFIER
