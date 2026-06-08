"""Cases for `classify_query` — the heuristic query classifier.

Each case returns `(query, expected_kind)`.  Tags drive which
test function consumes them:

- `identifier` / `concept` / `code` — correctly classified.
- `misclassified` — known heuristic trade-offs where the
  classifier gets it wrong.  Consumed by an `xfail` test so
  improvements that fix them surface as XPASS.
"""

from __future__ import annotations

from pytest_cases import case, parametrize

from rbtr.index.classify import QueryKind

# ── IDENTIFIER (correctly classified) ────────────────────────────────


@case(tags=["identifier"])
@parametrize(
    query=[
        # Python identifiers
        "fuse_scores",
        "build_index",
        "_next_element_sibling",
        "handleRequest",
        "DaemonClient",
        # Qualified / dotted names
        "IndexStore.has_indexed",
        "homeassistant.components.rapt_ble",
        "Pdm.setup",
        # Rust generics (single word, no angle-pair signal)
        "Installer",
        # Single common words
        "description",
        "batched",
        "model",
        "u",
        # Hyphenated / CSS / config keys
        "http-proxy",
        ".thinking-collapsed",
        "@smithy/smithy-client",
        # Two-word queries (below 3-word threshold)
        "retry logic",
        "error handling",
        # Multi-word structured names — punct ratio > 0.04 blocks CONCEPT.
        "Changelog > 0.11.11.Bug fixes",
        "Architecture > Data model.Completion tracking",
        ".select2-container--default .select2-selection--multiple .select2-selection__choice",
        "Claude API \u2014 Python.Error Handling",
        # HTML file paths (name provenance)
        "tests/forms_tests/templates/forms_tests/legend_test.html",
        # JSON keys
        "hit_at_1",
        "search_p95_ms",
        # Keyword at start, no syntax (score 1)
        "import os",
        "class Foo",
        # Empty / whitespace
        "",
        "   ",
    ],
)
def case_identifier(query: str) -> tuple[str, QueryKind]:
    return query, QueryKind.IDENTIFIER


# ── CONCEPT (correctly classified) ───────────────────────────────────


@case(tags=["concept"])
@parametrize(
    query=[
        # Threshold (3 words)
        "shorten conversation history",
        "are embeddings normalised",
        "search for retry",
        # Natural-language questions
        "how does idle unload work",
        "what is the default timeout",
        "where is the config loaded",
        "which function handles auth",
        # Imperative phrases
        "find functions that handle authentication",
        "check if user is admin",
        "handle timeout errors gracefully",
        "remove stale cache entries",
        # Multi-language concept queries from eval
        "how to adjust select element margins for right-to-left layouts",
        "how to use legend tag instead of label tag for radio select fields",
        "create a predicate that checks if a DOM element is an input of a specific type",
        "extract Go function symbol from source code",
        "display a search result with file location and relevance score",
        # Keyword at start, no syntax (score 1, 3+ words)
        "return build dependencies needed for editable installs",
        # Breadcrumb separator — doc-section name, not code
        "Form widgets > Widget attributes",
    ],
)
def case_concept(query: str) -> tuple[str, QueryKind]:
    return query, QueryKind.CONCEPT


# ── CODE (correctly classified) ──────────────────────────────────────


@case(tags=["code"])
@parametrize(
    query=[
        # Python
        "def fuse_scores(",
        "async def run(",
        "class Engine:",
        "for i in range(10):",
        # JavaScript / TypeScript
        "function update() {",
        "function updateRelatedObjectLinks(triggeringLink) {",
        # CSS
        "body {\n\tfont-size: 16px;",
        # Go
        "if err != nil {",
        # Operators
        "x = foo(bar)",
        "a == b",
        # Rust / TS generics
        "result <T>",
        # Multi-language body fragments from eval
        "dd {\n    margin-bottom: 0.8em;\n}",
        'function U(e){return e&&"undefined"!=typeof e.getElementsByTagName&&e}',
    ],
)
def case_code(query: str) -> tuple[str, QueryKind]:
    return query, QueryKind.CODE


# ── Known misclassifications (heuristic trade-offs) ──────────────────
#
# These are queries where the heuristic gets the wrong answer.
# Each documents the trade-off: why the expected classification
# differs from the actual one and why we accept the gap.


@case(tags=["misclassified"])
@parametrize(
    "query, expected_kind",
    [
        # Keyword-only code (score 1, below threshold 2).
        # Accepting: IDENTIFIER still gets keyword expansion.
        ("import os", QueryKind.CODE),
        ("return self.value", QueryKind.CODE),
        # Stopword keyword + trailing colon (score 1).
        # 'for' is a stopword to avoid "for loop best practices" → CODE.
        ("for item in self._cache:", QueryKind.CODE),
        # Markdown email addresses — angle brackets trigger CODE.
        ("J. Smith <jsmith@example.com>, 2024", QueryKind.IDENTIFIER),
        # RST section headings with backtick function refs.
        (
            "Built-in Expressions > ``F()`` expressions > ``F()`` assignments",
            QueryKind.IDENTIFIER,
        ),
        # Punct-ratio gate regression: hyphen pushes ratio to 0.042.
        ("is the cache thread-safe", QueryKind.CONCEPT),
        # Punct-ratio gate regression: triple-quotes and parens push ratio to 0.231.
        ('"""Linear interpolation (no easing)."""', QueryKind.CONCEPT),
        # Dot-joined section name with low punct ratio (0.038) slips past gate.
        ("rbtr search-weight tuning report.Impact by dimension", QueryKind.IDENTIFIER),
        # JSDoc {Type} braces trigger CODE.
        (
            "/**\n * Returns a function to use in pseudos\n * @param {String} type\n */",
            QueryKind.CONCEPT,
        ),
        # Concept query with function-call syntax in it.
        (
            "how are F() expression values updated on model instances after saving",
            QueryKind.CONCEPT,
        ),
    ],
)
def case_misclassified(query: str, expected_kind: QueryKind) -> tuple[str, QueryKind]:
    return query, expected_kind
