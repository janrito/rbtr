"""Shared test data for fact store tests.

Realistic facts derived from the rbtr codebase and actual review
sessions.  Organised into groups that exercise deduplication,
confirmation, supersession, and non-match scenarios.

Each group is a list of dicts with ``content`` and ``scope``
(``'global'`` or ``'owner/repo'``).  The groups document the
expected dedup behaviour so tests can verify BM25 matching
against concrete data.

Data only — no fixtures, no I/O.  Fixtures that seed a store
with this data live in ``conftest.py``.
"""

from __future__ import annotations

# ── Scope values ─────────────────────────────────────────────────────
#
# ``'global'`` for cross-project, ``'owner/repo'`` for repo-scoped.

GLOBAL = "global"
RBTR_KEY = "alejandrogiacometti/rbtr"
OTHER_KEY = "alejandrogiacometti/other-project"

# ── Baseline facts ───────────────────────────────────────────────────
#
# A realistic set of facts that would accumulate over several review
# sessions of the rbtr codebase.  Covers tooling, conventions,
# architecture, and preferences.

RBTR_FACTS: list[dict[str, str]] = [
    {
        "content": "Uses pytest with pytest-mock for testing. No unittest.mock.",
        "scope": RBTR_KEY,
    },
    {
        "content": "Run `just check` after every meaningful change. Do not batch.",
        "scope": RBTR_KEY,
    },
    {
        "content": "Primary language is Python 3.13+. Use modern features directly.",
        "scope": RBTR_KEY,
    },
    {
        "content": "Sessions use SQLite (stdlib sqlite3) for OLTP persistence.",
        "scope": RBTR_KEY,
    },
    {
        "content": "Code index uses DuckDB with PyArrow for OLAP workloads.",
        "scope": RBTR_KEY,
    },
    {
        "content": "Persisted history is immutable. Repairs applied transiently in memory.",
        "scope": RBTR_KEY,
    },
    {
        "content": "SQL lives in .sql files loaded via importlib.resources. Never inline.",
        "scope": RBTR_KEY,
    },
    {
        "content": "Use StrEnum for finite sets. Literal only when enum is impractical.",
        "scope": RBTR_KEY,
    },
]

GLOBAL_FACTS: list[dict[str, str]] = [
    {
        "content": "Prefer composition over inheritance. Functions over methods.",
        "scope": GLOBAL,
    },
    {
        "content": "No Any or object as lazy type escapes. Use the real type.",
        "scope": GLOBAL,
    },
]

OTHER_REPO_FACTS: list[dict[str, str]] = [
    {
        "content": "Uses Go 1.22 with standard library HTTP server.",
        "scope": OTHER_KEY,
    },
    {
        "content": "CI runs on GitLab CI with Docker executors.",
        "scope": OTHER_KEY,
    },
]

ALL_BASELINE = RBTR_FACTS + GLOBAL_FACTS + OTHER_REPO_FACTS


# ── Deduplication pairs ──────────────────────────────────────────────
#
# Each tuple is (existing_fact_content, new_extraction, should_match).
#
# ``should_match=True``:  the new extraction is semantically the same
#     fact as the existing one.  FTS5 BM25 should surface it as a
#     dedup candidate.  The correct action is to confirm, not insert.
#
# ``should_match=False``: the new extraction shares keywords but is
#     a different fact.  FTS5 should not treat it as a duplicate.

DEDUP_PAIRS: list[tuple[str, str, bool]] = [
    # ── Exact or near-exact rewording (should match) ─────────────
    (
        "Uses pytest with pytest-mock for testing. No unittest.mock.",
        "Test framework is pytest with the pytest-mock fixture. unittest.mock is not used.",
        True,
    ),
    (
        "Run `just check` after every meaningful change. Do not batch.",
        "Always run `just check` after each change. Never batch multiple changes.",
        True,
    ),
    (
        "Primary language is Python 3.13+. Use modern features directly.",
        "Python 3.13+ is the target. Use modern language features.",
        True,
    ),
    (
        "Sessions use SQLite (stdlib sqlite3) for OLTP persistence.",
        "Sessions stored in SQLite stdlib sqlite3 for persistence.",
        True,
    ),
    (
        "SQL lives in .sql files loaded via importlib.resources. Never inline.",
        "SQL queries are in separate .sql files loaded with importlib.resources.",
        True,
    ),
    # ── Different facts sharing keywords (should NOT match) ──────
    (
        "Sessions use SQLite (stdlib sqlite3) for OLTP persistence.",
        "Code index uses DuckDB with PyArrow for OLAP workloads.",
        False,
    ),
    (
        "Uses pytest with pytest-mock for testing. No unittest.mock.",
        "Use StrEnum for finite sets. Literal only when enum is impractical.",
        False,
    ),
    (
        "Run `just check` after every meaningful change. Do not batch.",
        "Use `just fmt` and `just lint` for formatting and linting.",
        False,
    ),
    (
        "Primary language is Python 3.13+. Use modern features directly.",
        "Uses Go 1.22 with standard library HTTP server.",
        False,
    ),
    # ── Partial overlap — tricky cases ───────────────────────────
    #
    # These share significant keyword overlap but are genuinely
    # different facts.  BM25 may score them moderately.  The dedup
    # threshold must be high enough to avoid false positives.
    (
        "Sessions use SQLite (stdlib sqlite3) for OLTP persistence.",
        "SQLite FTS5 is used for fact deduplication via BM25 ranking.",
        False,
    ),
    (
        "Persisted history is immutable. Repairs applied transiently in memory.",
        "Facts are mutable. They can be confirmed, superseded, or pruned.",
        False,
    ),
]


# ── Supersession pairs ──────────────────────────────────────────────
#
# (old_content, new_content) — the new fact should supersede the old.
# The old fact gets ``superseded_by`` set; the new one is inserted.

SUPERSESSION_PAIRS: list[tuple[str, str]] = [
    (
        "Primary language is Python 3.12. Use modern features.",
        "Primary language is Python 3.13+. Use modern features directly.",
    ),
    (
        "Sessions stored as JSONL flat files in .rbtr/sessions/.",
        "Sessions use SQLite (stdlib sqlite3) for OLTP persistence.",
    ),
    (
        "Schema version uses integer counter, bumped on migration.",
        "Schema version uses date-based format YYYYMMDD0R for migrations.",
    ),
    (
        "Compaction deletes old messages from the database.",
        "Compaction marks old messages with compacted_by FK. Non-destructive.",
    ),
]


# ── Confirmation sequence ────────────────────────────────────────────
#
# A fact extracted across multiple sessions.  Each entry is a slight
# rewording that should match the original via FTS5 and trigger a
# confirm (bump ``last_confirmed_at`` and ``confirm_count``) rather
# than a new insert.

CONFIRMATION_SEQUENCE: list[str] = [
    # Session 1: first extraction
    "Uses pytest with pytest-mock for testing. No unittest.mock.",
    # Session 2: slightly different wording
    "Testing uses pytest and the pytest-mock plugin. No unittest.mock allowed.",
    # Session 3: even more different but same core fact
    "Test framework: pytest. Mocking: pytest-mock fixture only, not unittest.mock.",
    # Session 4: terse
    "pytest + pytest-mock. No unittest.mock.",
]


# ── Cross-scope isolation ────────────────────────────────────────────
#
# Facts that are textually similar but have different ``scope`` values.
# Dedup must respect scope boundaries — a repo fact should not match
# a global fact even if the content overlaps.

CROSS_SCOPE_SIMILAR: list[dict[str, str]] = [
    {
        "content": "Prefer composition over inheritance.",
        "scope": GLOBAL,
    },
    {
        "content": "Prefer composition over inheritance. Use Protocol for interfaces.",
        "scope": RBTR_KEY,
    },
]
