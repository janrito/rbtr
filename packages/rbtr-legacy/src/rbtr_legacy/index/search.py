"""Unified search scoring and fusion.

Combines lexical (BM25), semantic (cosine), and name-match signals
into a single ranked result list.  Each signal is normalised to
[0, 1] via min-max scaling, then fused with a convex combination.
Post-fusion multipliers for chunk kind and file category adjust
the final score.

All scoring helpers are pure functions — easy to test in isolation.
Orchestration lives in `search` (`store.py`).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import StrEnum

from rbtr_legacy.index.models import Chunk, ChunkKind

# ── Score normalisation ──────────────────────────────────────────────


def normalise_scores(scores: list[float]) -> list[float]:
    """Min-max normalise *scores* to [0, 1].

    Returns an empty list for empty input.  If all values are
    equal, returns all zeros (no signal).
    """
    if not scores:
        return []
    lo = min(scores)
    hi = max(scores)
    span = hi - lo
    if span == 0.0:
        return [0.0] * len(scores)
    return [(s - lo) / span for s in scores]


# ── Name matching ────────────────────────────────────────────────────


def name_score(query: str, name: str) -> float:
    """Score how well *query* matches chunk *name*.

    Checks whole-query matching first (exact > prefix > substring),
    then falls back to token-level matching: if every token in the
    query appears in the name, scores 0.4.  This handles multi-word
    concept queries like `"import edge"` matching
    `"infer_import_edges"`.

    Returns:
        1.0  exact match (case-insensitive),
        0.8  prefix match,
        0.5  substring match,
        0.4  all query tokens found in name,
        0.0  no match.
    """
    q = query.lower()
    n = name.lower()
    if q == n:
        return 1.0
    if n.startswith(q):
        return 0.8
    if q in n:
        return 0.5
    # Token-level: every query token must appear in the name.
    tokens = q.split()
    if len(tokens) > 1 and all(t in n for t in tokens):
        return 0.4
    return 0.0


# ── Query classification ─────────────────────────────────────────────


class QueryKind(StrEnum):
    """Broad query category used to adjust fusion weights."""

    IDENTIFIER = "identifier"
    CONCEPT = "concept"
    PATTERN = "pattern"


# Regex metacharacters that signal a pattern query.  Excludes `.`
# (used in dotted identifiers like `a.b.c`) and `_` (used in
# snake_case identifiers).
_PATTERN_CHARS = re.compile(r"[\\*+?\[\]{}^$|()]")

# Identifier: single token, may contain camelCase, snake_case, dots.
# No spaces allowed.
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")


def classify_query(query: str) -> QueryKind:
    """Classify *query* to select fusion weights.

    - **pattern** — contains regex metacharacters (not `.`).
    - **identifier** — single token matching an identifier pattern.
    - **concept** — everything else (natural language, keywords).
    """
    stripped = query.strip()
    if _PATTERN_CHARS.search(stripped):
        return QueryKind.PATTERN
    if _IDENTIFIER_RE.match(stripped):
        return QueryKind.IDENTIFIER
    return QueryKind.CONCEPT


# Per-kind fusion weights: (alpha=semantic, beta=lexical, gamma=name).
# Tuned via scripts/tune_search.py against the eval set.
_KIND_WEIGHTS: dict[QueryKind, tuple[float, float, float]] = {
    # Identifiers: name match dominates.  BM25 adds noise because
    # import chunks inflate TF for identifier terms.  Semantic
    # weight reserved for when embeddings are available.
    QueryKind.IDENTIFIER: (0.1, 0.0, 0.9),
    # Concepts: multi-word keyword queries.  Semantic favoured
    # to bridge vocabulary gaps (e.g. "shorten conversation" →
    # compact_history).  Small lexical weight for BM25 fallback
    # when embeddings are unavailable.  Name channel valuable
    # for token-level matching (e.g. "import edge" →
    # infer_import_edges).
    QueryKind.CONCEPT: (0.4, 0.1, 0.5),
    # Pattern queries: routed to grep in practice.  When they
    # reach search, semantic is the best available signal.
    QueryKind.PATTERN: (0.5, 0.3, 0.2),
}


def weights_for_query(query: str) -> tuple[float, float, float]:
    """Return `(alpha, beta, gamma)` fusion weights for *query*."""
    return _KIND_WEIGHTS[classify_query(query)]


# ── Kind boost ───────────────────────────────────────────────────────

_KIND_BOOST: dict[ChunkKind, float] = {
    ChunkKind.CLASS: 1.5,
    ChunkKind.FUNCTION: 1.3,
    ChunkKind.METHOD: 1.3,
    ChunkKind.TEST_FUNCTION: 0.7,
    ChunkKind.IMPORT: 0.3,
    ChunkKind.RAW_CHUNK: 0.5,
    ChunkKind.DOC_SECTION: 0.8,
    ChunkKind.CONFIG_KEY: 0.6,
    ChunkKind.VARIABLE: 1.0,
    ChunkKind.MIGRATION: 0.4,
    ChunkKind.API_ENDPOINT: 1.0,
}


def kind_boost(kind: ChunkKind) -> float:
    """Return the ranking boost multiplier for *kind*."""
    return _KIND_BOOST.get(kind, 1.0)


# ── File category penalty ────────────────────────────────────────────


class FileCategory(StrEnum):
    """Broad category derived from a file path."""

    SOURCE = "source"
    TEST = "test"
    VENDOR = "vendor"
    GENERATED = "generated"
    DOC = "doc"
    CONFIG = "config"


_CATEGORY_PENALTY: dict[FileCategory, float] = {
    FileCategory.SOURCE: 1.0,
    FileCategory.TEST: 0.5,
    FileCategory.VENDOR: 0.3,
    FileCategory.GENERATED: 0.3,
    FileCategory.DOC: 0.8,
    FileCategory.CONFIG: 0.7,
}

# Patterns for file classification.  Checked in order — first
# match wins.  Tuples of (substring_or_suffix, category).
_TEST_INDICATORS = ("test_", "/tests/", "_test.", "/test/", "tests.", "conftest.")
_VENDOR_INDICATORS = (
    "vendor/",
    "/vendor/",
    "node_modules/",
    "/node_modules/",
    "third_party/",
    "/third_party/",
)
_GENERATED_SUFFIXES = ("_pb2.py", ".pb.go", "_generated.", ".gen.")
_DOC_SUFFIXES = (".md", ".rst", ".txt", ".adoc")
_CONFIG_SUFFIXES = (".json", ".yaml", ".yml", ".toml", ".ini", ".cfg")


def file_category(path: str) -> FileCategory:
    """Classify *path* into a broad file category."""
    lower = path.lower()

    for indicator in _VENDOR_INDICATORS:
        if indicator in lower:
            return FileCategory.VENDOR

    for suffix in _GENERATED_SUFFIXES:
        if lower.endswith(suffix):
            return FileCategory.GENERATED

    for indicator in _TEST_INDICATORS:
        if indicator in lower:
            return FileCategory.TEST

    for suffix in _DOC_SUFFIXES:
        if lower.endswith(suffix):
            return FileCategory.DOC

    for suffix in _CONFIG_SUFFIXES:
        if lower.endswith(suffix):
            return FileCategory.CONFIG

    return FileCategory.SOURCE


def file_category_penalty(path: str) -> float:
    """Return the ranking penalty multiplier for the file at *path*."""
    return _CATEGORY_PENALTY[file_category(path)]


# ── Importance (inbound-degree) ──────────────────────────────────────


def importance_score(inbound_degree: int) -> float:
    """Ranking boost from inbound edge count.

    Uses `log2(1 + degree)` scaled so that zero edges gives 1.0
    (neutral) and higher degree gives a multiplicative boost.
    Capped at 3.0 to avoid runaway dominance.

    Examples:
        0 edges → 1.0  (no boost)
        1 edge  → 1.0 + log2(2)/4 = 1.25
        3 edges → 1.0 + log2(4)/4 = 1.50
        7 edges → 1.0 + log2(8)/4 = 1.75
        15 edges → 1.0 + log2(16)/4 = 2.00
    """
    if inbound_degree <= 0:
        return 1.0
    return min(1.0 + math.log2(1 + inbound_degree) / 4, 3.0)


# ── Diff proximity ──────────────────────────────────────────────────


def proximity_score(
    chunk_file: str,
    changed_files: set[str],
    has_edge_to_changed: bool = False,
) -> float:
    """Ranking boost from proximity to the current diff.

    Returns:
        1.5  chunk is in a changed file,
        1.2  chunk has an edge to a changed file,
        1.1  chunk is in the same directory as a changed file,
        1.0  no proximity (neutral).
    """
    if not changed_files:
        return 1.0
    if chunk_file in changed_files:
        return 1.5
    if has_edge_to_changed:
        return 1.2
    # Same directory heuristic.
    chunk_dir = chunk_file.rsplit("/", 1)[0] if "/" in chunk_file else ""
    for cf in changed_files:
        cf_dir = cf.rsplit("/", 1)[0] if "/" in cf else ""
        if chunk_dir and chunk_dir == cf_dir:
            return 1.1
    return 1.0


# ── Score fusion ─────────────────────────────────────────────────────

# Starting fusion weights.  Tuned against the eval set.
DEFAULT_ALPHA = 0.3  # semantic
DEFAULT_BETA = 0.3  # lexical (BM25)
DEFAULT_GAMMA = 0.4  # name match


@dataclass(frozen=True)
class ScoredResult:
    """A search result with full signal breakdown."""

    chunk: Chunk
    score: float
    lexical: float
    semantic: float
    name: float
    kind_boost: float
    file_penalty: float
    importance: float = 1.0
    proximity: float = 1.0


def fuse_scores(
    results: dict[str, Chunk],
    lexical_scores: dict[str, float],
    semantic_scores: dict[str, float],
    name_scores: dict[str, float],
    *,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
    gamma: float = DEFAULT_GAMMA,
    top_k: int = 10,
    importance_scores: dict[str, float] | None = None,
    proximity_scores: dict[str, float] | None = None,
) -> list[ScoredResult]:
    """Fuse three signal channels into a ranked result list.

    Each `*_scores` dict maps chunk ID → raw score.  Scores
    are normalised independently, then combined::

        base = a*semantic + b*lexical + g*name
        final = base * kind_boost * file_penalty * importance * proximity

    Args:
        results:            All candidate chunks, keyed by ID.
        lexical_scores:     BM25 scores per chunk ID.
        semantic_scores:    Cosine similarity scores per chunk ID.
        name_scores:        Name-match scores per chunk ID.
        alpha:              Semantic weight.
        beta:               Lexical weight.
        gamma:              Name-match weight.
        top_k:              Maximum results to return.
        importance_scores:  Inbound-degree boost per chunk ID.
        proximity_scores:   Diff-proximity boost per chunk ID.
    """
    if not results:
        return []

    imp = importance_scores or {}
    prox = proximity_scores or {}
    all_ids = list(results.keys())

    # Collect raw scores (0.0 for chunks absent from a channel).
    raw_lex = [lexical_scores.get(cid, 0.0) for cid in all_ids]
    raw_sem = [semantic_scores.get(cid, 0.0) for cid in all_ids]
    raw_name = [name_scores.get(cid, 0.0) for cid in all_ids]

    # Normalise each channel independently.
    norm_lex = normalise_scores(raw_lex)
    norm_sem = normalise_scores(raw_sem)
    norm_name = normalise_scores(raw_name)

    scored: list[ScoredResult] = []
    for i, cid in enumerate(all_ids):
        chunk = results[cid]
        kb = kind_boost(chunk.kind)
        fp = file_category_penalty(chunk.file_path)
        imp_s = imp.get(cid, 1.0)
        prox_s = prox.get(cid, 1.0)

        base = alpha * norm_sem[i] + beta * norm_lex[i] + gamma * norm_name[i]
        final = base * kb * fp * imp_s * prox_s

        scored.append(
            ScoredResult(
                chunk=chunk,
                score=final,
                lexical=norm_lex[i],
                semantic=norm_sem[i],
                name=norm_name[i],
                kind_boost=kb,
                file_penalty=fp,
                importance=imp_s,
                proximity=prox_s,
            )
        )

    scored.sort(key=lambda r: r.score, reverse=True)
    return scored[:top_k]
