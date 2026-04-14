"""Tests for unified search scoring and fusion.

Pure function tests first, then integration tests that verify
the full pipeline produces correct rankings.
"""

from __future__ import annotations

import pytest

from rbtr.index.models import Chunk, ChunkKind, Edge, EdgeKind
from rbtr.index.search import (
    DEFAULT_ALPHA,
    DEFAULT_BETA,
    DEFAULT_GAMMA,
    FileCategory,
    QueryKind,
    ScoredResult,
    classify_query,
    file_category,
    file_category_penalty,
    fuse_scores,
    importance_score,
    kind_boost,
    name_score,
    normalise_scores,
    proximity_score,
    weights_for_query,
)
from rbtr.index.store import IndexStore
from rbtr.index.tokenise import tokenise_code

# ── normalise_scores ─────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("scores", "expected"),
    [
        ([], []),
        ([5.0], [0.0]),
        ([1.0, 3.0], [0.0, 1.0]),
        ([3.0, 1.0], [1.0, 0.0]),
        ([1.0, 2.0, 3.0], [0.0, 0.5, 1.0]),
        # All same → no signal.
        ([7.0, 7.0, 7.0], [0.0, 0.0, 0.0]),
        # Negative values work.
        ([-1.0, 0.0, 1.0], [0.0, 0.5, 1.0]),
    ],
    ids=lambda v: str(v)[:40],
)
def test_normalise_scores(scores: list[float], expected: list[float]) -> None:
    result = normalise_scores(scores)
    assert len(result) == len(expected)
    for r, e in zip(result, expected, strict=True):
        assert r == pytest.approx(e)


# ── name_score ───────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("query", "name", "expected"),
    [
        # Exact match (case-insensitive).
        ("IndexStore", "IndexStore", 1.0),
        ("indexstore", "IndexStore", 1.0),
        ("INDEXSTORE", "IndexStore", 1.0),
        # Prefix match.
        ("Index", "IndexStore", 0.8),
        ("index", "IndexStore", 0.8),
        # Substring match.
        ("Store", "IndexStore", 0.5),
        ("store", "IndexStore", 0.5),
        # No match.
        ("xyz", "IndexStore", 0.0),
        ("store", "insert_chunks", 0.0),
        # Single character.
        ("I", "IndexStore", 0.8),
        ("e", "IndexStore", 0.5),  # 'e' is a substring of 'indexstore'
        # Token-level match: every query token is a substring of name.
        ("import edge", "infer_import_edges", 0.4),
        ("import edges", "infer_import_edges", 0.4),
        ("deep merge", "_deep_merge", 0.4),
        ("build model", "build_model", 0.4),  # token-level (space ≠ underscore)
        # Token-level: partial token match → 0.0.
        ("store chunk", "insert_chunks", 0.0),  # 'store' not in 'insert_chunks'
        ("import xyz", "infer_import_edges", 0.0),
        ("foo edge", "infer_import_edges", 0.0),
        # Single-word queries: substring check, not token-level.
        ("import", "infer_import_edges", 0.5),  # 'import' is a substring
    ],
    ids=lambda v: str(v)[:30],
)
def test_name_score(query: str, name: str, expected: float) -> None:
    assert name_score(query, name) == pytest.approx(expected)


# ── classify_query ───────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        # Identifiers: single token, may contain camelCase/snake_case/dots.
        ("IndexStore", QueryKind.IDENTIFIER),
        ("build_model", QueryKind.IDENTIFIER),
        ("getUserById", QueryKind.IDENTIFIER),
        ("_deep_merge", QueryKind.IDENTIFIER),
        ("Config", QueryKind.IDENTIFIER),
        ("a.b.c", QueryKind.IDENTIFIER),
        # Concepts: multi-word, no regex metacharacters.
        ("how does auth work", QueryKind.CONCEPT),
        ("database storage for session", QueryKind.CONCEPT),
        ("import edge", QueryKind.CONCEPT),
        ("store chunk", QueryKind.CONCEPT),
        ("class config", QueryKind.CONCEPT),
        # Patterns: regex metacharacters present.
        ("def test_", QueryKind.CONCEPT),  # no regex chars, just concept
        ("def test_.*", QueryKind.PATTERN),
        ("raise.*Error", QueryKind.PATTERN),
        ("TODO:", QueryKind.CONCEPT),  # colon is not a regex metachar
        ("foo(bar)", QueryKind.PATTERN),
        ("^import", QueryKind.PATTERN),
        ("config\\.toml", QueryKind.PATTERN),
    ],
    ids=lambda v: str(v)[:30],
)
def test_classify_query(query: str, expected: QueryKind) -> None:
    assert classify_query(query) == expected


def test_weights_for_query_sum_to_one() -> None:
    """All weight tuples are valid convex combinations."""
    for q in ["IndexStore", "how does auth work", "raise.*Error"]:
        a, b, g = weights_for_query(q)
        assert pytest.approx(1.0) == a + b + g


def test_weights_identifier_favours_name() -> None:
    """Identifier queries assign highest weight to name match."""
    a, b, g = weights_for_query("IndexStore")
    assert g > a
    assert g > b


def test_weights_concept_favours_name() -> None:
    """Concept queries assign highest weight to name match."""
    a, b, g = weights_for_query("how does auth work")
    assert g >= a
    assert g >= b


# ── kind_boost ───────────────────────────────────────────────────────


def test_kind_boost_covers_all_kinds() -> None:
    """Every ChunkKind has an explicit boost value — no missing kinds."""
    for kind in ChunkKind:
        boost = kind_boost(kind)
        assert isinstance(boost, float)
        assert boost > 0.0, f"{kind} has zero or negative boost"


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        (ChunkKind.CLASS, 1.5),
        (ChunkKind.FUNCTION, 1.3),
        (ChunkKind.METHOD, 1.3),
        (ChunkKind.TEST_FUNCTION, 0.7),
        (ChunkKind.IMPORT, 0.3),
        (ChunkKind.RAW_CHUNK, 0.5),
        (ChunkKind.DOC_SECTION, 0.8),
        (ChunkKind.CONFIG_KEY, 0.6),
        (ChunkKind.VARIABLE, 1.0),
        (ChunkKind.MIGRATION, 0.4),
        (ChunkKind.API_ENDPOINT, 1.0),
    ],
)
def test_kind_boost_values(kind: ChunkKind, expected: float) -> None:
    assert kind_boost(kind) == pytest.approx(expected)


# ── file_category ────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        # Source files.
        ("src/rbtr/index/store.py", FileCategory.SOURCE),
        ("lib/main.go", FileCategory.SOURCE),
        ("app/models/user.rb", FileCategory.SOURCE),
        # Test files.
        ("src/tests/index/test_store.py", FileCategory.TEST),
        ("tests/test_handler.py", FileCategory.TEST),
        ("src/rbtr/store_test.go", FileCategory.TEST),
        ("src/tests/conftest.py", FileCategory.TEST),
        # Vendor.
        ("vendor/lib/foo.py", FileCategory.VENDOR),
        ("node_modules/react/index.js", FileCategory.VENDOR),
        ("third_party/proto/api.proto", FileCategory.VENDOR),
        # Generated.
        ("api/models_pb2.py", FileCategory.GENERATED),
        ("proto/service.pb.go", FileCategory.GENERATED),
        # Docs.
        ("docs/guide.md", FileCategory.DOC),
        ("README.md", FileCategory.DOC),
        ("CHANGELOG.rst", FileCategory.DOC),
        # Config.
        ("pyproject.toml", FileCategory.CONFIG),
        (".github/workflows/ci.yaml", FileCategory.CONFIG),
        ("package.json", FileCategory.CONFIG),
        ("config/settings.yml", FileCategory.CONFIG),
    ],
    ids=lambda v: str(v)[:40],
)
def test_file_category(path: str, expected: FileCategory) -> None:
    assert file_category(path) == expected


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("src/rbtr/index/store.py", 1.0),
        ("src/tests/index/test_store.py", 0.5),
        ("vendor/lib/foo.py", 0.3),
        ("docs/guide.md", 0.8),
        ("pyproject.toml", 0.7),
    ],
)
def test_file_category_penalty(path: str, expected: float) -> None:
    assert file_category_penalty(path) == pytest.approx(expected)


# ── importance_score ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("degree", "expected"),
    [
        (0, 1.0),
        (1, pytest.approx(1.25, abs=0.01)),
        (3, pytest.approx(1.50, abs=0.01)),
        (7, pytest.approx(1.75, abs=0.01)),
        (15, pytest.approx(2.00, abs=0.01)),
        (1000, 3.0),  # capped
    ],
    ids=lambda v: str(v),
)
def test_importance_score(degree: int, expected: float) -> None:
    assert importance_score(degree) == expected


def test_importance_score_monotonic() -> None:
    """Higher degree always gives higher or equal score."""
    prev = importance_score(0)
    for d in range(1, 50):
        curr = importance_score(d)
        assert curr >= prev
        prev = curr


# ── proximity_score ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("chunk_file", "changed", "has_edge", "expected"),
    [
        # In changed file.
        ("src/store.py", {"src/store.py"}, False, 1.5),
        # Has edge to changed file.
        ("src/orchestrator.py", {"src/store.py"}, True, 1.2),
        # Same directory, no edge.
        ("src/models.py", {"src/store.py"}, False, 1.1),
        # Different directory, no edge.
        ("cli/main.py", {"src/store.py"}, False, 1.0),
        # Empty changed set.
        ("src/store.py", set(), False, 1.0),
        # Root-level files — not considered "same directory".
        ("README.md", {"setup.py"}, False, 1.0),
    ],
    ids=lambda v: str(v)[:30],
)
def test_proximity_score(
    chunk_file: str,
    changed: set[str],
    has_edge: bool,
    expected: float,
) -> None:
    assert proximity_score(chunk_file, changed, has_edge) == expected


# ── fuse_scores ──────────────────────────────────────────────────────

# Helpers to create minimal chunks for fusion tests.


def _chunk(
    chunk_id: str,
    *,
    kind: ChunkKind = ChunkKind.FUNCTION,
    name: str = "fn",
    file_path: str = "src/lib.py",
) -> Chunk:
    return Chunk(
        id=chunk_id,
        blob_sha="blob",
        file_path=file_path,
        kind=kind,
        name=name,
        content="",
        line_start=1,
        line_end=1,
    )


def test_fuse_empty() -> None:
    """Empty input produces empty output."""
    assert fuse_scores({}, {}, {}, {}) == []


def test_fuse_single_result() -> None:
    """Single result gets a score based on its signals."""
    c = _chunk("a")
    result = fuse_scores(
        {"a": c},
        lexical_scores={"a": 1.0},
        semantic_scores={"a": 1.0},
        name_scores={"a": 1.0},
    )
    assert len(result) == 1
    assert result[0].chunk.id == "a"
    # Single item normalises to 0.0 on all channels → score = 0.
    assert result[0].score == pytest.approx(0.0)


def test_fuse_two_results_lexical_only() -> None:
    """With only lexical signal, the higher-scored chunk ranks first."""
    c1 = _chunk("a")
    c2 = _chunk("b")
    result = fuse_scores(
        {"a": c1, "b": c2},
        lexical_scores={"a": 1.0, "b": 5.0},
        semantic_scores={},
        name_scores={},
        alpha=0.0,
        beta=1.0,
        gamma=0.0,
    )
    assert result[0].chunk.id == "b"
    assert result[1].chunk.id == "a"


def test_fuse_name_match_beats_lexical() -> None:
    """Exact name match outranks higher lexical score."""
    source_def = _chunk("def", kind=ChunkKind.FUNCTION, name="IndexStore")
    test_mention = _chunk(
        "test",
        kind=ChunkKind.FUNCTION,
        name="test_search",
        file_path="tests/test_store.py",
    )
    result = fuse_scores(
        {"def": source_def, "test": test_mention},
        lexical_scores={"def": 1.0, "test": 10.0},
        semantic_scores={},
        name_scores={"def": 1.0, "test": 0.0},
    )
    assert result[0].chunk.id == "def"


def test_fuse_class_outranks_import() -> None:
    """A class definition outranks an import of the same name."""
    cls = _chunk("cls", kind=ChunkKind.CLASS, name="Engine")
    imp = _chunk("imp", kind=ChunkKind.IMPORT, name="from .core import Engine")
    result = fuse_scores(
        {"cls": cls, "imp": imp},
        lexical_scores={"cls": 1.0, "imp": 1.0},
        semantic_scores={},
        name_scores={"cls": 0.5, "imp": 0.5},
    )
    # Both have identical lexical and name scores, but CLASS
    # boost (1.5) beats IMPORT boost (0.3).
    assert result[0].chunk.id == "cls"


def test_fuse_source_outranks_test() -> None:
    """A source function outranks a test function with higher TF."""
    source = _chunk("src", name="build_index", file_path="src/orchestrator.py")
    test = _chunk("tst", name="test_build_index", file_path="tests/test_orchestrator.py")
    result = fuse_scores(
        {"src": source, "tst": test},
        lexical_scores={"src": 1.0, "tst": 44.0},  # test has 44x TF
        semantic_scores={},
        name_scores={"src": 0.5, "tst": 0.0},
    )
    assert result[0].chunk.id == "src"


def test_fuse_score_breakdown_populated() -> None:
    """Every ScoredResult has all signal fields populated."""
    c = _chunk("a", kind=ChunkKind.CLASS, file_path="src/lib.py")
    c2 = _chunk("b", kind=ChunkKind.IMPORT, file_path="tests/test.py")
    results = fuse_scores(
        {"a": c, "b": c2},
        lexical_scores={"a": 2.0, "b": 1.0},
        semantic_scores={"a": 0.8, "b": 0.2},
        name_scores={"a": 1.0, "b": 0.0},
    )
    for r in results:
        assert isinstance(r, ScoredResult)
        assert r.kind_boost > 0.0
        assert r.file_penalty > 0.0
        assert r.score >= 0.0


def test_fuse_top_k_limits_output() -> None:
    """fuse_scores respects the top_k parameter."""
    chunks = {f"c{i}": _chunk(f"c{i}") for i in range(20)}
    lex = {f"c{i}": float(i) for i in range(20)}
    result = fuse_scores(chunks, lex, {}, {}, top_k=5)
    assert len(result) == 5


def test_fuse_weights_sum_to_one() -> None:
    """Default fusion weights are a valid convex combination."""
    assert pytest.approx(1.0) == DEFAULT_ALPHA + DEFAULT_BETA + DEFAULT_GAMMA


def test_fuse_missing_channels() -> None:
    """Chunks absent from a channel get 0.0 raw score, not an error."""
    c1 = _chunk("a")
    c2 = _chunk("b")
    # c1 only in lexical, c2 only in semantic.
    result = fuse_scores(
        {"a": c1, "b": c2},
        lexical_scores={"a": 5.0},
        semantic_scores={"b": 0.9},
        name_scores={},
    )
    assert len(result) == 2
    # Both should have some score.
    ids = [r.chunk.id for r in result]
    assert "a" in ids
    assert "b" in ids


# ── Integration: store.search() ──────────────────────────────────────
#
# End-to-end tests through IndexStore.search() verifying that the
# fusion pipeline produces correct rankings with real FTS indexes.


def _make_chunk(
    chunk_id: str,
    *,
    kind: ChunkKind = ChunkKind.FUNCTION,
    name: str = "fn",
    file_path: str = "src/lib.py",
    content: str = "pass",
) -> Chunk:
    return Chunk(
        id=chunk_id,
        blob_sha=f"blob_{chunk_id}",
        file_path=file_path,
        kind=kind,
        name=name,
        content=content,
        content_tokens=tokenise_code(content),
        name_tokens=tokenise_code(name),
        line_start=1,
        line_end=1,
    )


def _seed(store: IndexStore, chunks: list[Chunk], commit: str = "head") -> None:
    """Insert chunks and snapshots into the store."""
    store.insert_chunks(chunks)
    for c in chunks:
        store.insert_snapshot(commit, c.file_path, c.blob_sha)


def test_store_search_class_outranks_import() -> None:
    """store.search() ranks a class definition above an import."""
    store = IndexStore()
    cls = _make_chunk(
        "cls",
        kind=ChunkKind.CLASS,
        name="Engine",
        file_path="src/core.py",
        content="class Engine:\n    pass",
    )
    imp = _make_chunk(
        "imp",
        kind=ChunkKind.IMPORT,
        name="from .core import Engine",
        file_path="src/tools.py",
        content="from .core import Engine",
    )
    _seed(store, [cls, imp])

    results = store.search("head", "Engine", top_k=5)
    assert len(results) >= 2
    assert results[0].chunk.id == "cls"


def test_store_search_source_outranks_test() -> None:
    """store.search() ranks source above test despite higher test TF."""
    store = IndexStore()
    source = _make_chunk(
        "src",
        name="build_index",
        file_path="src/orchestrator.py",
        content="def build_index(repo): pass",
    )
    test = _make_chunk(
        "tst",
        name="test_build_index",
        file_path="tests/test_orchestrator.py",
        content="build_index() " * 20,  # 20x more mentions
    )
    _seed(store, [source, test])

    results = store.search("head", "build_index", top_k=5)
    assert len(results) >= 2
    assert results[0].chunk.id == "src"


def test_store_search_exact_name_ranks_first() -> None:
    """store.search() ranks an exact name match first."""
    store = IndexStore()
    exact = _make_chunk(
        "exact",
        name="_embed_missing",
        file_path="src/orchestrator.py",
        content="def _embed_missing(): pass",
    )
    partial = _make_chunk(
        "partial",
        name="embed_text",
        file_path="src/embeddings.py",
        content="def embed_text(text): return embed(text) if not missing else None",
    )
    _seed(store, [exact, partial])

    results = store.search("head", "_embed_missing", top_k=5)
    assert len(results) >= 1
    assert results[0].chunk.id == "exact"


def test_store_search_returns_scored_results() -> None:
    """store.search() returns ScoredResult with signal breakdown."""
    store = IndexStore()
    c = _make_chunk("a", name="parse_config", content="def parse_config(): pass")
    _seed(store, [c])

    results = store.search("head", "parse_config", top_k=5)
    assert len(results) >= 1
    r = results[0]
    assert r.chunk.id == "a"
    assert r.score >= 0.0
    assert r.kind_boost > 0.0
    assert r.file_penalty > 0.0


def test_store_search_no_results() -> None:
    """store.search() returns empty list for gibberish query."""
    store = IndexStore()
    c = _make_chunk("a", content="def real_function(): pass")
    _seed(store, [c])

    results = store.search("head", "zzz_nonexistent_xyz", top_k=5)
    assert results == []


def test_store_search_graceful_without_embeddings() -> None:
    """store.search() works when embeddings are unavailable."""
    store = IndexStore()
    c = _make_chunk("a", name="handle_request", content="async def handle_request(): pass")
    _seed(store, [c])

    # No embeddings loaded — semantic channel should be skipped.
    results = store.search("head", "handle_request", top_k=5)
    assert len(results) >= 1
    assert results[0].chunk.id == "a"


def test_store_search_importance_boosts_central_chunk() -> None:
    """Chunk with more inbound edges ranks higher than an isolated one."""

    store = IndexStore()
    # Two classes with same name match quality.
    central = _make_chunk(
        "central",
        kind=ChunkKind.CLASS,
        name="Config",
        file_path="src/config.py",
        content="class Config: pass",
    )
    peripheral = _make_chunk(
        "peripheral",
        kind=ChunkKind.CLASS,
        name="ConfigHelper",
        file_path="src/helper.py",
        content="class ConfigHelper(Config): pass",
    )
    # Several chunks that import central but not peripheral.
    importers: list[Chunk] = []
    for i in range(5):
        imp = _make_chunk(
            f"imp{i}",
            kind=ChunkKind.IMPORT,
            name="from config import Config",
            file_path=f"src/mod{i}.py",
            content="from config import Config",
        )
        importers.append(imp)

    _seed(store, [central, peripheral, *importers])

    # Add edges: each importer → central.
    edges = [
        Edge(source_id=f"imp{i}", target_id="central", kind=EdgeKind.IMPORTS) for i in range(5)
    ]
    store.insert_edges(edges, "head")

    results = store.search("head", "Config", top_k=5)
    # Central should outrank peripheral due to importance boost.
    code_results = [r for r in results if r.chunk.kind == ChunkKind.CLASS]
    assert len(code_results) >= 2
    assert code_results[0].chunk.id == "central"
    assert code_results[0].importance > code_results[1].importance


def test_store_search_proximity_boosts_changed_file() -> None:
    """Chunks in changed files rank higher with proximity boost."""
    store = IndexStore()
    changed = _make_chunk(
        "changed",
        name="do_work",
        file_path="src/store.py",
        content="def do_work(data): return process(data)",
    )
    other = _make_chunk(
        "other",
        name="do_work_helper",
        file_path="lib/utils.py",
        content="def do_work_helper(items): return transform(items)",
    )
    _seed(store, [changed, other])

    # With changed_files: the one in the changed file ranks higher.
    results_diff = store.search("head", "do_work", top_k=5, changed_files={"src/store.py"})
    assert len(results_diff) >= 2
    assert results_diff[0].chunk.id == "changed"
    assert results_diff[0].proximity > results_diff[1].proximity


def test_store_search_importance_field_populated() -> None:
    """ScoredResult includes importance and proximity fields."""
    store = IndexStore()
    c = _make_chunk("a", name="parse", content="def parse(): pass")
    _seed(store, [c])

    results = store.search("head", "parse", top_k=5)
    assert len(results) >= 1
    r = results[0]
    assert r.importance >= 1.0
    assert r.proximity >= 1.0
