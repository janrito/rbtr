"""Tests for engine/tools.py — LLM tool functions.

Uses a data-first approach: realistic, semantically distinct code
chunks with meaningful content so we can verify ranking, edge
traversal, and output correctness — not just "does it return something".
"""

from __future__ import annotations

import tempfile

import pygit2

from rbtr.engine.session import Session
from rbtr.engine.tools import (
    commit_log,
    detect_language,
    diff,
    get_blast_radius,
    get_callers,
    get_dependents,
    list_indexed_files,
    read_symbol,
    search_codebase,
    search_similar,
    search_symbols,
    semantic_diff,
)
from rbtr.index.models import Chunk, ChunkKind, Edge, EdgeKind
from rbtr.index.store import IndexStore
from rbtr.models import BranchTarget

# ── Helpers ──────────────────────────────────────────────────────────


class _FakeCtx:
    """Minimal stand-in for RunContext[AgentDeps] in tool tests."""

    def __init__(self, session: Session) -> None:
        self.deps = _FakeDeps(session)


class _FakeDeps:
    def __init__(self, session: Session) -> None:
        self.session = session


# ── Shared test data ─────────────────────────────────────────────────
#
# A small but realistic codebase: an HTTP handler module, a math
# utilities module, a test file, and a README doc section.
# Edges model real relationships: test→code, import, doc→code.

_HANDLER = Chunk(
    id="handler_1",
    blob_sha="blob_handler",
    file_path="src/api/handler.py",
    kind=ChunkKind.FUNCTION,
    name="handle_request",
    content="""\
async def handle_request(request: Request) -> Response:
    data = await request.json()
    result = process_data(data)
    return Response(status=200, body=result)
""",
    line_start=10,
    line_end=14,
)

_PROCESS = Chunk(
    id="process_1",
    blob_sha="blob_handler",
    file_path="src/api/handler.py",
    kind=ChunkKind.FUNCTION,
    name="process_data",
    scope="",
    content="""\
def process_data(data: dict) -> dict:
    validated = validate_schema(data)
    return transform(validated)
""",
    line_start=20,
    line_end=23,
)

_MATH_MEAN = Chunk(
    id="math_mean_1",
    blob_sha="blob_math",
    file_path="src/stats/math_utils.py",
    kind=ChunkKind.FUNCTION,
    name="calculate_mean",
    content="""\
def calculate_mean(values: list[float]) -> float:
    if not values:
        raise ValueError("empty list")
    return sum(values) / len(values)
""",
    line_start=1,
    line_end=4,
)

_MATH_STDDEV = Chunk(
    id="math_stddev_1",
    blob_sha="blob_math",
    file_path="src/stats/math_utils.py",
    kind=ChunkKind.FUNCTION,
    name="calculate_standard_deviation",
    content="""\
def calculate_standard_deviation(values: list[float]) -> float:
    mean = calculate_mean(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return variance ** 0.5
""",
    line_start=8,
    line_end=12,
)

_STATS_CLASS = Chunk(
    id="stats_class_1",
    blob_sha="blob_math",
    file_path="src/stats/math_utils.py",
    kind=ChunkKind.CLASS,
    name="StatisticsCalculator",
    content="""\
class StatisticsCalculator:
    def __init__(self, data: list[float]):
        self.data = data
    def mean(self) -> float:
        return calculate_mean(self.data)
""",
    line_start=16,
    line_end=21,
)

_TEST_HANDLER = Chunk(
    id="test_handler_1",
    blob_sha="blob_test",
    file_path="tests/test_handler.py",
    kind=ChunkKind.FUNCTION,
    name="test_handle_request",
    content="""\
async def test_handle_request():
    response = await handle_request(fake_request)
    assert response.status == 200
""",
    line_start=1,
    line_end=3,
)

_TEST_MATH = Chunk(
    id="test_math_1",
    blob_sha="blob_test_math",
    file_path="tests/test_math.py",
    kind=ChunkKind.FUNCTION,
    name="test_calculate_mean",
    content="""\
def test_calculate_mean():
    assert calculate_mean([1, 2, 3]) == 2.0
""",
    line_start=1,
    line_end=2,
)

_DOC_SECTION = Chunk(
    id="doc_api_1",
    blob_sha="blob_readme",
    file_path="docs/api.md",
    kind=ChunkKind.DOC_SECTION,
    name="API Reference",
    content="""\
## API Reference

The `handle_request` function processes incoming HTTP requests
and delegates to `process_data` for validation.
""",
    line_start=1,
    line_end=5,
)

ALL_CHUNKS = [
    _HANDLER,
    _PROCESS,
    _MATH_MEAN,
    _MATH_STDDEV,
    _STATS_CLASS,
    _TEST_HANDLER,
    _TEST_MATH,
    _DOC_SECTION,
]

# Embedding vectors — orthogonal axes for clean ranking.
_VEC_HTTP = [1.0, 0.0, 0.0]
_VEC_MATH = [0.0, 1.0, 0.0]
_VEC_DOC = [0.0, 0.0, 1.0]

ALL_EDGES = [
    Edge(source_id="test_handler_1", target_id="handler_1", kind=EdgeKind.TESTS),
    Edge(source_id="test_math_1", target_id="math_mean_1", kind=EdgeKind.TESTS),
    Edge(source_id="process_1", target_id="handler_1", kind=EdgeKind.IMPORTS),
    Edge(source_id="doc_api_1", target_id="handler_1", kind=EdgeKind.DOCUMENTS),
    Edge(source_id="math_stddev_1", target_id="math_mean_1", kind=EdgeKind.IMPORTS),
]


def _make_session(*, embed: bool = False) -> tuple[Session, IndexStore]:
    """Build an in-memory store with the full test dataset."""
    store = IndexStore()
    session = Session()
    session.index = store
    session.review_target = BranchTarget(
        base_branch="main",
        head_branch="feature",
        updated_at=0,
    )

    store.insert_chunks(ALL_CHUNKS)
    for c in ALL_CHUNKS:
        store.insert_snapshot("feature", c.file_path, c.blob_sha)
    store.insert_edges(ALL_EDGES, "feature")
    store.rebuild_fts_index()

    if embed:
        store.update_embedding("handler_1", _VEC_HTTP)
        store.update_embedding("process_1", _VEC_HTTP)
        store.update_embedding("math_mean_1", _VEC_MATH)
        store.update_embedding("math_stddev_1", _VEC_MATH)
        store.update_embedding("stats_class_1", _VEC_MATH)
        store.update_embedding("test_handler_1", _VEC_HTTP)
        store.update_embedding("test_math_1", _VEC_MATH)
        store.update_embedding("doc_api_1", _VEC_DOC)

    return session, store


# ── search_symbols ───────────────────────────────────────────────────


def test_search_symbols_finds_exact_name() -> None:
    session, store = _make_session()
    ctx = _FakeCtx(session)
    result = search_symbols(ctx, "handle_request")  # type: ignore[arg-type]
    assert "handle_request" in result
    assert "src/api/handler.py:10" in result
    store.close()


def test_search_symbols_finds_partial_name() -> None:
    session, store = _make_session()
    ctx = _FakeCtx(session)
    result = search_symbols(ctx, "calculate")  # type: ignore[arg-type]
    # Should find both math functions.
    assert "calculate_mean" in result
    assert "calculate_standard_deviation" in result
    store.close()


def test_search_symbols_no_match() -> None:
    session, store = _make_session()
    ctx = _FakeCtx(session)
    result = search_symbols(ctx, "zzz_nonexistent_zzz")  # type: ignore[arg-type]
    assert "No symbols" in result
    store.close()


# ── search_codebase (BM25) ───────────────────────────────────────────


def test_search_codebase_ranks_by_relevance() -> None:
    """Searching 'variance' should rank math_stddev highest."""
    session, store = _make_session()
    ctx = _FakeCtx(session)
    result = search_codebase(ctx, "variance")  # type: ignore[arg-type]
    lines = result.strip().split("\n")
    # First result should be the stddev function (contains "variance").
    assert "calculate_standard_deviation" in lines[0]
    store.close()


def test_search_codebase_finds_http_content() -> None:
    session, store = _make_session()
    ctx = _FakeCtx(session)
    result = search_codebase(ctx, "Response status")  # type: ignore[arg-type]
    assert "handle_request" in result
    store.close()


def test_search_codebase_no_match() -> None:
    session, store = _make_session()
    ctx = _FakeCtx(session)
    result = search_codebase(ctx, "zzz_gibberish_xyz_999")  # type: ignore[arg-type]
    assert "No results" in result
    store.close()


# ── search_similar (embedding) ───────────────────────────────────────


def test_search_similar_ranks_math_first(mocker) -> None:
    """Query near the math axis ranks math chunks above HTTP."""
    session, store = _make_session(embed=True)
    mocker.patch("rbtr.index.embeddings.embed_text", return_value=[0.1, 0.9, 0.0])
    ctx = _FakeCtx(session)
    result = search_similar(ctx, "statistics calculations")  # type: ignore[arg-type]

    lines = result.strip().split("\n")
    # First results should be math-related (mean, stddev, StatisticsCalculator).
    top_three = " ".join(lines[:3])
    assert "calculate_mean" in top_three or "calculate_standard_deviation" in top_three
    # HTTP handler should not be in the top results.
    assert "handle_request" not in top_three
    store.close()


def test_search_similar_ranks_http_first(mocker) -> None:
    """Query near the HTTP axis ranks HTTP chunks above math."""
    session, store = _make_session(embed=True)
    mocker.patch("rbtr.index.embeddings.embed_text", return_value=[0.9, 0.1, 0.0])
    ctx = _FakeCtx(session)
    result = search_similar(ctx, "web request handling")  # type: ignore[arg-type]

    lines = result.strip().split("\n")
    top_two = " ".join(lines[:2])
    assert "handle_request" in top_two or "process_data" in top_two
    store.close()


def test_search_similar_no_embeddings(mocker) -> None:
    """Store with no embeddings returns no results."""
    session, store = _make_session(embed=False)
    mocker.patch("rbtr.index.embeddings.embed_text", return_value=[1.0, 0.0, 0.0])
    ctx = _FakeCtx(session)
    result = search_similar(ctx, "anything")  # type: ignore[arg-type]
    assert "No similar" in result
    store.close()


# ── get_dependents ───────────────────────────────────────────────────


def test_get_dependents_finds_importers() -> None:
    """process_data imports handle_request → shows as dependent."""
    session, store = _make_session()
    ctx = _FakeCtx(session)
    result = get_dependents(ctx, "handle_request")  # type: ignore[arg-type]
    assert "process_data" in result
    store.close()


def test_get_dependents_finds_stddev_imports_mean() -> None:
    """calculate_standard_deviation imports calculate_mean."""
    session, store = _make_session()
    ctx = _FakeCtx(session)
    result = get_dependents(ctx, "calculate_mean")  # type: ignore[arg-type]
    assert "calculate_standard_deviation" in result
    store.close()


def test_get_dependents_no_importers() -> None:
    """calculate_standard_deviation has no inbound import edges."""
    session, store = _make_session()
    ctx = _FakeCtx(session)
    result = get_dependents(ctx, "calculate_standard_deviation")  # type: ignore[arg-type]
    assert "No dependents" in result
    store.close()


def test_get_dependents_unknown_symbol() -> None:
    session, store = _make_session()
    ctx = _FakeCtx(session)
    result = get_dependents(ctx, "zzz_nonexistent")  # type: ignore[arg-type]
    assert "not found" in result
    store.close()


# ── get_callers ──────────────────────────────────────────────────────


def test_get_callers_finds_tests_and_docs() -> None:
    """handle_request has both a test and a doc reference."""
    session, store = _make_session()
    ctx = _FakeCtx(session)
    result = get_callers(ctx, "handle_request")  # type: ignore[arg-type]
    assert "[tests]" in result
    assert "test_handle_request" in result
    assert "[documents]" in result
    assert "API Reference" in result
    store.close()


def test_get_callers_test_only() -> None:
    """calculate_mean has a test but no doc reference."""
    session, store = _make_session()
    ctx = _FakeCtx(session)
    result = get_callers(ctx, "calculate_mean")  # type: ignore[arg-type]
    assert "test_calculate_mean" in result
    assert "[documents]" not in result
    store.close()


def test_get_callers_none() -> None:
    """StatisticsCalculator has no test or doc edges."""
    session, store = _make_session()
    ctx = _FakeCtx(session)
    result = get_callers(ctx, "StatisticsCalculator")  # type: ignore[arg-type]
    assert "No tests or docs" in result
    store.close()


# ── get_blast_radius ─────────────────────────────────────────────────


def test_blast_radius_handler_file() -> None:
    """handler.py has inbound test, import, and doc edges."""
    session, store = _make_session()
    ctx = _FakeCtx(session)
    result = get_blast_radius(ctx, "src/api/handler.py")  # type: ignore[arg-type]
    assert "test_handle_request" in result
    assert "process_data" in result
    assert "API Reference" in result
    store.close()


def test_blast_radius_math_file() -> None:
    """math_utils.py has inbound test and import edges."""
    session, store = _make_session()
    ctx = _FakeCtx(session)
    result = get_blast_radius(ctx, "src/stats/math_utils.py")  # type: ignore[arg-type]
    assert "test_calculate_mean" in result
    assert "calculate_standard_deviation" in result
    store.close()


def test_blast_radius_test_file() -> None:
    """Test files have no inbound edges."""
    session, store = _make_session()
    ctx = _FakeCtx(session)
    result = get_blast_radius(ctx, "tests/test_handler.py")  # type: ignore[arg-type]
    assert "Nothing depends" in result
    store.close()


def test_blast_radius_unknown_file() -> None:
    session, store = _make_session()
    ctx = _FakeCtx(session)
    result = get_blast_radius(ctx, "nonexistent.py")  # type: ignore[arg-type]
    assert "No indexed symbols" in result
    store.close()


# ── read_symbol ──────────────────────────────────────────────────────


def test_read_symbol_returns_full_source() -> None:
    session, store = _make_session()
    ctx = _FakeCtx(session)
    result = read_symbol(ctx, "handle_request")  # type: ignore[arg-type]
    assert "async def handle_request" in result
    assert "Response(status=200" in result
    assert "src/api/handler.py:10-14" in result
    store.close()


def test_read_symbol_prefers_code_over_tests() -> None:
    """When both a function and its test match, return the function."""
    session, store = _make_session()
    ctx = _FakeCtx(session)
    result = read_symbol(ctx, "handle_request")  # type: ignore[arg-type]
    # Should be the function, not the test.
    assert "src/api/handler.py" in result
    store.close()


def test_read_symbol_not_found() -> None:
    session, store = _make_session()
    ctx = _FakeCtx(session)
    result = read_symbol(ctx, "zzz_nonexistent")  # type: ignore[arg-type]
    assert "No symbol" in result
    store.close()


# ── list_indexed_files ────────────────────────────────────────────────


def test_list_indexed_files_all() -> None:
    session, store = _make_session()
    ctx = _FakeCtx(session)
    result = list_indexed_files(ctx, "")  # type: ignore[arg-type]
    assert "docs/api.md" in result
    assert "src/api/handler.py" in result
    assert "src/stats/math_utils.py" in result
    assert "tests/test_handler.py" in result
    store.close()


def test_list_indexed_files_filtered() -> None:
    session, store = _make_session()
    ctx = _FakeCtx(session)
    result = list_indexed_files(ctx, "math")  # type: ignore[arg-type]
    assert "math_utils.py" in result
    assert "test_math.py" in result
    assert "handler.py" not in result
    store.close()


def test_list_indexed_files_no_match() -> None:
    session, store = _make_session()
    ctx = _FakeCtx(session)
    result = list_indexed_files(ctx, "nonexistent_xyz")  # type: ignore[arg-type]
    assert "No indexed files" in result
    store.close()


def test_list_indexed_files_shows_symbol_count() -> None:
    session, store = _make_session()
    ctx = _FakeCtx(session)
    result = list_indexed_files(ctx, "handler.py")  # type: ignore[arg-type]
    # handler.py has 2 symbols: handle_request + process_data
    assert "2 symbols" in result
    store.close()


# ── detect_language ──────────────────────────────────────────────────


def test_detect_language_python() -> None:
    session, store = _make_session()
    ctx = _FakeCtx(session)
    assert detect_language(ctx, "src/app.py") == "python"  # type: ignore[arg-type]
    store.close()


def test_detect_language_unknown() -> None:
    session, store = _make_session()
    ctx = _FakeCtx(session)
    result = detect_language(ctx, "mystery.xyz")  # type: ignore[arg-type]
    assert "Unknown" in result
    store.close()


# ── Git tool helpers ─────────────────────────────────────────────────


def _make_repo_two_commits(tmp: str) -> tuple[pygit2.Repository, str, str]:
    """Create a repo with two commits: initial a.py, then modify a.py + add b.py."""
    repo = pygit2.init_repository(tmp)
    sig = pygit2.Signature("Test", "test@test.com")

    b1 = repo.create_blob(b"x = 1\n")
    tb1 = repo.TreeBuilder()
    tb1.insert("a.py", b1, pygit2.GIT_FILEMODE_BLOB)
    c1 = repo.create_commit("refs/heads/main", sig, sig, "initial", tb1.write(), [])
    repo.set_head("refs/heads/main")

    b2 = repo.create_blob(b"x = 2\ny = 3\n")
    b3 = repo.create_blob(b"def helper():\n    pass\n")
    tb2 = repo.TreeBuilder()
    tb2.insert("a.py", b2, pygit2.GIT_FILEMODE_BLOB)
    tb2.insert("b.py", b3, pygit2.GIT_FILEMODE_BLOB)
    c2 = repo.create_commit("refs/heads/feature", sig, sig, "add b and change a", tb2.write(), [c1])

    return repo, str(c1), str(c2)


def _git_session(repo: pygit2.Repository) -> Session:
    session = Session(repo=repo, owner="o", repo_name="r")
    session.review_target = BranchTarget(
        base_branch="main",
        head_branch="feature",
        updated_at=0,
    )
    return session


# ── diff ─────────────────────────────────────────────────────────────


def test_diff_shows_both_changed_files() -> None:
    """Default diff (base → head) shows both a.py and b.py changes."""
    with tempfile.TemporaryDirectory() as tmp:
        repo, _, _ = _make_repo_two_commits(tmp)
        ctx = _FakeCtx(_git_session(repo))
        result = diff(ctx, "")  # type: ignore[arg-type]
        assert "a.py" in result
        assert "b.py" in result
        assert "files changed" in result


def test_diff_single_ref_shows_commit() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        repo, _, c2 = _make_repo_two_commits(tmp)
        ctx = _FakeCtx(_git_session(repo))
        result = diff(ctx, c2[:8])  # type: ignore[arg-type]
        assert "files changed" in result
        assert "a.py" in result


def test_diff_range_syntax() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        repo, c1, c2 = _make_repo_two_commits(tmp)
        ctx = _FakeCtx(_git_session(repo))
        result = diff(ctx, f"{c1[:8]}..{c2[:8]}")  # type: ignore[arg-type]
        assert "files changed" in result


def test_diff_bad_ref() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        repo, _, _ = _make_repo_two_commits(tmp)
        ctx = _FakeCtx(_git_session(repo))
        result = diff(ctx, "nonexistent")  # type: ignore[arg-type]
        assert "not found" in result


def test_diff_no_target() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        repo, _, _ = _make_repo_two_commits(tmp)
        session = Session(repo=repo, owner="o", repo_name="r")
        ctx = _FakeCtx(session)
        result = diff(ctx, "")  # type: ignore[arg-type]
        assert "No review target" in result


# ── commit_log ───────────────────────────────────────────────────────


def test_commit_log_shows_message() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        repo, _, _ = _make_repo_two_commits(tmp)
        ctx = _FakeCtx(_git_session(repo))
        result = commit_log(ctx)  # type: ignore[arg-type]
        assert "add b and change a" in result


def test_commit_log_identical_branches() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        repo = pygit2.init_repository(tmp)
        sig = pygit2.Signature("T", "t@t.com")
        tree = repo.TreeBuilder().write()
        repo.create_commit("refs/heads/main", sig, sig, "init", tree, [])
        repo.set_head("refs/heads/main")
        repo.branches.local.create("feature", repo.head.peel(pygit2.Commit))

        ctx = _FakeCtx(_git_session(repo))
        result = commit_log(ctx)  # type: ignore[arg-type]
        assert "No commits" in result or "identical" in result.lower()


def test_commit_log_no_target() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        repo = pygit2.init_repository(tmp)
        sig = pygit2.Signature("T", "t@t.com")
        tree = repo.TreeBuilder().write()
        repo.create_commit("refs/heads/main", sig, sig, "init", tree, [])
        repo.set_head("refs/heads/main")

        session = Session(repo=repo, owner="o", repo_name="r")
        ctx = _FakeCtx(session)
        result = commit_log(ctx)  # type: ignore[arg-type]
        assert "No review target" in result


# ── semantic_diff ────────────────────────────────────────────────────


def _make_diff_session() -> tuple[Session, IndexStore]:
    """Build a store with base (v1) and head (v2) for semantic diff tests.

    Base has: handler (v1), calculate_mean.
    Head has: handler (v2, modified), calculate_mean (unchanged),
              new_endpoint (added). handler's test exists but
              new_endpoint has no test.
    """
    store = IndexStore()
    session = Session()
    session.index = store
    session.review_target = BranchTarget(
        base_branch="main",
        head_branch="feature",
        updated_at=0,
    )

    # Base versions.
    base_handler = Chunk(
        id="h_base",
        blob_sha="blob_base",
        file_path="src/api/handler.py",
        kind=ChunkKind.FUNCTION,
        name="handle_request",
        content="async def handle_request(): pass",
        line_start=1,
        line_end=1,
    )
    base_math = Chunk(
        id="m_base",
        blob_sha="blob_math",
        file_path="src/math.py",
        kind=ChunkKind.FUNCTION,
        name="calculate_mean",
        content="def calculate_mean(v): return sum(v)/len(v)",
        line_start=1,
        line_end=1,
    )
    store.insert_chunks([base_handler, base_math])
    store.insert_snapshots(
        [
            ("main", "src/api/handler.py", "blob_base"),
            ("main", "src/math.py", "blob_math"),
        ]
    )

    # Head versions — handler modified, math unchanged, new_endpoint added.
    head_handler = Chunk(
        id="h_head",
        blob_sha="blob_head",
        file_path="src/api/handler.py",
        kind=ChunkKind.FUNCTION,
        name="handle_request",
        content="async def handle_request(): return Response(200)",
        line_start=1,
        line_end=1,
    )
    new_endpoint = Chunk(
        id="ep_new",
        blob_sha="blob_head",
        file_path="src/api/handler.py",
        kind=ChunkKind.FUNCTION,
        name="new_endpoint",
        content="async def new_endpoint(): ...",
        line_start=10,
        line_end=10,
    )
    store.insert_chunks([head_handler, new_endpoint])
    store.insert_snapshots(
        [
            ("feature", "src/api/handler.py", "blob_head"),
            ("feature", "src/math.py", "blob_math"),
        ]
    )

    # Test edge for handle_request but NOT for new_endpoint.
    test_chunk = Chunk(
        id="t_handler",
        blob_sha="blob_test",
        file_path="tests/test_handler.py",
        kind=ChunkKind.FUNCTION,
        name="test_handle_request",
        content="def test_handle_request(): ...",
        line_start=1,
        line_end=1,
    )
    store.insert_chunks([test_chunk])
    store.insert_snapshots(
        [
            ("feature", "tests/test_handler.py", "blob_test"),
        ]
    )
    store.insert_edges(
        [Edge(source_id="t_handler", target_id="h_head", kind=EdgeKind.TESTS)],
        "feature",
    )

    return session, store


def test_semantic_diff_detects_added_symbol() -> None:
    session, store = _make_diff_session()
    ctx = _FakeCtx(session)
    result = semantic_diff(ctx)  # type: ignore[arg-type]
    assert "Added" in result
    assert "new_endpoint" in result
    store.close()


def test_semantic_diff_detects_modified_symbol() -> None:
    session, store = _make_diff_session()
    ctx = _FakeCtx(session)
    result = semantic_diff(ctx)  # type: ignore[arg-type]
    assert "Modified" in result
    assert "handle_request" in result
    store.close()


def test_semantic_diff_reports_missing_tests() -> None:
    """new_endpoint has no test edge → reported as missing."""
    session, store = _make_diff_session()
    ctx = _FakeCtx(session)
    result = semantic_diff(ctx)  # type: ignore[arg-type]
    assert "Missing tests" in result
    assert "new_endpoint" in result
    # test_handle_request is a test function — should NOT be flagged.
    missing_section = result[result.index("Missing tests") :]
    assert "test_handle_request" not in missing_section
    store.close()


def test_semantic_diff_no_changes() -> None:
    """Same blob at base and head → no structural differences."""
    store = IndexStore()
    session = Session()
    session.index = store
    session.review_target = BranchTarget(
        base_branch="main",
        head_branch="feature",
        updated_at=0,
    )

    chunk = Chunk(
        id="f1",
        blob_sha="b1",
        file_path="src/app.py",
        kind=ChunkKind.FUNCTION,
        name="handler",
        content="def handler(): pass",
        line_start=1,
        line_end=1,
    )
    store.insert_chunks([chunk])
    store.insert_snapshots(
        [
            ("main", "src/app.py", "b1"),
            ("feature", "src/app.py", "b1"),
        ]
    )

    ctx = _FakeCtx(session)
    result = semantic_diff(ctx)  # type: ignore[arg-type]
    assert "No structural differences" in result
    store.close()
