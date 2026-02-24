"""Tests for engine/tools.py — LLM tool functions.

Uses a data-first approach: realistic, semantically distinct code
chunks with meaningful content so we can verify ranking, edge
traversal, and output correctness — not just "does it return something".
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pygit2
import pytest
from pytest_mock import MockerFixture

from rbtr.engine.state import EngineState
from rbtr.engine.tools import (
    changed_files,
    changed_symbols,
    commit_log,
    diff,
    edit,
    find_references,
    grep,
    list_files,
    list_symbols,
    read_file,
    read_symbol,
    search_codebase,
    search_similar,
    search_symbols,
)
from rbtr.index.models import Chunk, ChunkKind, Edge, EdgeKind
from rbtr.index.store import IndexStore
from rbtr.models import BranchTarget

# ── Helpers ──────────────────────────────────────────────────────────


class _FakeCtx:
    """Minimal stand-in for RunContext[AgentDeps] in tool tests."""

    def __init__(self, state: EngineState) -> None:
        self.deps = _FakeDeps(state)


class _FakeDeps:
    def __init__(self, state: EngineState) -> None:
        self.state = state


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


def _make_state(*, embed: bool = False) -> tuple[EngineState, IndexStore]:
    """Build an in-memory store with the full test dataset."""
    store = IndexStore()
    state = EngineState()
    state.index = store
    state.review_target = BranchTarget(
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

    return state, store


# ── search_symbols ───────────────────────────────────────────────────


def test_search_symbols_finds_exact_name() -> None:
    state, store = _make_state()
    ctx = _FakeCtx(state)
    result = search_symbols(ctx, "handle_request")  # type: ignore[arg-type]
    assert "handle_request" in result
    assert "src/api/handler.py:10" in result
    store.close()


def test_search_symbols_finds_partial_name() -> None:
    state, store = _make_state()
    ctx = _FakeCtx(state)
    result = search_symbols(ctx, "calculate")  # type: ignore[arg-type]
    # Should find both math functions.
    assert "calculate_mean" in result
    assert "calculate_standard_deviation" in result
    store.close()


def test_search_symbols_no_match() -> None:
    state, store = _make_state()
    ctx = _FakeCtx(state)
    result = search_symbols(ctx, "zzz_nonexistent_zzz")  # type: ignore[arg-type]
    assert "No symbols" in result
    store.close()


# ── search_codebase (BM25) ───────────────────────────────────────────


def test_search_codebase_ranks_by_relevance() -> None:
    """Searching 'variance' should rank math_stddev highest."""
    state, store = _make_state()
    ctx = _FakeCtx(state)
    result = search_codebase(ctx, "variance")  # type: ignore[arg-type]
    lines = result.strip().split("\n")
    # First result should be the stddev function (contains "variance").
    assert "calculate_standard_deviation" in lines[0]
    store.close()


def test_search_codebase_finds_http_content() -> None:
    state, store = _make_state()
    ctx = _FakeCtx(state)
    result = search_codebase(ctx, "Response status")  # type: ignore[arg-type]
    assert "handle_request" in result
    store.close()


def test_search_codebase_no_match() -> None:
    state, store = _make_state()
    ctx = _FakeCtx(state)
    result = search_codebase(ctx, "zzz_gibberish_xyz_999")  # type: ignore[arg-type]
    assert "No results" in result
    store.close()


# ── search_similar (embedding) ───────────────────────────────────────


def test_search_similar_ranks_math_first(mocker: MockerFixture) -> None:
    """Query near the math axis ranks math chunks above HTTP."""
    state, store = _make_state(embed=True)
    mocker.patch("rbtr.index.embeddings.embed_text", return_value=[0.1, 0.9, 0.0])
    ctx = _FakeCtx(state)
    result = search_similar(ctx, "statistics calculations")  # type: ignore[arg-type]

    lines = result.strip().split("\n")
    # First results should be math-related (mean, stddev, StatisticsCalculator).
    top_three = " ".join(lines[:3])
    assert "calculate_mean" in top_three or "calculate_standard_deviation" in top_three
    # HTTP handler should not be in the top results.
    assert "handle_request" not in top_three
    store.close()


def test_search_similar_ranks_http_first(mocker: MockerFixture) -> None:
    """Query near the HTTP axis ranks HTTP chunks above math."""
    state, store = _make_state(embed=True)
    mocker.patch("rbtr.index.embeddings.embed_text", return_value=[0.9, 0.1, 0.0])
    ctx = _FakeCtx(state)
    result = search_similar(ctx, "web request handling")  # type: ignore[arg-type]

    lines = result.strip().split("\n")
    top_two = " ".join(lines[:2])
    assert "handle_request" in top_two or "process_data" in top_two
    store.close()


def test_search_similar_no_embeddings(mocker: MockerFixture) -> None:
    """Store with no embeddings returns no results."""
    state, store = _make_state(embed=False)
    mocker.patch("rbtr.index.embeddings.embed_text", return_value=[1.0, 0.0, 0.0])
    ctx = _FakeCtx(state)
    result = search_similar(ctx, "anything")  # type: ignore[arg-type]
    assert "No similar" in result
    store.close()


# ── find_references ──────────────────────────────────────────────────


def test_find_references_all_edges() -> None:
    """handle_request has import, test, and doc edges — all returned."""
    state, store = _make_state()
    ctx = _FakeCtx(state)
    result = find_references(ctx, "handle_request")  # type: ignore[arg-type]
    assert "[imports]" in result
    assert "process_data" in result
    assert "[tests]" in result
    assert "test_handle_request" in result
    assert "[documents]" in result
    assert "API Reference" in result
    store.close()


def test_find_references_filter_imports() -> None:
    """kind='imports' returns only import edges."""
    state, store = _make_state()
    ctx = _FakeCtx(state)
    result = find_references(ctx, "handle_request", kind="imports")  # type: ignore[arg-type]
    assert "process_data" in result
    assert "[imports]" in result
    # Should NOT include test or doc edges.
    assert "[tests]" not in result
    assert "[documents]" not in result
    store.close()


def test_find_references_filter_tests() -> None:
    """kind='tests' returns only test edges."""
    state, store = _make_state()
    ctx = _FakeCtx(state)
    result = find_references(ctx, "handle_request", kind="tests")  # type: ignore[arg-type]
    assert "test_handle_request" in result
    assert "[tests]" in result
    assert "[imports]" not in result
    store.close()


def test_find_references_filter_documents() -> None:
    """kind='documents' returns only doc edges."""
    state, store = _make_state()
    ctx = _FakeCtx(state)
    result = find_references(ctx, "handle_request", kind="documents")  # type: ignore[arg-type]
    assert "API Reference" in result
    assert "[documents]" in result
    assert "[tests]" not in result
    store.close()


def test_find_references_imports_mean() -> None:
    """calculate_standard_deviation imports calculate_mean."""
    state, store = _make_state()
    ctx = _FakeCtx(state)
    result = find_references(ctx, "calculate_mean", kind="imports")  # type: ignore[arg-type]
    assert "calculate_standard_deviation" in result
    store.close()


def test_find_references_no_match() -> None:
    """Symbol with no inbound edges returns clear message."""
    state, store = _make_state()
    ctx = _FakeCtx(state)
    result = find_references(ctx, "calculate_standard_deviation")  # type: ignore[arg-type]
    assert "No references" in result
    store.close()


def test_find_references_no_match_with_kind() -> None:
    """StatisticsCalculator has no test edges."""
    state, store = _make_state()
    ctx = _FakeCtx(state)
    result = find_references(ctx, "StatisticsCalculator", kind="tests")  # type: ignore[arg-type]
    assert "No 'tests' references" in result
    store.close()


def test_find_references_unknown_symbol() -> None:
    state, store = _make_state()
    ctx = _FakeCtx(state)
    result = find_references(ctx, "zzz_nonexistent")  # type: ignore[arg-type]
    assert "not found" in result
    store.close()


def test_find_references_invalid_kind() -> None:
    """Invalid kind value returns helpful error."""
    state, store = _make_state()
    ctx = _FakeCtx(state)
    result = find_references(ctx, "handle_request", kind="bogus")  # type: ignore[arg-type]
    assert "Unknown edge kind" in result
    assert "imports" in result  # lists valid kinds
    store.close()


# ── Index ref tests — shared fixture ─────────────────────────────────
#
# Models a realistic base→head evolution for an HTTP API module:
#
# Base (main):
#   - src/api/handler.py: parse_request (validates JSON payloads)
#   - src/api/handler.py: format_response (builds HTTP responses)
#   - tests/test_handler.py: test_parse_request (TESTS → parse_request)
#   - src/api/client.py: api_client (IMPORTS → parse_request)
#
# Head (feature):
#   - src/api/handler.py: parse_request (MODIFIED — added schema arg)
#   - src/api/handler.py: format_response (unchanged)
#   - src/api/handler.py: validate_schema (ADDED — new helper)
#   - tests/test_handler.py: test_parse_request (TESTS → parse_request)
#   - src/api/middleware.py: auth_middleware (IMPORTS → parse_request, new file)
#   - (src/api/client.py removed — old importer gone)


def _make_two_ref_state() -> tuple[EngineState, IndexStore]:
    """Build a store with base and head snapshots for ref tests."""
    store = IndexStore()
    state = EngineState()
    state.index = store
    state.review_target = BranchTarget(
        base_branch="main",
        head_branch="feature",
        updated_at=0,
    )

    # ── Base symbols ─────────────────────────────────────────────
    base_parse = Chunk(
        id="parse_b",
        blob_sha="handler_v1",
        file_path="src/api/handler.py",
        kind=ChunkKind.FUNCTION,
        name="parse_request",
        content="""\
def parse_request(raw: bytes) -> dict:
    return json.loads(raw)
""",
        line_start=5,
        line_end=7,
    )
    base_format = Chunk(
        id="format_b",
        blob_sha="handler_v1",
        file_path="src/api/handler.py",
        kind=ChunkKind.FUNCTION,
        name="format_response",
        content="""\
def format_response(data: dict, status: int = 200) -> Response:
    return Response(body=json.dumps(data), status=status)
""",
        line_start=10,
        line_end=12,
    )
    base_test = Chunk(
        id="test_parse_b",
        blob_sha="test_v1",
        file_path="tests/test_handler.py",
        kind=ChunkKind.FUNCTION,
        name="test_parse_request",
        content="""\
def test_parse_request():
    result = parse_request(b'{"key": "value"}')
    assert result == {"key": "value"}
""",
        line_start=1,
        line_end=3,
    )
    base_client = Chunk(
        id="client_b",
        blob_sha="client_v1",
        file_path="src/api/client.py",
        kind=ChunkKind.FUNCTION,
        name="api_client",
        content="""\
def api_client(endpoint: str, payload: bytes) -> dict:
    parsed = parse_request(payload)
    return send_to(endpoint, parsed)
""",
        line_start=1,
        line_end=3,
    )

    store.insert_chunks([base_parse, base_format, base_test, base_client])
    store.insert_snapshots(
        [
            ("main", "src/api/handler.py", "handler_v1"),
            ("main", "tests/test_handler.py", "test_v1"),
            ("main", "src/api/client.py", "client_v1"),
        ]
    )
    store.insert_edges(
        [
            Edge(source_id="test_parse_b", target_id="parse_b", kind=EdgeKind.TESTS),
            Edge(source_id="client_b", target_id="parse_b", kind=EdgeKind.IMPORTS),
        ],
        "main",
    )

    # ── Head symbols ─────────────────────────────────────────────
    head_parse = Chunk(
        id="parse_h",
        blob_sha="handler_v2",
        file_path="src/api/handler.py",
        kind=ChunkKind.FUNCTION,
        name="parse_request",
        content="""\
def parse_request(raw: bytes, schema: dict | None = None) -> dict:
    data = json.loads(raw)
    if schema:
        validate_schema(data, schema)
    return data
""",
        line_start=5,
        line_end=10,
    )
    head_format = Chunk(
        id="format_h",
        blob_sha="handler_v2",
        file_path="src/api/handler.py",
        kind=ChunkKind.FUNCTION,
        name="format_response",
        content="""\
def format_response(data: dict, status: int = 200) -> Response:
    return Response(body=json.dumps(data), status=status)
""",
        line_start=13,
        line_end=15,
    )
    head_validate = Chunk(
        id="validate_h",
        blob_sha="handler_v2",
        file_path="src/api/handler.py",
        kind=ChunkKind.FUNCTION,
        name="validate_schema",
        content="""\
def validate_schema(data: dict, schema: dict) -> None:
    for key in schema:
        if key not in data:
            raise ValueError(f"missing key: {key}")
""",
        line_start=18,
        line_end=22,
    )
    head_test = Chunk(
        id="test_parse_h",
        blob_sha="test_v1",
        file_path="tests/test_handler.py",
        kind=ChunkKind.FUNCTION,
        name="test_parse_request",
        content="""\
def test_parse_request():
    result = parse_request(b'{"key": "value"}')
    assert result == {"key": "value"}
""",
        line_start=1,
        line_end=3,
    )
    head_middleware = Chunk(
        id="mw_h",
        blob_sha="mw_v1",
        file_path="src/api/middleware.py",
        kind=ChunkKind.FUNCTION,
        name="auth_middleware",
        content="""\
def auth_middleware(request: Request) -> Request:
    parsed = parse_request(request.body)
    if "token" not in parsed:
        raise Unauthorized("missing token")
    return request
""",
        line_start=1,
        line_end=5,
    )

    store.insert_chunks(
        [
            head_parse,
            head_format,
            head_validate,
            head_test,
            head_middleware,
        ]
    )
    store.insert_snapshots(
        [
            ("feature", "src/api/handler.py", "handler_v2"),
            ("feature", "tests/test_handler.py", "test_v1"),
            ("feature", "src/api/middleware.py", "mw_v1"),
            # NOTE: no client.py in head — it was removed.
        ]
    )
    store.insert_edges(
        [
            Edge(source_id="test_parse_h", target_id="parse_h", kind=EdgeKind.TESTS),
            Edge(source_id="mw_h", target_id="parse_h", kind=EdgeKind.IMPORTS),
        ],
        "feature",
    )

    return state, store


def test_find_references_ref_base_vs_head() -> None:
    """Base has api_client importing parse_request; head has auth_middleware."""
    state, store = _make_two_ref_state()
    ctx = _FakeCtx(state)

    base_result = find_references(ctx, "parse_request", kind="imports", ref="base")  # type: ignore[arg-type]
    head_result = find_references(ctx, "parse_request", kind="imports", ref="head")  # type: ignore[arg-type]

    # Base: api_client imports parse_request.
    assert "api_client" in base_result
    assert "auth_middleware" not in base_result

    # Head: auth_middleware imports parse_request (client.py removed).
    assert "auth_middleware" in head_result
    assert "api_client" not in head_result
    store.close()


# ── ref scoping — snapshot isolation ─────────────────────────────────
#
# These tests verify that querying at a ref returns the *full state*
# at that snapshot and never leaks chunks or edges from the other ref.
# The _make_two_ref_state fixture has carefully separated data:
#
# Only in base: api_client (src/api/client.py)
# Only in head: validate_schema, auth_middleware (src/api/middleware.py)
# Modified:     parse_request (different content + line range per ref)
# Unchanged:    format_response, test_parse_request


def test_ref_scoping_search_symbols_base_excludes_head_only() -> None:
    """search_symbols at base must not find head-only symbols."""
    state, store = _make_two_ref_state()
    # search_symbols always uses head — verify it finds head-only symbols.
    ctx = _FakeCtx(state)
    result = search_symbols(ctx, "validate_schema")  # type: ignore[arg-type]
    assert "validate_schema" in result
    # Now verify via read_symbol that validate_schema is invisible at base.
    base_result = read_symbol(ctx, "validate_schema", ref="base")  # type: ignore[arg-type]
    assert "No symbol" in base_result
    store.close()


def test_ref_scoping_list_symbols_removed_file() -> None:
    """client.py exists at base but not head — list_symbols reflects this."""
    state, store = _make_two_ref_state()
    ctx = _FakeCtx(state)

    base_result = list_symbols(ctx, "src/api/client.py", ref="base")  # type: ignore[arg-type]
    head_result = list_symbols(ctx, "src/api/client.py", ref="head")  # type: ignore[arg-type]

    # Base: client.py has api_client.
    assert "api_client" in base_result
    assert "1 symbols" in base_result or "1 symbol" in base_result

    # Head: client.py was removed — no symbols.
    assert "No symbols" in head_result
    store.close()


def test_ref_scoping_list_symbols_added_file() -> None:
    """middleware.py exists at head but not base — list_symbols reflects this."""
    state, store = _make_two_ref_state()
    ctx = _FakeCtx(state)

    base_result = list_symbols(ctx, "src/api/middleware.py", ref="base")  # type: ignore[arg-type]
    head_result = list_symbols(ctx, "src/api/middleware.py", ref="head")  # type: ignore[arg-type]

    # Base: middleware.py doesn't exist.
    assert "No symbols" in base_result

    # Head: middleware.py has auth_middleware.
    assert "auth_middleware" in head_result
    store.close()


def test_ref_scoping_read_symbol_content_matches_ref() -> None:
    """Modified symbol returns different content per ref — not a mix."""
    state, store = _make_two_ref_state()
    ctx = _FakeCtx(state)

    base_result = read_symbol(ctx, "parse_request", ref="base")  # type: ignore[arg-type]
    head_result = read_symbol(ctx, "parse_request", ref="head")  # type: ignore[arg-type]

    # Base content: simple two-line function.
    assert "return json.loads(raw)" in base_result
    assert "5-7" in base_result  # base line range
    # Must NOT contain head-only content.
    assert "schema" not in base_result
    assert "validate_schema" not in base_result

    # Head content: five-line function with schema validation.
    assert "schema: dict | None = None" in head_result
    assert "validate_schema(data, schema)" in head_result
    assert "5-10" in head_result  # head line range
    # Must NOT contain base-only content (the simple return).
    # (Both have json.loads, so we check the unique parts.)
    assert "return json.loads(raw)\n" not in head_result
    store.close()


def test_ref_scoping_find_references_edges_isolated() -> None:
    """Edges are scoped per commit — base and head edges don't mix."""
    state, store = _make_two_ref_state()
    ctx = _FakeCtx(state)

    # All references at base.
    base_all = find_references(ctx, "parse_request", ref="base")  # type: ignore[arg-type]
    # All references at head.
    head_all = find_references(ctx, "parse_request", ref="head")  # type: ignore[arg-type]

    # Base edges: test_parse_request (TESTS) + api_client (IMPORTS).
    assert "test_parse_request" in base_all
    assert "api_client" in base_all
    assert "auth_middleware" not in base_all

    # Head edges: test_parse_request (TESTS) + auth_middleware (IMPORTS).
    assert "test_parse_request" in head_all
    assert "auth_middleware" in head_all
    assert "api_client" not in head_all
    store.close()


def test_ref_scoping_unchanged_symbol_same_at_both_refs() -> None:
    """format_response is unchanged — same content at both refs."""
    state, store = _make_two_ref_state()
    ctx = _FakeCtx(state)

    base_result = read_symbol(ctx, "format_response", ref="base")  # type: ignore[arg-type]
    head_result = read_symbol(ctx, "format_response", ref="head")  # type: ignore[arg-type]

    # Both should contain the same implementation.
    assert "json.dumps(data)" in base_result
    assert "json.dumps(data)" in head_result
    # But line numbers differ (shifted by the added schema code).
    assert "10-12" in base_result
    assert "13-15" in head_result
    store.close()


def test_ref_scoping_search_symbols_only_returns_head() -> None:
    """search_symbols always queries head — finds head-only, misses base-only."""
    state, store = _make_two_ref_state()
    ctx = _FakeCtx(state)

    # validate_schema only exists at head — search_symbols must find it.
    result = search_symbols(ctx, "validate_schema")  # type: ignore[arg-type]
    assert "validate_schema" in result

    # api_client only exists at base — search_symbols must NOT find it.
    result = search_symbols(ctx, "api_client")  # type: ignore[arg-type]
    assert "No symbols" in result
    store.close()


def test_ref_scoping_search_codebase_only_returns_head() -> None:
    """search_codebase (BM25) always queries head — misses base-only content."""
    state, store = _make_two_ref_state()
    store.rebuild_fts_index()
    ctx = _FakeCtx(state)

    # "missing key" only appears in validate_schema (head-only).
    result = search_codebase(ctx, "missing key")  # type: ignore[arg-type]
    assert "validate_schema" in result

    # "send_to" only appears in api_client (base-only) — must not be found.
    result = search_codebase(ctx, "send_to")  # type: ignore[arg-type]
    assert "No results" in result
    store.close()


def test_ref_scoping_find_references_edge_without_visible_source() -> None:
    """Edge source chunk not in current ref's snapshots is silently skipped."""
    store = IndexStore()
    state = EngineState()
    state.index = store
    state.review_target = BranchTarget(
        base_branch="main",
        head_branch="feature",
        updated_at=0,
    )

    # Target visible at head.
    target = Chunk(
        id="tgt",
        blob_sha="b1",
        file_path="src/lib.py",
        kind=ChunkKind.FUNCTION,
        name="target_fn",
        content="def target_fn(): pass",
        line_start=1,
        line_end=1,
    )
    # Source NOT registered in any snapshot — orphan chunk.
    orphan_source = Chunk(
        id="orphan",
        blob_sha="b_orphan",
        file_path="src/gone.py",
        kind=ChunkKind.FUNCTION,
        name="orphan_caller",
        content="def orphan_caller(): target_fn()",
        line_start=1,
        line_end=1,
    )
    store.insert_chunks([target, orphan_source])
    store.insert_snapshot("feature", "src/lib.py", "b1")
    # Deliberately NOT inserting snapshot for orphan_source.

    # Edge points from orphan → target.
    store.insert_edges(
        [Edge(source_id="orphan", target_id="tgt", kind=EdgeKind.IMPORTS)],
        "feature",
    )

    ctx = _FakeCtx(state)
    result = find_references(ctx, "target_fn", ref="head")  # type: ignore[arg-type]

    # orphan_caller should NOT appear — its chunk isn't in head's snapshots.
    # get_chunks(resolved) won't return it, so all_chunks won't contain it.
    assert "orphan_caller" not in result
    # The edge exists but source is invisible → effectively no references.
    assert "No references" in result
    store.close()


# ── Stale data from unrelated reviews ────────────────────────────────
#
# The DuckDB store is persistent — it accumulates snapshots, chunks,
# and edges from past reviews on different branches.  These must not
# leak into the current review's queries.
#
# This test populates the store with data from an "old-pr" review
# (different branch names, different files/symbols/edges), then runs
# a new review on "main"/"feature" and verifies every tool returns
# only current-review data.


def test_stale_review_data_does_not_leak() -> None:
    """Chunks and edges from an older, unrelated review are invisible."""
    store = IndexStore()
    state = EngineState()
    state.index = store

    # ── Old review residue (old-base / old-head) ─────────────────
    stale_chunk = Chunk(
        id="stale_fn",
        blob_sha="stale_blob",
        file_path="src/billing/invoice.py",
        kind=ChunkKind.FUNCTION,
        name="generate_invoice",
        content="""\
def generate_invoice(order_id: int) -> Invoice:
    order = fetch_order(order_id)
    return Invoice(total=order.total, tax=order.tax)
""",
        line_start=10,
        line_end=13,
    )
    stale_caller = Chunk(
        id="stale_caller",
        blob_sha="stale_blob2",
        file_path="src/billing/reports.py",
        kind=ChunkKind.FUNCTION,
        name="monthly_report",
        content="""\
def monthly_report():
    invoices = [generate_invoice(oid) for oid in get_order_ids()]
    return summarise(invoices)
""",
        line_start=1,
        line_end=3,
    )
    store.insert_chunks([stale_chunk, stale_caller])
    store.insert_snapshots(
        [
            ("old-base", "src/billing/invoice.py", "stale_blob"),
            ("old-head", "src/billing/invoice.py", "stale_blob"),
            ("old-head", "src/billing/reports.py", "stale_blob2"),
        ]
    )
    store.insert_edges(
        [Edge(source_id="stale_caller", target_id="stale_fn", kind=EdgeKind.IMPORTS)],
        "old-head",
    )

    # ── Current review (main / feature) ──────────────────────────
    state.review_target = BranchTarget(
        base_branch="main",
        head_branch="feature",
        updated_at=0,
    )

    current_fn = Chunk(
        id="cur_fn",
        blob_sha="cur_blob",
        file_path="src/api/handler.py",
        kind=ChunkKind.FUNCTION,
        name="handle_request",
        content="""\
def handle_request(req):
    return Response(200, req.body)
""",
        line_start=1,
        line_end=2,
    )
    store.insert_chunks([current_fn])
    store.insert_snapshots(
        [
            ("main", "src/api/handler.py", "cur_blob"),
            ("feature", "src/api/handler.py", "cur_blob"),
        ]
    )

    ctx = _FakeCtx(state)

    # search_symbols (always head) must not find stale symbols.
    sym_result = search_symbols(ctx, "generate_invoice")  # type: ignore[arg-type]
    assert "No symbols" in sym_result

    sym_result = search_symbols(ctx, "monthly_report")  # type: ignore[arg-type]
    assert "No symbols" in sym_result

    sym_result = search_symbols(ctx, "handle_request")  # type: ignore[arg-type]
    assert "handle_request" in sym_result

    # read_symbol at both refs must not find stale symbols.
    for ref in ("base", "head"):
        rs = read_symbol(ctx, "generate_invoice", ref=ref)  # type: ignore[arg-type]
        assert "No symbol" in rs

    # list_symbols for stale file paths must return nothing.
    ls = list_symbols(ctx, "src/billing/invoice.py", ref="head")  # type: ignore[arg-type]
    assert "No symbols" in ls

    # find_references must not find stale edges.
    fr = find_references(ctx, "generate_invoice", ref="head")  # type: ignore[arg-type]
    assert "not found" in fr

    # Current data is accessible.
    rs = read_symbol(ctx, "handle_request", ref="head")  # type: ignore[arg-type]
    assert "Response(200" in rs

    store.close()


# ── read_symbol ──────────────────────────────────────────────────────


def test_read_symbol_returns_full_source() -> None:
    state, store = _make_state()
    ctx = _FakeCtx(state)
    result = read_symbol(ctx, "handle_request")  # type: ignore[arg-type]
    assert "async def handle_request" in result
    assert "Response(status=200" in result
    assert "src/api/handler.py:10-14" in result
    store.close()


def test_read_symbol_prefers_code_over_tests() -> None:
    """When both a function and its test match, return the function."""
    state, store = _make_state()
    ctx = _FakeCtx(state)
    result = read_symbol(ctx, "handle_request")  # type: ignore[arg-type]
    # Should be the function, not the test.
    assert "src/api/handler.py" in result
    store.close()


def test_read_symbol_not_found() -> None:
    state, store = _make_state()
    ctx = _FakeCtx(state)
    result = read_symbol(ctx, "zzz_nonexistent")  # type: ignore[arg-type]
    assert "No symbol" in result
    store.close()


def test_read_symbol_ref_base_vs_head() -> None:
    """Base parse_request has simple body; head has schema validation."""
    state, store = _make_two_ref_state()
    ctx = _FakeCtx(state)

    base_result = read_symbol(ctx, "parse_request", ref="base")  # type: ignore[arg-type]
    head_result = read_symbol(ctx, "parse_request", ref="head")  # type: ignore[arg-type]

    # Base: simple json.loads.
    assert "json.loads(raw)" in base_result
    assert "src/api/handler.py:5-7" in base_result
    assert "schema" not in base_result

    # Head: added schema parameter and validate_schema call.
    assert "schema: dict | None = None" in head_result
    assert "validate_schema(data, schema)" in head_result
    assert "src/api/handler.py:5-10" in head_result
    store.close()


def test_read_symbol_ref_base_not_found() -> None:
    """validate_schema only exists in head → base returns not-found."""
    state, store = _make_two_ref_state()
    ctx = _FakeCtx(state)

    base_result = read_symbol(ctx, "validate_schema", ref="base")  # type: ignore[arg-type]
    head_result = read_symbol(ctx, "validate_schema", ref="head")  # type: ignore[arg-type]

    assert "No symbol" in base_result
    assert "validate_schema" in head_result
    assert "missing key" in head_result  # verify content, not just name
    store.close()


# ── list_symbols ─────────────────────────────────────────────────────


def test_list_symbols_shows_symbols() -> None:
    """Indexed file shows symbols with line numbers."""
    state, store = _make_state()
    ctx = _FakeCtx(state)
    result = list_symbols(ctx, "src/api/handler.py")  # type: ignore[arg-type]
    assert "2 symbols" in result
    assert "handle_request" in result
    assert "process_data" in result
    assert "10" in result  # line number
    assert "20" in result  # line number
    store.close()


def test_list_symbols_math_file() -> None:
    """Math file shows all three symbols (2 functions + 1 class)."""
    state, store = _make_state()
    ctx = _FakeCtx(state)
    result = list_symbols(ctx, "src/stats/math_utils.py")  # type: ignore[arg-type]
    assert "3 symbols" in result
    assert "calculate_mean" in result
    assert "calculate_standard_deviation" in result
    assert "StatisticsCalculator" in result
    store.close()


def test_list_symbols_no_symbols() -> None:
    """File with no indexed code symbols returns clear message."""
    state, store = _make_state()
    ctx = _FakeCtx(state)
    result = list_symbols(ctx, "config/settings.toml")  # type: ignore[arg-type]
    assert "No symbols" in result
    store.close()


def test_list_symbols_path_traversal_rejected() -> None:
    """Paths with '..' are rejected."""
    state, store = _make_state()
    ctx = _FakeCtx(state)
    result = list_symbols(ctx, "../etc/passwd")  # type: ignore[arg-type]
    assert "'..' " in result or "contains '..'" in result
    store.close()


def test_list_symbols_ref_base_vs_head() -> None:
    """Base handler.py has 2 symbols; head has 3 (validate_schema added)."""
    state, store = _make_two_ref_state()
    ctx = _FakeCtx(state)

    base_result = list_symbols(ctx, "src/api/handler.py", ref="base")  # type: ignore[arg-type]
    head_result = list_symbols(ctx, "src/api/handler.py", ref="head")  # type: ignore[arg-type]

    # Base: parse_request + format_response.
    assert "2 symbols" in base_result
    assert "parse_request" in base_result
    assert "format_response" in base_result
    assert "validate_schema" not in base_result

    # Head: parse_request + format_response + validate_schema.
    assert "3 symbols" in head_result
    assert "parse_request" in head_result
    assert "format_response" in head_result
    assert "validate_schema" in head_result
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


def _git_state(repo: pygit2.Repository) -> EngineState:
    state = EngineState(repo=repo, owner="o", repo_name="r")
    state.review_target = BranchTarget(
        base_branch="main",
        head_branch="feature",
        updated_at=0,
    )
    return state


# ── diff ─────────────────────────────────────────────────────────────


def test_diff_shows_both_changed_files() -> None:
    """Default diff (base → head) shows both a.py and b.py changes."""
    with tempfile.TemporaryDirectory() as tmp:
        repo, _, _ = _make_repo_two_commits(tmp)
        ctx = _FakeCtx(_git_state(repo))
        result = diff(ctx)  # type: ignore[arg-type]
        assert "a.py" in result
        assert "b.py" in result
        assert "files changed" in result


def test_diff_single_ref_shows_commit() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        repo, _, c2 = _make_repo_two_commits(tmp)
        ctx = _FakeCtx(_git_state(repo))
        result = diff(ctx, ref=c2[:8])  # type: ignore[arg-type]
        assert "files changed" in result
        assert "a.py" in result


def test_diff_range_syntax() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        repo, c1, c2 = _make_repo_two_commits(tmp)
        ctx = _FakeCtx(_git_state(repo))
        result = diff(ctx, ref=f"{c1[:8]}..{c2[:8]}")  # type: ignore[arg-type]
        assert "files changed" in result


def test_diff_bad_ref() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        repo, _, _ = _make_repo_two_commits(tmp)
        ctx = _FakeCtx(_git_state(repo))
        result = diff(ctx, ref="nonexistent")  # type: ignore[arg-type]
        assert "Cannot resolve ref" in result


def test_diff_no_target() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        repo, _, _ = _make_repo_two_commits(tmp)
        state = EngineState(repo=repo, owner="o", repo_name="r")
        ctx = _FakeCtx(state)
        result = diff(ctx)  # type: ignore[arg-type]
        assert "No review target" in result


# ── changed_files ────────────────────────────────────────────────────


def test_changed_files_lists_modified_and_added() -> None:
    """changed_files returns both modified and added files."""
    with tempfile.TemporaryDirectory() as tmp:
        repo, _, _ = _make_repo_two_commits(tmp)
        ctx = _FakeCtx(_git_state(repo))
        result = changed_files(ctx)  # type: ignore[arg-type]
        assert "a.py" in result  # modified
        assert "b.py" in result  # added
        assert "Changed files (2)" in result


def test_changed_files_includes_deleted() -> None:
    """Deleted files appear in the changed list."""
    with tempfile.TemporaryDirectory() as tmp:
        repo = pygit2.init_repository(tmp)
        sig = pygit2.Signature("T", "t@t.com")

        # Base: two files.
        b1 = repo.create_blob(b"x = 1\n")
        b2 = repo.create_blob(b"y = 2\n")
        tb1 = repo.TreeBuilder()
        tb1.insert("keep.py", b1, pygit2.GIT_FILEMODE_BLOB)
        tb1.insert("remove.py", b2, pygit2.GIT_FILEMODE_BLOB)
        c1 = repo.create_commit("refs/heads/main", sig, sig, "init", tb1.write(), [])
        repo.set_head("refs/heads/main")

        # Head: only keep.py remains.
        tb2 = repo.TreeBuilder()
        tb2.insert("keep.py", b1, pygit2.GIT_FILEMODE_BLOB)
        repo.create_commit("refs/heads/feature", sig, sig, "delete", tb2.write(), [c1])

        ctx = _FakeCtx(_git_state(repo))
        result = changed_files(ctx)  # type: ignore[arg-type]
        assert "remove.py" in result
        assert "keep.py" not in result  # unchanged — should not appear


def test_changed_files_identical_branches() -> None:
    """Identical branches report no changes."""
    with tempfile.TemporaryDirectory() as tmp:
        repo = pygit2.init_repository(tmp)
        sig = pygit2.Signature("T", "t@t.com")
        tree = repo.TreeBuilder().write()
        repo.create_commit("refs/heads/main", sig, sig, "init", tree, [])
        repo.set_head("refs/heads/main")
        repo.branches.local.create("feature", repo.head.peel(pygit2.Commit))

        ctx = _FakeCtx(_git_state(repo))
        result = changed_files(ctx)  # type: ignore[arg-type]
        assert "No files changed" in result


def test_changed_files_no_target() -> None:
    """No review target returns message."""
    with tempfile.TemporaryDirectory() as tmp:
        repo, _, _ = _make_repo_two_commits(tmp)
        state = EngineState(repo=repo, owner="o", repo_name="r")
        ctx = _FakeCtx(state)
        result = changed_files(ctx)  # type: ignore[arg-type]
        assert "No review target" in result


# ── commit_log ───────────────────────────────────────────────────────


def test_commit_log_shows_message() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        repo, _, _ = _make_repo_two_commits(tmp)
        ctx = _FakeCtx(_git_state(repo))
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

        ctx = _FakeCtx(_git_state(repo))
        result = commit_log(ctx)  # type: ignore[arg-type]
        assert "No commits" in result or "identical" in result.lower()


def test_commit_log_no_target() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        repo = pygit2.init_repository(tmp)
        sig = pygit2.Signature("T", "t@t.com")
        tree = repo.TreeBuilder().write()
        repo.create_commit("refs/heads/main", sig, sig, "init", tree, [])
        repo.set_head("refs/heads/main")

        state = EngineState(repo=repo, owner="o", repo_name="r")
        ctx = _FakeCtx(state)
        result = commit_log(ctx)  # type: ignore[arg-type]
        assert "No review target" in result


# ── changed_symbols ────────────────────────────────────────────────────


def _make_diff_state() -> tuple[EngineState, IndexStore]:
    """Build a store with base (v1) and head (v2) for semantic diff tests.

    Base has: handler (v1), calculate_mean.
    Head has: handler (v2, modified), calculate_mean (unchanged),
              new_endpoint (added). handler's test exists but
              new_endpoint has no test.
    """
    store = IndexStore()
    state = EngineState()
    state.index = store
    state.review_target = BranchTarget(
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

    return state, store


def test_changed_symbols_detects_added_symbol() -> None:
    state, store = _make_diff_state()
    ctx = _FakeCtx(state)
    result = changed_symbols(ctx)  # type: ignore[arg-type]
    assert "Added" in result
    assert "new_endpoint" in result
    store.close()


def test_changed_symbols_detects_modified_symbol() -> None:
    state, store = _make_diff_state()
    ctx = _FakeCtx(state)
    result = changed_symbols(ctx)  # type: ignore[arg-type]
    assert "Modified" in result
    assert "handle_request" in result
    store.close()


def test_changed_symbols_reports_missing_tests() -> None:
    """new_endpoint has no test edge → reported as missing."""
    state, store = _make_diff_state()
    ctx = _FakeCtx(state)
    result = changed_symbols(ctx)  # type: ignore[arg-type]
    assert "Missing tests" in result
    assert "new_endpoint" in result
    # test_handle_request is a test function — should NOT be flagged.
    missing_section = result[result.index("Missing tests") :]
    assert "test_handle_request" not in missing_section
    store.close()


def test_changed_symbols_no_changes() -> None:
    """Same blob at base and head → no structural differences."""
    store = IndexStore()
    state = EngineState()
    state.index = store
    state.review_target = BranchTarget(
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

    ctx = _FakeCtx(state)
    result = changed_symbols(ctx)  # type: ignore[arg-type]
    assert "No structural differences" in result
    store.close()


# ── Prepare functions (tool hiding) ──────────────────────────────────


def test_require_index_hides_when_no_index() -> None:
    """_require_index returns None when no index is loaded."""
    import asyncio

    state = EngineState()
    state.review_target = BranchTarget(base_branch="main", head_branch="f", updated_at=0)
    assert state.index is None
    ctx = _FakeCtx(state)

    from rbtr.engine.tools import _require_index

    tool_def = object()  # stand-in
    result = asyncio.run(_require_index(ctx, tool_def))  # type: ignore[arg-type]
    assert result is None


def test_require_index_hides_when_no_target() -> None:
    """_require_index returns None when no review target is set."""
    import asyncio

    state = EngineState()
    state.index = IndexStore()
    assert state.review_target is None
    ctx = _FakeCtx(state)

    from rbtr.engine.tools import _require_index

    tool_def = object()
    result = asyncio.run(_require_index(ctx, tool_def))  # type: ignore[arg-type]
    assert result is None
    state.index.close()


def test_require_index_returns_tool_when_ready() -> None:
    """_require_index returns the tool definition when both index and target exist."""
    import asyncio

    state, store = _make_state()
    ctx = _FakeCtx(state)

    from rbtr.engine.tools import _require_index

    tool_def = object()
    result = asyncio.run(_require_index(ctx, tool_def))  # type: ignore[arg-type]
    assert result is tool_def
    store.close()


def test_require_repo_hides_when_no_repo() -> None:
    """_require_repo returns None when no repo is loaded."""
    import asyncio

    state = EngineState()
    state.review_target = BranchTarget(base_branch="main", head_branch="f", updated_at=0)
    ctx = _FakeCtx(state)

    from rbtr.engine.tools import _require_repo

    tool_def = object()
    result = asyncio.run(_require_repo(ctx, tool_def))  # type: ignore[arg-type]
    assert result is None


def test_require_repo_hides_when_no_target() -> None:
    """_require_repo returns None when no review target is set."""
    import asyncio

    with tempfile.TemporaryDirectory() as tmp:
        repo, _, _ = _make_repo_two_commits(tmp)
        state = EngineState(repo=repo)
        ctx = _FakeCtx(state)

        from rbtr.engine.tools import _require_repo

        tool_def = object()
        result = asyncio.run(_require_repo(ctx, tool_def))  # type: ignore[arg-type]
        assert result is None


# ── search_similar edge cases ────────────────────────────────────────


def test_search_similar_embedding_error(mocker: MockerFixture) -> None:
    """Embedding model failure returns a helpful fallback message."""
    state, store = _make_state()
    mocker.patch(
        "rbtr.index.store.IndexStore.search_by_text",
        side_effect=RuntimeError("model not loaded"),
    )
    ctx = _FakeCtx(state)
    result = search_similar(ctx, "anything")  # type: ignore[arg-type]
    assert "Semantic search unavailable" in result
    store.close()


# ── changed_symbols edge cases ─────────────────────────────────────────


def test_changed_symbols_no_review_target() -> None:
    """changed_symbols without a target returns message."""
    store = IndexStore()
    state = EngineState()
    state.index = store
    ctx = _FakeCtx(state)
    result = changed_symbols(ctx)  # type: ignore[arg-type]
    assert "No review target" in result
    store.close()


def test_changed_symbols_detects_removed_symbol() -> None:
    """A symbol present in base but not head is reported as removed."""
    store = IndexStore()
    state = EngineState()
    state.index = store
    state.review_target = BranchTarget(base_branch="main", head_branch="feature", updated_at=0)

    # Base has two functions; head has only one.
    base_a = Chunk(
        id="a_base",
        blob_sha="b1",
        file_path="src/a.py",
        kind=ChunkKind.FUNCTION,
        name="func_a",
        content="def func_a(): pass",
        line_start=1,
        line_end=1,
    )
    base_b = Chunk(
        id="b_base",
        blob_sha="b1",
        file_path="src/a.py",
        kind=ChunkKind.FUNCTION,
        name="func_b",
        content="def func_b(): pass",
        line_start=5,
        line_end=5,
    )
    store.insert_chunks([base_a, base_b])
    store.insert_snapshot("main", "src/a.py", "b1")

    head_a = Chunk(
        id="a_head",
        blob_sha="b2",
        file_path="src/a.py",
        kind=ChunkKind.FUNCTION,
        name="func_a",
        content="def func_a(): pass",
        line_start=1,
        line_end=1,
    )
    store.insert_chunks([head_a])
    store.insert_snapshot("feature", "src/a.py", "b2")

    ctx = _FakeCtx(state)
    result = changed_symbols(ctx)  # type: ignore[arg-type]
    assert "Removed" in result
    assert "func_b" in result
    store.close()


def test_changed_symbols_detects_stale_docs() -> None:
    """Doc that references a modified symbol is flagged as stale."""
    store = IndexStore()
    state = EngineState()
    state.index = store
    state.review_target = BranchTarget(base_branch="main", head_branch="feature", updated_at=0)

    # Base: handler v1 + doc
    base_handler = Chunk(
        id="h_b",
        blob_sha="blob_b",
        file_path="src/h.py",
        kind=ChunkKind.FUNCTION,
        name="handler",
        content="def handler(): v1",
        line_start=1,
        line_end=1,
    )
    doc = Chunk(
        id="doc_b",
        blob_sha="blob_doc",
        file_path="docs/api.md",
        kind=ChunkKind.DOC_SECTION,
        name="API",
        content="## handler docs",
        line_start=1,
        line_end=1,
    )
    store.insert_chunks([base_handler, doc])
    store.insert_snapshot("main", "src/h.py", "blob_b")
    store.insert_snapshot("main", "docs/api.md", "blob_doc")

    # Head: handler v2 (modified), doc unchanged
    head_handler = Chunk(
        id="h_h",
        blob_sha="blob_h",
        file_path="src/h.py",
        kind=ChunkKind.FUNCTION,
        name="handler",
        content="def handler(): v2",
        line_start=1,
        line_end=1,
    )
    store.insert_chunks([head_handler])
    store.insert_snapshot("feature", "src/h.py", "blob_h")
    store.insert_snapshot("feature", "docs/api.md", "blob_doc")

    # Doc → handler edge
    store.insert_edges(
        [Edge(source_id="doc_b", target_id="h_h", kind=EdgeKind.DOCUMENTS)],
        "feature",
    )

    ctx = _FakeCtx(state)
    result = changed_symbols(ctx)  # type: ignore[arg-type]
    assert "Stale docs" in result
    assert "API" in result
    store.close()


def test_changed_symbols_detects_broken_edges() -> None:
    """Import edge pointing at a removed symbol is flagged as broken."""
    store = IndexStore()
    state = EngineState()
    state.index = store
    state.review_target = BranchTarget(base_branch="main", head_branch="feature", updated_at=0)

    # Base: two modules that import each other
    base_a = Chunk(
        id="a_b",
        blob_sha="b1",
        file_path="src/a.py",
        kind=ChunkKind.FUNCTION,
        name="func_a",
        content="def func_a(): pass",
        line_start=1,
        line_end=1,
    )
    base_b = Chunk(
        id="b_b",
        blob_sha="b2",
        file_path="src/b.py",
        kind=ChunkKind.FUNCTION,
        name="func_b",
        content="def func_b(): func_a()",
        line_start=1,
        line_end=1,
    )
    store.insert_chunks([base_a, base_b])
    store.insert_snapshot("main", "src/a.py", "b1")
    store.insert_snapshot("main", "src/b.py", "b2")

    # Head: func_a removed, func_b unchanged, edge still points at a_b
    head_b = Chunk(
        id="b_h",
        blob_sha="b2",
        file_path="src/b.py",
        kind=ChunkKind.FUNCTION,
        name="func_b",
        content="def func_b(): func_a()",
        line_start=1,
        line_end=1,
    )
    store.insert_chunks([head_b])
    store.insert_snapshot("feature", "src/b.py", "b2")
    # No src/a.py in feature snapshot — it's deleted

    # Edge from b→a (a is removed in head)
    store.insert_edges(
        [Edge(source_id="b_h", target_id="a_b", kind=EdgeKind.IMPORTS)],
        "feature",
    )

    ctx = _FakeCtx(state)
    result = changed_symbols(ctx)  # type: ignore[arg-type]
    assert "Broken edges" in result
    store.close()


# ── diff edge cases ─────────────────────────────────────────────────


def test_diff_initial_commit() -> None:
    """Diffing the initial commit (no parent) returns a message."""
    with tempfile.TemporaryDirectory() as tmp:
        repo = pygit2.init_repository(tmp)
        sig = pygit2.Signature("T", "t@t.com")
        blob = repo.create_blob(b"x = 1\n")
        tb = repo.TreeBuilder()
        tb.insert("a.py", blob, pygit2.GIT_FILEMODE_BLOB)
        c = repo.create_commit("refs/heads/main", sig, sig, "init", tb.write(), [])
        repo.set_head("refs/heads/main")

        state = _git_state(repo)
        ctx = _FakeCtx(state)
        result = diff(ctx, ref=str(c)[:8])  # type: ignore[arg-type]
        assert "no parent" in result or "initial commit" in result


def test_diff_bad_range_refs() -> None:
    """Unresolvable refs in range syntax returns error."""
    with tempfile.TemporaryDirectory() as tmp:
        repo, _, _ = _make_repo_two_commits(tmp)
        ctx = _FakeCtx(_git_state(repo))
        result = diff(ctx, ref="badref1..badref2")  # type: ignore[arg-type]
        assert "Cannot resolve ref" in result


def test_diff_truncation(config_path: Path) -> None:
    """Large diffs are limited at max_lines."""
    from rbtr.config import config as cfg

    cfg.tools.max_lines = 5  # tiny limit for test

    with tempfile.TemporaryDirectory() as tmp:
        repo = pygit2.init_repository(tmp)
        sig = pygit2.Signature("T", "t@t.com")

        b1 = repo.create_blob(b"")
        tb1 = repo.TreeBuilder()
        tb1.insert("a.py", b1, pygit2.GIT_FILEMODE_BLOB)
        c1 = repo.create_commit("refs/heads/main", sig, sig, "init", tb1.write(), [])
        repo.set_head("refs/heads/main")

        # Make a big change
        big = ("x = 1\n" * 200).encode()
        b2 = repo.create_blob(big)
        tb2 = repo.TreeBuilder()
        tb2.insert("a.py", b2, pygit2.GIT_FILEMODE_BLOB)
        repo.create_commit("refs/heads/feature", sig, sig, "big", tb2.write(), [c1])

        state = _git_state(repo)
        ctx = _FakeCtx(state)
        result = diff(ctx)  # type: ignore[arg-type]
        assert "... limited" in result
        assert "offset=" in result


# ── diff with path ───────────────────────────────────────────────────


def test_diff_path_filters_to_single_file() -> None:
    """path='a.py' shows only that file's diff."""
    with tempfile.TemporaryDirectory() as tmp:
        repo, _, _ = _make_repo_two_commits(tmp)
        ctx = _FakeCtx(_git_state(repo))
        result = diff(ctx, path="a.py")  # type: ignore[arg-type]
        assert "a.py" in result
        assert "b.py" not in result
        assert "1 files changed" in result


def test_diff_path_empty_shows_full() -> None:
    """Empty path (default) shows the full diff as before."""
    with tempfile.TemporaryDirectory() as tmp:
        repo, _, _ = _make_repo_two_commits(tmp)
        ctx = _FakeCtx(_git_state(repo))
        result = diff(ctx)  # type: ignore[arg-type]
        assert "a.py" in result
        assert "b.py" in result
        assert "2 files changed" in result


def test_diff_path_nonexistent() -> None:
    """Nonexistent path returns empty diff."""
    with tempfile.TemporaryDirectory() as tmp:
        repo, _, _ = _make_repo_two_commits(tmp)
        ctx = _FakeCtx(_git_state(repo))
        result = diff(ctx, path="nonexistent.py")  # type: ignore[arg-type]
        assert "0 files changed" in result


def test_diff_path_with_single_ref() -> None:
    """path also works in single-ref mode."""
    with tempfile.TemporaryDirectory() as tmp:
        repo, _, c2 = _make_repo_two_commits(tmp)
        ctx = _FakeCtx(_git_state(repo))
        result = diff(ctx, path="a.py", ref=c2[:8])  # type: ignore[arg-type]
        assert "a.py" in result
        assert "b.py" not in result


def test_diff_path_with_range() -> None:
    """path also works with range syntax."""
    with tempfile.TemporaryDirectory() as tmp:
        repo, c1, c2 = _make_repo_two_commits(tmp)
        ctx = _FakeCtx(_git_state(repo))
        result = diff(ctx, path="b.py", ref=f"{c1[:8]}..{c2[:8]}")  # type: ignore[arg-type]
        assert "b.py" in result
        assert "a.py" not in result


# ── commit_log edge cases ────────────────────────────────────────────


def test_commit_log_bad_refs() -> None:
    """Unresolvable branch refs returns error."""
    with tempfile.TemporaryDirectory() as tmp:
        repo = pygit2.init_repository(tmp)
        sig = pygit2.Signature("T", "t@t.com")
        tree = repo.TreeBuilder().write()
        repo.create_commit("refs/heads/main", sig, sig, "init", tree, [])
        repo.set_head("refs/heads/main")

        state = EngineState(repo=repo, owner="o", repo_name="r")
        state.review_target = BranchTarget(
            base_branch="main",
            head_branch="nonexistent",
            updated_at=0,
        )
        ctx = _FakeCtx(state)
        result = commit_log(ctx)  # type: ignore[arg-type]
        assert "Cannot resolve ref" in result


def test_commit_log_truncation(config_path: Path) -> None:
    """Long commit log is limited at max_results."""
    from rbtr.config import config as cfg

    cfg.tools.max_results = 2  # tiny limit

    with tempfile.TemporaryDirectory() as tmp:
        repo = pygit2.init_repository(tmp)
        sig = pygit2.Signature("T", "t@t.com")
        tree = repo.TreeBuilder().write()
        c = repo.create_commit("refs/heads/main", sig, sig, "base", tree, [])
        repo.set_head("refs/heads/main")

        # Add 5 commits on feature
        parent_id = c
        for i in range(5):
            parent_id = repo.create_commit(
                "refs/heads/feature",
                sig,
                sig,
                f"commit {i}",
                tree,
                [parent_id],
            )

        state = _git_state(repo)
        ctx = _FakeCtx(state)
        result = commit_log(ctx)  # type: ignore[arg-type]
        assert "... limited" in result
        assert "offset=" in result


# ── File tools — shared fixture ──────────────────────────────────────
#
# A realistic two-branch git repo with semantically distinct files:
#   - src/api/handler.py   — Python handler, modified between base/head
#   - src/api/routes.py    — Python routes, head-only (new file)
#   - config/settings.toml — non-code config file
#   - docs/README.md       — prose with known headings
#   - assets/logo.png      — binary blob (null bytes)
#
# Base (main): handler v1, settings, README, logo
# Head (feature): handler v2, routes (new), settings, README, logo

_HANDLER_V1 = """\
from api.utils import validate

async def handle_request(request):
    \"\"\"Handle incoming HTTP requests.\"\"\"
    data = await request.json()
    validated = validate(data)
    return Response(status=200, body=validated)

async def health_check():
    \"\"\"Simple health endpoint.\"\"\"
    return Response(status=200, body="ok")
"""

_HANDLER_V2 = """\
from api.utils import validate
from api.auth import require_auth

@require_auth
async def handle_request(request):
    \"\"\"Handle incoming HTTP requests with auth.\"\"\"
    data = await request.json()
    validated = validate(data)
    return Response(status=200, body=validated)

async def health_check():
    \"\"\"Simple health endpoint.\"\"\"
    return Response(status=200, body="ok")

async def shutdown():
    \"\"\"Graceful shutdown handler.\"\"\"
    await cleanup_connections()
"""

_ROUTES = """\
from api.handler import handle_request, health_check, shutdown

ROUTES = [
    ("POST", "/api/handle", handle_request),
    ("GET", "/health", health_check),
    ("POST", "/shutdown", shutdown),
]
"""

_SETTINGS = """\
[server]
host = "0.0.0.0"
port = 8080
workers = 4

[database]
url = "postgresql://localhost/mydb"
pool_size = 10

[logging]
level = "INFO"
format = "json"
"""

_README = """\
# My Project

## Overview

A simple HTTP API server with authentication.

## API Endpoints

The `handle_request` function processes incoming data.
The `health_check` endpoint returns server status.

## Configuration

See `config/settings.toml` for server and database settings.

## Development

Run tests with `pytest`.  See CONTRIBUTING.md for guidelines.
"""

# 20 bytes with nulls — enough to trigger binary detection.
_BINARY_LOGO = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x10"


def _make_file_repo(tmp: str) -> tuple[pygit2.Repository, str, str]:
    """Create a two-branch repo with realistic file content.

    Returns (repo, main_sha, feature_sha).
    """
    repo = pygit2.init_repository(tmp)
    sig = pygit2.Signature("Test", "test@test.com")

    # ── Base commit (main) ───────────────────────────────────────
    blobs_base = {
        "src/api/handler.py": repo.create_blob(_HANDLER_V1.encode()),
        "config/settings.toml": repo.create_blob(_SETTINGS.encode()),
        "docs/README.md": repo.create_blob(_README.encode()),
        "assets/logo.png": repo.create_blob(_BINARY_LOGO),
    }

    # Build nested trees bottom-up.
    api_tb = repo.TreeBuilder()
    api_tb.insert("handler.py", blobs_base["src/api/handler.py"], pygit2.GIT_FILEMODE_BLOB)
    src_tb = repo.TreeBuilder()
    src_tb.insert("api", api_tb.write(), pygit2.GIT_FILEMODE_TREE)

    config_tb = repo.TreeBuilder()
    config_tb.insert("settings.toml", blobs_base["config/settings.toml"], pygit2.GIT_FILEMODE_BLOB)

    docs_tb = repo.TreeBuilder()
    docs_tb.insert("README.md", blobs_base["docs/README.md"], pygit2.GIT_FILEMODE_BLOB)

    assets_tb = repo.TreeBuilder()
    assets_tb.insert("logo.png", blobs_base["assets/logo.png"], pygit2.GIT_FILEMODE_BLOB)

    root_tb = repo.TreeBuilder()
    root_tb.insert("src", src_tb.write(), pygit2.GIT_FILEMODE_TREE)
    root_tb.insert("config", config_tb.write(), pygit2.GIT_FILEMODE_TREE)
    root_tb.insert("docs", docs_tb.write(), pygit2.GIT_FILEMODE_TREE)
    root_tb.insert("assets", assets_tb.write(), pygit2.GIT_FILEMODE_TREE)

    c1 = repo.create_commit("refs/heads/main", sig, sig, "initial", root_tb.write(), [])
    repo.set_head("refs/heads/main")

    # ── Head commit (feature) ────────────────────────────────────
    blobs_head = {
        "src/api/handler.py": repo.create_blob(_HANDLER_V2.encode()),
        "src/api/routes.py": repo.create_blob(_ROUTES.encode()),
        "config/settings.toml": blobs_base["config/settings.toml"],
        "docs/README.md": blobs_base["docs/README.md"],
        "assets/logo.png": blobs_base["assets/logo.png"],
    }

    api_tb2 = repo.TreeBuilder()
    api_tb2.insert("handler.py", blobs_head["src/api/handler.py"], pygit2.GIT_FILEMODE_BLOB)
    api_tb2.insert("routes.py", blobs_head["src/api/routes.py"], pygit2.GIT_FILEMODE_BLOB)
    src_tb2 = repo.TreeBuilder()
    src_tb2.insert("api", api_tb2.write(), pygit2.GIT_FILEMODE_TREE)

    config_tb2 = repo.TreeBuilder()
    config_tb2.insert("settings.toml", blobs_head["config/settings.toml"], pygit2.GIT_FILEMODE_BLOB)

    docs_tb2 = repo.TreeBuilder()
    docs_tb2.insert("README.md", blobs_head["docs/README.md"], pygit2.GIT_FILEMODE_BLOB)

    assets_tb2 = repo.TreeBuilder()
    assets_tb2.insert("logo.png", blobs_head["assets/logo.png"], pygit2.GIT_FILEMODE_BLOB)

    root_tb2 = repo.TreeBuilder()
    root_tb2.insert("src", src_tb2.write(), pygit2.GIT_FILEMODE_TREE)
    root_tb2.insert("config", config_tb2.write(), pygit2.GIT_FILEMODE_TREE)
    root_tb2.insert("docs", docs_tb2.write(), pygit2.GIT_FILEMODE_TREE)
    root_tb2.insert("assets", assets_tb2.write(), pygit2.GIT_FILEMODE_TREE)

    c2 = repo.create_commit(
        "refs/heads/feature", sig, sig, "add routes and auth", root_tb2.write(), [c1]
    )

    return repo, str(c1), str(c2)


def _file_state(repo: pygit2.Repository) -> EngineState:
    """EngineState with repo and review target for file tool tests."""
    state = EngineState(repo=repo, owner="o", repo_name="r")
    state.review_target = BranchTarget(
        base_branch="main",
        head_branch="feature",
        updated_at=0,
    )
    return state


# ── read_file ────────────────────────────────────────────────────────


def test_read_file_full() -> None:
    """Full file read returns numbered lines with correct content."""
    with tempfile.TemporaryDirectory() as tmp:
        repo, _, _ = _make_file_repo(tmp)
        ctx = _FakeCtx(_file_state(repo))
        result = read_file(ctx, "config/settings.toml")  # type: ignore[arg-type]
        assert "config/settings.toml" in result
        assert 'host = "0.0.0.0"' in result
        assert "pool_size = 10" in result
        # Line numbers present.
        assert "│" in result


def test_read_file_line_range() -> None:
    """Line range returns exact slice with correct line numbers."""
    with tempfile.TemporaryDirectory() as tmp:
        repo, _, _ = _make_file_repo(tmp)
        ctx = _FakeCtx(_file_state(repo))
        result = read_file(ctx, "config/settings.toml", offset=5, max_lines=3)  # type: ignore[arg-type]
        # Lines 6-8 are the [database] section.
        assert "[database]" in result
        assert 'url = "postgresql://localhost/mydb"' in result
        assert "pool_size = 10" in result
        # Should NOT contain [server] section (lines 1-4).
        assert "[server]" not in result


def test_read_file_head_vs_base() -> None:
    """Head and base refs return different content for modified file."""
    with tempfile.TemporaryDirectory() as tmp:
        repo, _, _ = _make_file_repo(tmp)
        ctx = _FakeCtx(_file_state(repo))
        head_result = read_file(ctx, "src/api/handler.py", ref="head")  # type: ignore[arg-type]
        base_result = read_file(ctx, "src/api/handler.py", ref="base")  # type: ignore[arg-type]
        # Head has @require_auth and shutdown; base does not.
        assert "require_auth" in head_result
        assert "shutdown" in head_result
        assert "require_auth" not in base_result
        assert "shutdown" not in base_result


def test_read_file_raw_ref() -> None:
    """Raw commit SHA resolves correctly."""
    with tempfile.TemporaryDirectory() as tmp:
        repo, c1, _ = _make_file_repo(tmp)
        ctx = _FakeCtx(_file_state(repo))
        result = read_file(ctx, "src/api/handler.py", ref=c1[:8])  # type: ignore[arg-type]
        # c1 is the base commit — should have v1 content.
        assert "require_auth" not in result
        assert "handle_request" in result


@pytest.mark.parametrize("bad_path", ["../etc/passwd", "src/../../../secret", "foo/../../bar"])
def test_read_file_rejects_traversal(bad_path: str) -> None:
    """Paths with '..' are rejected."""
    with tempfile.TemporaryDirectory() as tmp:
        repo, _, _ = _make_file_repo(tmp)
        ctx = _FakeCtx(_file_state(repo))
        result = read_file(ctx, bad_path)  # type: ignore[arg-type]
        assert "'..' " in result or "traversal" in result.lower() or "contains '..'" in result


def test_read_file_binary_rejection() -> None:
    """Binary files are rejected with a clear message."""
    with tempfile.TemporaryDirectory() as tmp:
        repo, _, _ = _make_file_repo(tmp)
        ctx = _FakeCtx(_file_state(repo))
        result = read_file(ctx, "assets/logo.png")  # type: ignore[arg-type]
        assert "binary" in result.lower()


def test_read_file_not_found() -> None:
    """Nonexistent path returns not-found message."""
    with tempfile.TemporaryDirectory() as tmp:
        repo, _, _ = _make_file_repo(tmp)
        ctx = _FakeCtx(_file_state(repo))
        result = read_file(ctx, "nonexistent/file.py")  # type: ignore[arg-type]
        assert "not found" in result.lower()


def test_read_file_truncation(config_path: Path) -> None:
    """Files exceeding max_lines are limited with pagination hint."""
    from rbtr.config import config as cfg

    cfg.tools.max_lines = 3  # tiny limit

    with tempfile.TemporaryDirectory() as tmp:
        repo, _, _ = _make_file_repo(tmp)
        ctx = _FakeCtx(_file_state(repo))
        result = read_file(ctx, "src/api/handler.py")  # type: ignore[arg-type]
        assert "... limited" in result
        assert "offset=" in result


def test_read_file_bad_ref() -> None:
    """Unresolvable ref returns error."""
    with tempfile.TemporaryDirectory() as tmp:
        repo, _, _ = _make_file_repo(tmp)
        ctx = _FakeCtx(_file_state(repo))
        result = read_file(ctx, "src/api/handler.py", ref="nonexistent_branch")  # type: ignore[arg-type]
        assert "not found" in result.lower()


# ── grep ─────────────────────────────────────────────────────────────


def test_grep_single_match() -> None:
    """Single match returns the line with context."""
    with tempfile.TemporaryDirectory() as tmp:
        repo, _, _ = _make_file_repo(tmp)
        ctx = _FakeCtx(_file_state(repo))
        result = grep(ctx, "CONTRIBUTING", path="docs/README.md")  # type: ignore[arg-type]
        assert "1 match" in result
        assert "CONTRIBUTING" in result
        # Should include surrounding context.
        assert "Development" in result


def test_grep_multiple_matches() -> None:
    """Multiple matches each appear in output."""
    with tempfile.TemporaryDirectory() as tmp:
        repo, _, _ = _make_file_repo(tmp)
        ctx = _FakeCtx(_file_state(repo))
        # "handle_request" appears in both handler.py imports and the ROUTES list.
        result = grep(ctx, "handle_request", path="src/api/routes.py")  # type: ignore[arg-type]
        assert "handle_request" in result
        # Should show match count.
        assert "2 matches" in result


def test_grep_overlapping_context_merge() -> None:
    """Matches close together produce merged context (no duplicate lines)."""
    with tempfile.TemporaryDirectory() as tmp:
        repo, _, _ = _make_file_repo(tmp)
        ctx = _FakeCtx(_file_state(repo))
        # In README, "handle_request" and "health_check" are on adjacent lines.
        result = grep(ctx, "handle_request", path="docs/README.md", context_lines=3)  # type: ignore[arg-type]
        # Count line number markers — should not have duplicate line numbers.
        numbered = [line for line in result.split("\n") if "│" in line]
        line_nums = [line.split("│")[0].strip() for line in numbered]
        assert len(line_nums) == len(set(line_nums)), "duplicate line numbers in merged context"


def test_grep_custom_context_lines() -> None:
    """Custom context_lines overrides the default window size."""
    with tempfile.TemporaryDirectory() as tmp:
        repo, _, _ = _make_file_repo(tmp)
        ctx = _FakeCtx(_file_state(repo))
        # Search with tiny context (1 line) vs default (10).
        small = grep(ctx, "CONTRIBUTING", path="docs/README.md", context_lines=1)  # type: ignore[arg-type]
        large = grep(ctx, "CONTRIBUTING", path="docs/README.md", context_lines=10)  # type: ignore[arg-type]
        # Small context should have fewer lines.
        small_lines = [ln for ln in small.split("\n") if "│" in ln]
        large_lines = [ln for ln in large.split("\n") if "│" in ln]
        assert len(small_lines) < len(large_lines)


def test_grep_no_match() -> None:
    """No matches returns a clear message."""
    with tempfile.TemporaryDirectory() as tmp:
        repo, _, _ = _make_file_repo(tmp)
        ctx = _FakeCtx(_file_state(repo))
        result = grep(ctx, "zzz_nonexistent_zzz", path="docs/README.md")  # type: ignore[arg-type]
        assert "No matches" in result


def test_grep_case_insensitive() -> None:
    """Search is case-insensitive."""
    with tempfile.TemporaryDirectory() as tmp:
        repo, _, _ = _make_file_repo(tmp)
        ctx = _FakeCtx(_file_state(repo))
        result = grep(ctx, "overview", path="docs/README.md")  # type: ignore[arg-type]
        # "## Overview" has capital O but search is lowercase.
        assert "Overview" in result
        assert "1 match" in result


@pytest.mark.parametrize("bad_path", ["../etc/passwd", "src/../../../secret"])
def test_grep_rejects_traversal(bad_path: str) -> None:
    """Paths with '..' are rejected."""
    with tempfile.TemporaryDirectory() as tmp:
        repo, _, _ = _make_file_repo(tmp)
        ctx = _FakeCtx(_file_state(repo))
        result = grep(ctx, "anything", path=bad_path)  # type: ignore[arg-type]
        assert "'..' " in result or "contains '..'" in result


def test_grep_binary_rejection() -> None:
    """Binary files are rejected."""
    with tempfile.TemporaryDirectory() as tmp:
        repo, _, _ = _make_file_repo(tmp)
        ctx = _FakeCtx(_file_state(repo))
        result = grep(ctx, "PNG", path="assets/logo.png")  # type: ignore[arg-type]
        assert "binary" in result.lower()


# ── grep repo-wide ──────────────────────────────────────────────────


def test_grep_repo_wide_no_path() -> None:
    """Empty path searches all files, finds matches across multiple files."""
    with tempfile.TemporaryDirectory() as tmp:
        repo, _, _ = _make_file_repo(tmp)
        ctx = _FakeCtx(_file_state(repo))
        # "handle_request" appears in handler.py, routes.py, and README.md.
        result = grep(ctx, "handle_request")  # type: ignore[arg-type]
        assert "src/api/handler.py" in result
        assert "src/api/routes.py" in result
        assert "docs/README.md" in result
        assert "Found" in result


def test_grep_directory_prefix() -> None:
    """Directory prefix scopes search to subtree."""
    with tempfile.TemporaryDirectory() as tmp:
        repo, _, _ = _make_file_repo(tmp)
        ctx = _FakeCtx(_file_state(repo))
        # Search only in src/ — should find handler.py and routes.py but not README.md.
        result = grep(ctx, "handle_request", path="src/")  # type: ignore[arg-type]
        assert "src/api/handler.py" in result
        assert "src/api/routes.py" in result
        assert "README.md" not in result


def test_grep_exact_file_still_works() -> None:
    """Exact file path behaves as single-file grep."""
    with tempfile.TemporaryDirectory() as tmp:
        repo, _, _ = _make_file_repo(tmp)
        ctx = _FakeCtx(_file_state(repo))
        result = grep(ctx, "pool_size", path="config/settings.toml")  # type: ignore[arg-type]
        assert "1 match" in result
        assert "pool_size" in result
        assert "config/settings.toml" in result


def test_grep_repo_wide_skips_binary() -> None:
    """Binary files are silently skipped in repo-wide search."""
    with tempfile.TemporaryDirectory() as tmp:
        repo, _, _ = _make_file_repo(tmp)
        ctx = _FakeCtx(_file_state(repo))
        # "PNG" is in the binary logo — should not appear.
        result = grep(ctx, "pool_size")  # type: ignore[arg-type]
        assert "logo.png" not in result
        assert "pool_size" in result


def test_grep_repo_wide_ref_base() -> None:
    """ref='base' searches the base snapshot — misses head-only content."""
    with tempfile.TemporaryDirectory() as tmp:
        repo, _, _ = _make_file_repo(tmp)
        ctx = _FakeCtx(_file_state(repo))
        # "require_auth" only exists in handler v2 (head).
        head_result = grep(ctx, "require_auth", ref="head")  # type: ignore[arg-type]
        base_result = grep(ctx, "require_auth", ref="base")  # type: ignore[arg-type]
        assert "require_auth" in head_result
        assert "No matches" in base_result


def test_grep_repo_wide_no_match() -> None:
    """No matches across all files returns clear message."""
    with tempfile.TemporaryDirectory() as tmp:
        repo, _, _ = _make_file_repo(tmp)
        ctx = _FakeCtx(_file_state(repo))
        result = grep(ctx, "zzz_nonexistent_zzz")  # type: ignore[arg-type]
        assert "No matches" in result


def test_grep_repo_wide_truncation(config_path: Path) -> None:
    """Repo-wide results are limited at max_grep_hits."""
    from rbtr.config import config as cfg

    cfg.tools.max_grep_hits = 1  # tiny limit — only one match group

    with tempfile.TemporaryDirectory() as tmp:
        repo, _, _ = _make_file_repo(tmp)
        ctx = _FakeCtx(_file_state(repo))
        # "handle_request" matches in multiple files — should limit.
        result = grep(ctx, "handle_request")  # type: ignore[arg-type]
        assert "... limited" in result
        assert "offset=" in result


# ── list_files ───────────────────────────────────────────────────────


def test_list_files_root() -> None:
    """Root listing returns all files."""
    with tempfile.TemporaryDirectory() as tmp:
        repo, _, _ = _make_file_repo(tmp)
        ctx = _FakeCtx(_file_state(repo))
        result = list_files(ctx)  # type: ignore[arg-type]
        assert "src/api/handler.py" in result
        assert "src/api/routes.py" in result
        assert "config/settings.toml" in result
        assert "docs/README.md" in result
        assert "assets/logo.png" in result


def test_list_files_directory_prefix() -> None:
    """Directory prefix filters to files under that path."""
    with tempfile.TemporaryDirectory() as tmp:
        repo, _, _ = _make_file_repo(tmp)
        ctx = _FakeCtx(_file_state(repo))
        result = list_files(ctx, path="src/api")  # type: ignore[arg-type]
        assert "handler.py" in result
        assert "routes.py" in result
        # Should NOT include files outside src/api.
        assert "settings.toml" not in result
        assert "README.md" not in result


def test_list_files_config_prefix() -> None:
    """Prefix 'config' returns only config files."""
    with tempfile.TemporaryDirectory() as tmp:
        repo, _, _ = _make_file_repo(tmp)
        ctx = _FakeCtx(_file_state(repo))
        result = list_files(ctx, path="config")  # type: ignore[arg-type]
        assert "settings.toml" in result
        assert "handler.py" not in result


def test_list_files_base_ref_omits_new_files() -> None:
    """Base ref doesn't include head-only files (routes.py)."""
    with tempfile.TemporaryDirectory() as tmp:
        repo, _, _ = _make_file_repo(tmp)
        ctx = _FakeCtx(_file_state(repo))
        result = list_files(ctx, ref="base")  # type: ignore[arg-type]
        assert "handler.py" in result
        assert "routes.py" not in result


def test_list_files_head_ref_includes_new_files() -> None:
    """Head ref includes new files added on the feature branch."""
    with tempfile.TemporaryDirectory() as tmp:
        repo, _, _ = _make_file_repo(tmp)
        ctx = _FakeCtx(_file_state(repo))
        result = list_files(ctx, ref="head")  # type: ignore[arg-type]
        assert "routes.py" in result


def test_list_files_truncation(config_path: Path) -> None:
    """More than max_results entries triggers limitation."""
    from rbtr.config import config as cfg

    cfg.tools.max_results = 2  # tiny limit

    with tempfile.TemporaryDirectory() as tmp:
        repo, _, _ = _make_file_repo(tmp)
        ctx = _FakeCtx(_file_state(repo))
        result = list_files(ctx)  # type: ignore[arg-type]
        assert "... limited" in result
        assert "offset=" in result


def test_list_files_no_match() -> None:
    """Nonexistent directory returns empty message."""
    with tempfile.TemporaryDirectory() as tmp:
        repo, _, _ = _make_file_repo(tmp)
        ctx = _FakeCtx(_file_state(repo))
        result = list_files(ctx, path="nonexistent/dir")  # type: ignore[arg-type]
        assert "No files" in result


def test_list_files_bad_ref() -> None:
    """Unresolvable ref returns error."""
    with tempfile.TemporaryDirectory() as tmp:
        repo, _, _ = _make_file_repo(tmp)
        ctx = _FakeCtx(_file_state(repo))
        result = list_files(ctx, ref="nonexistent_branch")  # type: ignore[arg-type]
        assert "not found" in result.lower()


# ── edit tool ────────────────────────────────────────────────────────


def _edit_ctx() -> _FakeCtx:
    """Minimal context — edit doesn't need repo or index."""
    state = EngineState()
    return _FakeCtx(state)


def test_edit_create_new_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Creating a new REVIEW- file writes content and returns confirmation."""
    monkeypatch.chdir(tmp_path)
    ctx = _edit_ctx()
    result = edit(ctx, ".rbtr/REVIEW-plan.md", "# Plan\n\n- Step 1\n")  # type: ignore[arg-type]
    assert "Created" in result
    content = (tmp_path / ".rbtr" / "REVIEW-plan.md").read_text()
    assert content == "# Plan\n\n- Step 1\n"


def test_edit_append_to_existing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Appending to an existing file concatenates content."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".rbtr").mkdir()
    (tmp_path / ".rbtr" / "REVIEW-notes.md").write_text("# Notes\n")
    ctx = _edit_ctx()
    result = edit(ctx, ".rbtr/REVIEW-notes.md", "\n- Finding 1\n")  # type: ignore[arg-type]
    assert "Appended" in result
    content = (tmp_path / ".rbtr" / "REVIEW-notes.md").read_text()
    assert content == "# Notes\n\n- Finding 1\n"


def test_edit_replace_exact_match(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Exact old_text match is replaced with new_text."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".rbtr").mkdir()
    original = "# Plan\n\n- [ ] Review handler\n- [ ] Review config\n"
    (tmp_path / ".rbtr" / "REVIEW-plan.md").write_text(original)
    ctx = _edit_ctx()
    result = edit(
        ctx,  # type: ignore[arg-type]
        ".rbtr/REVIEW-plan.md",
        "- [x] Review handler  ✓\n",
        old_text="- [ ] Review handler\n",
    )
    assert "Replaced" in result
    content = (tmp_path / ".rbtr" / "REVIEW-plan.md").read_text()
    assert "- [x] Review handler  ✓" in content
    assert "- [ ] Review config" in content  # untouched


def test_edit_replace_not_found(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """old_text that doesn't exist returns an error."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".rbtr").mkdir()
    (tmp_path / ".rbtr" / "REVIEW-plan.md").write_text("# Plan\n")
    ctx = _edit_ctx()
    result = edit(
        ctx,  # type: ignore[arg-type]
        ".rbtr/REVIEW-plan.md",
        "replacement",
        old_text="nonexistent text",
    )
    assert "not found" in result


def test_edit_replace_ambiguous(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """old_text matching multiple times returns an error."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".rbtr").mkdir()
    (tmp_path / ".rbtr" / "REVIEW-plan.md").write_text("AAA\nBBB\nAAA\n")
    ctx = _edit_ctx()
    result = edit(
        ctx,  # type: ignore[arg-type]
        ".rbtr/REVIEW-plan.md",
        "CCC",
        old_text="AAA",
    )
    assert "2 times" in result


def test_edit_replace_file_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Replacing in a nonexistent file returns an error."""
    monkeypatch.chdir(tmp_path)
    ctx = _edit_ctx()
    result = edit(
        ctx,  # type: ignore[arg-type]
        ".rbtr/REVIEW-plan.md",
        "new",
        old_text="old",
    )
    assert "does not exist" in result


def test_edit_rejects_path_outside_rbtr() -> None:
    """Paths not inside .rbtr/ are rejected."""
    ctx = _edit_ctx()
    result = edit(ctx, "src/main.py", "content")  # type: ignore[arg-type]
    assert "must be inside .rbtr/" in result


def test_edit_rejects_non_review_filename() -> None:
    """Filenames not starting with REVIEW- are rejected."""
    ctx = _edit_ctx()
    result = edit(ctx, ".rbtr/notes.md", "content")  # type: ignore[arg-type]
    assert "REVIEW-" in result


def test_edit_rejects_path_traversal() -> None:
    """Paths with '..' are rejected."""
    ctx = _edit_ctx()
    result = edit(ctx, ".rbtr/../REVIEW-escape.md", "content")  # type: ignore[arg-type]
    assert "not allowed" in result


def test_edit_creates_subdirectory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Nested paths under .rbtr/ create intermediate directories."""
    monkeypatch.chdir(tmp_path)
    ctx = _edit_ctx()
    result = edit(ctx, ".rbtr/drafts/REVIEW-comments.md", "# Draft\n")  # type: ignore[arg-type]
    assert "Created" in result
    assert (tmp_path / ".rbtr" / "drafts" / "REVIEW-comments.md").exists()


# ── Workspace file access (.rbtr/) ──────────────────────────────────
#
# read_file, grep, and list_files read from the local filesystem
# when paths start with .rbtr/ — these files aren't in the git tree.


def _make_workspace(tmp_path: Path) -> None:
    """Create .rbtr/ workspace with review notes for testing."""
    rbtr = tmp_path / ".rbtr"
    rbtr.mkdir()
    (rbtr / "REVIEW-plan.md").write_text(
        """\
# Review Plan

## Phase 1: Orientation
- Read PR description
- Check changed files

## Phase 2: Deep dive
- handler.py: check error paths
- config.py: verify defaults
"""
    )
    (rbtr / "REVIEW-findings.md").write_text(
        """\
# Findings

## blocker: Missing null check in handler
The `handle_request` function does not validate
that `data` is non-empty before processing.

## suggestion: Config defaults are stale
The `pool_size` default of 5 was appropriate for
the old architecture but should be revisited.
"""
    )


def test_read_file_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """read_file reads .rbtr/ files from the local filesystem."""
    monkeypatch.chdir(tmp_path)
    _make_workspace(tmp_path)
    with tempfile.TemporaryDirectory() as repo_tmp:
        repo, _, _ = _make_file_repo(repo_tmp)
        ctx = _FakeCtx(_file_state(repo))
        result = read_file(ctx, ".rbtr/REVIEW-plan.md")  # type: ignore[arg-type]
        assert "# Review Plan" in result
        assert "Phase 1" in result
        assert "handler.py" in result


def test_read_file_workspace_not_found(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """read_file returns error for nonexistent .rbtr/ file."""
    monkeypatch.chdir(tmp_path)
    with tempfile.TemporaryDirectory() as repo_tmp:
        repo, _, _ = _make_file_repo(repo_tmp)
        ctx = _FakeCtx(_file_state(repo))
        result = read_file(ctx, ".rbtr/REVIEW-nonexistent.md")  # type: ignore[arg-type]
        assert "not found" in result.lower()


def test_read_file_workspace_ignores_ref(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """ref parameter is ignored for .rbtr/ workspace files."""
    monkeypatch.chdir(tmp_path)
    _make_workspace(tmp_path)
    with tempfile.TemporaryDirectory() as repo_tmp:
        repo, _, _ = _make_file_repo(repo_tmp)
        ctx = _FakeCtx(_file_state(repo))
        # ref="base" would fail for git files, but is ignored for .rbtr/
        result = read_file(ctx, ".rbtr/REVIEW-plan.md", ref="base")  # type: ignore[arg-type]
        assert "# Review Plan" in result


def test_grep_workspace_single_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """grep searches a single .rbtr/ file from the filesystem."""
    monkeypatch.chdir(tmp_path)
    _make_workspace(tmp_path)
    with tempfile.TemporaryDirectory() as repo_tmp:
        repo, _, _ = _make_file_repo(repo_tmp)
        ctx = _FakeCtx(_file_state(repo))
        result = grep(ctx, "null check", path=".rbtr/REVIEW-findings.md")  # type: ignore[arg-type]
        assert "null check" in result.lower()
        assert "REVIEW-findings.md" in result


def test_grep_workspace_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """grep searches all files under .rbtr/ when given a directory prefix."""
    monkeypatch.chdir(tmp_path)
    _make_workspace(tmp_path)
    with tempfile.TemporaryDirectory() as repo_tmp:
        repo, _, _ = _make_file_repo(repo_tmp)
        ctx = _FakeCtx(_file_state(repo))
        # "handler" appears in both plan and findings
        result = grep(ctx, "handler", path=".rbtr/")  # type: ignore[arg-type]
        assert "handler" in result.lower()
        # Should show matches from at least one file
        assert "REVIEW-" in result


def test_grep_workspace_no_match(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """grep returns no-match message for workspace files."""
    monkeypatch.chdir(tmp_path)
    _make_workspace(tmp_path)
    with tempfile.TemporaryDirectory() as repo_tmp:
        repo, _, _ = _make_file_repo(repo_tmp)
        ctx = _FakeCtx(_file_state(repo))
        result = grep(ctx, "xyznonexistent", path=".rbtr/")  # type: ignore[arg-type]
        assert "No matches" in result


def test_list_files_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """list_files lists .rbtr/ files from the local filesystem."""
    monkeypatch.chdir(tmp_path)
    _make_workspace(tmp_path)
    with tempfile.TemporaryDirectory() as repo_tmp:
        repo, _, _ = _make_file_repo(repo_tmp)
        ctx = _FakeCtx(_file_state(repo))
        result = list_files(ctx, path=".rbtr/")  # type: ignore[arg-type]
        assert "REVIEW-plan.md" in result
        assert "REVIEW-findings.md" in result


def test_list_files_workspace_empty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """list_files returns empty message for nonexistent .rbtr/ directory."""
    monkeypatch.chdir(tmp_path)
    with tempfile.TemporaryDirectory() as repo_tmp:
        repo, _, _ = _make_file_repo(repo_tmp)
        ctx = _FakeCtx(_file_state(repo))
        result = list_files(ctx, path=".rbtr/")  # type: ignore[arg-type]
        assert "No files" in result


# ── Filesystem fallback — git-first, then filesystem ─────────────────
#
# When a prefix matches git files, git wins.
# When nothing is in git, filesystem is tried.
# read_file: git blob → filesystem.
# list_files: git tree → filesystem.
# grep: git blob/tree → filesystem.


def test_list_files_git_prefix_wins_over_filesystem(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When git tree has files under the prefix, filesystem is NOT used."""
    monkeypatch.chdir(tmp_path)
    # Create a filesystem file under src/api/ (same prefix as git files).
    (tmp_path / "src" / "api").mkdir(parents=True)
    (tmp_path / "src" / "api" / "local_only.py").write_text("# local\n")
    with tempfile.TemporaryDirectory() as repo_tmp:
        repo, _, _ = _make_file_repo(repo_tmp)
        ctx = _FakeCtx(_file_state(repo))
        result = list_files(ctx, path="src/api")  # type: ignore[arg-type]
        # Git files should be listed.
        assert "handler.py" in result
        # Filesystem-only file should NOT appear — git wins.
        assert "local_only.py" not in result


def test_list_files_falls_back_to_filesystem(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When prefix has no git files, filesystem is used as fallback."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "local_dir").mkdir()
    (tmp_path / "local_dir" / "notes.txt").write_text("hello\n")
    with tempfile.TemporaryDirectory() as repo_tmp:
        repo, _, _ = _make_file_repo(repo_tmp)
        ctx = _FakeCtx(_file_state(repo))
        result = list_files(ctx, path="local_dir")  # type: ignore[arg-type]
        assert "notes.txt" in result


def test_read_file_git_blob_wins_over_filesystem(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When a file exists in both git and filesystem, git version is used."""
    monkeypatch.chdir(tmp_path)
    # Create a local file with different content than the git blob.
    (tmp_path / "src" / "api").mkdir(parents=True)
    (tmp_path / "src" / "api" / "handler.py").write_text("# FILESYSTEM VERSION\n")
    with tempfile.TemporaryDirectory() as repo_tmp:
        repo, _, _ = _make_file_repo(repo_tmp)
        ctx = _FakeCtx(_file_state(repo))
        result = read_file(ctx, "src/api/handler.py")  # type: ignore[arg-type]
        # Git content has handle_request; filesystem has "FILESYSTEM VERSION".
        assert "handle_request" in result
        assert "FILESYSTEM VERSION" not in result


def test_read_file_falls_back_to_filesystem(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When a file is not in git, falls back to local filesystem."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "untracked.txt").write_text("local content here\n")
    with tempfile.TemporaryDirectory() as repo_tmp:
        repo, _, _ = _make_file_repo(repo_tmp)
        ctx = _FakeCtx(_file_state(repo))
        result = read_file(ctx, "untracked.txt")  # type: ignore[arg-type]
        assert "local content here" in result


def test_read_file_not_in_git_or_filesystem() -> None:
    """When a file is in neither git nor filesystem, returns git error."""
    with tempfile.TemporaryDirectory() as tmp:
        repo, _, _ = _make_file_repo(tmp)
        ctx = _FakeCtx(_file_state(repo))
        result = read_file(ctx, "totally_missing.txt")  # type: ignore[arg-type]
        assert "not found" in result.lower()


def test_grep_git_prefix_wins_over_filesystem(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When git tree has files under prefix, grep searches git only."""
    monkeypatch.chdir(tmp_path)
    # Create filesystem file with unique content under the same prefix.
    (tmp_path / "src" / "api").mkdir(parents=True)
    (tmp_path / "src" / "api" / "local.py").write_text("UNIQUE_LOCAL_MARKER\n")
    with tempfile.TemporaryDirectory() as repo_tmp:
        repo, _, _ = _make_file_repo(repo_tmp)
        ctx = _FakeCtx(_file_state(repo))
        result = grep(ctx, "UNIQUE_LOCAL_MARKER", path="src/api")  # type: ignore[arg-type]
        # Git has files under src/api but none contain this marker.
        assert "No matches" in result


def test_grep_falls_back_to_filesystem(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When prefix has no git files, grep searches filesystem."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "local_dir").mkdir()
    (tmp_path / "local_dir" / "notes.txt").write_text("important finding\n")
    with tempfile.TemporaryDirectory() as repo_tmp:
        repo, _, _ = _make_file_repo(repo_tmp)
        ctx = _FakeCtx(_file_state(repo))
        result = grep(ctx, "important", path="local_dir")  # type: ignore[arg-type]
        assert "important finding" in result
        assert "notes.txt" in result


def test_grep_single_file_falls_back_to_filesystem(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When an exact file path isn't in git, grep searches filesystem."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "local.txt").write_text("needle in haystack\n")
    with tempfile.TemporaryDirectory() as repo_tmp:
        repo, _, _ = _make_file_repo(repo_tmp)
        ctx = _FakeCtx(_file_state(repo))
        result = grep(ctx, "needle", path="local.txt")  # type: ignore[arg-type]
        assert "needle in haystack" in result
