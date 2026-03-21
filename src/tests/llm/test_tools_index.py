"""Tests for index tools — search, read_symbol, list_symbols, find_references, changed_symbols."""

from __future__ import annotations

from pydantic_ai import RunContext
from pytest_mock import MockerFixture

from rbtr.index.models import Chunk, ChunkKind, Edge, EdgeKind
from rbtr.index.store import IndexStore
from rbtr.llm.deps import AgentDeps
from rbtr.llm.tools.index import changed_symbols, find_references, list_symbols, read_symbol, search
from rbtr.models import BranchTarget
from rbtr.sessions.store import SessionStore
from rbtr.state import EngineState

from .ctx import tool_ctx

# ── search_symbols ───────────────────────────────────────────────────


def test_search_symbols_finds_exact_name(index_ctx: RunContext[AgentDeps]) -> None:
    result = search(index_ctx, "handle_request")
    assert "handle_request" in result
    assert "src/api/handler.py:10" in result


def test_search_symbols_finds_partial_name(index_ctx: RunContext[AgentDeps]) -> None:
    result = search(index_ctx, "calculate")
    # Should find both math functions.
    assert "calculate_mean" in result
    assert "calculate_standard_deviation" in result


def test_search_symbols_no_match(index_ctx: RunContext[AgentDeps]) -> None:
    result = search(index_ctx, "zzz_nonexistent_zzz")
    assert "No results" in result


# ── search_codebase (BM25) ───────────────────────────────────────────


def test_search_codebase_ranks_by_relevance(index_ctx: RunContext[AgentDeps]) -> None:
    """Searching 'variance' should rank math_stddev highest."""
    result = search(index_ctx, "variance")
    sections = result.strip().split("\n\n")
    # First result should be the stddev function (contains "variance").
    assert "calculate_standard_deviation" in sections[0]


def test_search_codebase_finds_http_content(index_ctx: RunContext[AgentDeps]) -> None:
    result = search(index_ctx, "Response status")
    assert "handle_request" in result


def test_search_codebase_no_match(index_ctx: RunContext[AgentDeps]) -> None:
    result = search(index_ctx, "zzz_gibberish_xyz_999")
    assert "No results" in result


# ── search_similar (embedding) ───────────────────────────────────────


def test_search_similar_ranks_math_first(
    index_ctx_embedded: RunContext[AgentDeps], mocker: MockerFixture
) -> None:
    """Query near the math axis ranks math chunks above HTTP."""
    mocker.patch("rbtr.index.embeddings.embed_text", return_value=[0.1, 0.9, 0.0])
    result = search(index_ctx_embedded, "statistics calculations")

    sections = result.strip().split("\n\n")
    # First results should be math-related (mean, stddev, StatisticsCalculator).
    top_three = " ".join(sections[:3])
    assert "calculate_mean" in top_three or "calculate_standard_deviation" in top_three
    # HTTP handler should not be in the top results.
    assert "handle_request" not in top_three


def test_search_similar_ranks_http_first(
    index_ctx_embedded: RunContext[AgentDeps], mocker: MockerFixture
) -> None:
    """Query near the HTTP axis ranks HTTP chunks above math."""
    mocker.patch("rbtr.index.embeddings.embed_text", return_value=[0.9, 0.1, 0.0])
    result = search(index_ctx_embedded, "web request handling")

    sections = result.strip().split("\n\n")
    top_two = " ".join(sections[:2])
    assert "handle_request" in top_two or "process_data" in top_two


def test_search_similar_no_embeddings(
    index_ctx: RunContext[AgentDeps], mocker: MockerFixture
) -> None:
    """Unified search still returns results when embeddings are unavailable."""
    mocker.patch("rbtr.index.embeddings.embed_text", side_effect=RuntimeError("no model"))
    # Without embeddings, search falls back to BM25 + name match.
    result = search(index_ctx, "handle_request")
    assert "handle_request" in result


# ── find_references ──────────────────────────────────────────────────


def test_find_references_all_edges(index_ctx: RunContext[AgentDeps]) -> None:
    """handle_request has import, test, and doc edges — all returned."""
    result = find_references(index_ctx, "handle_request")
    assert "[imports]" in result
    assert "process_data" in result
    assert "[tests]" in result
    assert "test_handle_request" in result
    assert "[documents]" in result
    assert "API Reference" in result


def test_find_references_filter_imports(index_ctx: RunContext[AgentDeps]) -> None:
    """kind=IMPORTS returns only import edges."""
    result = find_references(index_ctx, "handle_request", kind=EdgeKind.IMPORTS)
    assert "process_data" in result
    assert "[imports]" in result
    # Should NOT include test or doc edges.
    assert "[tests]" not in result
    assert "[documents]" not in result


def test_find_references_filter_tests(index_ctx: RunContext[AgentDeps]) -> None:
    """kind=TESTS returns only test edges."""
    result = find_references(index_ctx, "handle_request", kind=EdgeKind.TESTS)
    assert "test_handle_request" in result
    assert "[tests]" in result
    assert "[imports]" not in result


def test_find_references_filter_documents(index_ctx: RunContext[AgentDeps]) -> None:
    """kind=DOCUMENTS returns only doc edges."""
    result = find_references(index_ctx, "handle_request", kind=EdgeKind.DOCUMENTS)
    assert "API Reference" in result
    assert "[documents]" in result
    assert "[tests]" not in result


def test_find_references_imports_mean(index_ctx: RunContext[AgentDeps]) -> None:
    """calculate_standard_deviation imports calculate_mean."""
    result = find_references(index_ctx, "calculate_mean", kind=EdgeKind.IMPORTS)
    assert "calculate_standard_deviation" in result


def test_find_references_no_match(index_ctx: RunContext[AgentDeps]) -> None:
    """Symbol with no inbound edges returns clear message."""
    result = find_references(index_ctx, "calculate_standard_deviation")
    assert "No references" in result


def test_find_references_no_match_with_kind(index_ctx: RunContext[AgentDeps]) -> None:
    """StatisticsCalculator has no test edges."""
    result = find_references(index_ctx, "StatisticsCalculator", kind=EdgeKind.TESTS)
    assert "No 'tests' references" in result


def test_find_references_unknown_symbol(index_ctx: RunContext[AgentDeps]) -> None:
    result = find_references(index_ctx, "zzz_nonexistent")
    assert "not found" in result


# ── Index ref tests ──────────────────────────────────────────────────


def test_find_references_ref_base_vs_head(two_ref_ctx: RunContext[AgentDeps]) -> None:
    """Base has api_client importing parse_request; head has auth_middleware."""

    base_result = find_references(two_ref_ctx, "parse_request", kind=EdgeKind.IMPORTS, ref="base")
    head_result = find_references(two_ref_ctx, "parse_request", kind=EdgeKind.IMPORTS, ref="head")

    # Base: api_client imports parse_request.
    assert "api_client" in base_result
    assert "auth_middleware" not in base_result

    # Head: auth_middleware imports parse_request (client.py removed).
    assert "auth_middleware" in head_result
    assert "api_client" not in head_result


# ── ref scoping — snapshot isolation ─────────────────────────────────
#
# These tests verify that querying at a ref returns the *full state*
# at that snapshot and never leaks chunks or edges from the other ref.
# The make_two_ref_state factory has carefully separated data:
#
# Only in base: api_client (src/api/client.py)
# Only in head: validate_schema, auth_middleware (src/api/middleware.py)
# Modified:     parse_request (different content + line range per ref)
# Unchanged:    format_response, test_parse_request


def test_ref_scoping_search_symbols_base_excludes_head_only(
    two_ref_ctx: RunContext[AgentDeps],
) -> None:
    """search_symbols at base must not find head-only symbols."""
    # search_symbols always uses head — verify it finds head-only symbols.
    result = search(two_ref_ctx, "validate_schema")
    assert "validate_schema" in result
    # Now verify via read_symbol that validate_schema is invisible at base.
    base_result = read_symbol(two_ref_ctx, "validate_schema", ref="base")
    assert "No symbol" in base_result


def test_ref_scoping_list_symbols_removed_file(two_ref_ctx: RunContext[AgentDeps]) -> None:
    """client.py exists at base but not head — list_symbols reflects this."""

    base_result = list_symbols(two_ref_ctx, "src/api/client.py", ref="base")
    head_result = list_symbols(two_ref_ctx, "src/api/client.py", ref="head")

    # Base: client.py has api_client.
    assert "api_client" in base_result
    assert "1 symbols" in base_result or "1 symbol" in base_result

    # Head: client.py was removed — no symbols.
    assert "No symbols" in head_result


def test_ref_scoping_list_symbols_added_file(two_ref_ctx: RunContext[AgentDeps]) -> None:
    """middleware.py exists at head but not base — list_symbols reflects this."""

    base_result = list_symbols(two_ref_ctx, "src/api/middleware.py", ref="base")
    head_result = list_symbols(two_ref_ctx, "src/api/middleware.py", ref="head")

    # Base: middleware.py doesn't exist.
    assert "No symbols" in base_result

    # Head: middleware.py has auth_middleware.
    assert "auth_middleware" in head_result


def test_ref_scoping_read_symbol_content_matches_ref(two_ref_ctx: RunContext[AgentDeps]) -> None:
    """Modified symbol returns different content per ref — not a mix."""

    base_result = read_symbol(two_ref_ctx, "parse_request", ref="base")
    head_result = read_symbol(two_ref_ctx, "parse_request", ref="head")

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


def test_ref_scoping_find_references_edges_isolated(two_ref_ctx: RunContext[AgentDeps]) -> None:
    """Edges are scoped per commit — base and head edges don't mix."""

    # All references at base.
    base_all = find_references(two_ref_ctx, "parse_request", ref="base")
    # All references at head.
    head_all = find_references(two_ref_ctx, "parse_request", ref="head")

    # Base edges: test_parse_request (TESTS) + api_client (IMPORTS).
    assert "test_parse_request" in base_all
    assert "api_client" in base_all
    assert "auth_middleware" not in base_all

    # Head edges: test_parse_request (TESTS) + auth_middleware (IMPORTS).
    assert "test_parse_request" in head_all
    assert "auth_middleware" in head_all
    assert "api_client" not in head_all


def test_ref_scoping_unchanged_symbol_same_at_both_refs(two_ref_ctx: RunContext[AgentDeps]) -> None:
    """format_response is unchanged — same content at both refs."""

    base_result = read_symbol(two_ref_ctx, "format_response", ref="base")
    head_result = read_symbol(two_ref_ctx, "format_response", ref="head")

    # Both should contain the same implementation.
    assert "json.dumps(data)" in base_result
    assert "json.dumps(data)" in head_result
    # But line numbers differ (shifted by the added schema code).
    assert "10-12" in base_result
    assert "13-15" in head_result


def test_ref_scoping_search_symbols_only_returns_head(two_ref_ctx: RunContext[AgentDeps]) -> None:
    """search_symbols always queries head — finds head-only, misses base-only."""

    # validate_schema only exists at head — search_symbols must find it.
    result = search(two_ref_ctx, "validate_schema")
    assert "validate_schema" in result

    # api_client only exists at base — search must NOT find it.
    result = search(two_ref_ctx, "api_client")
    assert "No results" in result


def test_ref_scoping_search_codebase_only_returns_head(
    two_ref_ctx_fts: RunContext[AgentDeps],
) -> None:
    """search_codebase (BM25) always queries head — misses base-only content."""
    # "missing key" only appears in validate_schema (head-only).
    result = search(two_ref_ctx_fts, "missing key")
    assert "validate_schema" in result

    # "send_to" only appears in api_client (base-only) — must not be found.
    result = search(two_ref_ctx_fts, "send_to")
    assert "No results" in result


def test_ref_scoping_find_references_edge_without_visible_source(store: SessionStore) -> None:
    """Edge source chunk not in current ref's snapshots is silently skipped."""
    store = IndexStore()
    state = EngineState()
    state.index = store
    state.review_target = BranchTarget(
        base_branch="main",
        head_branch="feature",
        base_commit="main",
        head_commit="feature",
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

    ctx = tool_ctx(state, store)
    result = find_references(ctx, "target_fn", ref="head")

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


def test_stale_review_data_does_not_leak(store: SessionStore) -> None:
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
        base_commit="main",
        head_commit="feature",
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

    ctx = tool_ctx(state, store)

    # search (always head) must not find stale symbols.
    sym_result = search(ctx, "generate_invoice")
    assert "No results" in sym_result

    sym_result = search(ctx, "monthly_report")
    assert "No results" in sym_result

    sym_result = search(ctx, "handle_request")
    assert "handle_request" in sym_result

    # read_symbol at both refs must not find stale symbols.
    for ref in ("base", "head"):
        rs = read_symbol(ctx, "generate_invoice", ref=ref)
        assert "No symbol" in rs

    # list_symbols for stale file paths must return nothing.
    ls = list_symbols(ctx, "src/billing/invoice.py", ref="head")
    assert "No symbols" in ls

    # find_references must not find stale edges.
    fr = find_references(ctx, "generate_invoice", ref="head")
    assert "not found" in fr

    # Current data is accessible.
    rs = read_symbol(ctx, "handle_request", ref="head")
    assert "Response(200" in rs

    store.close()


# ── read_symbol ──────────────────────────────────────────────────────


def test_read_symbol_returns_full_source(index_ctx: RunContext[AgentDeps]) -> None:
    result = read_symbol(index_ctx, "handle_request")
    assert "async def handle_request" in result
    assert "Response(status=200" in result
    assert "src/api/handler.py:10-14" in result


def test_read_symbol_prefers_code_over_tests(index_ctx: RunContext[AgentDeps]) -> None:
    """When both a function and its test match, return the function."""
    result = read_symbol(index_ctx, "handle_request")
    # Should be the function, not the test.
    assert "src/api/handler.py" in result


def test_read_symbol_not_found(index_ctx: RunContext[AgentDeps]) -> None:
    result = read_symbol(index_ctx, "zzz_nonexistent")
    assert "No symbol" in result


def test_read_symbol_ref_base_vs_head(two_ref_ctx: RunContext[AgentDeps]) -> None:
    """Base parse_request has simple body; head has schema validation."""

    base_result = read_symbol(two_ref_ctx, "parse_request", ref="base")
    head_result = read_symbol(two_ref_ctx, "parse_request", ref="head")

    # Base: simple json.loads.
    assert "json.loads(raw)" in base_result
    assert "src/api/handler.py:5-7" in base_result
    assert "schema" not in base_result

    # Head: added schema parameter and validate_schema call.
    assert "schema: dict | None = None" in head_result
    assert "validate_schema(data, schema)" in head_result
    assert "src/api/handler.py:5-10" in head_result


def test_read_symbol_ref_base_not_found(two_ref_ctx: RunContext[AgentDeps]) -> None:
    """validate_schema only exists in head → base returns not-found."""

    base_result = read_symbol(two_ref_ctx, "validate_schema", ref="base")
    head_result = read_symbol(two_ref_ctx, "validate_schema", ref="head")

    assert "No symbol" in base_result
    assert "validate_schema" in head_result
    assert "missing key" in head_result  # verify content, not just name


# ── list_symbols ─────────────────────────────────────────────────────


def test_list_symbols_shows_symbols(index_ctx: RunContext[AgentDeps]) -> None:
    """Indexed file shows symbols with line numbers."""
    result = list_symbols(index_ctx, "src/api/handler.py")
    assert "2 symbols" in result
    assert "handle_request" in result
    assert "process_data" in result
    assert "10" in result  # line number
    assert "20" in result  # line number


def test_list_symbols_math_file(index_ctx: RunContext[AgentDeps]) -> None:
    """Math file shows all three symbols (2 functions + 1 class)."""
    result = list_symbols(index_ctx, "src/stats/math_utils.py")
    assert "3 symbols" in result
    assert "calculate_mean" in result
    assert "calculate_standard_deviation" in result
    assert "StatisticsCalculator" in result


def test_list_symbols_no_symbols(index_ctx: RunContext[AgentDeps]) -> None:
    """File with no indexed code symbols returns clear message."""
    result = list_symbols(index_ctx, "config/settings.toml")
    assert "No symbols" in result


def test_list_symbols_path_traversal_rejected(index_ctx: RunContext[AgentDeps]) -> None:
    """Paths with '..' are rejected."""
    result = list_symbols(index_ctx, "../etc/passwd")
    assert "'..' " in result or "contains '..'" in result


def test_list_symbols_ref_base_vs_head(two_ref_ctx: RunContext[AgentDeps]) -> None:
    """Base handler.py has 2 symbols; head has 3 (validate_schema added)."""

    base_result = list_symbols(two_ref_ctx, "src/api/handler.py", ref="base")
    head_result = list_symbols(two_ref_ctx, "src/api/handler.py", ref="head")

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
        base_commit="main",
        head_commit="feature",
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


def test_changed_symbols_detects_added_symbol(store: SessionStore) -> None:
    state, store = _make_diff_state()
    ctx = tool_ctx(state, store)
    result = changed_symbols(ctx)
    assert "Added" in result
    assert "new_endpoint" in result
    store.close()


def test_changed_symbols_detects_modified_symbol(store: SessionStore) -> None:
    state, store = _make_diff_state()
    ctx = tool_ctx(state, store)
    result = changed_symbols(ctx)
    assert "Modified" in result
    assert "handle_request" in result
    store.close()


def test_changed_symbols_reports_missing_tests(store: SessionStore) -> None:
    """new_endpoint has no test edge → reported as missing."""
    state, store = _make_diff_state()
    ctx = tool_ctx(state, store)
    result = changed_symbols(ctx)
    assert "Missing tests" in result
    assert "new_endpoint" in result
    # test_handle_request is a test function — should NOT be flagged.
    missing_section = result[result.index("Missing tests") :]
    assert "test_handle_request" not in missing_section
    store.close()


def test_changed_symbols_no_changes(store: SessionStore) -> None:
    """Same blob at base and head → no structural differences."""
    store = IndexStore()
    state = EngineState()
    state.index = store
    state.review_target = BranchTarget(
        base_branch="main",
        head_branch="feature",
        base_commit="main",
        head_commit="feature",
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

    ctx = tool_ctx(state, store)
    result = changed_symbols(ctx)
    assert "No structural differences" in result
    store.close()


# ── search_similar edge cases ────────────────────────────────────────


def test_search_similar_embedding_error(
    index_ctx: RunContext[AgentDeps], mocker: MockerFixture, store: SessionStore
) -> None:
    """Embedding model failure gracefully falls back to BM25 + name."""
    mocker.patch(
        "rbtr.index.store.IndexStore.search_by_text",
        side_effect=RuntimeError("model not loaded"),
    )
    # Unified search falls back to BM25 + name match without crashing.
    result = search(index_ctx, "handle_request")
    assert "handle_request" in result


# ── changed_symbols edge cases ─────────────────────────────────────────


def test_changed_symbols_no_review_target(store: SessionStore) -> None:
    """changed_symbols without a target returns message."""
    store = IndexStore()
    state = EngineState()
    state.index = store
    ctx = tool_ctx(state, store)
    result = changed_symbols(ctx)
    assert "No diff target" in result
    store.close()


def test_changed_symbols_detects_removed_symbol(store: SessionStore) -> None:
    """A symbol present in base but not head is reported as removed."""
    store = IndexStore()
    state = EngineState()
    state.index = store
    state.review_target = BranchTarget(
        base_branch="main",
        head_branch="feature",
        base_commit="main",
        head_commit="feature",
        updated_at=0,
    )

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

    ctx = tool_ctx(state, store)
    result = changed_symbols(ctx)
    assert "Removed" in result
    assert "func_b" in result
    store.close()


def test_changed_symbols_detects_stale_docs(store: SessionStore) -> None:
    """Doc that references a modified symbol is flagged as stale."""
    store = IndexStore()
    state = EngineState()
    state.index = store
    state.review_target = BranchTarget(
        base_branch="main",
        head_branch="feature",
        base_commit="main",
        head_commit="feature",
        updated_at=0,
    )

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

    ctx = tool_ctx(state, store)
    result = changed_symbols(ctx)
    assert "Stale docs" in result
    assert "API" in result
    store.close()


def test_changed_symbols_detects_broken_edges(store: SessionStore) -> None:
    """Import edge pointing at a removed symbol is flagged as broken."""
    store = IndexStore()
    state = EngineState()
    state.index = store
    state.review_target = BranchTarget(
        base_branch="main",
        head_branch="feature",
        base_commit="main",
        head_commit="feature",
        updated_at=0,
    )

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

    ctx = tool_ctx(state, store)
    result = changed_symbols(ctx)
    assert "Broken edges" in result
    store.close()
