"""Shared fixtures for LLM pipeline tests."""

from __future__ import annotations

from collections.abc import Generator

import pytest

from rbtr.index.models import Chunk, ChunkKind, Edge, EdgeKind
from rbtr.index.store import IndexStore
from rbtr.index.tokenise import tokenise_code
from rbtr.models import BranchTarget
from rbtr.sessions.store import SessionStore
from rbtr.state import EngineState
from tests.conftest import drain, has_event_type, output_texts  # noqa: F401

# ── Tool-test duck type ──────────────────────────────────────────────


class FakeCtx:
    """Minimal stand-in for ``RunContext[AgentDeps]`` in tool tests.

    Tools access ``ctx.deps.state`` and optionally ``ctx.deps.store``,
    so this duck type is sufficient without pulling in a real
    ``Model`` instance.
    """

    def __init__(
        self,
        state: EngineState,
        store: SessionStore | None = None,
    ) -> None:
        self.deps = _FakeDeps(state, store)


class _FakeDeps:
    def __init__(
        self,
        state: EngineState,
        store: SessionStore | None = None,
    ) -> None:
        self.state = state
        self.store = store or SessionStore()


# ── Store helpers ────────────────────────────────────────────────────


def _tokenise_and_insert(store: IndexStore, chunks: list[Chunk]) -> None:
    """Populate token fields and insert chunks into the store."""
    for c in chunks:
        c.content_tokens = tokenise_code(c.content)
        c.name_tokens = tokenise_code(c.name)
    store.insert_chunks(chunks)


def _build_index_store(*, embed: bool = False) -> IndexStore:
    """Single-branch index store — HTTP handler + math utils + tests + docs.

    Eight chunks on the ``feature`` branch with realistic edges
    (test→code, import, doc→code).  Optionally populates embedding
    vectors on orthogonal axes for clean similarity ranking.
    """
    store = IndexStore()

    # ── Chunks ───────────────────────────────────────────────────
    handler = Chunk(
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
    process = Chunk(
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
    math_mean = Chunk(
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
    math_stddev = Chunk(
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
    stats_class = Chunk(
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
    test_handler = Chunk(
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
    test_math = Chunk(
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
    doc_section = Chunk(
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

    chunks = [
        handler, process, math_mean, math_stddev,
        stats_class, test_handler, test_math, doc_section,
    ]  # fmt: skip

    # ── Populate store ───────────────────────────────────────────
    _tokenise_and_insert(store, chunks)
    for c in chunks:
        store.insert_snapshot("feature", c.file_path, c.blob_sha)
    store.insert_edges(
        [
            Edge(source_id="test_handler_1", target_id="handler_1", kind=EdgeKind.TESTS),
            Edge(source_id="test_math_1", target_id="math_mean_1", kind=EdgeKind.TESTS),
            Edge(source_id="process_1", target_id="handler_1", kind=EdgeKind.IMPORTS),
            Edge(source_id="doc_api_1", target_id="handler_1", kind=EdgeKind.DOCUMENTS),
            Edge(source_id="math_stddev_1", target_id="math_mean_1", kind=EdgeKind.IMPORTS),
        ],
        "feature",
    )
    store.rebuild_fts_index()

    if embed:
        for cid, vec in [
            ("handler_1", [1.0, 0.0, 0.0]),
            ("process_1", [1.0, 0.0, 0.0]),
            ("math_mean_1", [0.0, 1.0, 0.0]),
            ("math_stddev_1", [0.0, 1.0, 0.0]),
            ("stats_class_1", [0.0, 1.0, 0.0]),
            ("test_handler_1", [1.0, 0.0, 0.0]),
            ("test_math_1", [0.0, 1.0, 0.0]),
            ("doc_api_1", [0.0, 0.0, 1.0]),
        ]:
            store.update_embedding(cid, vec)

    return store


def _build_two_ref_store() -> IndexStore:
    """Two-branch index store — base (main) vs head (feature).

    Models a realistic PR diff: ``parse_request`` is modified,
    ``format_response`` unchanged, ``validate_schema`` added on
    head, ``api_client`` removed from head, ``auth_middleware``
    added on head.
    """
    store = IndexStore()

    # ── Base symbols (main) ──────────────────────────────────────
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

    base_chunks = [base_parse, base_format, base_test, base_client]
    _tokenise_and_insert(store, base_chunks)
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

    # ── Head symbols (feature) ───────────────────────────────────
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

    head_chunks = [head_parse, head_format, head_validate, head_test, head_middleware]
    _tokenise_and_insert(store, head_chunks)
    store.insert_snapshots(
        [
            ("feature", "src/api/handler.py", "handler_v2"),
            ("feature", "tests/test_handler.py", "test_v1"),
            ("feature", "src/api/middleware.py", "mw_v1"),
        ]
    )
    store.insert_edges(
        [
            Edge(source_id="test_parse_h", target_id="parse_h", kind=EdgeKind.TESTS),
            Edge(source_id="mw_h", target_id="parse_h", kind=EdgeKind.IMPORTS),
        ],
        "feature",
    )

    return store


def _review_state(store: IndexStore) -> EngineState:
    """EngineState wired to *store* with a main→feature review target."""
    state = EngineState()
    state.index = store
    state.review_target = BranchTarget(
        base_branch="main",
        head_branch="feature",
        base_commit="main",
        head_commit="feature",
        updated_at=0,
    )
    return state


# ``engine``, ``llm_ctx``, and ``llm_engine`` live in the root
# conftest — available to all test packages.


# ── Index dataset (single branch) ───────────────────────────────────


@pytest.fixture
def index_ctx() -> Generator[FakeCtx]:
    """FakeCtx with the single-branch index dataset (no embeddings)."""
    store = _build_index_store()
    yield FakeCtx(_review_state(store))
    store.close()


@pytest.fixture
def index_ctx_embedded() -> Generator[FakeCtx]:
    """FakeCtx with the single-branch index dataset + embeddings."""
    store = _build_index_store(embed=True)
    yield FakeCtx(_review_state(store))
    store.close()


# ── Two-ref dataset (base vs head) ──────────────────────────────────


@pytest.fixture
def two_ref_ctx() -> Generator[FakeCtx]:
    """FakeCtx with base/head snapshots for ref-scoping tests."""
    store = _build_two_ref_store()
    yield FakeCtx(_review_state(store))
    store.close()


@pytest.fixture
def two_ref_ctx_fts() -> Generator[FakeCtx]:
    """Same as ``two_ref_ctx`` with the FTS index rebuilt for BM25."""
    store = _build_two_ref_store()
    store.rebuild_fts_index()
    yield FakeCtx(_review_state(store))
    store.close()
