"""Shared fixtures for ``tests/index/``.

Two datasets served from here:

1. A realistic multi-chunk project used by the E2E ranking and
   structural search tests.  The ``ranking_store`` fixture loads
   a seeded ``IndexStore``.  Individual chunk fixtures
   (``ranking_config_class``, ``ranking_load_config``, ...) are
   exposed for the rare test that wants to reference one directly.

2. Small named chunk / edge / vector fixtures consumed by the
   ``case_store_*`` / ``test_store_*`` families.

No module-level test data.  Everything is a fixture so dependency
graphs are explicit.
"""

from __future__ import annotations

import pytest

from rbtr.index.models import Chunk, ChunkKind, Edge, EdgeKind
from rbtr.index.store import IndexStore
from rbtr.index.tokenise import tokenise_code


def _tokenise(chunk: Chunk) -> Chunk:
    """Return *chunk* with content/name token fields filled in."""
    return chunk.model_copy(
        update={
            "content_tokens": tokenise_code(chunk.content),
            "name_tokens": tokenise_code(chunk.name),
        }
    )


# ═════════════════════════════════════════════════════════════════════
# Ranking dataset (for test_search_ranking.py, test_search_structural.py)
# ═════════════════════════════════════════════════════════════════════
#
# Small realistic project with deliberate ranking conflicts:
#   * CLASS vs IMPORT of the same name  → kind boost
#   * source vs test mentioning a fn    → file penalty
#   * high-df term ("config") in 5/6    → IDF neutralised
#   * exact name match vs content       → name score
#   * 3 inbound edges vs 0              → importance
#   * diff touching src/server.py       → proximity


@pytest.fixture
def ranking_commit() -> str:
    return "abc123"


@pytest.fixture
def ranking_config_class() -> Chunk:
    return _tokenise(
        Chunk(
            id="config_class",
            blob_sha="blob_config",
            file_path="src/config.py",
            kind=ChunkKind.CLASS,
            name="AppConfig",
            content=(
                "class AppConfig:\n"
                "    database_url: str\n"
                "    max_retries: int = 3\n"
                "    timeout: float = 30.0\n"
            ),
            line_start=1,
            line_end=5,
        )
    )


@pytest.fixture
def ranking_load_config() -> Chunk:
    return _tokenise(
        Chunk(
            id="load_config",
            blob_sha="blob_config",
            file_path="src/config.py",
            kind=ChunkKind.FUNCTION,
            name="load_config",
            content=(
                "def load_config(path: str) -> AppConfig:\n"
                "    with open(path) as f:\n"
                "        data = json.load(f)\n"
                "    return AppConfig(**data)\n"
            ),
            line_start=10,
            line_end=14,
        )
    )


@pytest.fixture
def ranking_import_config() -> Chunk:
    return _tokenise(
        Chunk(
            id="import_config",
            blob_sha="blob_server",
            file_path="src/server.py",
            kind=ChunkKind.IMPORT,
            name="from config import AppConfig",
            content="from config import AppConfig, load_config",
            line_start=1,
            line_end=1,
        )
    )


@pytest.fixture
def ranking_start_server() -> Chunk:
    return _tokenise(
        Chunk(
            id="start_server",
            blob_sha="blob_server",
            file_path="src/server.py",
            kind=ChunkKind.FUNCTION,
            name="start_server",
            content=(
                "def start_server(config: AppConfig) -> None:\n"
                "    app = create_app(config)\n"
                '    app.run(host="0.0.0.0", port=config.port)\n'
            ),
            line_start=5,
            line_end=8,
        )
    )


@pytest.fixture
def ranking_test_config() -> Chunk:
    return _tokenise(
        Chunk(
            id="test_config",
            blob_sha="blob_test_config",
            file_path="tests/test_config.py",
            kind=ChunkKind.FUNCTION,
            name="test_load_config",
            content=(
                "def test_load_config():\n"
                '    config = load_config("test.json")\n'
                "    assert isinstance(config, AppConfig)\n"
                "    assert config.max_retries == 3\n"
                '    config = load_config("other.json")\n'
                "    assert config.timeout == 30.0\n"
                '    config = load_config("empty.json")\n'
            ),
            line_start=1,
            line_end=7,
        )
    )


@pytest.fixture
def ranking_doc_section() -> Chunk:
    return _tokenise(
        Chunk(
            id="doc_config",
            blob_sha="blob_docs",
            file_path="docs/setup.md",
            kind=ChunkKind.DOC_SECTION,
            name="Configuration",
            content=(
                "## Configuration\n"
                "\n"
                "Use `load_config` to load an `AppConfig` from a JSON file.\n"
                "Set `database_url` and `max_retries` as needed.\n"
            ),
            line_start=1,
            line_end=5,
        )
    )


@pytest.fixture
def ranking_chunks(
    ranking_config_class: Chunk,
    ranking_load_config: Chunk,
    ranking_import_config: Chunk,
    ranking_start_server: Chunk,
    ranking_test_config: Chunk,
    ranking_doc_section: Chunk,
) -> list[Chunk]:
    return [
        ranking_config_class,
        ranking_load_config,
        ranking_import_config,
        ranking_start_server,
        ranking_test_config,
        ranking_doc_section,
    ]


@pytest.fixture
def ranking_edges() -> list[Edge]:
    """Edge graph used by importance / structural ranking tests.

    ``config_class`` receives 2 inbound edges (import, start_server).
    ``load_config`` receives 3 inbound edges (import, test, doc).
    ``start_server`` receives 0.
    """
    return [
        Edge(
            source_id="import_config",
            target_id="config_class",
            kind=EdgeKind.IMPORTS,
        ),
        Edge(
            source_id="import_config",
            target_id="load_config",
            kind=EdgeKind.IMPORTS,
        ),
        Edge(
            source_id="start_server",
            target_id="config_class",
            kind=EdgeKind.IMPORTS,
        ),
        Edge(
            source_id="test_config",
            target_id="load_config",
            kind=EdgeKind.TESTS,
        ),
        Edge(
            source_id="doc_config",
            target_id="load_config",
            kind=EdgeKind.DOCUMENTS,
        ),
    ]


@pytest.fixture
def ranking_store(
    ranking_commit: str,
    ranking_chunks: list[Chunk],
    ranking_edges: list[Edge],
) -> IndexStore:
    """An in-memory IndexStore pre-loaded with the ranking dataset."""
    store = IndexStore()
    store.insert_chunks(ranking_chunks)
    for c in ranking_chunks:
        store.insert_snapshot(ranking_commit, c.file_path, c.blob_sha)
    store.insert_edges(ranking_edges, ranking_commit)
    store.rebuild_fts_index()
    return store


# ═════════════════════════════════════════════════════════════════════
# Store-family fixtures (case_store_* / test_store_*)
# ═════════════════════════════════════════════════════════════════════


@pytest.fixture
def math_func() -> Chunk:
    return Chunk(
        id="math_1",
        blob_sha="blob_math",
        file_path="src/math_utils.py",
        kind=ChunkKind.FUNCTION,
        name="calculate_standard_deviation",
        content=(
            "def calculate_standard_deviation(values: list[float]) -> float:\n"
            "    mean = sum(values) / len(values)\n"
            "    variance = sum((x - mean) ** 2 for x in values) / len(values)\n"
            "    return variance ** 0.5\n"
        ),
        line_start=1,
        line_end=4,
    )


@pytest.fixture
def http_func() -> Chunk:
    return Chunk(
        id="http_1",
        blob_sha="blob_http",
        file_path="src/api/client.py",
        kind=ChunkKind.FUNCTION,
        name="fetch_json_from_endpoint",
        content=(
            "async def fetch_json_from_endpoint(url: str, headers: dict) -> dict:\n"
            "    async with httpx.AsyncClient() as client:\n"
            "        response = await client.get(url, headers=headers)\n"
            "        response.raise_for_status()\n"
            "        return response.json()\n"
        ),
        line_start=10,
        line_end=15,
    )


@pytest.fixture
def string_func() -> Chunk:
    return Chunk(
        id="string_1",
        blob_sha="blob_string",
        file_path="src/text/normalize.py",
        kind=ChunkKind.FUNCTION,
        name="normalize_whitespace",
        content=(
            "def normalize_whitespace(text: str) -> str:\n"
            "    import re\n"
            "    collapsed = re.sub(r'\\s+', ' ', text)\n"
            "    return collapsed.strip()\n"
        ),
        line_start=1,
        line_end=4,
    )


@pytest.fixture
def math_class() -> Chunk:
    """Shares ``blob_sha='blob_math'`` with ``math_func`` on purpose."""
    return Chunk(
        id="math_class_1",
        blob_sha="blob_math",
        file_path="src/math_utils.py",
        kind=ChunkKind.CLASS,
        name="StatisticsCalculator",
        content=(
            "class StatisticsCalculator:\n"
            "    def __init__(self, data: list[float]):\n"
            "        self.data = data\n"
            "    def mean(self) -> float:\n"
            "        return sum(self.data) / len(self.data)\n"
        ),
        line_start=10,
        line_end=15,
    )


@pytest.fixture
def all_store_chunks(
    math_func: Chunk,
    http_func: Chunk,
    string_func: Chunk,
    math_class: Chunk,
) -> list[Chunk]:
    return [math_func, http_func, string_func, math_class]


@pytest.fixture
def vec_math() -> list[float]:
    return [1.0, 0.0, 0.0]


@pytest.fixture
def vec_http() -> list[float]:
    return [0.0, 1.0, 0.0]


@pytest.fixture
def vec_string() -> list[float]:
    return [0.0, 0.0, 1.0]


# ── GC building-block chunks ─────────────────────────────────────────
#
# Blob-distinct minimal chunks used by GC scenarios.


@pytest.fixture
def gc_chunk_x() -> Chunk:
    return Chunk(
        id="cx",
        blob_sha="blob_x",
        file_path="x.py",
        kind=ChunkKind.FUNCTION,
        name="f_x",
        content="def f_x(): pass",
        line_start=1,
        line_end=1,
    )


@pytest.fixture
def gc_chunk_y() -> Chunk:
    return Chunk(
        id="cy",
        blob_sha="blob_y",
        file_path="y.py",
        kind=ChunkKind.FUNCTION,
        name="f_y",
        content="def f_y(): pass",
        line_start=1,
        line_end=1,
    )


@pytest.fixture
def gc_chunk_z() -> Chunk:
    return Chunk(
        id="cz",
        blob_sha="blob_z",
        file_path="z.py",
        kind=ChunkKind.FUNCTION,
        name="f_z",
        content="def f_z(): pass",
        line_start=1,
        line_end=1,
    )


# ── Edge fixtures used by edge-family cases ─────────────────────────


@pytest.fixture
def edge_math_calls_class() -> Edge:
    return Edge(
        source_id="math_1", target_id="math_class_1", kind=EdgeKind.CALLS
    )


@pytest.fixture
def edge_a_calls_b() -> Edge:
    return Edge(source_id="a", target_id="b", kind=EdgeKind.CALLS)


@pytest.fixture
def edge_c_imports_d() -> Edge:
    return Edge(source_id="c", target_id="d", kind=EdgeKind.IMPORTS)
