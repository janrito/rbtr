"""Shared test dataset for search ranking and structural tests.

Models a small but realistic project with deliberate ranking
conflicts.  Every chunk is semantically distinct and exercises
a specific scoring signal: kind boost, file-category penalty,
importance (inbound degree), and proximity (diff distance).

All E2E search tests import the ``seeded_store`` fixture, which
returns an :class:`IndexStore` pre-loaded with this dataset.
"""

from __future__ import annotations

import pytest

from rbtr.index.models import Chunk, ChunkKind, Edge, EdgeKind
from rbtr.index.store import IndexStore
from rbtr.index.tokenise import tokenise_code

# ── Commit SHA used for all snapshots ────────────────────────────────

COMMIT = "abc123"


# ── Chunks ───────────────────────────────────────────────────────────
#
# Six chunks across four files: source module, a consumer module,
# a test file, and a doc file.  Deliberate conflicts:
#
#   * CLASS vs IMPORT of the same name → kind boost resolves
#   * Source vs test mentioning the same function → file penalty resolves
#   * High-df term ("config") in 5/6 chunks → IDF neutralised
#   * Exact name match vs content-only mention → name score resolves
#   * 3 inbound edges (config_class) vs 0 (start_server) → importance
#   * Diff touching src/server.py → proximity

CONFIG_CLASS = Chunk(
    id="config_class",
    blob_sha="blob_config",
    file_path="src/config.py",
    kind=ChunkKind.CLASS,
    name="AppConfig",
    content="""\
class AppConfig:
    database_url: str
    max_retries: int = 3
    timeout: float = 30.0
""",
    line_start=1,
    line_end=5,
)

LOAD_CONFIG = Chunk(
    id="load_config",
    blob_sha="blob_config",
    file_path="src/config.py",
    kind=ChunkKind.FUNCTION,
    name="load_config",
    content="""\
def load_config(path: str) -> AppConfig:
    with open(path) as f:
        data = json.load(f)
    return AppConfig(**data)
""",
    line_start=10,
    line_end=14,
)

IMPORT_CONFIG = Chunk(
    id="import_config",
    blob_sha="blob_server",
    file_path="src/server.py",
    kind=ChunkKind.IMPORT,
    name="from config import AppConfig",
    content="from config import AppConfig, load_config",
    line_start=1,
    line_end=1,
)

START_SERVER = Chunk(
    id="start_server",
    blob_sha="blob_server",
    file_path="src/server.py",
    kind=ChunkKind.FUNCTION,
    name="start_server",
    content="""\
def start_server(config: AppConfig) -> None:
    app = create_app(config)
    app.run(host="0.0.0.0", port=config.port)
""",
    line_start=5,
    line_end=8,
)

TEST_CONFIG = Chunk(
    id="test_config",
    blob_sha="blob_test_config",
    file_path="tests/test_config.py",
    kind=ChunkKind.FUNCTION,
    name="test_load_config",
    content="""\
def test_load_config():
    config = load_config("test.json")
    assert isinstance(config, AppConfig)
    assert config.max_retries == 3
    config = load_config("other.json")
    assert config.timeout == 30.0
    config = load_config("empty.json")
""",
    line_start=1,
    line_end=7,
)

DOC_SECTION = Chunk(
    id="doc_config",
    blob_sha="blob_docs",
    file_path="docs/setup.md",
    kind=ChunkKind.DOC_SECTION,
    name="Configuration",
    content="""\
## Configuration

Use `load_config` to load an `AppConfig` from a JSON file.
Set `database_url` and `max_retries` as needed.
""",
    line_start=1,
    line_end=5,
)

ALL_CHUNKS: list[Chunk] = [
    CONFIG_CLASS,
    LOAD_CONFIG,
    IMPORT_CONFIG,
    START_SERVER,
    TEST_CONFIG,
    DOC_SECTION,
]

# Pre-tokenise once at import — deterministic and avoids mutation
# inside the fixture.
for _c in ALL_CHUNKS:
    _c.content_tokens = tokenise_code(_c.content)
    _c.name_tokens = tokenise_code(_c.name)


# ── Edges ────────────────────────────────────────────────────────────
#
# config_class receives 2 inbound edges (import, start_server).
# load_config receives 3 inbound edges (import, test, doc).
# start_server receives 0.

ALL_EDGES: list[Edge] = [
    # import_config imports from config.py
    Edge(source_id="import_config", target_id="config_class", kind=EdgeKind.IMPORTS),
    Edge(source_id="import_config", target_id="load_config", kind=EdgeKind.IMPORTS),
    # start_server depends on AppConfig
    Edge(source_id="start_server", target_id="config_class", kind=EdgeKind.IMPORTS),
    # test_config tests load_config
    Edge(source_id="test_config", target_id="load_config", kind=EdgeKind.TESTS),
    # doc references load_config
    Edge(source_id="doc_config", target_id="load_config", kind=EdgeKind.DOCUMENTS),
]


# ── Fixture ──────────────────────────────────────────────────────────


@pytest.fixture
def seeded_store() -> IndexStore:
    """An in-memory IndexStore pre-loaded with the shared dataset.

    Chunks have tokenised content/name fields, FTS is rebuilt,
    and edges are inserted.  No embeddings.
    """
    store = IndexStore()
    store.insert_chunks(ALL_CHUNKS)
    for c in ALL_CHUNKS:
        store.insert_snapshot(COMMIT, c.file_path, c.blob_sha)
    store.insert_edges(ALL_EDGES, COMMIT)
    store.rebuild_fts_index()
    return store
