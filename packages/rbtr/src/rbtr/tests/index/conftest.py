"""Shared fixtures for `tests/index/`.

Two fixture families:

1. **Ranking dataset** — a seeded IndexStore for the search
   ranking and structural-boost tests.

2. **Small named chunks / edges** — building blocks consumed
   by case files for store-level behavioural tests.

No module-level test data.  Everything is a fixture so
dependency graphs are explicit.
"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pygit2
import pytest

from rbtr.domain.models import (
    Chunk,
    ChunkKind,
    Edge,
    EdgeKind,
    FileSnapshot,
    SnapshotRef,
)
from rbtr.domain.tokenise import tokenise_code
from rbtr.index.build import build_index
from rbtr.index.staging import TokenisedChunk
from rbtr.index.store import IndexStore

# ═════════════════════════════════════════════════════════════════════
# Builders (no side effects, no hidden state)
# ═════════════════════════════════════════════════════════════════════


def make_chunk(
    handle: str,
    *,
    name: str = "",
    content: str = "",
    path: str = "f.py",
    blob: str = "",
    kind: ChunkKind = ChunkKind.FUNCTION,
    language: str = "",
    file_language: str | None = None,
) -> TokenisedChunk:
    """Build a minimal `TokenisedChunk` with auto-tokenised fields.

    *handle* is a short label for the chunk within a test: it seeds the
    default `name` and `blob_sha`, which is what keeps two fixture chunks
    distinct. It is not the id — that is derived from the content, so
    assert against a fixture's `.id` or its `name`, never a literal.

    Chunks are content-addressed and carry no `repo_id`; which repo a
    chunk belongs to is decided by the snapshot that references its
    blob (see `seed_store`'s `repo_id`).

    `file_language` defaults to *language*, which is what a file in one
    language yields. Pass it separately for an embedded chunk, whose own
    language differs from its file's — and match it to the
    `detected_language` of the `make_snap` rows that reference the blob,
    or the two will not pair up.
    """
    name = name or handle
    content = content or f"def {name}(): pass"
    return TokenisedChunk(
        blob_sha=blob or f"blob_{handle}",
        file_path=path,
        kind=kind,
        name=name,
        content=content,
        content_tokens=tokenise_code(content),
        name_tokens=tokenise_code(name),
        line_start=1,
        line_end=1,
        language=language,
        file_language=language if file_language is None else file_language,
    )


def make_snap(sha: str, path: str, blob: str, language: str = "") -> FileSnapshot:
    """Build a `FileSnapshot`.

    *language* is the file's detected language, which pairs the snapshot
    with the chunks extracted from it.
    """
    return FileSnapshot(snapshot_sha=sha, file_path=path, blob_sha=blob, detected_language=language)


def seed_store(
    store: IndexStore,
    chunks: list[TokenisedChunk],
    *,
    snapshot_sha: str = "head",
    mark_indexed: bool = True,
    repo_id: int | None = None,
) -> int:
    """Insert chunks + snapshots into a store via session; return the repo id.

    Snapshots reference the chunks' blobs under *repo_id*.  Seed several
    repos by calling once per repo (registering each first and passing its
    id); pass the same chunk to two repos to model content shared across
    worktrees/clones.

    When *repo_id* is omitted a repo is registered here, because per-repo
    rows must hang off a real `repos` row — the foreign key rejects an id
    that was never registered.  The path is synthetic: these are
    store-level tests that never read git, and registration is about
    identity, not about a directory existing on disk.
    """
    with store.session() as ws:
        if repo_id is None:
            repo_id = ws.register_repo("/repo")
        for c in chunks:
            ws.add_chunk(c)
        ws.insert_snapshots(
            [
                FileSnapshot(snapshot_sha=snapshot_sha, file_path=c.file_path, blob_sha=c.blob_sha)
                for c in chunks
            ],
            repo_id=repo_id,
        )
        if mark_indexed:
            ws.mark_indexed(repo_id, snapshot_sha)
    return repo_id


# ═════════════════════════════════════════════════════════════════════
# Cross-repo content sharing
# ═════════════════════════════════════════════════════════════════════


@pytest.fixture
def shared_chunk() -> TokenisedChunk:
    """The single chunk that `shared_chunk_store`'s two repos share."""
    return make_chunk("shared_fn", path="x.py", blob="b_shared")


@pytest.fixture
def shared_chunk_store(store: IndexStore, shared_chunk: TokenisedChunk) -> IndexStore:
    """A store where one chunk is shared by repo 1 and repo 2.

    Models byte-identical content in two worktrees/clones of one
    repository: the blob (and so the chunk `id`) coincides. The chunk
    is inserted **once**; each repo records a snapshot referencing the
    same blob — mirroring the `blob_is_current` skip that happens when a
    second repo indexes already-chunked content.
    """
    with store.session() as ws:
        ws.register_repo("/repo1")
        ws.register_repo("/repo2")
        ws.add_chunk(shared_chunk)
        ws.insert_snapshots([make_snap("head", "x.py", "b_shared")], repo_id=1)
        ws.insert_snapshots([make_snap("head", "x.py", "b_shared")], repo_id=2)
        ws.mark_indexed(1, "head")
        ws.mark_indexed(2, "head")
    return store


# ═════════════════════════════════════════════════════════════════════
# Commits over the shared `git_repo` fixture (defined in the root
# conftest, alongside `store` — both are reused beyond `tests/index/`).
# ═════════════════════════════════════════════════════════════════════


@pytest.fixture
def snapshot_sha(git_repo: pygit2.Repository) -> str:
    """SHA of the initial commit."""
    return str(git_repo.head.target)


@pytest.fixture
def two_commits(git_repo: pygit2.Repository, tmp_path: Path) -> tuple[str, str]:
    """Add a second commit that modifies utils.py and adds a new file."""
    base_sha = str(git_repo.head.target)

    utils_path = tmp_path / "src" / "utils.py"
    utils_path.write_bytes(b"""\
\"\"\"Utility functions.\"\"\"

def helper():
    return 42

def format_name(name):
    return name.strip()

def new_func():
    return "new"
""")

    new_path = tmp_path / "src" / "service.py"
    new_path.parent.mkdir(parents=True, exist_ok=True)
    new_path.write_bytes(b"""\
\"\"\"Service layer.\"\"\"

def serve():
    return True
""")

    index = git_repo.index
    index.add("src/utils.py")
    index.add("src/service.py")
    index.write()
    tree_oid = index.write_tree()
    sig = pygit2.Signature("Test", "test@test.com")
    parent = git_repo.get(git_repo.head.target)
    assert parent is not None
    git_repo.create_commit("HEAD", sig, sig, "Add new_func and service", tree_oid, [parent.id])

    head_sha = str(git_repo.head.target)
    return base_sha, head_sha


# ═════════════════════════════════════════════════════════════════════
# Symbol-diff dataset (for test_diff_symbols.py)
# ═════════════════════════════════════════════════════════════════════


@pytest.fixture
def diff_repo(tmp_path: Path) -> pygit2.Repository:
    """Empty git repo (no commits) for symbol-diff scenarios."""
    return pygit2.init_repository(str(tmp_path), bare=False, initial_head="main")


# ═════════════════════════════════════════════════════════════════════
# Duplicated content (for test_duplicate_content.py)
#
# Copies land in a *second* commit because that is when the bug bites:
# the originals are already committed, so the dedup gate sees their
# blobs and skips the copies.  In one commit the copies survive by
# accident, the chunk buffer not having flushed yet.
#
# `src/caller.py` is a weaker match for the ranking query: min-max
# normalisation scores every candidate 0.0 when they all tie, so a
# penalty needs something to separate.


@pytest.fixture
def dup_store(tmp_path: Path, store: IndexStore) -> IndexStore:
    """Originals, then verbatim copies at new paths, both built."""
    shared = b'''\
"""Shared helpers."""


def parse_manifest(text):
    """Read a manifest and return its entries."""
    return [line for line in text.splitlines() if line]
'''
    api = b"""\
export function describeEndpoint(name) {
  return `endpoint ${name}`;
}
"""
    dup = b'''\
def normalise_widget(widget):
    """Normalise a widget before serialisation."""
    return widget.strip().lower()
'''
    caller = b'''\
from dup import normalise_widget


def render_widget(widget):
    """Render a widget for display."""
    return normalise_widget(widget)
'''

    repo = pygit2.init_repository(str(tmp_path), bare=False, initial_head="main")
    sig = pygit2.Signature("Test", "test@test.com")
    commits: list[str] = []
    for message, files in (
        (
            "Originals",
            {
                "src/shared.py": shared,
                "types/api.d.ts": api,
                "src/dup.py": dup,
                "src/caller.py": caller,
            },
        ),
        ("Copies", {"lib/shared.py": shared, "dist/api.js": api, "node_modules/dup.py": dup}),
    ):
        for path, content in files.items():
            full = tmp_path / path
            full.parent.mkdir(parents=True, exist_ok=True)
            full.write_bytes(content)
            repo.index.add(path)
        repo.index.write()
        parents = [repo.head.target] if commits else []
        tree = repo.index.write_tree()
        commits.append(str(repo.create_commit("HEAD", sig, sig, message, tree, parents)))

    build_index(repo.workdir, commits[0], store)
    build_index(repo.workdir, commits[1], store, base_sha=commits[0])
    return store


@pytest.fixture
def dup_ref(dup_store: IndexStore) -> SnapshotRef:
    """The duplicated repo at its head commit."""
    ref = dup_store.latest_ref(dup_store.list_repos()[0])
    assert ref is not None
    return ref


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
    name = "AppConfig"
    content = """\
class AppConfig:
    database_url: str
    max_retries: int = 3
    timeout: float = 30.0
"""
    return TokenisedChunk(
        blob_sha="blob_config",
        file_path="src/config.py",
        kind=ChunkKind.CLASS,
        name=name,
        content=content,
        content_tokens=tokenise_code(content),
        name_tokens=tokenise_code(name),
        line_start=1,
        line_end=5,
    )


@pytest.fixture
def ranking_load_config() -> Chunk:
    name = "load_config"
    content = """\
def load_config(path: str) -> AppConfig:
    with open(path) as f:
        data = json.load(f)
    return AppConfig(**data)
"""
    return TokenisedChunk(
        blob_sha="blob_config",
        file_path="src/config.py",
        kind=ChunkKind.FUNCTION,
        name=name,
        content=content,
        content_tokens=tokenise_code(content),
        name_tokens=tokenise_code(name),
        line_start=10,
        line_end=14,
    )


@pytest.fixture
def ranking_import_config() -> Chunk:
    name = "from config import AppConfig"
    content = "from config import AppConfig, load_config"
    return TokenisedChunk(
        blob_sha="blob_server",
        file_path="src/server.py",
        kind=ChunkKind.IMPORT,
        name=name,
        content=content,
        content_tokens=tokenise_code(content),
        name_tokens=tokenise_code(name),
        line_start=1,
        line_end=1,
    )


@pytest.fixture
def ranking_start_server() -> Chunk:
    name = "start_server"
    content = """\
def start_server(config: AppConfig) -> None:
    app = create_app(config)
    app.run(host="0.0.0.0", port=config.port)
"""
    return TokenisedChunk(
        blob_sha="blob_server",
        file_path="src/server.py",
        kind=ChunkKind.FUNCTION,
        name=name,
        content=content,
        content_tokens=tokenise_code(content),
        name_tokens=tokenise_code(name),
        line_start=5,
        line_end=8,
    )


@pytest.fixture
def ranking_test_config() -> Chunk:
    name = "test_load_config"
    content = """\
def test_load_config():
    config = load_config("test.json")
    assert isinstance(config, AppConfig)
    assert config.max_retries == 3
    config = load_config("other.json")
    assert config.timeout == 30.0
    config = load_config("empty.json")
"""
    return TokenisedChunk(
        blob_sha="blob_test_config",
        file_path="tests/test_config.py",
        kind=ChunkKind.FUNCTION,
        name=name,
        content=content,
        content_tokens=tokenise_code(content),
        name_tokens=tokenise_code(name),
        line_start=1,
        line_end=7,
    )


@pytest.fixture
def ranking_doc_section() -> Chunk:
    name = "Configuration"
    content = """\
## Configuration

Use `load_config` to load an `AppConfig` from a JSON file.
Set `database_url` and `max_retries` as needed.
"""
    return TokenisedChunk(
        blob_sha="blob_docs",
        file_path="docs/setup.md",
        kind=ChunkKind.DOC_SECTION,
        name=name,
        content=content,
        content_tokens=tokenise_code(content),
        name_tokens=tokenise_code(name),
        line_start=1,
        line_end=5,
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
def ranking_edges(
    ranking_config_class: Chunk,
    ranking_load_config: Chunk,
    ranking_import_config: Chunk,
    ranking_start_server: Chunk,
    ranking_test_config: Chunk,
    ranking_doc_section: Chunk,
) -> list[Edge]:
    """Edge graph used by importance / structural ranking tests.

    The config class receives 2 inbound edges (import, start_server),
    `load_config` 3 (import, call, doc), and `start_server` none. Edges
    carry the chunks' real ids, so they point at rows that exist.
    """
    return [
        Edge(
            source_id=ranking_import_config.id,
            target_id=ranking_config_class.id,
            kind=EdgeKind.IMPORTS,
            source_path=ranking_import_config.file_path,
            target_path=ranking_config_class.file_path,
        ),
        Edge(
            source_id=ranking_import_config.id,
            target_id=ranking_load_config.id,
            kind=EdgeKind.IMPORTS,
            source_path=ranking_import_config.file_path,
            target_path=ranking_load_config.file_path,
        ),
        Edge(
            source_id=ranking_start_server.id,
            target_id=ranking_config_class.id,
            kind=EdgeKind.IMPORTS,
            source_path=ranking_start_server.file_path,
            target_path=ranking_config_class.file_path,
        ),
        Edge(
            source_id=ranking_test_config.id,
            target_id=ranking_load_config.id,
            kind=EdgeKind.CALLS,
            source_path=ranking_test_config.file_path,
            target_path=ranking_load_config.file_path,
        ),
        Edge(
            source_id=ranking_doc_section.id,
            target_id=ranking_load_config.id,
            kind=EdgeKind.DOCUMENTS,
            source_path=ranking_doc_section.file_path,
            target_path=ranking_load_config.file_path,
        ),
    ]


@pytest.fixture
def ranking_store(
    ranking_commit: str,
    ranking_chunks: list[Chunk],
    ranking_edges: list[Edge],
) -> Generator[IndexStore]:
    """An in-memory IndexStore pre-loaded with the ranking dataset."""
    store = IndexStore(writable=True)
    with store.session() as ws:
        ws.register_repo("/repo")
        for c in ranking_chunks:
            tc = c if isinstance(c, TokenisedChunk) else TokenisedChunk(**c.model_dump())
            ws.add_chunk(tc)
        ws.insert_snapshots(
            [
                FileSnapshot(
                    snapshot_sha=ranking_commit, file_path=c.file_path, blob_sha=c.blob_sha
                )
                for c in ranking_chunks
            ],
            repo_id=1,
        )
        ws.insert_edges(ranking_edges, ranking_commit, repo_id=1)
    yield store
    store.close()


# ═════════════════════════════════════════════════════════════════════
# Small named chunks / edges (for case_store_* families)
# ═════════════════════════════════════════════════════════════════════


@pytest.fixture
def math_func() -> TokenisedChunk:
    return TokenisedChunk(
        blob_sha="blob_math",
        file_path="src/math_utils.py",
        kind=ChunkKind.FUNCTION,
        name="calculate_standard_deviation",
        content="""\
def calculate_standard_deviation(values: list[float]) -> float:
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return variance ** 0.5
""",
        line_start=1,
        line_end=4,
    )


@pytest.fixture
def http_func() -> TokenisedChunk:
    return TokenisedChunk(
        blob_sha="blob_http",
        file_path="src/api/client.py",
        kind=ChunkKind.FUNCTION,
        name="fetch_json_from_endpoint",
        content="""\
async def fetch_json_from_endpoint(url: str, headers: dict) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers)
        response.raise_for_status()
        return response.json()
""",
        line_start=10,
        line_end=15,
    )


@pytest.fixture
def string_func() -> TokenisedChunk:
    return TokenisedChunk(
        blob_sha="blob_string",
        file_path="src/text/normalize.py",
        kind=ChunkKind.FUNCTION,
        name="normalize_whitespace",
        content="""\
def normalize_whitespace(text: str) -> str:
    import re
    collapsed = re.sub(r'\\s+', ' ', text)
    return collapsed.strip()
""",
        line_start=1,
        line_end=4,
    )


@pytest.fixture
def math_class() -> Chunk:
    """Shares `blob_sha='blob_math'` with `math_func` on purpose."""
    return TokenisedChunk(
        blob_sha="blob_math",
        file_path="src/math_utils.py",
        kind=ChunkKind.CLASS,
        name="StatisticsCalculator",
        content="""\
class StatisticsCalculator:
    def __init__(self, data: list[float]):
        self.data = data
    def mean(self) -> float:
        return sum(self.data) / len(self.data)
""",
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


# ── GC building-block chunks ─────────────────────────────────────────


@pytest.fixture
def gc_chunk_x() -> Chunk:
    return TokenisedChunk(
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
    return TokenisedChunk(
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
    return TokenisedChunk(
        blob_sha="blob_z",
        file_path="z.py",
        kind=ChunkKind.FUNCTION,
        name="f_z",
        content="def f_z(): pass",
        line_start=1,
        line_end=1,
    )


# ── Edge fixtures ────────────────────────────────────────────────────


@pytest.fixture
def edge_math_calls_class() -> Edge:
    return Edge(
        source_id="math_1",
        target_id="math_class_1",
        kind=EdgeKind.CALLS,
        source_path="src/math_1.py",
        target_path="src/math_class_1.py",
    )


@pytest.fixture
def edge_a_calls_b() -> Edge:
    return Edge(
        source_id="a",
        target_id="b",
        kind=EdgeKind.CALLS,
        source_path="src/a.py",
        target_path="src/b.py",
    )


@pytest.fixture
def edge_c_imports_d() -> Edge:
    return Edge(
        source_id="c",
        target_id="d",
        kind=EdgeKind.IMPORTS,
        source_path="src/c.py",
        target_path="src/d.py",
    )
