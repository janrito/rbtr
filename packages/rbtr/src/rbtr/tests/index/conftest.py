"""Shared fixtures for `tests/index/`.

Two fixture families:

1. **Ranking dataset** — a seeded Index for the search
   ranking and structural-boost tests.

2. **Small named chunks / edges** — building blocks consumed
   by case files for store-level behavioural tests.

No module-level test data.  Everything is a fixture so
dependency graphs are explicit.
"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import polars as pl
import pygit2
import pytest

from rbtr.domain.models import (
    Chunk,
    ChunkKind,
    Definition,
    EdgeKind,
    FileSnapshot,
    Import,
    ImportMeta,
    Section,
    SnapshotRef,
    Span,
)
from rbtr.domain.tokenise import tokenise_code
from rbtr.git import normalise_repo_path
from rbtr.index.build import build_index
from rbtr.index.index import Index
from rbtr.index.refs import latest_ref
from rbtr.index.staging import TokenisedChunk
from rbtr.languages.edges import InferredEdgeRow

from ..conftest import make_commit

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
    scope: str = "",
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
    blob (see `seed_index`'s `ref`).

    `file_language` defaults to *language*, which is what a file in one
    language yields. Pass it separately for an embedded chunk, whose own
    language differs from its file's — and match it to the
    `detected_language` of the `make_snap` rows that reference the blob,
    or the two will not pair up.
    """
    name = name or handle
    content = content or f"def {name}(): pass"
    return TokenisedChunk.model_validate(
        {
            "blob_sha": blob or f"blob_{handle}",
            "file_path": path,
            "role": {"kind": kind, "name": name, "meta": {}},
            "scope": scope,
            "content": content,
            "content_tokens": tokenise_code(content),
            "name_tokens": tokenise_code(name),
            "span": {"start": 1, "end": 1},
            "language": language,
            "file_language": language if file_language is None else file_language,
        }
    )


def make_snap(sha: str, path: str, blob: str, language: str = "") -> FileSnapshot:
    """Build a `FileSnapshot`.

    *language* is the file's detected language, which pairs the snapshot
    with the chunks extracted from it.
    """
    return FileSnapshot(snapshot_sha=sha, file_path=path, blob_sha=blob, detected_language=language)


def seed_index(index: Index, chunks: list[TokenisedChunk], ref: SnapshotRef) -> None:
    """Insert chunks, and snapshots referencing their blobs, under *ref*.

    *ref* is marked indexed, so the seeded snapshot is one a read can
    find.

    The repo named by *ref* must already be registered: its rows live in
    its project's file, which registration makes.  `head_ref` does that
    registration for the common single-repo case; a test seeding several
    repos registers each one and takes a ref from the repo it gets back,
    then calls this once per ref.  Passing the same chunk under two refs
    models content shared across worktrees or clones.
    """
    with index.session(ref.repo) as ws:
        for c in chunks:
            ws.add_chunk(c)
        ws.insert_snapshots(
            [
                FileSnapshot(
                    snapshot_sha=ref.snapshot_sha,
                    file_path=c.file_path,
                    blob_sha=c.blob_sha,
                    detected_language=c.file_language,
                )
                for c in chunks
            ],
        )
        ws.mark_indexed(snapshot_sha=ref.snapshot_sha)


@pytest.fixture
def head_ref(index: Index) -> SnapshotRef:
    """Repo `/repo` registered in `index`, at snapshot `head`.

    The repo comes back from the registration, so the ref names a repo
    that exists.  The path is synthetic: these are store-level tests
    that never read git, and registration establishes identity.
    """
    return index.register("/repo").at("head")


# ═════════════════════════════════════════════════════════════════════
# Cross-repo content sharing
# ═════════════════════════════════════════════════════════════════════


@pytest.fixture
def shared_sha(git_repo: pygit2.Repository) -> str:
    """A commit of `git_repo` holding only `x.py`, which defines `shared_fn`."""
    x = b"""\
def shared_fn():
    pass
"""
    return str(make_commit(git_repo, {"x.py": x}, ref="refs/heads/shared"))


@pytest.fixture
def shared_chunk_index(
    index: Index, shared_sha: str, repo_path: str, linked_worktree: pygit2.Repository
) -> Index:
    """Both checkouts of one project built at `shared_sha`, so they share one chunk.

    The main checkout is repo 1 and the linked worktree repo 2. The
    second build finds the blob already chunked and only records its
    snapshot, as a second checkout does in production.
    """
    build_index(repo_path, shared_sha, index)
    build_index(normalise_repo_path(linked_worktree.workdir), shared_sha, index)
    return index


@pytest.fixture
def shared_chunk_id(shared_chunk_index: Index, shared_sha: str) -> str:
    """The id of `shared_fn`, the one chunk `shared_chunk_index`'s repos share."""
    repo1 = shared_chunk_index.repos()[0]
    return str(shared_chunk_index.chunks(at=repo1.at(shared_sha), name="shared_fn")["id"].item())


@pytest.fixture
def m_py() -> bytes:
    """`m.py`, defining `fn`: byte-identical wherever it is committed."""
    return b"""\
def fn():
    return 1
"""


@pytest.fixture
def importer_a_sha(git_repo: pygit2.Repository, m_py: bytes) -> str:
    """A commit holding `m.py` and `a.py`, which imports `fn` from it."""
    a = b"""\
from m import fn
"""
    return str(make_commit(git_repo, {"m.py": m_py, "a.py": a}, ref="refs/heads/a"))


@pytest.fixture
def importer_b_sha(git_repo: pygit2.Repository, m_py: bytes) -> str:
    """A commit holding `m.py` and `b.py`, which imports `fn` from it."""
    b = b"""\
from m import fn
"""
    return str(make_commit(git_repo, {"m.py": m_py, "b.py": b}, ref="refs/heads/b"))


@pytest.fixture
def shared_target(
    index: Index,
    repo_path: str,
    linked_worktree: pygit2.Repository,
    importer_a_sha: str,
    importer_b_sha: str,
) -> Index:
    """Two checkouts of one project, each importing `fn` from its own copy of `m.py`.

    The main checkout (repo 1) is built at `importer_a_sha` and the
    linked worktree (repo 2) at `importer_b_sha`. `m.py` is the same in
    both, so `fn` is one chunk the two snapshots share.
    """
    build_index(repo_path, importer_a_sha, index)
    build_index(normalise_repo_path(linked_worktree.workdir), importer_b_sha, index)
    return index


# ═════════════════════════════════════════════════════════════════════
# Edges built from source
# ═════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="module")
def import_graph(module_index: Index, module_repo: pygit2.Repository) -> SnapshotRef:
    """A snapshot, built into `module_index`, where code imports a function and prose links to its file.

    `src/app.py` imports `fn` from `src/m.py`, an `imports` edge;
    `README.md` links to `src/m.py`, a `documents` edge to `fn`.
    """
    m = b"""\
def fn():
    return 1
"""
    app = b"""\
from src.m import fn


def run():
    return fn()
"""
    readme = b"""\
# Guide

See [fn](src/m.py).
"""
    files = {"src/m.py": m, "src/app.py": app, "README.md": readme}
    sha = str(make_commit(module_repo, files, ref="refs/heads/graph"))
    path = normalise_repo_path(module_repo.workdir)
    build_index(path, sha, module_index)
    return module_index.register(path).at(sha)


# ═════════════════════════════════════════════════════════════════════
# Commits over the shared `git_repo` fixture (defined in the root
# conftest, alongside `index` — both are reused beyond `tests/index/`).
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

    git_index = git_repo.index
    git_index.add("src/utils.py")
    git_index.add("src/service.py")
    git_index.write()
    tree_oid = git_index.write_tree()
    sig = pygit2.Signature("Test", "test@test.com")
    parent = git_repo.get(git_repo.head.target)
    assert parent is not None
    git_repo.create_commit("HEAD", sig, sig, "Add new_func and service", tree_oid, [parent.id])

    head_sha = str(git_repo.head.target)
    return base_sha, head_sha


# ═════════════════════════════════════════════════════════════════════
# Symbol-diff dataset (for test_symbol_diff.py)
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
def dup_index(tmp_path: Path, index: Index) -> Index:
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

    build_index(repo.workdir, commits[0], index)
    build_index(repo.workdir, commits[1], index, base_sha=commits[0])
    return index


@pytest.fixture
def dup_ref(dup_index: Index) -> SnapshotRef:
    """The duplicated repo at its head commit."""
    ref = latest_ref(dup_index, dup_index.repos()[0])
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
        role=Definition(kind=ChunkKind.CLASS, name=name),
        content=content,
        content_tokens=tokenise_code(content),
        name_tokens=tokenise_code(name),
        span=Span(start=1, end=5),
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
        role=Definition(kind=ChunkKind.FUNCTION, name=name),
        content=content,
        content_tokens=tokenise_code(content),
        name_tokens=tokenise_code(name),
        span=Span(start=10, end=14),
    )


@pytest.fixture
def ranking_import_config() -> Chunk:
    name = "from config import AppConfig"
    content = "from config import AppConfig, load_config"
    return TokenisedChunk(
        blob_sha="blob_server",
        file_path="src/server.py",
        role=Import(name=name, meta=ImportMeta()),
        content=content,
        content_tokens=tokenise_code(content),
        name_tokens=tokenise_code(name),
        span=Span(start=1, end=1),
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
        role=Definition(kind=ChunkKind.FUNCTION, name=name),
        content=content,
        content_tokens=tokenise_code(content),
        name_tokens=tokenise_code(name),
        span=Span(start=5, end=8),
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
        role=Definition(kind=ChunkKind.FUNCTION, name=name),
        content=content,
        content_tokens=tokenise_code(content),
        name_tokens=tokenise_code(name),
        span=Span(start=1, end=7),
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
        role=Section(kind=ChunkKind.DOC_SECTION, name=name),
        content=content,
        content_tokens=tokenise_code(content),
        name_tokens=tokenise_code(name),
        span=Span(start=1, end=5),
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
) -> list[dict[str, str]]:
    """Edge graph used by importance / structural ranking tests.

    The config class receives 2 inbound edges (import, start_server),
    `load_config` 3 (import, call, doc), and `start_server` none. Edges
    carry the chunks' real ids, so they point at rows that exist.
    """
    return [
        {
            "source_id": ranking_import_config.id,
            "target_id": ranking_config_class.id,
            "kind": EdgeKind.IMPORTS,
            "source_path": ranking_import_config.file_path,
            "target_path": ranking_config_class.file_path,
        },
        {
            "source_id": ranking_import_config.id,
            "target_id": ranking_load_config.id,
            "kind": EdgeKind.IMPORTS,
            "source_path": ranking_import_config.file_path,
            "target_path": ranking_load_config.file_path,
        },
        {
            "source_id": ranking_start_server.id,
            "target_id": ranking_config_class.id,
            "kind": EdgeKind.IMPORTS,
            "source_path": ranking_start_server.file_path,
            "target_path": ranking_config_class.file_path,
        },
        {
            "source_id": ranking_test_config.id,
            "target_id": ranking_load_config.id,
            "kind": EdgeKind.CALLS,
            "source_path": ranking_test_config.file_path,
            "target_path": ranking_load_config.file_path,
        },
        {
            "source_id": ranking_doc_section.id,
            "target_id": ranking_load_config.id,
            "kind": EdgeKind.DOCUMENTS,
            "source_path": ranking_doc_section.file_path,
            "target_path": ranking_load_config.file_path,
        },
    ]


@pytest.fixture
def ranking_index(
    ranking_commit: str,
    ranking_chunks: list[Chunk],
    ranking_edges: list[dict[str, str]],
) -> Generator[Index]:
    """An in-memory Index pre-loaded with the ranking dataset."""
    # Written by hand, not built: `ranking_edges` holds a `calls` edge,
    # and no build infers `calls` edges yet.
    index = Index(writable=True)
    repo = index.register("/repo")
    with index.session(repo) as ws:
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
        )
        ws.insert_edges(
            InferredEdgeRow.validate(pl.DataFrame(ranking_edges), cast=True),
            snapshot_sha=ranking_commit,
        )
    yield index
    index.close()


# ═════════════════════════════════════════════════════════════════════
# Small named chunks / edges (for case_store_* families)
# ═════════════════════════════════════════════════════════════════════


@pytest.fixture
def math_func() -> TokenisedChunk:
    return TokenisedChunk(
        blob_sha="blob_math",
        file_path="src/math_utils.py",
        role=Definition(kind=ChunkKind.FUNCTION, name="calculate_standard_deviation"),
        content="""\
def calculate_standard_deviation(values: list[float]) -> float:
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return variance ** 0.5
""",
        span=Span(start=1, end=4),
    )


@pytest.fixture
def http_func() -> TokenisedChunk:
    return TokenisedChunk(
        blob_sha="blob_http",
        file_path="src/api/client.py",
        role=Definition(kind=ChunkKind.FUNCTION, name="fetch_json_from_endpoint"),
        content="""\
async def fetch_json_from_endpoint(url: str, headers: dict) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers)
        response.raise_for_status()
        return response.json()
""",
        span=Span(start=10, end=15),
    )


@pytest.fixture
def string_func() -> TokenisedChunk:
    return TokenisedChunk(
        blob_sha="blob_string",
        file_path="src/text/normalize.py",
        role=Definition(kind=ChunkKind.FUNCTION, name="normalize_whitespace"),
        content="""\
def normalize_whitespace(text: str) -> str:
    import re
    collapsed = re.sub(r'\\s+', ' ', text)
    return collapsed.strip()
""",
        span=Span(start=1, end=4),
    )
