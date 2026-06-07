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
from pygit2.enums import FileMode

from rbtr.index.models import (
    Chunk,
    ChunkKind,
    Edge,
    EdgeKind,
    Snapshot,
    TokenisedChunk,
)
from rbtr.index.store import IndexStore
from rbtr.index.tokenise import tokenise_code

# ═════════════════════════════════════════════════════════════════════
# Builders (no side effects, no hidden state)
# ═════════════════════════════════════════════════════════════════════


def make_chunk(
    chunk_id: str,
    *,
    name: str = "",
    content: str = "",
    path: str = "f.py",
    blob: str = "",
    kind: ChunkKind = ChunkKind.FUNCTION,
    repo_id: int = 1,
) -> TokenisedChunk:
    """Build a minimal `TokenisedChunk` with auto-tokenised fields."""
    name = name or chunk_id
    content = content or f"def {name}(): pass"
    return TokenisedChunk(
        id=chunk_id,
        repo_id=repo_id,
        blob_sha=blob or f"blob_{chunk_id}",
        file_path=path,
        kind=kind,
        name=name,
        content=content,
        content_tokens=tokenise_code(content),
        name_tokens=tokenise_code(name),
        line_start=1,
        line_end=1,
    )


def make_snap(sha: str, path: str, blob: str) -> Snapshot:
    """Build a `Snapshot`."""
    return Snapshot(commit_sha=sha, file_path=path, blob_sha=blob)


def seed_store(
    store: IndexStore,
    chunks: list[TokenisedChunk],
    *,
    commit_sha: str = "head",
    mark_indexed: bool = True,
) -> None:
    """Insert chunks + snapshots into a store via session.

    Each chunk's own `repo_id` decides where it lands; a mixed-repo
    list seeds several repos in one call.  Snapshots and
    `mark_indexed` are derived per repo from the chunks.
    """
    with store.session() as ws:
        for c in chunks:
            ws.add_chunk(c)
        by_repo: dict[int, list[TokenisedChunk]] = {}
        for c in chunks:
            by_repo.setdefault(c.repo_id, []).append(c)
        for repo_id, repo_chunks in by_repo.items():
            ws.insert_snapshots(
                [
                    Snapshot(commit_sha=commit_sha, file_path=c.file_path, blob_sha=c.blob_sha)
                    for c in repo_chunks
                ],
                repo_id=repo_id,
            )
            if mark_indexed:
                ws.mark_indexed(repo_id, commit_sha)


# ═════════════════════════════════════════════════════════════════════
# Store fixture
# ═════════════════════════════════════════════════════════════════════


@pytest.fixture
def store() -> Generator[IndexStore]:
    """In-memory writable IndexStore."""
    s = IndexStore(writable=True)
    yield s
    s.close()


# ═════════════════════════════════════════════════════════════════════
# Shared git repo (build, embed, diff, incremental tests)
# ═════════════════════════════════════════════════════════════════════


@pytest.fixture
def git_repo(tmp_path: Path) -> pygit2.Repository:
    """Git repo with 3 Python files, tests, and README.

    Writes to working directory (needed by index tests that
    exercise tree-sitter extraction via `build_index`).
    """
    repo = pygit2.init_repository(str(tmp_path), bare=False)

    files = {
        "src/models.py": b"""\
\"\"\"Data models.\"\"\"

class User:
    pass

class Order:
    pass
""",
        "src/utils.py": b"""\
\"\"\"Utility functions.\"\"\"

def helper():
    return 42

def format_name(name):
    return name.strip()
""",
        "src/main.py": b"""\
\"\"\"Main module.\"\"\"

from src.models import User
from src.utils import helper

def run():
    u = User()
    return helper()
""",
        "tests/test_utils.py": b"""\
\"\"\"Tests for utils.\"\"\"

from src.utils import helper, format_name

def test_helper():
    assert helper() == 42

def test_format():
    assert format_name(\"  hi  \") == \"hi\"
""",
        "README.md": b"""\
# My Project

This project uses `helper` and `User` for things.

## Setup

Run `format_name` to clean strings.
""",
    }

    index = repo.index
    for path, content in files.items():
        full = tmp_path / path
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_bytes(content)
        index.add(path)

    index.write()
    tree_oid = index.write_tree()
    sig = pygit2.Signature("Test", "test@test.com")
    repo.create_commit("HEAD", sig, sig, "Initial commit", tree_oid, [])

    return repo


@pytest.fixture
def commit_sha(git_repo: pygit2.Repository) -> str:
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


def commit_file_set(repo: pygit2.Repository, files: dict[str, bytes], *, parents: list[str]) -> str:
    """Create a dangling commit whose tree is exactly *files*.

    The tree is built from an in-memory index, so the working
    directory is never touched — `build_index` reads file content
    from the git tree at the SHA, not the workdir.
    """
    index = pygit2.Index()
    for path, content in files.items():
        blob_oid = repo.create_blob(content)
        index.add(pygit2.IndexEntry(path, blob_oid, FileMode.BLOB))
    tree_oid = index.write_tree(repo)
    sig = pygit2.Signature("Test", "test@test.com")
    commit_oid = repo.create_commit(
        None, sig, sig, "case", tree_oid, [pygit2.Oid(hex=p) for p in parents]
    )
    return str(commit_oid)


@pytest.fixture
def diff_repo(tmp_path: Path) -> pygit2.Repository:
    """Empty git repo (no commits) for symbol-diff scenarios."""
    return pygit2.init_repository(str(tmp_path), bare=False)


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
    name = "AppConfig"
    content = """\
class AppConfig:
    database_url: str
    max_retries: int = 3
    timeout: float = 30.0
"""
    return TokenisedChunk(
        id="config_class",
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
        id="load_config",
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
        id="import_config",
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
        id="start_server",
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
        id="test_config",
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
        id="doc_config",
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
def ranking_edges() -> list[Edge]:
    """Edge graph used by importance / structural ranking tests.

    `config_class` receives 2 inbound edges (import, start_server).
    `load_config` receives 3 inbound edges (import, test, doc).
    `start_server` receives 0.
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
) -> Generator[IndexStore]:
    """An in-memory IndexStore pre-loaded with the ranking dataset."""
    store = IndexStore(writable=True)
    with store.session() as ws:
        for c in ranking_chunks:
            tc = c if isinstance(c, TokenisedChunk) else TokenisedChunk(**c.model_dump())
            ws.add_chunk(tc)
        ws.insert_snapshots(
            [
                Snapshot(commit_sha=ranking_commit, file_path=c.file_path, blob_sha=c.blob_sha)
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
        id="math_1",
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
        id="http_1",
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
        id="string_1",
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
        id="math_class_1",
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
    return TokenisedChunk(
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
    return TokenisedChunk(
        id="cz",
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
    return Edge(source_id="math_1", target_id="math_class_1", kind=EdgeKind.CALLS)


@pytest.fixture
def edge_a_calls_b() -> Edge:
    return Edge(source_id="a", target_id="b", kind=EdgeKind.CALLS)


@pytest.fixture
def edge_c_imports_d() -> Edge:
    return Edge(source_id="c", target_id="d", kind=EdgeKind.IMPORTS)
