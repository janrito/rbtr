"""Data models and enums for the code index."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, Field, TypeAdapter

# ── Enums ────────────────────────────────────────────────────────────


class ChunkKind(StrEnum):
    """Kind of indexed chunk."""

    FUNCTION = "function"
    CLASS = "class"
    METHOD = "method"
    VARIABLE = "variable"
    IMPORT = "import"
    DOC_SECTION = "doc_section"
    CONFIG_KEY = "config_key"
    MIGRATION = "migration"
    TEST_FUNCTION = "test_function"
    API_ENDPOINT = "api_endpoint"
    RAW_CHUNK = "raw_chunk"


CODE_KINDS: frozenset[ChunkKind] = frozenset(
    {ChunkKind.FUNCTION, ChunkKind.CLASS, ChunkKind.METHOD},
)


class ChangeKind(StrEnum):
    """How a symbol changed between two indexed commits."""

    ADDED = "added"
    MODIFIED = "modified"
    REMOVED = "removed"


class EdgeKind(StrEnum):
    """Kind of relationship between chunks."""

    CALLS = "calls"
    IMPORTS = "imports"
    INHERITS = "inherits"
    TESTS = "tests"
    DOCUMENTS = "documents"
    CONFIGURES = "configures"


class IndexPhase(StrEnum):
    """Current phase of background indexing."""

    IDLE = "idle"
    INDEXING = "indexing"
    READY = "ready"
    FAILED = "failed"


# ── Structured metadata ──────────────────────────────────────────────


class ImportMeta(BaseModel):
    """Structured import data extracted by tree-sitter.

    All fields default to empty — different import styles
    populate different subsets.  `edges.py` reads these fields
    without knowing the source language.
    """

    module: str = ""
    """Module path after stripping relative prefixes."""
    names: str = ""
    """Comma-separated imported symbol names."""
    dots: str = ""
    """Relative import depth as a string (empty for absolute)."""
    language_hint: str = ""
    """Target language when known from structure (e.g. HTML
    `<script src>` → `'javascript'`).  Empty means unknown —
    resolver uses `import_targets`."""


# ── Data models ──────────────────────────────────────────────────────


class Chunk(BaseModel):
    """A single indexed unit of code, documentation, or configuration."""

    id: str
    blob_sha: str
    file_path: str
    kind: ChunkKind
    name: str
    scope: str = ""
    language: str = ""
    content: str
    line_start: int
    line_end: int
    metadata: ImportMeta = Field(default_factory=ImportMeta)
    embedding: Annotated[list[float], Field(exclude=True)] = []


class ScoredChunk(BaseModel, frozen=True):
    """A search result: chunk data plus full signal breakdown.

    `repo_path` attributes the result to its repo in cross-repo
    search; it is `None` for single-repo (workspace) searches.
    """

    id: str
    blob_sha: str
    repo_path: str | None = None
    file_path: str
    kind: ChunkKind
    name: str
    scope: str = ""
    language: str = ""
    content: str
    line_start: int
    line_end: int
    metadata: ImportMeta = Field(default_factory=ImportMeta)
    embedding: Annotated[list[float], Field(exclude=True)] = []
    score: float
    lexical: float
    semantic: float
    name_match: float
    kind_boost: float
    file_penalty: float
    importance: float = 1.0
    proximity: float = 1.0
    fusion: float = 0.0
    reranker: float = 0.0


class TokenisedChunk(Chunk):
    """Chunk with the extra columns the `chunks` table needs.

    Written during extraction, stored in DB, consumed by FTS.
    No code outside the extraction loop reads these fields
    from the model — they exist only to flow into DuckDB.
    The four added fields split by role: `content_tokens` and
    `name_tokens` are the code-aware tokenisations BM25/FTS
    queries against; `repo_id` and `language_plugin_version` are
    storage columns, not part of chunk identity, which is derived
    from file/blob/name/line only.  `repo_id` defaults to 1, the
    sole repo in the common single-repo case.
    """

    repo_id: int = 1
    content_tokens: str = ""
    name_tokens: str = ""
    language_plugin_version: int = 1


class Snapshot(BaseModel):
    """A file in a commit's tree, mapping path to blob SHA."""

    commit_sha: str
    file_path: str
    blob_sha: str
    detected_language: str = ""


class Edge(BaseModel):
    """A directed relationship between two chunks."""

    source_id: str
    target_id: str
    kind: EdgeKind


@dataclass(frozen=True, slots=True, kw_only=True)
class RepoRef:
    """A repo paired with the indexed ref to read it at.

    Internal-only transport: built at the daemon's handler
    boundary (where a client `path` is resolved to a `repo_id`)
    and consumed by the store's search SQL.  It never crosses the
    RPC boundary — clients name repos by path, never by numeric
    `repo_id` — so the integer key stays inside the process.

    `commit_sha` is the snapshot identity (a commit SHA or a
    dirty worktree tree SHA) that search and edge queries join
    through `file_snapshots`.  `kw_only` forbids positional /
    tuple-style construction and unpacking.
    """

    repo_id: int
    commit_sha: str


Chunks = TypeAdapter(list[Chunk])
ScoredChunks = TypeAdapter(list[ScoredChunk])
TokenisedChunks = TypeAdapter(list[TokenisedChunk])
Snapshots = TypeAdapter(list[Snapshot])
Edges = TypeAdapter(list[Edge])


class IndexStats(BaseModel):
    """Summary statistics for a completed index."""

    total_chunks: int = 0
    total_edges: int = 0
    total_files: int = 0
    skipped_files: int = 0
    parsed_files: int = 0
    embedded_chunks: int = 0
    elapsed_seconds: float = 0.0


class IndexStatus(BaseModel):
    """Current state of the index, readable by the UI."""

    phase: IndexPhase = IndexPhase.IDLE
    files_indexed: int = 0
    total_files: int = 0
    skipped_files: int = 0
    stats: IndexStats | None = None
    error: str = ""


# ── GC / session types ────────────────────────────────────────


@dataclass(frozen=True)
class GcCounts:
    """Rows removed by a garbage-collection operation."""

    commits: int = 0
    snapshots: int = 0
    edges: int = 0
    chunks: int = 0

    def __add__(self, other: GcCounts) -> GcCounts:
        return GcCounts(
            commits=self.commits + other.commits,
            snapshots=self.snapshots + other.snapshots,
            edges=self.edges + other.edges,
            chunks=self.chunks + other.chunks,
        )


# ── Pipeline result types ────────────────────────────────────────────


@dataclass
class IndexResult:
    """Outcome of an index build or update."""

    stats: IndexStats = field(default_factory=IndexStats)
    errors: list[str] = field(default_factory=list)
