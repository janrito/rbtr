"""Data models and enums for the code index."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Annotated, Any

from pydantic import BaseModel, Field, TypeAdapter, field_validator, model_validator

from rbtr.index.identity import compose_scope, make_chunk_id

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


class QueryKind(StrEnum):
    """Query processing tier.

    `CONCEPT`    — natural-language question ("how does fusion work").
    `IDENTIFIER` — a symbol name ("fuse_scores", "Embedder").
    `CODE`       — a code fragment ("def fuse_scores(").
    """

    CONCEPT = "concept"
    IDENTIFIER = "identifier"
    CODE = "code"


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
    """A single indexed unit of code, documentation, or configuration.

    The model owns its identity: `scope` is composed from enclosing-scope
    segments (a list is joined into a `::` address; a string passes
    through), and `id` is derived from `(file_path, blob_sha, name,
    line_start - 1)` when not supplied. Synthetic chunks (raw/link/ref)
    that need a decorated id pass it explicitly, and reads from storage
    carry the stored id — both are kept as-is.
    """

    id: str = ""
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

    @field_validator("scope", mode="before")
    @classmethod
    def _compose_scope(cls, value: str | Sequence[str]) -> str:
        """Join enclosing-scope segments into the `::` address.

        A plain string (storage read, scope-less chunk) passes through;
        a sequence of segment names is composed. The `str` guard avoids
        treating a string as a sequence of characters.
        """
        if isinstance(value, str):
            return value
        return compose_scope(value)

    @model_validator(mode="before")
    @classmethod
    def _derive_id(cls, data: Any) -> Any:
        """Derive `id` from identity fields when one is not supplied."""
        if isinstance(data, dict) and not data.get("id"):
            return {
                **data,
                "id": make_chunk_id(
                    data["file_path"], data["blob_sha"], data["name"], data["line_start"] - 1
                ),
            }
        return data


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
    query_kind: QueryKind
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
    # Preview anchor: where the query literally matched the content.
    # Populated only when the search passes a lexical query; not a
    # ranking signal.
    match_line_offset: int | None = None
    matched_terms: list[str] = Field(default_factory=list)


class TokenisedChunk(Chunk):
    """Chunk with the extra columns the `chunks` table needs.

    Written during extraction, stored in DB, consumed by FTS.
    No code outside the extraction loop reads these fields
    from the model — they exist only to flow into DuckDB.
    The added fields split by role: `content_tokens` and
    `name_tokens` are the code-aware tokenisations BM25/FTS
    queries against; `language_plugin_version` is a storage
    column, not part of chunk identity, which is derived from
    file/blob/name/line only.  Chunks carry no `repo_id` — the
    store is content-addressed and repo attribution lives in
    `file_snapshots`.
    """

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
    """Rows removed by a garbage-collection operation.

    `commits`, `snapshots`, and `edges` are per-repo (summed when a global
    GC visits several repos). `chunks` is the number of chunks actually
    freed from the content-addressed pool — a global figure, since a chunk
    dies only when no `file_snapshots` row in any repo references it.
    """

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
