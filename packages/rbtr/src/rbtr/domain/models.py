"""Data models and enums for the code index."""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from pydantic import (
    BaseModel,
    Field,
    TypeAdapter,
    computed_field,
    field_validator,
    model_validator,
)

from rbtr.domain.identity import compose_scope

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
    COMMENT = "comment"
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
    DOCUMENTS = "documents"
    CONFIGURES = "configures"


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


class Chunk(BaseModel, frozen=True):
    """A single indexed unit of code, documentation, or configuration.

    The model owns its identity: `scope` is composed from enclosing-scope
    segments (a list is joined into a `::` address; a string passes
    through), and `id` is *computed* from the fields it hashes, so a
    chunk cannot carry an id that disagrees with its own content. Every one of those fields is a stored column, so a chunk
    read back from storage recomputes the id it was stored under.

    `file_path` is where this chunk was found, and is *not* part of its
    identity: it comes from `file_snapshots` on read, and one chunk can
    be reached at several paths.

    Frozen, because a chunk *is* its content: two with the same content
    are the same chunk, and changing one of the fields that says so makes
    it a different chunk rather than the same one altered. Labelling one
    therefore returns a copy.
    """

    blob_sha: str
    file_path: str
    kind: ChunkKind
    name: str
    scope: str = ""
    language: str = ""
    file_language: str = ""
    """The language the *file* was extracted as.

    Distinct from `language`, which is the chunk's own: a python fence in
    a markdown document has `language="python"` and
    `file_language="markdown"`. Stamped by `extract_file`, the only place
    that knows it — a chunker is passed the language it is extracting,
    which for an embedded block is the block's, not the file's.
    """
    content: str
    line_start: int
    line_end: int
    metadata: ImportMeta = Field(default_factory=ImportMeta)
    has_embedding: bool = Field(default=False, exclude=True)

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

    @computed_field  # type: ignore[prop-decorator]  # pydantic needs the property last
    @property
    def id(self) -> str:
        """A hash of what this chunk is: content, language, role, span.

        Byte-identical files share every chunk, so a copy of an indexed
        file resolves to the row already there. Location is the province
        of `file_snapshots`, which holds every path a chunk is reachable
        at.

        `file_language` — the language the file was read as — is part of
        it because the same bytes read as two languages are two
        extractions: the empty blob is one chunk under python and
        another under html, and the dedup gate asks after each
        separately. `kind` and `line_end` are part of it because a chunk
        is a span with a role, and an rst `doc_section` and a `:mod:`
        reference can share a name and a starting line while spanning
        different lines and meaning different things.
        """
        raw = (
            f"{self.blob_sha}:{self.file_language}:{self.kind}:"
            f"{self.name}:{self.line_start}:{self.line_end}"
        )
        return hashlib.blake2b(raw.encode(), digest_size=8).hexdigest()

    @model_validator(mode="after")
    def _check_unnamed_kinds(self) -> Chunk:
        """Reject a name on a kind with no identifier to hold one.

        A remark and a slice of lines have nothing to name. Every other
        kind *may* be unnamed — an arrow function, a CSS `@media` block, a
        `@charset` rule, a heading-less paragraph — so this runs one way.
        """
        if self.kind in {ChunkKind.COMMENT, ChunkKind.RAW_CHUNK} and self.name:
            msg = f"{self.kind} carries no name, got {self.name!r}"
            raise ValueError(msg)
        return self


class ScoredChunk(BaseModel, frozen=True):
    """A search result: chunk data plus full signal breakdown.

    `repo_path` attributes the result to its repo in cross-repo
    search; it is `None` for single-repo (workspace) searches.

    `file_paths` holds every location this content was found at, sorted,
    with no primary: a chunk is its content, so the same bytes at four
    paths are one result reachable four ways rather than four results.
    Ranking still sees each location — candidates are scored per path
    and collapsed afterwards, so a copy under `node_modules` cannot drag
    down the source it duplicates.
    """

    id: str
    blob_sha: str
    repo_path: str | None = None
    file_paths: list[str] = Field(min_length=1)
    kind: ChunkKind
    query_kind: QueryKind
    name: str
    scope: str = ""
    language: str = ""
    content: str
    line_start: int
    line_end: int
    metadata: ImportMeta = Field(default_factory=ImportMeta)
    has_embedding: bool = Field(default=False, exclude=True)
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


class FileSnapshot(BaseModel, frozen=True):
    """A file in a snapshot's tree, mapping path to blob SHA."""

    snapshot_sha: str
    file_path: str
    blob_sha: str
    detected_language: str = ""


class Edge(BaseModel, frozen=True):
    """A directed relationship between content at two locations.

    A chunk is content and sits at as many paths as hold those bytes, so
    the paths are part of the relationship: `docx/helpers.py` importing
    its sibling and `xlsx/helpers.py` importing a different one are two
    edges from a single import chunk, distinguished by where each end
    sits. `find-refs` reports the location that did the referring.
    """

    source_id: str
    target_id: str
    kind: EdgeKind
    source_path: str
    target_path: str


@dataclass(frozen=True, slots=True, kw_only=True)
class SnapshotRef:
    """A repo paired with the indexed ref to read it at.

    Internal-only transport: built at the daemon's handler
    boundary (where a client `path` is resolved to a `repo_id`)
    and consumed by the store's search SQL.  It never crosses the
    RPC boundary — clients name repos by path, never by numeric
    `repo_id` — so the integer key stays inside the process.

    `snapshot_sha` is the snapshot identity (a commit SHA or a
    dirty worktree tree SHA) that search and edge queries join
    through `file_snapshots`.  `kw_only` forbids positional /
    tuple-style construction and unpacking.
    """

    repo_id: int
    snapshot_sha: str


@dataclass(frozen=True, slots=True, kw_only=True)
class Repo:
    """A repository registered in the index: its surrogate id and path.

    The `(repo_id, repo_path)` correspondence is fixed at registration and
    guaranteed by the `repos` table (`path` UNIQUE, `id` its PK). Taking a
    `Repo` rather than the two as separate parameters makes pairing an id
    with the wrong path impossible to express.
    """

    repo_id: int
    repo_path: str


Chunks = TypeAdapter(list[Chunk])
ScoredChunks = TypeAdapter(list[ScoredChunk])
FileSnapshots = TypeAdapter(list[FileSnapshot])
Edges = TypeAdapter(list[Edge])


class FileOutcome(StrEnum):
    """What a build did with one file.

    Exactly one is recorded per file, so the counts derived from the
    tally agree with each other and with the loop. Every path through
    the loop has a member here, including `EXTRACTED_EMPTY` — a file
    that was extracted and yielded nothing, which a build log names so
    that a run reporting few parsed files says why.
    """

    PARSED = "parsed"
    SKIPPED_UNCHANGED = "skipped_unchanged"  # not in the incremental diff
    SKIPPED_CURRENT = "skipped_current"  # the blob was already extracted
    EXTRACTED_EMPTY = "extracted_empty"  # ran, produced nothing
    FAILED = "failed"


class IndexStats(BaseModel):
    """Summary statistics for a completed index.

    `outcomes` holds one entry per file and is the only thing stored or
    sent; the file counts are properties over it, so they cannot
    disagree with each other or with what the loop did.
    """

    total_chunks: int = 0
    total_edges: int = 0
    outcomes: Counter[FileOutcome] = Field(default_factory=Counter)
    embedded_chunks: int = 0
    elapsed_seconds: float = 0.0

    def record(self, outcome: FileOutcome) -> None:
        """Tally one file's outcome."""
        self.outcomes[outcome] += 1

    @property
    def total_files(self) -> int:
        """Files the build considered."""
        return sum(self.outcomes.values())

    @property
    def parsed_files(self) -> int:
        """Files that were extracted and produced at least one chunk."""
        return self.outcomes[FileOutcome.PARSED]

    @property
    def skipped_files(self) -> int:
        """Files the build did not extract, for either reason."""
        return (
            self.outcomes[FileOutcome.SKIPPED_UNCHANGED]
            + self.outcomes[FileOutcome.SKIPPED_CURRENT]
        )


# ── GC / session types ────────────────────────────────────────


class GcMode(StrEnum):
    """What a `rbtr gc` invocation is allowed to delete."""

    HEAD_ONLY = "head_only"  # keep current HEAD, drop the rest
    KEEP = "keep"  # keep only listed refs
    ORPHANS = "orphans"  # sweep residue only, drop no commits
    WATCHED = "watched"  # default: HEAD + local branches/tags/notes + watched
    WATCHED_ONLY = "watched_only"  # HEAD + resolved watched_refs only


@dataclass(frozen=True)
class GcCounts:
    """Rows removed by a garbage-collection operation.

    `snapshots`, `file_snapshots`, and `edges` are per-repo (summed when a
    global GC visits several repos). `chunks` is the number of chunks actually
    freed from the content-addressed pool — a global figure, since a chunk
    dies only when no `file_snapshots` row in any repo references it.
    """

    snapshots: int = 0
    file_snapshots: int = 0
    edges: int = 0
    chunks: int = 0

    def __add__(self, other: GcCounts) -> GcCounts:
        return GcCounts(
            snapshots=self.snapshots + other.snapshots,
            file_snapshots=self.file_snapshots + other.file_snapshots,
            edges=self.edges + other.edges,
            chunks=self.chunks + other.chunks,
        )


# ── Pipeline result types ────────────────────────────────────────────


@dataclass
class IndexResult:
    """Outcome of an index build or update."""

    stats: IndexStats = field(default_factory=IndexStats)
    errors: list[str] = field(default_factory=list)
