"""Data models and enums for the code index."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Discriminator,
    Field,
    TypeAdapter,
    computed_field,
    field_validator,
    model_validator,
)

from rbtr.domain.identity import chunk_id, compose_scope


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
    FRAGMENT = "fragment"


CODE_KINDS: frozenset[ChunkKind] = frozenset(
    {ChunkKind.FUNCTION, ChunkKind.CLASS, ChunkKind.METHOD},
)

KIND_NARROWS: Mapping[ChunkKind, frozenset[ChunkKind]] = {
    kind: frozenset(
        {ChunkKind.IMPORT}
        | (
            {ChunkKind.FUNCTION, ChunkKind.METHOD}
            if kind in {ChunkKind.TEST_FUNCTION, ChunkKind.API_ENDPOINT, ChunkKind.MIGRATION}
            else set()
        )
    )
    for kind in ChunkKind
    if kind is not ChunkKind.IMPORT
}
"""The kinds each kind narrows: of the kinds captured on one span, the one that narrows every other names its chunk.

Every kind narrows `import`, as a key or a section that imports is that
key or section, carrying the import; a test, an endpoint or a migration
narrows the function or method it is. Two kinds neither narrows take
the first in `ChunkKind` order, and the overlap is the plugin's to fix.
The only statement of kind precedence.
"""


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


class RefSource(StrEnum):
    """How a read chose the snapshot it read.

    `REQUESTED`      — the ref the caller named.
    `HEAD`           — no ref named and a clean working tree: HEAD.
    `WORKTREE`       — no ref named and a dirty, indexed working tree.
    `LATEST_INDEXED` — the snapshot the ref names is not indexed or
                       does not resolve, so the latest indexed commit.
    """

    REQUESTED = "requested"
    HEAD = "head"
    WORKTREE = "worktree"
    LATEST_INDEXED = "latest_indexed"


class EdgeKind(StrEnum):
    """Kind of relationship between chunks."""

    CALLS = "calls"
    IMPORTS = "imports"
    INHERITS = "inherits"
    DOCUMENTS = "documents"
    CONFIGURES = "configures"


class ModuleStyle(StrEnum):
    """How a language writes a module: as a path (`./b`, `b.css`), dotted (`pkg.b`) or scoped (`pkg::b`)."""

    PATH = "path"
    DOTTED = "dotted"
    SCOPED = "scoped"


class MemberOrder(StrEnum):
    """The order a language searches a class's bases for a member it does not define.

    `LISTED` and `REVERSED` walk the bases depth first, keeping each
    ancestor's last visit, so a base shared by two comes after both:
    `LISTED` takes them as written (Python, Java, C++, JavaScript, Rust),
    `REVERSED` last written first (Ruby, whose latest `include` precedes
    earlier ones and its superclass). `NEAREST` takes the shallowest, ties
    alike (Go's embedding).
    """

    LISTED = "listed"
    REVERSED = "reversed"
    NEAREST = "nearest"


class SectionNames(StrEnum):
    """How an import compares with a section's name.

    `EXACT` compares the name as written (an HTML `id`). `FOLDED` ignores
    case, accents, punctuation and leading digits, as renderers do when
    they make a heading's anchor (`Getting Started` → `#getting-started`).
    """

    EXACT = "exact"
    FOLDED = "folded"


class EmbeddingState(StrEnum):
    """Whether a chunk has an embedding, and whether its text was cut to make it.

    `TRUNCATED` is an embedding of the text's first part only: the
    chunk was longer than the model's context.
    """

    NONE = "none"
    FULL = "full"
    TRUNCATED = "truncated"


class Binding(BaseModel, frozen=True):
    """A name an import binds in the importing file, and what it names there.

    `local` is the name the file's code uses. `name` is what it names: for
    an import listing names, one of them (`from m import Chunk as C` binds
    `C` to `Chunk`); for one listing none, a module path (`import numpy as
    np` binds `np` to `numpy`).
    """

    name: str
    local: str


class ImportMeta(BaseModel):
    """Structured import data extracted by tree-sitter.

    All fields default to empty — different import styles
    populate different subsets.  Edge inference and module
    resolution read these fields without knowing the source language.
    """

    module: str = ""
    """Module path; a relative one starts with `./` (the file's directory) or `../`."""
    names: str = ""
    """Comma-separated imported symbol names."""
    language_hint: str = ""
    """Target language when known from structure (e.g. HTML
    `<script src>` → `'javascript'`).  Empty means unknown —
    resolver uses `import_targets`."""
    bindings: list[Binding] = Field(default_factory=list)
    """Every name it binds in the importing file; empty when its language records none."""


class Lookup(StrEnum):
    """Where a reference's path is looked up, from how it starts: name lookup, as language standards call it."""

    UNQUALIFIED = "unqualified"
    """In the enclosing scopes, then the file's imports: `foo()`, `models.CharField()`."""
    MEMBER = "member"
    """In the enclosing class's members: `self.save()`, `this.render()`."""
    BASE = "base"
    """In the enclosing class's bases: `super().__init__()`."""
    TYPE = "type"
    """In the written type of a function's local: `order.save()` with `order: Order`."""


class Definition(BaseModel, frozen=True):
    """A construct the code defines: a function, class, method or variable.

    `name` is empty for one written without a name, such as an arrow
    function assigned nowhere.
    """

    kind: Literal[
        ChunkKind.FUNCTION,
        ChunkKind.CLASS,
        ChunkKind.METHOD,
        ChunkKind.VARIABLE,
        ChunkKind.MIGRATION,
        ChunkKind.TEST_FUNCTION,
        ChunkKind.API_ENDPOINT,
    ]
    name: str


class Section(BaseModel, frozen=True):
    """A part of a document or a data file: a heading's section, a key.

    `name` is empty for one with no title, such as a heading-less
    paragraph or a CSS `@media` block.
    """

    kind: Literal[ChunkKind.DOC_SECTION, ChunkKind.CONFIG_KEY]
    name: str


class Import(BaseModel, frozen=True):
    """A statement or link bringing in another file, module or symbol."""

    kind: Literal[ChunkKind.IMPORT] = ChunkKind.IMPORT
    name: str
    meta: ImportMeta


class Comment(BaseModel, frozen=True):
    """A comment or docstring standing on its own, with nothing to name."""

    kind: Literal[ChunkKind.COMMENT] = ChunkKind.COMMENT

    @property
    def name(self) -> str:
        return ""


class RawSlice(BaseModel, frozen=True):
    """A run of lines cut from a file no grammar reads, with nothing to name."""

    kind: Literal[ChunkKind.RAW_CHUNK] = ChunkKind.RAW_CHUNK

    @property
    def name(self) -> str:
        return ""


class FragmentMeta(BaseModel, frozen=True):
    """The syntax a fragment holds, its named nodes' types in order once each."""

    node_types: tuple[str, ...]
    """Diagnostic: nothing in search or the edge graph reads it. rbtr-eval's
    index report tallies it to find syntax no query captures."""


class Fragment(BaseModel, frozen=True):
    """A run of a file's code that no query pattern names."""

    kind: Literal[ChunkKind.FRAGMENT] = ChunkKind.FRAGMENT
    meta: FragmentMeta

    @property
    def name(self) -> str:
        return ""


type Role = Annotated[
    Definition | Section | Import | Comment | RawSlice | Fragment,
    Discriminator("kind"),
]
"""What a chunk is, carrying exactly the data that kind of chunk has."""


class Span(BaseModel, frozen=True):
    """Where a chunk is in its file: its bytes, `start_byte` up to `end_byte`, and the lines they fall on.

    Lines are 1-based and inclusive; neither the bytes nor the lines end
    before they start.
    """

    start: int
    end: int
    start_byte: int
    end_byte: int

    @model_validator(mode="after")
    def ends_after_it_starts(self) -> Span:
        if self.end < self.start:
            msg = f"a span cannot end (line {self.end}) before it starts (line {self.start})"
            raise ValueError(msg)
        if self.end_byte < self.start_byte:
            msg = (
                f"a span cannot end (byte {self.end_byte}) before it starts"
                f" (byte {self.start_byte})"
            )
            raise ValueError(msg)
        return self


class Chunk(BaseModel, frozen=True):
    """A single indexed unit of code, documentation, or configuration, as a chunker yields it.

    The model owns its identity: `scope` is composed from enclosing-scope
    segments (a list is joined into a `::` address; a string passes
    through), and `id` is *computed* from the fields it hashes, the ones
    staging hashes for its row, so a chunk cannot carry an id that
    disagrees with the one it is stored under.

    `file_path` is where the chunker read it, and is *not* part of its
    identity: one chunk can be reached at several paths, which
    `file_snapshots` records.

    Frozen, because a chunk *is* a span of a blob read as one language:
    changing one of the fields that says so makes it a different chunk
    rather than the same one altered. Labelling one therefore returns a
    copy.
    """

    blob_sha: str
    file_path: str
    role: Role
    scope: str = ""
    language: str = ""
    file_language: str = ""
    """The language the *file* was extracted as.

    Distinct from `language`, which is the chunk's own: a python fence in
    a markdown document has `language="python"` and
    `file_language="markdown"`. Stamped from the file's row by the
    extraction rules (`rbtr.domain.rules`), the only place that knows it — a chunker is passed the language it is extracting,
    which for an embedded block is the block's, not the file's.
    """
    content: str
    span: Span

    @property
    def kind(self) -> ChunkKind:
        return self.role.kind

    @property
    def name(self) -> str:
        return self.role.name

    @property
    def line_start(self) -> int:
        return self.span.start

    @property
    def line_end(self) -> int:
        return self.span.end

    @field_validator("scope", mode="before")
    @classmethod
    def composed_scope(cls, value: str | Sequence[str]) -> str:
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
        """This chunk's `chunk_id`, from its blob, file language and byte span; its paths are `file_snapshots`'."""
        return chunk_id(self.blob_sha, self.file_language, self.span.start_byte, self.span.end_byte)


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
    imports: list[ImportMeta] = Field(default_factory=list)
    """The imports the chunk makes, empty when it makes none."""
    embedding_state: EmbeddingState = Field(default=EmbeddingState.NONE, exclude=True)
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


@dataclass(frozen=True, slots=True, kw_only=True)
class SnapshotRef:
    """A registered repo at one of its snapshots: what a read is scoped to.

    Made by `Repo.at`.  It never crosses the RPC boundary: clients name
    repos by path, and the daemon resolves a path to a `Repo` first.

    `snapshot_sha` is the snapshot identity (a commit SHA or a
    dirty worktree tree SHA) that search and edge queries join
    through `file_snapshots`.  `kw_only` forbids positional /
    tuple-style construction and unpacking.
    """

    repo: Repo
    snapshot_sha: str


@dataclass(frozen=True, slots=True, kw_only=True)
class SnapshotRange:
    """One repo, compared between two indexed snapshots.

    Built at the daemon's handler boundary and consumed by the store's
    diff SQL.  Both ends share the one repo the range carries, so a
    comparison spanning two repos cannot be constructed.  `kw_only`
    forbids positional / tuple-style construction and unpacking.
    """

    repo: Repo
    base_sha: str
    head_sha: str


@dataclass(frozen=True, slots=True, kw_only=True)
class SnapshotCounts:
    """How many chunks a snapshot holds, and how many carry embeddings.

    Counts chunks: content held at several paths counts once, because
    embedding it writes the one content-addressed row every path
    shares.  `kw_only` forbids positional / tuple-style
    construction and unpacking.

    `SnapshotCountsRow` establishes `embedded <= total` where the counts
    are read.
    """

    total: int
    embedded: int

    @property
    def unembedded(self) -> int:
        """Chunks still awaiting an embedding."""
        return self.total - self.embedded


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

    def at(self, snapshot_sha: str) -> SnapshotRef:
        """This repo at *snapshot_sha*."""
        return SnapshotRef(repo=self, snapshot_sha=snapshot_sha)


class ReplacedIndex(BaseModel, frozen=True):
    """An index replaced on open, because its schema is too old to migrate.

    `stored` is the schema version it recorded, `None` when it recorded
    none; `replaced_at` is when this rbtr replaced it with an empty one.
    """

    stored: str | None
    replaced_at: datetime


class BuildWork(BaseModel):
    """Building the snapshot `ref` of the repository at `repo_path`."""

    model_config = ConfigDict(extra="forbid")
    kind: Literal["build"] = "build"
    repo_path: str
    ref: str


class EmbedWork(BaseModel):
    """Embedding the chunks of the snapshot `ref` of the repository at `repo_path`."""

    model_config = ConfigDict(extra="forbid")
    kind: Literal["embed"] = "embed"
    repo_path: str
    ref: str


class CollectWork(BaseModel):
    """Collecting garbage, from one repository or from every one."""

    model_config = ConfigDict(extra="forbid")
    kind: Literal["collect"] = "collect"


class MigrateWork(BaseModel):
    """Migrating the index's files to the schema this rbtr reads."""

    model_config = ConfigDict(extra="forbid")
    kind: Literal["migrate"] = "migrate"


type Work = Annotated[BuildWork | EmbedWork | CollectWork | MigrateWork, Discriminator("kind")]
"""What a piece of background work does, and what it does it to.

Work on a snapshot names its repository; other work names none and
concerns every client.
"""


@dataclass(frozen=True, slots=True, kw_only=True)
class ChunkSplit:
    """A gc drop set's candidate chunks: those it frees, and those another ref keeps."""

    dropped: int
    kept: int


@dataclass(frozen=True, slots=True, kw_only=True)
class IndexedSnapshot:
    """A snapshot a repo has fully indexed, and when it finished (naive UTC)."""

    snapshot_sha: str
    indexed_at: datetime


type RefsByRepo = dict[str, list[str]]
"""Refs attributed to the repo that watched them, keyed by its path."""


class Scope(StrEnum):
    """Breadth of an operation over the shared store.

    `WORKSPACE` is the single repo identified by `repo_path`;
    `ALL` is every indexed repo.
    """

    WORKSPACE = "workspace"
    ALL = "all"


Chunks = TypeAdapter(list[Chunk])
ScoredChunks = TypeAdapter(list[ScoredChunk])
FileSnapshots = TypeAdapter(list[FileSnapshot])


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


class GcMode(StrEnum):
    """What a `rbtr gc` invocation is allowed to delete."""

    HEAD_ONLY = "head_only"  # keep current HEAD, drop the rest
    KEEP = "keep"  # keep only listed refs
    WATCHED = "watched"  # default: HEAD + local branches/tags/notes + watched
    WATCHED_ONLY = "watched_only"  # HEAD + resolved watched_refs only


# The retentions every repo states in its own terms, so the only ones a
# gc across every repo can apply.
type EveryRepoGcMode = Literal[GcMode.WATCHED, GcMode.WATCHED_ONLY]


@dataclass(frozen=True)
class GcCounts:
    """Rows removed by a garbage-collection operation.

    `snapshots`, `file_snapshots`, and `edges` are per-repo (summed when a
    GC visits several repos). `chunks` is the number of chunks actually
    freed from the projects' content-addressed stores — a per-project
    figure, since a chunk dies only when no `file_snapshots` row of any
    checkout of its project references it.
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


class KindOverlap(BaseModel, frozen=True):
    """Captures of more than one kind on one span of a file, and the kind its chunk took.

    `narrowed` is whether `KIND_NARROWS` decided it; one it did not took the
    first of `kinds` in `ChunkKind` order, and is a defect of the language's
    patterns.
    """

    path: str
    line: int
    kinds: tuple[ChunkKind, ...]
    kept: ChunkKind
    narrowed: bool


@dataclass
class IndexResult:
    """Outcome of an index build or update.

    `kind_overlaps` are the spans its extraction captured as more than one
    kind (`KindOverlap`).
    """

    stats: IndexStats = field(default_factory=IndexStats)
    errors: list[str] = field(default_factory=list)
    kind_overlaps: list[KindOverlap] = field(default_factory=list)
