"""Data models and enums for the code index."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Annotated, TypedDict

from pydantic import BaseModel, Field

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


class ImportMeta(TypedDict, total=False):
    """Structured import data extracted by tree-sitter.

    All keys are optional — different import styles populate
    different subsets.  `edges.py` reads these keys without
    knowing the source language.

    Keys:
        module: Module path after stripping relative prefixes.
                Dotted for Python/Java (`foo.bar`), slash-separated
                for Go/Rust (`std/collections`), bare name for
                JS/TS after `./`/`../` is parsed out (`models`).
        names:  Comma-separated imported symbol names
                (e.g. `Chunk,Edge`).
        dots:   Levels up from the importing file for relative
                imports (stored as a string).  1 = current directory
                (Python `from .`, JS `./`), 2 = parent directory
                (Python `from ..`, JS `../`, Rust `super::`),
                etc.  Absent for absolute imports.
    """

    module: str
    names: str
    dots: str


# ── Data models ──────────────────────────────────────────────────────


class Chunk(BaseModel):
    """A single indexed unit of code, documentation, or configuration."""

    id: str
    blob_sha: str
    file_path: str
    kind: ChunkKind
    name: str
    scope: str = ""
    content: str
    content_tokens: Annotated[str, Field(exclude=True)] = ""
    name_tokens: Annotated[str, Field(exclude=True)] = ""
    line_start: int
    line_end: int
    metadata: ImportMeta = {}
    embedding: Annotated[list[float], Field(exclude=True)] = []


class Edge(BaseModel):
    """A directed relationship between two chunks."""

    source_id: str
    target_id: str
    kind: EdgeKind


class IndexStats(BaseModel):
    """Summary statistics for a completed index."""

    total_chunks: int = 0
    total_edges: int = 0
    total_files: int = 0
    skipped_files: int = 0
    parsed_files: int = 0
    elapsed_seconds: float = 0.0


class IndexStatus(BaseModel):
    """Current state of the index, readable by the UI."""

    phase: IndexPhase = IndexPhase.IDLE
    files_indexed: int = 0
    total_files: int = 0
    skipped_files: int = 0
    stats: IndexStats | None = None
    error: str = ""


# ── Pipeline result types ────────────────────────────────────────────


@dataclass
class IndexResult:
    """Outcome of an index build or update."""

    stats: IndexStats = field(default_factory=IndexStats)
    errors: list[str] = field(default_factory=list)


@dataclass
class SemanticDiff:
    """Structural differences between two indexed commits."""

    added: list[Chunk] = field(default_factory=list)
    """Symbols that exist in head but not in base."""

    removed: list[Chunk] = field(default_factory=list)
    """Symbols that exist in base but not in head."""

    modified: list[Chunk] = field(default_factory=list)
    """Symbols at the same path whose content changed."""

    stale_docs: list[tuple[Chunk, Chunk]] = field(default_factory=list)
    """`(doc_chunk, code_chunk)` where the code changed but
    the doc referencing it did not."""

    missing_tests: list[Chunk] = field(default_factory=list)
    """New functions/methods with no `TESTS` edge."""

    broken_edges: list[Edge] = field(default_factory=list)
    """Import edges in head that pointed at symbols now removed."""
