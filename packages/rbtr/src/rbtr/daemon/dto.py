"""Output DTOs for the read commands.

These are the API/wire shape returned by the daemon and printed by the
CLI's `--json`. They are deliberately distinct from the storage models
in `rbtr.index.models`: storage rows carry identity hashes (`id`,
`blob_sha`), an embedding, ranking internals, and an always-present
`metadata` bag, none of which a caller needs. Projecting to a DTO keeps
the public contract minimal and low-noise — agents read these payloads
straight into their context.

Being output-only, a DTO never flows into the write/staging path, so it
is free to omit empty fields (which a storage model cannot — the same
`model_dump` builds its insert frame).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from rbtr.index.models import Chunk, ChunkKind, EdgeKind, ImportMeta, ScoredChunk

_STRICT = ConfigDict(extra="forbid")


class SymbolOut(BaseModel):
    """A symbol as returned by read-symbol, list-symbols, changed-symbols."""

    model_config = _STRICT

    name: str
    kind: ChunkKind
    file_path: str
    scope: str = ""
    language: str = ""
    content: str
    line_start: int
    line_end: int
    metadata: ImportMeta | None = Field(default=None, exclude_if=lambda v: v is None)

    @classmethod
    def from_chunk(cls, chunk: Chunk) -> SymbolOut:
        meta = chunk.metadata if chunk.metadata != ImportMeta() else None
        return cls(
            name=chunk.name,
            kind=chunk.kind,
            file_path=chunk.file_path,
            scope=chunk.scope,
            language=chunk.language,
            content=chunk.content,
            line_start=chunk.line_start,
            line_end=chunk.line_end,
            metadata=meta,
        )


class SearchSignals(BaseModel):
    """The per-signal ranking breakdown behind a hit's fused `score`.

    Returned only when a search requests `explain`; the weight and
    reranker tuners re-rank candidates offline from these components.
    Not part of the default payload.
    """

    model_config = _STRICT

    lexical: float
    semantic: float
    name_match: float
    kind_boost: float
    file_penalty: float
    importance: float
    proximity: float
    fusion: float
    reranker: float

    @classmethod
    def from_scored(cls, sc: ScoredChunk) -> SearchSignals:
        return cls(
            lexical=sc.lexical,
            semantic=sc.semantic,
            name_match=sc.name_match,
            kind_boost=sc.kind_boost,
            file_penalty=sc.file_penalty,
            importance=sc.importance,
            proximity=sc.proximity,
            fusion=sc.fusion,
            reranker=sc.reranker,
        )


class SearchHitOut(BaseModel):
    """A search result: a symbol plus its fused score.

    Carries the single final `score`. The ranking-signal breakdown
    (`signals`) is included only when the search requests `explain`,
    keeping the default payload low-noise.
    """

    model_config = _STRICT

    name: str
    kind: ChunkKind
    file_path: str
    scope: str = ""
    language: str = ""
    content: str
    line_start: int
    line_end: int
    match_line_offset: int | None = Field(default=None, exclude_if=lambda v: v is None)
    matched_terms: list[str] = Field(default_factory=list, exclude_if=lambda v: not v)
    metadata: ImportMeta | None = Field(default=None, exclude_if=lambda v: v is None)
    repo_path: str | None = Field(default=None, exclude_if=lambda v: v is None)
    score: float
    signals: SearchSignals | None = Field(default=None, exclude_if=lambda v: v is None)

    @classmethod
    def from_scored(cls, sc: ScoredChunk, *, explain: bool = False) -> SearchHitOut:
        meta = sc.metadata if sc.metadata != ImportMeta() else None
        return cls(
            name=sc.name,
            kind=sc.kind,
            file_path=sc.file_path,
            scope=sc.scope,
            language=sc.language,
            content=sc.content,
            line_start=sc.line_start,
            line_end=sc.line_end,
            match_line_offset=sc.match_line_offset,
            matched_terms=sc.matched_terms,
            metadata=meta,
            repo_path=sc.repo_path,
            score=sc.score,
            signals=SearchSignals.from_scored(sc) if explain else None,
        )


class RefOut(BaseModel):
    """A reference to the queried symbol, resolved to its referrer.

    `find-refs` answers "what references this symbol"; each edge's target
    is the queried symbol, so the legible part is the source. The source
    chunk's identity is surfaced here (rather than an opaque chunk-id
    hash) along with the relationship `edge` kind.
    """

    model_config = _STRICT

    name: str
    kind: ChunkKind
    file_path: str
    line_start: int
    edge: EdgeKind


RefOuts = TypeAdapter(list[RefOut])
"""Bulk validator for a `find-refs` result frame (one call, not per row)."""


class PluginInfo(BaseModel):
    """One installed language plugin, per language it registers.

    Package and version repeat across a package's languages; the
    `extraction_serial` is the language's own extraction-invalidation
    stamp, so it can differ between registrations of one package.
    """

    model_config = _STRICT

    language: str
    package: str
    version: str
    extraction_serial: int
