"""Daemon protocol — pydantic models with discriminated unions.

Three message categories:
- **Requests** (client → server via REQ/REP)
- **Responses** (server → client via REQ/REP)
- **Notifications** (server → clients via PUB)

Each category is a discriminated union on the `kind` field.
Serialisation uses `model_dump_json()` / `TypeAdapter.validate_json()`.
"""

from __future__ import annotations

import os
from collections.abc import Hashable
from enum import StrEnum
from pathlib import PurePath
from typing import Annotated, Any, Literal, Protocol, runtime_checkable

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationInfo,
    field_validator,
)
from pydantic.json_schema import JsonSchemaValue, models_json_schema
from pydantic_core import from_json

from rbtr.config import WeightTriple, config
from rbtr.daemon.dto import RefOut, SearchHitOut, SymbolOut
from rbtr.daemon.status import DaemonStatusReport
from rbtr.index.models import ChangeKind, IndexStats, QueryKind

# ── Error codes ──────────────────────────────────────────────────────


class ErrorCode(StrEnum):
    """Error codes for the daemon protocol."""

    INVALID_REQUEST = "invalid_request"
    INDEX_NOT_BUILT = "index_not_built"
    INDEX_IN_PROGRESS = "index_in_progress"
    REPO_NOT_FOUND = "repo_not_found"
    INTERNAL = "internal"


class Scope(StrEnum):
    """Breadth of a search or status request.

    `WORKSPACE` is the single repo identified by `repo_path`;
    `ALL` is every indexed repo in the shared store.
    """

    WORKSPACE = "workspace"
    ALL = "all"


# ── Base ─────────────────────────────────────────────────────────────

_STRICT = ConfigDict(extra="forbid")


@runtime_checkable
class HasRepoPath(Protocol):
    """Any message that carries a repository path."""

    repo_path: str


# ── Job types (work queue) ───────────────────────────────────────────


class BuildJob(BaseModel):
    """A build-index job for the unified work queue."""

    model_config = _STRICT
    kind: Literal["build"] = "build"
    repo_path: str
    refs: tuple[str, ...]
    embed: bool = True

    @property
    def dedupe_key(self) -> Hashable:
        return (self.repo_path, self.refs)


class EmbedJob(BaseModel):
    """An embed-index job for the unified work queue."""

    model_config = _STRICT
    kind: Literal["embed"] = "embed"
    repo_path: str
    repo_id: int
    ref: str

    @property
    def dedupe_key(self) -> Hashable:
        return (self.repo_id, self.ref)


Job = Annotated[BuildJob | EmbedJob, Field(discriminator="kind")]


def _decode_json_array(text: str) -> Any:
    """Return *text* JSON-decoded if it is a JSON array, else *text* itself.

    Uses pydantic-core's JSON parser (the same one pydantic uses for
    `model_validate_json`). Only decodes; the decoded value is validated
    by pydantic as usual, so no type-checking is duplicated here.
    """
    if not text.strip().startswith("["):
        return text
    try:
        return from_json(text)
    except ValueError:
        return text


def _unwrap_json_list(value: Any) -> Any:
    """Decode a list argument delivered as a JSON-encoded string.

    Some tool-call providers send a `list[str]` argument as a JSON
    string — bare (`'["a"]'`) or wrapped in a one-element list
    (`['["a"]']`) by an intermediate validator. Decode the JSON so
    pydantic's normal validation runs against the real list (and reports
    its own error for a genuinely wrong shape). Native lists and
    non-JSON strings pass through untouched.
    """
    if isinstance(value, str):
        return _decode_json_array(value)
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], str):
        decoded = _decode_json_array(value[0])
        # Unwrap only when the element really was JSON; a genuine
        # single value (identity unchanged) keeps its one-element list.
        return decoded if decoded is not value[0] else value
    return value


# A list-of-strings request field that tolerates a JSON-encoded-string
# delivery (see `_unwrap_json_list`). Optional and required variants.
OptStrList = Annotated[list[str] | None, BeforeValidator(_unwrap_json_list)]
StrList = Annotated[list[str], BeforeValidator(_unwrap_json_list)]


def _to_repo_relative(value: list[str] | None, info: ValidationInfo) -> list[str] | None:
    """Normalise scoping paths to repo-root-relative POSIX form.

    Chunk `file_path`s are stored repo-root-relative with POSIX
    separators, so scoping matches them exactly. Absolute paths are made
    relative to the request's `repo_path`; a leading `./` is stripped.
    Paths outside the repo root simply won't match any chunk. `None` or
    an empty list passes through unchanged.

    Relies on `repo_path` being declared before the annotated field so
    it is present in `info.data` by the time this runs.
    """
    if not value:
        return value
    repo_path = info.data["repo_path"]
    return [
        PurePath(os.path.relpath(p, repo_path) if os.path.isabs(p) else p).as_posix() for p in value
    ]


# Scoping paths, normalised to repo-root-relative POSIX form at validation
# time so handlers can match them against stored chunk paths directly.
RepoRelativePaths = Annotated[
    list[str] | None, BeforeValidator(_unwrap_json_list), AfterValidator(_to_repo_relative)
]


# ── Requests (client → server) ──────────────────────────────────────


class ShutdownRequest(BaseModel):
    model_config = _STRICT
    kind: Literal["shutdown"] = "shutdown"


class BuildIndexRequest(BaseModel):
    model_config = _STRICT
    kind: Literal["index"] = "index"
    repo_path: str
    refs: StrList = ["HEAD"]
    embed: bool = True


class SearchRequest(BaseModel):
    """Search the code index.

    `weights` overrides the per-`QueryKind` fusion weights for
    the duration of the call.  When `None`, per-kind defaults
    apply.
    """

    model_config = _STRICT
    kind: Literal["search"] = "search"
    repo_path: str
    query: str
    limit: int = 10
    ref: str | None = None
    weights: WeightTriple | None = None
    reranker_pool: int | None = None
    reranker_blend_weight: float | None = None
    keywords: OptStrList = None
    variants: OptStrList = None
    explain: bool = False
    scope: Scope = Scope.WORKSPACE
    query_kind: str | None = Field(
        default=None,
        description=(
            "Force expansion pipeline. One of concept|identifier|code. None means heuristic."
        ),
    )

    @field_validator("query_kind")
    @classmethod
    def _check_query_kind(cls, v: str | None) -> str | None:
        if v is not None and v not in QueryKind:
            msg = f"query_kind must be one of {list(QueryKind)}; got {v!r}"
            raise ValueError(msg)
        return v

    @field_validator("keywords", "variants")
    @classmethod
    def _strip_empty(cls, v: list[str] | None) -> list[str] | None:
        """Strip whitespace and drop empty strings."""
        if v is None:
            return None
        return [stripped for x in v if (stripped := x.strip())]

    @field_validator("keywords")
    @classmethod
    def _dedup_and_cap(cls, v: list[str] | None) -> list[str] | None:
        """Deduplicate preserving order, cap at `config.max_expansion_keywords`."""
        if v is None:
            return None
        return list(dict.fromkeys(v))[: config.max_expansion_keywords]


class ReadSymbolRequest(BaseModel):
    model_config = _STRICT
    kind: Literal["read_symbol"] = "read_symbol"
    repo_path: str
    symbol: str
    ref: str | None = None
    file_paths: RepoRelativePaths = None


class ListSymbolsRequest(BaseModel):
    model_config = _STRICT
    kind: Literal["list_symbols"] = "list_symbols"
    repo_path: str
    file_path: str
    ref: str | None = None


class FindRefsRequest(BaseModel):
    model_config = _STRICT
    kind: Literal["find_refs"] = "find_refs"
    repo_path: str
    symbol: str
    ref: str | None = None
    file_paths: RepoRelativePaths = None


class ChangedSymbolsRequest(BaseModel):
    model_config = _STRICT
    kind: Literal["changed_symbols"] = "changed_symbols"
    repo_path: str
    base: str
    head: str
    file_paths: RepoRelativePaths = None


class StatusRequest(BaseModel):
    model_config = _STRICT
    kind: Literal["status"] = "status"
    repo_path: str
    scope: Scope = Scope.WORKSPACE


class GcMode(StrEnum):
    """What a `rbtr gc` invocation is allowed to delete."""

    HEAD_ONLY = "head_only"  # keep current HEAD, drop the rest
    KEEP_REFS = "keep_refs"  # keep all local refs (branches/tags/notes/HEAD)
    KEEP = "keep"  # keep only listed refs
    DROP = "drop"  # drop only listed refs
    ORPHANS = "orphans"  # sweep residue only, drop no commits


class GcRequest(BaseModel):
    model_config = _STRICT
    kind: Literal["gc"] = "gc"
    repo_path: str
    mode: GcMode
    refs: list[str] = []
    dry_run: bool = False


Request = Annotated[
    ShutdownRequest
    | BuildIndexRequest
    | SearchRequest
    | ReadSymbolRequest
    | ListSymbolsRequest
    | FindRefsRequest
    | ChangedSymbolsRequest
    | StatusRequest
    | GcRequest,
    Field(discriminator="kind"),
]

request_adapter: TypeAdapter[Request] = TypeAdapter(Request)


# ── Responses (server → client) ─────────────────────────────────────


class ErrorResponse(BaseModel):
    model_config = _STRICT
    kind: Literal["error"] = "error"
    code: ErrorCode
    message: str


class OkResponse(BaseModel):
    model_config = _STRICT
    kind: Literal["ok"] = "ok"


class BuildIndexResponse(BaseModel):
    model_config = _STRICT
    kind: Literal["index"] = "index"
    resolved_refs: list[str]
    stats: IndexStats
    errors: list[str]


class SearchResponse(BaseModel):
    model_config = _STRICT
    kind: Literal["search"] = "search"
    results: list[SearchHitOut]
    query_kind: QueryKind | None = Field(default=None, exclude_if=lambda v: v is None)


class ReadSymbolResponse(BaseModel):
    model_config = _STRICT
    kind: Literal["read_symbol"] = "read_symbol"
    chunks: list[SymbolOut]


class ListSymbolsResponse(BaseModel):
    model_config = _STRICT
    kind: Literal["list_symbols"] = "list_symbols"
    chunks: list[SymbolOut]


class FindRefsResponse(BaseModel):
    model_config = _STRICT
    kind: Literal["find_refs"] = "find_refs"
    refs: list[RefOut]


class ChangedSymbol(BaseModel):
    """One changed symbol: its chunk plus how it changed."""

    model_config = _STRICT
    chunk: SymbolOut
    change: ChangeKind


class ChangedSymbolsResponse(BaseModel):
    model_config = _STRICT
    kind: Literal["changed_symbols"] = "changed_symbols"
    changes: list[ChangedSymbol]


class ActiveJob(BaseModel):
    """A build or embed job currently running in the daemon.

    Used for both `active_build` and `active_embed` on
    `StatusResponse` — the field name distinguishes them.

    `ref` is the resolved SHA under build/embed.  `phase`
    is the current stage (`"starting"`, `"parsing"`, or
    `"embedding"`).  `current` / `total` come from the
    orchestrator's progress callbacks; both are 0 until the
    first callback fires.  `elapsed_seconds` is computed by
    the daemon at snapshot time so the CLI doesn't need to
    reason about clock boundaries.
    """

    model_config = _STRICT
    repo_path: str
    ref: str
    phase: str
    current: int
    total: int
    elapsed_seconds: float


class IndexedRef(BaseModel):
    """Per-ref status: symbolic names, chunk count, embed completeness."""

    model_config = _STRICT
    sha: str
    names: list[str] = []
    total: int
    embedded: int
    repo_path: str | None = None


class StatusResponse(BaseModel):
    model_config = _STRICT
    kind: Literal["status"] = "status"
    db_path: str | None = None
    indexed_refs: list[IndexedRef] = []
    active_build: ActiveJob | None = None
    active_embed: ActiveJob | None = None


class GcResponse(BaseModel):
    model_config = _STRICT
    kind: Literal["gc"] = "gc"
    commits_dropped: int
    snapshots_dropped: int
    edges_dropped: int
    chunks_dropped: int
    elapsed_seconds: float
    dry_run: bool = False


Response = Annotated[
    ErrorResponse
    | OkResponse
    | BuildIndexResponse
    | SearchResponse
    | ReadSymbolResponse
    | ListSymbolsResponse
    | FindRefsResponse
    | ChangedSymbolsResponse
    | StatusResponse
    | GcResponse,
    Field(discriminator="kind"),
]

response_adapter: TypeAdapter[Response] = TypeAdapter(Response)


# ── Notifications (server → PUB) ────────────────────────────────────


class ProgressNotification(BaseModel):
    model_config = _STRICT
    kind: Literal["progress"] = "progress"
    repo_path: str
    phase: str
    current: int
    total: int


class ReadyNotification(BaseModel):
    model_config = _STRICT
    kind: Literal["ready"] = "ready"
    repo_path: str
    ref: str
    chunks: int
    embedded: int
    edges: int
    elapsed: float


class AutoRebuildNotification(BaseModel):
    model_config = _STRICT
    kind: Literal["auto_rebuild"] = "auto_rebuild"
    repo_path: str
    new_ref: str


class EmbedCompleteNotification(BaseModel):
    model_config = _STRICT
    kind: Literal["embed_complete"] = "embed_complete"
    repo_path: str
    ref: str
    chunks: int
    embedded: int


class IndexErrorNotification(BaseModel):
    model_config = _STRICT
    kind: Literal["index_error"] = "index_error"
    repo_path: str
    message: str


Notification = Annotated[
    ProgressNotification
    | ReadyNotification
    | EmbedCompleteNotification
    | AutoRebuildNotification
    | IndexErrorNotification,
    Field(discriminator="kind"),
]

notification_adapter: TypeAdapter[Notification] = TypeAdapter(Notification)


class ProtocolSchema(BaseModel):
    """The protocol's top-level types, bundled for one-pass schema gen.

    A single `models_json_schema` pass over this bundle emits every wire
    message union and the CLI-only `DaemonStatusReport` with one shared
    `$defs` — no per-union duplication for the TypeScript codegen to
    reconcile. Used only by `protocol_json_schema`; not a wire message.
    """

    request: Request
    response: Response
    notification: Notification
    daemon_status_report: DaemonStatusReport


def protocol_json_schema() -> JsonSchemaValue:
    """The protocol's JSON Schema (generation only), for codegen.

    One `models_json_schema` pass over `ProtocolSchema` yields a single
    schema document with shared definitions: the three message unions
    under `$defs.ProtocolSchema.properties`, every referenced model and
    `DaemonStatusReport` as shared `$defs` entries. Serialising it is
    the caller's concern.
    """
    _, schema = models_json_schema(
        [(ProtocolSchema, "validation")],
        ref_template="#/$defs/{model}",
    )
    return schema
