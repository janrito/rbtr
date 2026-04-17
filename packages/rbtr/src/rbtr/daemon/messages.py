"""Daemon protocol — pydantic models with discriminated unions.

Three message categories:
- **Requests** (client → server via REQ/REP)
- **Responses** (server → client via REQ/REP)
- **Notifications** (server → clients via PUB)

Each category is a discriminated union on the `kind` field.
Serialisation uses `model_dump_json()` / `TypeAdapter.validate_json()`.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from rbtr.index.models import Chunk, Edge, IndexStats
from rbtr.index.search import ScoredResult

# ── Error codes ──────────────────────────────────────────────────────


class ErrorCode(StrEnum):
    """Error codes for the daemon protocol."""

    INVALID_REQUEST = "invalid_request"
    INDEX_NOT_BUILT = "index_not_built"
    INDEX_IN_PROGRESS = "index_in_progress"
    REPO_NOT_FOUND = "repo_not_found"
    INTERNAL = "internal"


# ── Base ─────────────────────────────────────────────────────────────

_STRICT = ConfigDict(extra="forbid")


# ── Requests (client → server) ──────────────────────────────────────


class PingRequest(BaseModel):
    model_config = _STRICT
    kind: Literal["ping"] = "ping"


class ShutdownRequest(BaseModel):
    model_config = _STRICT
    kind: Literal["shutdown"] = "shutdown"


class BuildIndexRequest(BaseModel):
    model_config = _STRICT
    kind: Literal["build_index"] = "build_index"
    repo: str
    refs: list[str] = ["HEAD"]


class SearchRequest(BaseModel):
    model_config = _STRICT
    kind: Literal["search"] = "search"
    repo: str
    query: str
    limit: int = 10
    ref: str = "HEAD"


class ReadSymbolRequest(BaseModel):
    model_config = _STRICT
    kind: Literal["read_symbol"] = "read_symbol"
    repo: str
    name: str
    ref: str = "HEAD"


class ListSymbolsRequest(BaseModel):
    model_config = _STRICT
    kind: Literal["list_symbols"] = "list_symbols"
    repo: str
    file_path: str
    ref: str = "HEAD"


class FindRefsRequest(BaseModel):
    model_config = _STRICT
    kind: Literal["find_refs"] = "find_refs"
    repo: str
    symbol: str
    ref: str = "HEAD"


class ChangedSymbolsRequest(BaseModel):
    model_config = _STRICT
    kind: Literal["changed_symbols"] = "changed_symbols"
    repo: str
    base: str
    head: str


class StatusRequest(BaseModel):
    model_config = _STRICT
    kind: Literal["status"] = "status"
    repo: str


Request = Annotated[
    PingRequest
    | ShutdownRequest
    | BuildIndexRequest
    | SearchRequest
    | ReadSymbolRequest
    | ListSymbolsRequest
    | FindRefsRequest
    | ChangedSymbolsRequest
    | StatusRequest,
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


class PingResponse(BaseModel):
    model_config = _STRICT
    kind: Literal["ping"] = "ping"
    version: str
    uptime: float


class BuildIndexResponse(BaseModel):
    model_config = _STRICT
    kind: Literal["build_index"] = "build_index"
    refs: list[str]
    stats: IndexStats
    errors: list[str]


class SearchResponse(BaseModel):
    model_config = _STRICT
    kind: Literal["search"] = "search"
    results: list[ScoredResult]


class ReadSymbolResponse(BaseModel):
    model_config = _STRICT
    kind: Literal["read_symbol"] = "read_symbol"
    chunks: list[Chunk]


class ListSymbolsResponse(BaseModel):
    model_config = _STRICT
    kind: Literal["list_symbols"] = "list_symbols"
    chunks: list[Chunk]


class FindRefsResponse(BaseModel):
    model_config = _STRICT
    kind: Literal["find_refs"] = "find_refs"
    edges: list[Edge]


class ChangedSymbolsResponse(BaseModel):
    model_config = _STRICT
    kind: Literal["changed_symbols"] = "changed_symbols"
    chunks: list[Chunk]


class StatusResponse(BaseModel):
    model_config = _STRICT
    kind: Literal["status"] = "status"
    exists: bool
    db_path: str | None = None
    total_chunks: int | None = None


Response = Annotated[
    ErrorResponse
    | OkResponse
    | PingResponse
    | BuildIndexResponse
    | SearchResponse
    | ReadSymbolResponse
    | ListSymbolsResponse
    | FindRefsResponse
    | ChangedSymbolsResponse
    | StatusResponse,
    Field(discriminator="kind"),
]

response_adapter: TypeAdapter[Response] = TypeAdapter(Response)


# ── Notifications (server → PUB) ────────────────────────────────────


class ProgressNotification(BaseModel):
    model_config = _STRICT
    kind: Literal["progress"] = "progress"
    repo: str
    phase: str
    current: int
    total: int


class ReadyNotification(BaseModel):
    model_config = _STRICT
    kind: Literal["ready"] = "ready"
    repo: str
    ref: str
    chunks: int
    edges: int
    elapsed: float


class AutoRebuildNotification(BaseModel):
    model_config = _STRICT
    kind: Literal["auto_rebuild"] = "auto_rebuild"
    repo: str
    new_ref: str


class IndexErrorNotification(BaseModel):
    model_config = _STRICT
    kind: Literal["index_error"] = "index_error"
    repo: str
    message: str


Notification = Annotated[
    ProgressNotification | ReadyNotification | AutoRebuildNotification | IndexErrorNotification,
    Field(discriminator="kind"),
]

notification_adapter: TypeAdapter[Notification] = TypeAdapter(Notification)
