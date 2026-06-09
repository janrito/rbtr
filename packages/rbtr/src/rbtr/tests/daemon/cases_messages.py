"""Cases for daemon protocol message routing and roundtrip.

Each case returns a `MessageScenario` — pure declarative data
describing a raw JSON payload, which `TypeAdapter` to validate it
against, the expected model type, and field-level checks.

No I/O, no helpers, no imports beyond the message models.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import TypeAdapter
from pytest_cases import case

from rbtr.daemon.messages import (
    AutoRebuildNotification,
    BuildIndexRequest,
    BuildIndexResponse,
    ChangedSymbolsRequest,
    ChangedSymbolsResponse,
    EmbedCompleteNotification,
    ErrorCode,
    ErrorResponse,
    FindRefsRequest,
    FindRefsResponse,
    GcMode,
    GcRequest,
    GcResponse,
    IndexErrorNotification,
    ListSymbolsRequest,
    ListSymbolsResponse,
    Notification,
    OkResponse,
    ProgressNotification,
    ReadSymbolRequest,
    ReadSymbolResponse,
    ReadyNotification,
    Request,
    Response,
    SearchRequest,
    SearchResponse,
    ShutdownRequest,
    StatusRequest,
    StatusResponse,
    notification_adapter,
    request_adapter,
    response_adapter,
)


@dataclass(frozen=True)
class MessageScenario:
    """Declarative description of a message routing test."""

    raw: bytes
    adapter: TypeAdapter[Request] | TypeAdapter[Response] | TypeAdapter[Notification]
    expected_type: type
    checks: dict[str, object] = field(default_factory=dict)


# ── Requests ─────────────────────────────────────────────────────────


@case(tags=["request"])
def case_shutdown() -> MessageScenario:
    return MessageScenario(
        raw=b'{"kind":"shutdown"}',
        adapter=request_adapter,
        expected_type=ShutdownRequest,
    )


@case(tags=["request"])
def case_build_index_defaults() -> MessageScenario:
    return MessageScenario(
        raw=b'{"kind":"index","repo_path":"/r"}',
        adapter=request_adapter,
        expected_type=BuildIndexRequest,
        checks={"refs": ["HEAD"], "embed": True},
    )


@case(tags=["request"])
def case_build_index_two_refs() -> MessageScenario:
    return MessageScenario(
        raw=b'{"kind":"index","repo_path":"/r","refs":["main","HEAD"]}',
        adapter=request_adapter,
        expected_type=BuildIndexRequest,
        checks={"refs": ["main", "HEAD"]},
    )


@case(tags=["request"])
def case_search_defaults() -> MessageScenario:
    return MessageScenario(
        raw=b'{"kind":"search","repo_path":"/r","query":"config"}',
        adapter=request_adapter,
        expected_type=SearchRequest,
        checks={"query": "config", "limit": 10, "ref": None},
    )


@case(tags=["request"])
def case_read_symbol() -> MessageScenario:
    return MessageScenario(
        raw=b'{"kind":"read_symbol","repo_path":"/r","symbol":"HttpClient"}',
        adapter=request_adapter,
        expected_type=ReadSymbolRequest,
        checks={"symbol": "HttpClient", "ref": None},
    )


@case(tags=["request"])
def case_list_symbols() -> MessageScenario:
    return MessageScenario(
        raw=b'{"kind":"list_symbols","repo_path":"/r","file_path":"src/app.py"}',
        adapter=request_adapter,
        expected_type=ListSymbolsRequest,
        checks={"file_path": "src/app.py"},
    )


@case(tags=["request"])
def case_find_refs() -> MessageScenario:
    return MessageScenario(
        raw=b'{"kind":"find_refs","repo_path":"/r","symbol":"load_config"}',
        adapter=request_adapter,
        expected_type=FindRefsRequest,
        checks={"symbol": "load_config"},
    )


@case(tags=["request"])
def case_changed_symbols() -> MessageScenario:
    return MessageScenario(
        raw=b'{"kind":"changed_symbols","repo_path":"/r","base":"abc","head":"def"}',
        adapter=request_adapter,
        expected_type=ChangedSymbolsRequest,
        checks={"base": "abc", "head": "def"},
    )


@case(tags=["request"])
def case_status() -> MessageScenario:
    return MessageScenario(
        raw=b'{"kind":"status","repo_path":"/r"}',
        adapter=request_adapter,
        expected_type=StatusRequest,
        checks={"repo_path": "/r"},
    )


@case(tags=["request"])
def case_gc() -> MessageScenario:
    return MessageScenario(
        raw=b'{"kind":"gc","repo_path":"/r","mode":"head_only"}',
        adapter=request_adapter,
        expected_type=GcRequest,
        checks={"mode": GcMode.HEAD_ONLY, "dry_run": False},
    )


# ── Responses ────────────────────────────────────────────────────────


@case(tags=["response"])
def case_ok() -> MessageScenario:
    return MessageScenario(
        raw=b'{"kind":"ok"}',
        adapter=response_adapter,
        expected_type=OkResponse,
    )


@case(tags=["response"])
def case_error() -> MessageScenario:
    return MessageScenario(
        raw=b'{"kind":"error","code":"internal","message":"boom"}',
        adapter=response_adapter,
        expected_type=ErrorResponse,
        checks={"code": ErrorCode.INTERNAL, "message": "boom"},
    )


@case(tags=["response"])
def case_build_index_response() -> MessageScenario:
    return MessageScenario(
        raw=(b'{"kind":"index","resolved_refs":["HEAD"],"stats":{},"errors":[]}'),
        adapter=response_adapter,
        expected_type=BuildIndexResponse,
        checks={"resolved_refs": ["HEAD"], "errors": []},
    )


@case(tags=["response"])
def case_search_response() -> MessageScenario:
    return MessageScenario(
        raw=b'{"kind":"search","results":[]}',
        adapter=response_adapter,
        expected_type=SearchResponse,
        checks={"results": []},
    )


@case(tags=["response"])
def case_read_symbol_response() -> MessageScenario:
    return MessageScenario(
        raw=b'{"kind":"read_symbol","chunks":[]}',
        adapter=response_adapter,
        expected_type=ReadSymbolResponse,
        checks={"chunks": []},
    )


@case(tags=["response"])
def case_list_symbols_response() -> MessageScenario:
    return MessageScenario(
        raw=b'{"kind":"list_symbols","chunks":[]}',
        adapter=response_adapter,
        expected_type=ListSymbolsResponse,
        checks={"chunks": []},
    )


@case(tags=["response"])
def case_find_refs_response() -> MessageScenario:
    return MessageScenario(
        raw=b'{"kind":"find_refs","edges":[]}',
        adapter=response_adapter,
        expected_type=FindRefsResponse,
        checks={"edges": []},
    )


@case(tags=["response"])
def case_changed_symbols_response() -> MessageScenario:
    return MessageScenario(
        raw=b'{"kind":"changed_symbols","changes":[]}',
        adapter=response_adapter,
        expected_type=ChangedSymbolsResponse,
        checks={"changes": []},
    )


@case(tags=["response"])
def case_status_response_empty() -> MessageScenario:
    return MessageScenario(
        raw=b'{"kind":"status"}',
        adapter=response_adapter,
        expected_type=StatusResponse,
        checks={"indexed_refs": []},
    )


@case(tags=["response"])
def case_status_response_with_refs() -> MessageScenario:
    return MessageScenario(
        raw=(
            b'{"kind":"status",'
            b'"indexed_refs":['
            b'{"sha":"abc123","total":50,"embedded":50},'
            b'{"sha":"def456","total":30,"embedded":10}'
            b"]}"
        ),
        adapter=response_adapter,
        expected_type=StatusResponse,
    )


@case(tags=["response"])
def case_gc_response() -> MessageScenario:
    return MessageScenario(
        raw=(
            b'{"kind":"gc","commits_dropped":2,"snapshots_dropped":5,'
            b'"edges_dropped":3,"chunks_dropped":10,'
            b'"elapsed_seconds":0.42,"dry_run":false}'
        ),
        adapter=response_adapter,
        expected_type=GcResponse,
        checks={"commits_dropped": 2, "chunks_dropped": 10, "dry_run": False},
    )


# ── Notifications ────────────────────────────────────────────────────


@case(tags=["notification"])
def case_progress() -> MessageScenario:
    return MessageScenario(
        raw=b'{"kind":"progress","repo_path":"/r","phase":"parsing","current":5,"total":10}',
        adapter=notification_adapter,
        expected_type=ProgressNotification,
        checks={"phase": "parsing", "current": 5, "total": 10},
    )


@case(tags=["notification"])
def case_ready() -> MessageScenario:
    return MessageScenario(
        raw=b'{"kind":"ready","repo_path":"/r","ref":"abc","chunks":100,"embedded":0,"edges":50,"elapsed":1.2}',
        adapter=notification_adapter,
        expected_type=ReadyNotification,
        checks={"ref": "abc", "chunks": 100},
    )


@case(tags=["notification"])
def case_auto_rebuild() -> MessageScenario:
    return MessageScenario(
        raw=b'{"kind":"auto_rebuild","repo_path":"/r","new_ref":"b"}',
        adapter=notification_adapter,
        expected_type=AutoRebuildNotification,
        checks={"new_ref": "b"},
    )


@case(tags=["notification"])
def case_embed_complete() -> MessageScenario:
    return MessageScenario(
        raw=b'{"kind":"embed_complete","repo_path":"/r","ref":"abc","chunks":100,"embedded":42}',
        adapter=notification_adapter,
        expected_type=EmbedCompleteNotification,
        checks={"ref": "abc", "embedded": 42},
    )


@case(tags=["notification"])
def case_index_error() -> MessageScenario:
    return MessageScenario(
        raw=b'{"kind":"index_error","repo_path":"/r","message":"parse failed"}',
        adapter=notification_adapter,
        expected_type=IndexErrorNotification,
        checks={"message": "parse failed"},
    )
