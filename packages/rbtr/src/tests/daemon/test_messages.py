"""Tests for daemon protocol messages — discriminated unions.

Data-first: tests use raw JSON bytes to verify the TypeAdapter
routes to the correct model type. Roundtrip tests construct
models and verify they survive serialise → deserialise.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from rbtr.daemon.messages import (
    AutoRebuildNotification,
    BuildIndexRequest,
    BuildIndexResponse,
    ErrorCode,
    ErrorResponse,
    OkResponse,
    PingRequest,
    PingResponse,
    ProgressNotification,
    ReadyNotification,
    SearchRequest,
    StatusResponse,
    notification_adapter,
    request_adapter,
    response_adapter,
)
from rbtr.index.models import IndexStats

# ── Request discriminator ────────────────────────────────────────────


def test_request_ping() -> None:
    req = request_adapter.validate_json(b'{"kind":"ping"}')
    assert isinstance(req, PingRequest)


def test_request_search_with_defaults() -> None:
    raw = b'{"kind":"search","repo":"/r","query":"config"}'
    req = request_adapter.validate_json(raw)
    assert isinstance(req, SearchRequest)
    assert req.repo == "/r"
    assert req.query == "config"
    assert req.limit == 10
    assert req.ref == "HEAD"


def test_request_build_index_with_defaults() -> None:
    raw = b'{"kind":"build_index","repo":"/r"}'
    req = request_adapter.validate_json(raw)
    assert isinstance(req, BuildIndexRequest)
    assert req.refs == ["HEAD"]


def test_request_build_index_two_refs() -> None:
    raw = b'{"kind":"build_index","repo":"/r","refs":["main","HEAD"]}'
    req = request_adapter.validate_json(raw)
    assert isinstance(req, BuildIndexRequest)
    assert req.refs == ["main", "HEAD"]


def test_request_unknown_kind_rejected() -> None:
    with pytest.raises(ValidationError):
        request_adapter.validate_json(b'{"kind":"unknown"}')


def test_request_extra_field_rejected() -> None:
    with pytest.raises(ValidationError, match="extra"):
        request_adapter.validate_json(b'{"kind":"ping","bogus":1}')


# ── Response discriminator ───────────────────────────────────────────


def test_response_ping() -> None:
    resp = response_adapter.validate_json(b'{"kind":"ping","version":"0.1.0","uptime":1.5}')
    assert isinstance(resp, PingResponse)
    assert resp.version == "0.1.0"


def test_response_ok() -> None:
    resp = response_adapter.validate_json(b'{"kind":"ok"}')
    assert isinstance(resp, OkResponse)


def test_response_error() -> None:
    resp = response_adapter.validate_json(b'{"kind":"error","code":"internal","message":"boom"}')
    assert isinstance(resp, ErrorResponse)
    assert resp.code == ErrorCode.INTERNAL


def test_response_status() -> None:
    resp = response_adapter.validate_json(b'{"kind":"status","exists":true,"total_chunks":100}')
    assert isinstance(resp, StatusResponse)
    assert resp.total_chunks == 100


# ── Notification discriminator ───────────────────────────────────────


def test_notification_progress() -> None:
    n = notification_adapter.validate_json(
        b'{"kind":"progress","repo":"/r","phase":"parsing","current":5,"total":10}'
    )
    assert isinstance(n, ProgressNotification)
    assert n.current == 5


def test_notification_ready() -> None:
    n = notification_adapter.validate_json(
        b'{"kind":"ready","repo":"/r","ref":"abc","chunks":100,"edges":50,"elapsed":1.2}'
    )
    assert isinstance(n, ReadyNotification)


def test_notification_auto_rebuild() -> None:
    n = notification_adapter.validate_json(
        b'{"kind":"auto_rebuild","repo":"/r","old_ref":"a","new_ref":"b"}'
    )
    assert isinstance(n, AutoRebuildNotification)


# ── Roundtrip ────────────────────────────────────────────────────────


def test_request_roundtrip() -> None:
    req = SearchRequest(repo="/r", query="config", limit=5)
    parsed = request_adapter.validate_json(req.model_dump_json())
    assert isinstance(parsed, SearchRequest)
    assert parsed.query == "config"
    assert parsed.limit == 5


def test_response_roundtrip() -> None:
    resp = BuildIndexResponse(refs=["HEAD"], stats=IndexStats(), errors=[])
    parsed = response_adapter.validate_json(resp.model_dump_json())
    assert isinstance(parsed, BuildIndexResponse)


def test_notification_roundtrip() -> None:
    n = ProgressNotification(repo="/r", phase="embedding", current=3, total=10)
    parsed = notification_adapter.validate_json(n.model_dump_json())
    assert isinstance(parsed, ProgressNotification)
    assert parsed.phase == "embedding"


# ── Error codes ──────────────────────────────────────────────────────


def test_all_error_codes_roundtrip() -> None:
    for code in ErrorCode:
        resp = ErrorResponse(code=code, message="test")
        parsed = response_adapter.validate_json(resp.model_dump_json())
        assert isinstance(parsed, ErrorResponse)
        assert parsed.code == code
