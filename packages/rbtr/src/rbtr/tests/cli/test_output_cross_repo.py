"""Cross-repo CLI rendering — behaviour through the `emit` surface.

Drives `emit()` and asserts on the rendered text captured by the
`rendered` fixture.  The render internals are never called
directly.
"""

from __future__ import annotations

from io import StringIO
from typing import get_args

import pytest
from pydantic import BaseModel
from pytest_cases import parametrize_with_cases

from rbtr.cli.output import emit
from rbtr.daemon.dto import ResolvedRef
from rbtr.daemon.messages import (
    ChangedSymbolsResponse,
    DaemonConfigResponse,
    ErrorResponse,
    FindRefsResponse,
    ForgetResponse,
    GcResponse,
    ListSymbolsResponse,
    OkResponse,
    ReadSymbolResponse,
    Response,
    SearchResponse,
    StatusResponse,
    UnwatchResponse,
    WatchResponse,
    WatchSetResponse,
)
from rbtr.domain.models import IndexStats, RefSource

from .cases_output import RenderScenario


@parametrize_with_cases("scenario", cases=".cases_output")
def test_emit_renders_repo_attribution(scenario: RenderScenario, rendered: StringIO) -> None:
    """Rendered output shows (and omits) the expected repo cues."""
    emit(scenario.model)

    out = rendered.getvalue()
    for text in scenario.expected:
        assert text in out
    for text in scenario.forbidden:
        assert text not in out


@pytest.fixture
def resolved() -> ResolvedRef:
    return ResolvedRef(sha="abc", source=RefSource.HEAD)


@pytest.fixture
def one_of_each_response(resolved: ResolvedRef) -> tuple[BaseModel, ...]:
    """An empty instance of every response `emit` can be handed."""
    return (
        OkResponse(),
        WatchResponse(resolved_refs=[], stats=IndexStats(), errors=[]),
        WatchSetResponse(watched=[]),
        SearchResponse(results=[], resolved=None),
        ReadSymbolResponse(chunks=[], resolved=resolved, file_paths=None),
        ListSymbolsResponse(chunks=[], resolved=resolved, file_path="a.py"),
        FindRefsResponse(refs=[], resolved=resolved, file_paths=None),
        ChangedSymbolsResponse(changes=[], base_sha="abc", head_sha="def", file_paths=None),
        StatusResponse(db_path="/db"),
        DaemonConfigResponse(rbtr_version="0", config={}, plugins=[]),
        GcResponse(
            snapshots_dropped=0,
            file_snapshots_dropped=0,
            edges_dropped=0,
            chunks_freed=0,
            elapsed_seconds=0.0,
        ),
        UnwatchResponse(),
        ForgetResponse(),
    )


def test_every_response_kind_has_a_renderer(
    one_of_each_response: tuple[BaseModel, ...], rendered: StringIO
) -> None:
    """`emit` renders every member of the `Response` union.

    The suite otherwise reaches `emit` through subprocesses, whose
    stdout is not a terminal and so takes the JSON path — a response
    model with no rich case raises only in a real terminal.

    `ErrorResponse` is excluded: every command reports it through
    `fail`, which exits, so it never reaches `emit`.
    """
    covered = {type(model) for model in one_of_each_response}
    assert covered == set(get_args(get_args(Response)[0])) - {ErrorResponse}

    for model in one_of_each_response:
        emit(model)
