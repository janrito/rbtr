"""The levelling rule on rows alone: each levelled capture's end and scope, from its rank and order."""

from __future__ import annotations

import polars as pl
from pytest_cases import parametrize_with_cases

from rbtr.domain.models import ChunkKind
from rbtr.domain.rules import CaptureRow, FileRow, levelled
from rbtr.languages.treesitter import Capture, Metadata

from .cases_levelled import LevelledScenario


@parametrize_with_cases("scenario", cases=".cases_levelled")
def test_a_levelled_capture_ends_at_the_next_and_sits_under_a_higher_one(
    scenario: LevelledScenario,
) -> None:
    fragment = Metadata(module=None, names=None, language_hint=None, node_types=[])
    rows = [
        Capture(
            file=0,
            source=scenario.source,
            order=order,
            match=None if kind == ChunkKind.FRAGMENT else order,
            language="doc",
            kind=kind,
            name=name,
            scope="",
            in_class=False,
            column=0,
            docs=[],
            start_byte=start_byte,
            end_byte=end_byte,
            line_start=line_start,
            line_end=line_end,
            metadata=fragment if kind == ChunkKind.FRAGMENT else None,
            level=level,
            level_style=level_style,
            source_end=scenario.source_end or len(scenario.content),
        )
        for order, (kind, name, start_byte, end_byte, line_start, line_end, level, level_style) in (
            enumerate(scenario.captures)
        )
    ]
    captures = pl.DataFrame(rows, schema=CaptureRow.create_empty().schema).pipe(
        CaptureRow.validate, cast=True
    )
    files = pl.DataFrame(
        {
            "file": [0],
            "path": ["doc"],
            "blob_sha": ["b"],
            "language": ["doc"],
            "content": [scenario.content],
        }
    ).pipe(FileRow.validate, cast=True)
    out = levelled(captures.lazy(), files).collect()
    assert out.sort("order").select("order", "end_byte", "line_end", "scope").rows() == (
        scenario.expected
    )
