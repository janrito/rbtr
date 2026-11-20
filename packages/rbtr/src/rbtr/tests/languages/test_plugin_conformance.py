"""Every installed language plugin, over every one of its sample projects: every line is in a chunk, and every span is one chunk.

A plugin's samples are the `tests/samples/` directory of its package. A
project is each directory under it but `imports/` and `calls/`, each
directory under those two, and each file directly under it, alone. Each
project is built with every installed language, as a user's repository
would be, and read back from the index.
"""

from __future__ import annotations

import importlib.metadata
import importlib.resources
from pathlib import Path

import pytest

from rbtr.index.refs import resolve_read_ref
from rbtr.testing import Indexed
from rbtr.testing.assertions import (
    assert_every_line_in_a_chunk,
    assert_no_undecided_kind_overlaps,
)


@pytest.mark.parametrize(
    "samples",
    sorted(
        {
            Path(str(importlib.resources.files(ep.module.split(".")[0]))) / "tests" / "samples"
            for ep in importlib.metadata.entry_points(group="rbtr.languages")
            if not ep.module.startswith("rbtr.")
        }
    ),
    ids=lambda samples: samples.parents[1].name,
)
def test_a_plugin_has_samples(samples: Path) -> None:
    """A plugin ships the sample projects the rules are checked over."""
    assert any(samples.rglob("*.*"))


@pytest.mark.parametrize(
    "project",
    [
        project
        for samples in sorted(
            {
                Path(str(importlib.resources.files(ep.module.split(".")[0]))) / "tests" / "samples"
                for ep in importlib.metadata.entry_points(group="rbtr.languages")
                if not ep.module.startswith("rbtr.")
            }
        )
        for pattern in ("*/", "imports/*/", "calls/*/", "*.*")
        for project in sorted(samples.glob(pattern))
        if project.name not in {"imports", "calls"}
    ],
    ids=lambda project: str(project).partition("/src/")[2],
)
def test_every_line_is_in_a_chunk_and_every_span_is_one(indexed: Indexed, project: Path) -> None:
    """Every non-blank line is in a chunk, no span has two kinds no narrowing decides, and each chunk is the text of its byte span."""
    files = project if project.is_dir() else {project.name: project.read_text()}
    with indexed(files) as build:
        assert_every_line_in_a_chunk(build)
        assert_no_undecided_kind_overlaps(build)
        [repo] = build.index.repos()
        chunks = build.index.chunks(at=resolve_read_ref(build.index, repo, None).at)
    disagreeing = [
        chunk
        for chunk in chunks.rows(named=True)
        for text in [(project / chunk["file_path"] if project.is_dir() else project).read_bytes()]
        if (chunk["content"], chunk["line_start"], chunk["line_end"])
        != (
            text[chunk["start_byte"] : chunk["end_byte"]].decode("utf-8", "replace"),
            text.count(b"\n", 0, chunk["start_byte"]) + 1,
            text.count(b"\n", 0, max(chunk["start_byte"], chunk["end_byte"] - 1)) + 1,
        )
    ]
    assert disagreeing == []
