"""The row schemas and role models refuse values whose fields disagree: each case is one, written in full."""

from __future__ import annotations

import dataframely as dy
import polars as pl
import pytest

from rbtr.domain.edge_frames import ExportRow, PathRow
from rbtr.domain.rules import CaptureRow, ChunkRow


@pytest.mark.parametrize(
    "row",
    [
        pytest.param(
            {
                "blob_sha": "b",
                "file_language": "python",
                "language": "python",
                "kind": "import",
                "name": "x",
                "scope": "",
                "content": "x",
                "line_start": 1,
                "line_end": 1,
                "metadata": {
                    "module": "os",
                    "names": "",
                    "language_hint": "",
                    "node_types": None,
                    "refs": None,
                    "bindings": [{"name": "os", "local": "os"}],
                },
                "docs": None,
            },
            id="an-import",
        ),
        pytest.param(
            {
                "blob_sha": "b",
                "file_language": "python",
                "language": "python",
                "kind": "class",
                "name": "x",
                "scope": "",
                "content": "x",
                "line_start": 1,
                "line_end": 1,
                "metadata": {
                    "module": None,
                    "names": None,
                    "language_hint": None,
                    "node_types": None,
                    "refs": [
                        {
                            "kind": "inherits",
                            "lookup": "unqualified",
                            "path": ["Base"],
                            "type": None,
                        },
                        {"kind": "calls", "lookup": "unqualified", "path": ["make"], "type": None},
                    ],
                    "bindings": None,
                },
                "docs": None,
            },
            id="a-class-with-a-base-and-a-call",
        ),
        pytest.param(
            {
                "blob_sha": "b",
                "file_language": "python",
                "language": "python",
                "kind": "function",
                "name": "x",
                "scope": "",
                "content": "x",
                "line_start": 1,
                "line_end": 1,
                "metadata": {
                    "module": None,
                    "names": None,
                    "language_hint": None,
                    "node_types": None,
                    "refs": [
                        {
                            "kind": "calls",
                            "lookup": "type",
                            "path": ["o", "save"],
                            "type": ["Order"],
                        },
                        {"kind": "calls", "lookup": "type", "path": ["f"], "type": None},
                        {"kind": "calls", "lookup": "member", "path": ["run"], "type": None},
                    ],
                    "bindings": None,
                },
                "docs": None,
            },
            id="a-function-calling-through-values-and-self",
        ),
        pytest.param(
            {
                "blob_sha": "b",
                "file_language": "python",
                "language": "python",
                "kind": "fragment",
                "name": "x",
                "scope": "",
                "content": "x",
                "line_start": 1,
                "line_end": 1,
                "metadata": {
                    "module": None,
                    "names": None,
                    "language_hint": None,
                    "node_types": ["expression_statement"],
                    "refs": [],
                    "bindings": None,
                },
                "docs": None,
            },
            id="a-fragment",
        ),
        pytest.param(
            {
                "blob_sha": "b",
                "file_language": "python",
                "language": "python",
                "kind": "doc_section",
                "name": "x",
                "scope": "",
                "content": "x",
                "line_start": 1,
                "line_end": 1,
                "metadata": None,
                "docs": None,
            },
            id="a-section",
        ),
    ],
)
def test_a_chunk_whose_fields_agree_is_accepted(row: dict) -> None:
    ChunkRow.validate(pl.DataFrame([row]), cast=True)


@pytest.mark.parametrize(
    ("row", "rule"),
    [
        pytest.param(
            {
                "blob_sha": "b",
                "file_language": "python",
                "language": "python",
                "kind": "function",
                "name": "x",
                "scope": "",
                "content": "x",
                "line_start": 1,
                "line_end": 1,
                "metadata": {
                    "module": "os",
                    "names": None,
                    "language_hint": None,
                    "node_types": None,
                    "refs": [],
                    "bindings": None,
                },
                "docs": None,
            },
            "import_fields_only_on_imports",
            id="a-module-on-a-function",
        ),
        pytest.param(
            {
                "blob_sha": "b",
                "file_language": "python",
                "language": "python",
                "kind": "import",
                "name": "x",
                "scope": "",
                "content": "x",
                "line_start": 1,
                "line_end": 1,
                "metadata": {
                    "module": "os",
                    "names": "",
                    "language_hint": "",
                    "node_types": None,
                    "refs": None,
                    "bindings": None,
                },
                "docs": None,
            },
            "import_fields_only_on_imports",
            id="an-import-without-bindings",
        ),
        pytest.param(
            {
                "blob_sha": "b",
                "file_language": "python",
                "language": "python",
                "kind": "class",
                "name": "x",
                "scope": "",
                "content": "x",
                "line_start": 1,
                "line_end": 1,
                "metadata": {
                    "module": None,
                    "names": None,
                    "language_hint": None,
                    "node_types": ["block"],
                    "refs": [],
                    "bindings": None,
                },
                "docs": None,
            },
            "node_types_only_on_fragments",
            id="node-types-on-a-class",
        ),
        pytest.param(
            {
                "blob_sha": "b",
                "file_language": "python",
                "language": "python",
                "kind": "import",
                "name": "x",
                "scope": "",
                "content": "x",
                "line_start": 1,
                "line_end": 1,
                "metadata": {
                    "module": "os",
                    "names": "",
                    "language_hint": "",
                    "node_types": None,
                    "refs": [],
                    "bindings": [],
                },
                "docs": None,
            },
            "refs_only_on_definitions_classes_and_fragments",
            id="refs-on-an-import",
        ),
        pytest.param(
            {
                "blob_sha": "b",
                "file_language": "python",
                "language": "python",
                "kind": "function",
                "name": "x",
                "scope": "",
                "content": "x",
                "line_start": 1,
                "line_end": 1,
                "metadata": {
                    "module": None,
                    "names": None,
                    "language_hint": None,
                    "node_types": None,
                    "refs": [
                        {
                            "kind": "inherits",
                            "lookup": "unqualified",
                            "path": ["Base"],
                            "type": None,
                        }
                    ],
                    "bindings": None,
                },
                "docs": None,
            },
            "bases_only_on_classes",
            id="a-base-on-a-function",
        ),
        pytest.param(
            {
                "blob_sha": "b",
                "file_language": "python",
                "language": "python",
                "kind": "function",
                "name": "x",
                "scope": "",
                "content": "x",
                "line_start": 1,
                "line_end": 1,
                "metadata": {
                    "module": None,
                    "names": None,
                    "language_hint": None,
                    "node_types": None,
                    "refs": [{"kind": "calls", "lookup": "member", "path": [], "type": None}],
                    "bindings": None,
                },
                "docs": None,
            },
            "refs_are_well_formed",
            id="an-empty-path",
        ),
        pytest.param(
            {
                "blob_sha": "b",
                "file_language": "python",
                "language": "python",
                "kind": "function",
                "name": "x",
                "scope": "",
                "content": "x",
                "line_start": 1,
                "line_end": 1,
                "metadata": {
                    "module": None,
                    "names": None,
                    "language_hint": None,
                    "node_types": None,
                    "refs": [
                        {"kind": "calls", "lookup": "unqualified", "path": ["f"], "type": ["Order"]}
                    ],
                    "bindings": None,
                },
                "docs": None,
            },
            "refs_are_well_formed",
            id="a-type-on-a-name",
        ),
        pytest.param(
            {
                "blob_sha": "b",
                "file_language": "python",
                "language": "python",
                "kind": "class",
                "name": "x",
                "scope": "",
                "content": "x",
                "line_start": 1,
                "line_end": 1,
                "metadata": {
                    "module": None,
                    "names": None,
                    "language_hint": None,
                    "node_types": None,
                    "refs": [
                        {"kind": "inherits", "lookup": "member", "path": ["Base"], "type": None}
                    ],
                    "bindings": None,
                },
                "docs": None,
            },
            "refs_are_well_formed",
            id="a-base-through-self",
        ),
    ],
)
def test_a_chunk_whose_fields_disagree_is_refused(row: dict, rule: str) -> None:
    with pytest.raises(dy.exc.ValidationError, match=rule):
        ChunkRow.validate(pl.DataFrame([row]), cast=True)


@pytest.mark.parametrize(
    ("row", "rule"),
    [
        pytest.param(
            {
                "file": 0,
                "source": 0,
                "order": 0,
                "match": 3,
                "language": "python",
                "kind": "fragment",
                "name": "",
                "scope": "",
                "in_class": False,
                "column": 0,
                "docs": [],
                "start_byte": 0,
                "end_byte": 1,
                "line_start": 1,
                "line_end": 1,
                "metadata": {
                    "module": None,
                    "names": None,
                    "language_hint": None,
                    "node_types": ["block"],
                    "refs": [],
                    "bindings": None,
                },
                "level": None,
                "level_style": None,
                "source_end": 1,
            },
            "a_match_except_on_fragments",
            id="a-fragment-with-a-match",
        ),
        pytest.param(
            {
                "file": 0,
                "source": 0,
                "order": 0,
                "match": None,
                "language": "python",
                "kind": "import",
                "name": "import os",
                "scope": "",
                "in_class": False,
                "column": 0,
                "docs": [],
                "start_byte": 0,
                "end_byte": 9,
                "line_start": 1,
                "line_end": 1,
                "metadata": {
                    "module": "os",
                    "names": "",
                    "language_hint": "",
                    "node_types": None,
                    "refs": None,
                    "bindings": [],
                },
                "level": None,
                "level_style": None,
                "source_end": 9,
            },
            "a_match_except_on_fragments",
            id="an-import-with-no-match",
        ),
        pytest.param(
            {
                "file": 0,
                "source": 0,
                "order": 0,
                "match": 0,
                "language": "doc",
                "kind": "doc_section",
                "name": "Guide",
                "scope": "",
                "in_class": False,
                "column": 0,
                "docs": [],
                "start_byte": 0,
                "end_byte": 11,
                "line_start": 1,
                "line_end": 2,
                "metadata": None,
                "level": 1,
                "level_style": "=",
                "source_end": 11,
            },
            "one_level_statement",
            id="a-level-both-stated-and-styled",
        ),
    ],
)
def test_a_capture_whose_fields_disagree_is_refused(row: dict, rule: str) -> None:
    with pytest.raises(dy.exc.ValidationError, match=rule):
        CaptureRow.validate(pl.DataFrame([row]), cast=True)


@pytest.mark.parametrize(
    "row",
    [
        pytest.param(
            {
                "file_path": "a",
                "scope": "",
                "name": "f",
                "def_id": "d",
                "def_path": None,
                "module_path": None,
            },
            id="an-id-without-its-path",
        ),
        pytest.param(
            {
                "file_path": "a",
                "scope": "",
                "name": "f",
                "def_id": "d",
                "def_path": "b",
                "module_path": "c",
            },
            id="a-definition-and-a-module",
        ),
        pytest.param(
            {
                "file_path": "a",
                "scope": "",
                "name": "f",
                "def_id": None,
                "def_path": None,
                "module_path": None,
            },
            id="nothing",
        ),
    ],
)
def test_a_name_reaches_exactly_one_definition_or_module(row: dict) -> None:
    with pytest.raises(dy.exc.ValidationError, match="reaches_one_definition_or_module"):
        ExportRow.validate(pl.DataFrame([row]), cast=True)


def test_a_base_is_only_a_name_path() -> None:
    row = {
        "source_id": "c",
        "source_path": "a",
        "language": "python",
        "kind": "inherits",
        "lookup": "member",
        "path": ["Base"],
        "caller_chain": "",
        "position": 0,
    }
    with pytest.raises(dy.exc.ValidationError, match="bases_are_unqualified_paths"):
        PathRow.validate(pl.DataFrame([row]), cast=True)
