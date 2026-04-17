"""Tests for edge inference — imports, tests, docs.

Pure helper functions (``_resolve_module_to_file``,
``_strip_test_prefix``, ``_find_source_file``) are tested directly
with inline parametrize.  The three ``infer_*`` functions are
covered by an ``EdgeCase`` scenario family in ``case_edges.py``.
"""

from __future__ import annotations

import pytest
from pytest_cases import fixture, parametrize_with_cases

from rbtr.index.edges import (
    _find_source_file,
    _resolve_module_to_file,
    _strip_test_prefix,
    infer_doc_edges,
    infer_import_edges,
    infer_test_edges,
)
from rbtr.index.models import Chunk, Edge
from tests.index.case_edges import ChunkSpec, EdgeCase, InferFn


# ── _resolve_module_to_file ──────────────────────────────────────────


@pytest.mark.parametrize(
    ("module", "files", "expected"),
    [
        (
            "rbtr/index/models",
            {"rbtr/index/models.py", "rbtr/index/__init__.py"},
            "rbtr/index/models.py",
        ),
        ("rbtr/index", {"rbtr/index/__init__.py"}, "rbtr/index/__init__.py"),
        ("rbtr/index/models", {"rbtr/other.py"}, None),
    ],
    ids=["direct_py", "init_py", "not_found"],
)
def test_resolve_module_to_file(
    module: str, files: set[str], expected: str | None
) -> None:
    assert _resolve_module_to_file(module, files) == expected


# ── _strip_test_prefix ───────────────────────────────────────────────


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("tests/test_foo.py", "foo"),
        ("src/tests/test_bar.py", "bar"),
        ("src/foo.py", None),
        ("test_baz.py", "baz"),
    ],
    ids=["simple", "nested", "no_prefix", "root"],
)
def test_strip_test_prefix(path: str, expected: str | None) -> None:
    assert _strip_test_prefix(path) == expected


# ── _find_source_file ────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("symbol", "test_file", "files", "expected"),
    [
        ("foo", "tests/test_foo.py", {"foo.py", "tests/test_foo.py"}, "foo.py"),
        ("foo", "tests/test_foo.py", {"src/foo.py"}, "src/foo.py"),
        ("foo", "src/tests/test_foo.py", {"src/foo.py"}, "src/foo.py"),
        ("foo_bar", "tests/test_foo_bar.py", {"src/foo/bar.py"}, "src/foo/bar.py"),
        ("foo", "tests/test_foo.py", {"deep/nested/foo.py"}, "deep/nested/foo.py"),
        ("foo", "tests/test_foo.py", {"bar.py"}, None),
    ],
    ids=[
        "direct",
        "in_src",
        "sibling_of_tests",
        "underscore_to_path",
        "suffix_fallback",
        "not_found",
    ],
)
def test_find_source_file(
    symbol: str, test_file: str, files: set[str], expected: str | None
) -> None:
    assert _find_source_file(symbol, test_file, files) == expected


# ── infer_* scenarios ───────────────────────────────────────────────


@fixture
@parametrize_with_cases("scenario", cases="tests.index.case_edges")
def inferred(scenario: EdgeCase) -> tuple[EdgeCase, list[Edge]]:
    chunks = [
        Chunk(
            id=spec.id,
            blob_sha=f"sha_{spec.id}",
            file_path=spec.file_path,
            kind=spec.kind,
            name=spec.name,
            content=spec.content,
            line_start=1,
            line_end=1,
            scope="",
            metadata=dict(spec.metadata),
        )
        for spec in scenario.chunks
    ]
    if scenario.fn is InferFn.IMPORT:
        edges = infer_import_edges(chunks, set(scenario.repo_files))
    elif scenario.fn is InferFn.TEST:
        edges = infer_test_edges(chunks, set(scenario.repo_files))
    else:
        edges = infer_doc_edges(chunks)
    return scenario, edges


def test_inference_produces_expected_edges(
    inferred: tuple[EdgeCase, list[Edge]],
) -> None:
    scenario, edges = inferred
    pairs = frozenset((e.source_id, e.target_id) for e in edges)
    assert pairs == scenario.expected


def test_inference_edge_kind_is_correct(
    inferred: tuple[EdgeCase, list[Edge]],
) -> None:
    from rbtr.index.models import EdgeKind

    scenario, edges = inferred
    if not edges:
        return
    expected_kind = {
        InferFn.IMPORT: EdgeKind.IMPORTS,
        InferFn.TEST: EdgeKind.TESTS,
        InferFn.DOC: EdgeKind.DOCUMENTS,
    }[scenario.fn]
    for e in edges:
        assert e.kind == expected_kind
