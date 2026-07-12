"""Tests for cross-file edge inference.

Private helpers that are difficult to reach through the public
`infer_*` functions (e.g. `_resolve_module_to_file`) are tested
directly because the project AGENTS permit it when doing so avoids
mocking the full pipeline.
"""

from __future__ import annotations

import pytest
from pytest_cases import fixture, parametrize_with_cases

from rbtr.index.edges import (
    ImportResolution,
    _resolve_module_to_file,
    infer_import_edges,
)
from rbtr.index.models import Chunk, Edge, EdgeKind
from rbtr.languages.registration import ModuleStyle

from .cases_edges import EdgeScenario

# ── _resolve_module_to_file ──────────────────────────────────────────


@pytest.mark.parametrize(
    ("module", "files", "resolution", "expected"),
    [
        (
            "src/models",
            {"src/models.ts", "src/models.js"},
            ImportResolution(
                extensions=(".ts", ".tsx", ".js"),
                index_files=("index.ts",),
                source_roots=("",),
                path_substitutions=(),
            ),
            "src/models.ts",
        ),
        (
            "src/models",
            {"src/models/index.ts"},
            ImportResolution(
                extensions=(".ts",),
                index_files=("index.ts",),
                source_roots=("",),
                path_substitutions=(),
            ),
            "src/models/index.ts",
        ),
        (
            "crate/models",
            {"src/models.rs"},
            ImportResolution(
                extensions=(".rs",),
                index_files=("mod.rs",),
                source_roots=("",),
                path_substitutions=(("crate/", "src/"),),
            ),
            "src/models.rs",
        ),
        (
            "com/example/Foo",
            {"src/main/java/com/example/Foo.java"},
            ImportResolution(
                extensions=(".java",),
                index_files=(),
                source_roots=("", "src/main/java"),
                path_substitutions=(),
            ),
            "src/main/java/com/example/Foo.java",
        ),
        (
            "utils",
            {"include/utils.h"},
            ImportResolution(
                extensions=(".h", ".c"),
                index_files=(),
                source_roots=("", "include", "src"),
                path_substitutions=(),
            ),
            "include/utils.h",
        ),
        (
            "src/models",
            {"other/file.ts"},
            ImportResolution(
                extensions=(".ts",),
                index_files=(),
                source_roots=("",),
                path_substitutions=(),
            ),
            None,
        ),
    ],
    ids=[
        "ts-direct",
        "ts-index",
        "rust-crate-sub",
        "java-source-root",
        "c-include-root",
        "not-found",
    ],
)
def test_resolve_module_to_file_with_resolution(
    module: str,
    files: set[str],
    resolution: ImportResolution,
    expected: str | None,
) -> None:
    assert _resolve_module_to_file(module, files, resolution) == expected


# ── infer_* scenarios ────────────────────────────────────────────────


@fixture
@parametrize_with_cases("scenario", cases=".cases_edges")
def inferred(scenario: EdgeScenario) -> tuple[EdgeScenario, list[Edge]]:
    chunks = [
        Chunk(
            id=spec.id,
            blob_sha=f"sha_{spec.id}",
            file_path=spec.file_path,
            kind=spec.kind,
            name=spec.name,
            language=spec.language,
            content=spec.content,
            line_start=1,
            line_end=1,
            scope="",
            metadata=spec.metadata,
        )
        for spec in scenario.chunks
    ]
    edges = infer_import_edges(chunks, set(scenario.repo_files), scenario.resolution_map)
    return scenario, edges


def test_inference_produces_expected_edges(
    inferred: tuple[EdgeScenario, list[Edge]],
) -> None:
    scenario, edges = inferred
    pairs = frozenset((e.source_id, e.target_id) for e in edges)
    assert pairs == scenario.expected


def test_inference_edge_kind_is_correct(
    inferred: tuple[EdgeScenario, list[Edge]],
) -> None:
    scenario, edges = inferred
    if not edges:
        return
    expected_kind = scenario.expected_edge_kind or EdgeKind.IMPORTS
    for e in edges:
        assert e.kind == expected_kind


# ── language_hint resolution ─────────────────────────────────────────


@pytest.mark.parametrize(
    ("module", "files", "hint", "expected"),
    [
        ("app", {"src/app.js", "src/app.ts"}, "javascript", "src/app.js"),
        ("styles", {"assets/styles.css"}, "css", "assets/styles.css"),
    ],
    ids=["hint-js", "hint-css"],
)
def test_language_hint_directs_resolution(
    module: str,
    files: set[str],
    hint: str,
    expected: str | None,
) -> None:
    """language_hint picks the correct ImportResolution."""
    resolution_map = {
        "javascript": ImportResolution(
            extensions=(".js",),
            index_files=(),
            source_roots=("src",),
            path_substitutions=(),
        ),
        "css": ImportResolution(
            extensions=(".css",),
            index_files=(),
            source_roots=("assets",),
            path_substitutions=(),
        ),
    }
    resolution = resolution_map.get(hint)
    assert resolution is not None
    result = _resolve_module_to_file(module, files, resolution)
    assert result == expected


# ── PATH-style resolution ────────────────────────────────────────────


@pytest.mark.parametrize(
    ("module", "files", "resolution", "expected"),
    [
        (
            "app.js",
            {"src/app.js"},
            ImportResolution(
                extensions=(".js",),
                index_files=(),
                source_roots=("src",),
                path_substitutions=(),
            ),
            "src/app.js",
        ),
        (
            "models",
            {"models.js"},
            ImportResolution(
                extensions=(".js",),
                index_files=(),
                source_roots=("",),
                path_substitutions=(),
            ),
            "models.js",
        ),
        (
            "components",
            {"components/index.js"},
            ImportResolution(
                extensions=(".js",),
                index_files=("index.js",),
                source_roots=("",),
                path_substitutions=(),
            ),
            "components/index.js",
        ),
        (
            "nonexistent.js",
            {"other.js"},
            ImportResolution(
                extensions=(".js",),
                index_files=(),
                source_roots=("",),
                path_substitutions=(),
            ),
            None,
        ),
        (
            "components/Button",
            {"src/ui/components/Button.js"},
            ImportResolution(
                extensions=(".js",),
                index_files=(),
                source_roots=("",),
                path_substitutions=(),
            ),
            "src/ui/components/Button.js",
        ),
    ],
    ids=["bare-filename", "extensionless", "index-file", "not-found", "nested-suffix"],
)
def test_resolve_path_style(
    module: str,
    files: set[str],
    resolution: ImportResolution,
    expected: str | None,
) -> None:
    """PATH-style resolution: bare filenames, extensions, index files."""
    assert _resolve_module_to_file(module, files, resolution) == expected


# ── DOTTED-style resolution ─────────────────────────────────────────


@pytest.mark.parametrize(
    ("module", "files", "resolution", "expected"),
    [
        (
            "os.path",
            {"os/path.py"},
            ImportResolution(
                extensions=(".py",),
                index_files=("__init__.py",),
                source_roots=("",),
                path_substitutions=(),
                module_style=ModuleStyle.DOTTED,
            ),
            "os/path.py",
        ),
        (
            "pathlib",
            {"pathlib.py"},
            ImportResolution(
                extensions=(".py",),
                index_files=("__init__.py",),
                source_roots=("",),
                path_substitutions=(),
                module_style=ModuleStyle.DOTTED,
            ),
            "pathlib.py",
        ),
        (
            "rbtr.index",
            {"rbtr/index/__init__.py"},
            ImportResolution(
                extensions=(".py",),
                index_files=("__init__.py",),
                source_roots=("",),
                path_substitutions=(),
                module_style=ModuleStyle.DOTTED,
            ),
            "rbtr/index/__init__.py",
        ),
        (
            "nonexistent.module",
            {"other.py"},
            ImportResolution(
                extensions=(".py",),
                index_files=("__init__.py",),
                source_roots=("",),
                path_substitutions=(),
                module_style=ModuleStyle.DOTTED,
            ),
            None,
        ),
        (
            "rbtr.index.store",
            {"packages/rbtr/src/rbtr/index/store.py"},
            ImportResolution(
                extensions=(".py",),
                index_files=("__init__.py",),
                source_roots=("", "src"),
                path_substitutions=(),
                module_style=ModuleStyle.DOTTED,
            ),
            "packages/rbtr/src/rbtr/index/store.py",
        ),
        (
            "rbtr.utils.helpers",
            {"packages/rbtr/src/rbtr/utils/helpers.py", "other/helpers.py"},
            ImportResolution(
                extensions=(".py",),
                index_files=("__init__.py",),
                source_roots=("", "src"),
                path_substitutions=(),
                module_style=ModuleStyle.DOTTED,
            ),
            "packages/rbtr/src/rbtr/utils/helpers.py",
        ),
        (
            "common.io",
            {"packages/a/src/common/io.py", "packages/b/src/common/io.py"},
            ImportResolution(
                extensions=(".py",),
                index_files=("__init__.py",),
                source_roots=("", "src"),
                path_substitutions=(),
                module_style=ModuleStyle.DOTTED,
            ),
            None,
        ),
        (
            "os",
            {"a/b/os.py"},
            ImportResolution(
                extensions=(".py",),
                index_files=("__init__.py",),
                source_roots=("",),
                path_substitutions=(),
                module_style=ModuleStyle.DOTTED,
            ),
            None,
        ),
        (
            "pydantic.main",
            {"src/app.py"},
            ImportResolution(
                extensions=(".py",),
                index_files=("__init__.py",),
                source_roots=("", "src"),
                path_substitutions=(),
                module_style=ModuleStyle.DOTTED,
            ),
            None,
        ),
        (
            "app.models",
            {"app/models.py", "deep/nested/app/models.py"},
            ImportResolution(
                extensions=(".py",),
                index_files=("__init__.py",),
                source_roots=("",),
                path_substitutions=(),
                module_style=ModuleStyle.DOTTED,
            ),
            "app/models.py",
        ),
    ],
    ids=[
        "multi-segment",
        "single-segment",
        "init-file",
        "not-found",
        "monorepo-suffix",
        "helpers-non-collision",
        "true-collision-dropped",
        "single-segment-guard-nested",
        "multi-seg-external-none",
        "tier2-prefix-wins",
    ],
)
def test_resolve_dotted_style(
    module: str,
    files: set[str],
    resolution: ImportResolution,
    expected: str | None,
) -> None:
    """DOTTED-style resolution: dot-to-slash conversion."""
    assert _resolve_module_to_file(module, files, resolution) == expected


# ── suffix-tier tie-break by importer (Question 5) ────────────────────


@pytest.mark.parametrize(
    ("module", "files", "resolution", "importer_ext", "expected"),
    [
        (
            "pkg.models",
            {"a/pkg/models.py", "a/pkg/models.pyi"},
            ImportResolution(
                extensions=(".py", ".pyi"),
                index_files=(),
                source_roots=("",),
                path_substitutions=(),
                module_style=ModuleStyle.DOTTED,
                own_extensions=frozenset({".py", ".pyi"}),
            ),
            ".py",
            "a/pkg/models.py",
        ),
        (
            "ui/Button",
            {"src/ui/Button.tsx", "src/ui/Button.js"},
            ImportResolution(
                extensions=(".ts", ".tsx", ".js"),
                index_files=(),
                source_roots=("",),
                path_substitutions=(),
                own_extensions=frozenset({".ts", ".tsx"}),
            ),
            ".ts",
            "src/ui/Button.tsx",
        ),
        (
            "ui/Button",
            {"src/ui/Button.js", "src/ui/Button.css"},
            ImportResolution(
                extensions=(".js", ".css"),
                index_files=(),
                source_roots=("",),
                path_substitutions=(),
                own_extensions=frozenset({".js", ".jsx"}),
            ),
            ".js",
            "src/ui/Button.js",
        ),
        (
            "pkg.models",
            {"a/pkg/models.py", "b/pkg/models.py"},
            ImportResolution(
                extensions=(".py",),
                index_files=(),
                source_roots=("",),
                path_substitutions=(),
                module_style=ModuleStyle.DOTTED,
                own_extensions=frozenset({".py"}),
            ),
            ".py",
            None,
        ),
    ],
    ids=[
        "py-over-pyi",
        "tsx-over-js-same-language",
        "js-over-css",
        "same-ext-two-paths-dropped",
    ],
)
def test_resolve_suffix_tie_break(
    module: str,
    files: set[str],
    resolution: ImportResolution,
    importer_ext: str,
    expected: str | None,
) -> None:
    """Suffix-tier same-path collisions resolve toward the importing file."""
    assert _resolve_module_to_file(module, files, resolution, importer_ext) == expected
