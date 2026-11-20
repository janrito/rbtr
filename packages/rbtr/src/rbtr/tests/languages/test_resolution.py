"""Module resolution: which repository files an import's module names."""

from __future__ import annotations

import pytest

from rbtr.domain.models import ImportMeta
from rbtr.languages.registration import ModuleStyle
from rbtr.languages.resolution import ImportResolution, import_files, module_files


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
def test_resolve_by_source_roots_and_substitutions(
    module: str,
    files: set[str],
    resolution: ImportResolution,
    expected: str | None,
) -> None:
    assert module_files(module, files, resolution) == ([] if expected is None else [expected])


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
    """An import's `language_hint` resolves by the hinted language's rules, not its file's."""
    resolution_map = {
        "html": ImportResolution(
            extensions=(".html",),
            index_files=(),
            source_roots=("",),
            path_substitutions=(),
        ),
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
    meta = ImportMeta(module=module, language_hint=hint)
    found = import_files(meta, "page.html", "html", files, resolution_map)
    assert found == ([] if expected is None else [expected])


@pytest.mark.parametrize(
    ("module", "dots", "expected"),
    [
        ("models", "1", "pkg/sub/models.py"),
        ("models", "2", "pkg/models.py"),
        ("models", "4", None),
        ("", "1", "pkg/sub/__init__.py"),
    ],
    ids=["sibling", "parent", "past-the-root", "package-itself"],
)
def test_relative_depth_climbs_from_the_importer(
    module: str, dots: str, expected: str | None
) -> None:
    """An import's depth climbs from its file's directory; a depth past the root reaches nothing."""
    # A root `models.py` that a climb stopping at the root would reach.
    files = {
        "models.py",
        "pkg/models.py",
        "pkg/sub/models.py",
        "pkg/sub/__init__.py",
        "pkg/sub/app.py",
    }
    resolution = ImportResolution(
        extensions=(".py",),
        index_files=("__init__.py",),
        source_roots=("",),
        path_substitutions=(),
        module_style=ModuleStyle.DOTTED,
    )
    meta = ImportMeta(module=module, dots=dots)
    found = import_files(meta, "pkg/sub/app.py", "lang", files, {"lang": resolution})
    assert found == ([] if expected is None else [expected])


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
    assert module_files(module, files, resolution) == ([] if expected is None else [expected])


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
    assert module_files(module, files, resolution) == ([] if expected is None else [expected])


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
    assert module_files(module, files, resolution, importer_ext) == (
        [] if expected is None else [expected]
    )


@pytest.mark.parametrize(
    ("names", "expected"),
    [("usage", ["guide.md"]), ("", [])],
    ids=["naming-something", "naming-nothing"],
)
def test_a_file_reaches_itself_only_to_name_something(names: str, expected: list[str]) -> None:
    """An import of its own file reaches it when it names something in it, never whole."""
    resolution = ImportResolution(
        extensions=(".md",), index_files=(), source_roots=("",), path_substitutions=()
    )
    meta = ImportMeta(module="guide.md", names=names)
    found = import_files(meta, "guide.md", "lang", {"guide.md"}, {"lang": resolution})
    assert found == expected
