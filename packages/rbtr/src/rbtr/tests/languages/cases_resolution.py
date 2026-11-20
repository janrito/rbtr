"""One import at a source path, the rules of the languages involved, and the files it reaches.

Each case states one rule of how `resolved_files` turns an import into
repository files: where a module is looked for under source roots,
substitutions, extensions and index files, which candidate wins, which
language's rules apply, how a written `./` or `../` prefix anchors the
module, and when a file reaches itself.
"""

from __future__ import annotations

from dataclasses import dataclass

from pytest_cases import parametrize

from rbtr.domain.models import ModuleStyle
from rbtr.languages.registration import LanguageRegistration


@dataclass(frozen=True)
class ImportScenario:
    """An import written in `language` at `source_path`, and the files it resolves to.

    A source path without an extension prefers no candidate's extension.
    """

    source_path: str
    language: str
    module: str
    registrations: list[LanguageRegistration]
    files: frozenset[str]
    expected: list[str]
    names: str = ""
    language_hint: str = ""


@parametrize(
    "module,files,registrations,expected",
    [
        (
            "src/models",
            frozenset({"src/models.js", "src/models.ts"}),
            [
                LanguageRegistration(
                    id="lang",
                    extensions=frozenset({".ts", ".tsx"}),
                    index_files=frozenset({"index.ts"}),
                    import_targets=frozenset({"lang", "js"}),
                ),
                LanguageRegistration(id="js", extensions=frozenset({".js"})),
            ],
            ["src/models.ts"],
        ),
        (
            "src/models",
            frozenset({"src/models/index.ts"}),
            [
                LanguageRegistration(
                    id="lang", extensions=frozenset({".ts"}), index_files=frozenset({"index.ts"})
                )
            ],
            ["src/models/index.ts"],
        ),
        (
            "crate/models",
            frozenset({"src/models.rs"}),
            [
                LanguageRegistration(
                    id="lang",
                    extensions=frozenset({".rs"}),
                    index_files=frozenset({"mod.rs"}),
                    path_substitutions=(("crate/", "src/"),),
                )
            ],
            ["src/models.rs"],
        ),
        (
            "com/example/Foo",
            frozenset({"src/main/java/com/example/Foo.java"}),
            [
                LanguageRegistration(
                    id="lang", extensions=frozenset({".java"}), source_roots=("", "src/main/java")
                )
            ],
            ["src/main/java/com/example/Foo.java"],
        ),
        (
            "utils",
            frozenset({"include/utils.h"}),
            [
                LanguageRegistration(
                    id="lang",
                    extensions=frozenset({".c", ".h"}),
                    source_roots=("", "include", "src"),
                )
            ],
            ["include/utils.h"],
        ),
        (
            "src/models",
            frozenset({"other/file.ts"}),
            [LanguageRegistration(id="lang", extensions=frozenset({".ts"}))],
            [],
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
def case_a_module_is_looked_for_under_roots_substitutions_and_index_files(
    module: str,
    files: frozenset[str],
    registrations: list[LanguageRegistration],
    expected: list[str],
) -> ImportScenario:
    """A module is a file under a source root, with a substitution applied, an extension added or a directory's index file."""
    return ImportScenario(
        source_path="importer",
        language=registrations[0].id,
        module=module,
        registrations=registrations,
        files=files,
        expected=expected,
    )


@parametrize(
    "module,files,registrations,expected",
    [
        (
            "app.js",
            frozenset({"src/app.js"}),
            [LanguageRegistration(id="lang", extensions=frozenset({".js"}), source_roots=("src",))],
            ["src/app.js"],
        ),
        (
            "models",
            frozenset({"models.js"}),
            [LanguageRegistration(id="lang", extensions=frozenset({".js"}))],
            ["models.js"],
        ),
        (
            "components",
            frozenset({"components/index.js"}),
            [
                LanguageRegistration(
                    id="lang", extensions=frozenset({".js"}), index_files=frozenset({"index.js"})
                )
            ],
            ["components/index.js"],
        ),
        (
            "nonexistent.js",
            frozenset({"other.js"}),
            [LanguageRegistration(id="lang", extensions=frozenset({".js"}))],
            [],
        ),
        (
            "components/Button",
            frozenset({"src/ui/components/Button.js"}),
            [LanguageRegistration(id="lang", extensions=frozenset({".js"}))],
            ["src/ui/components/Button.js"],
        ),
        (
            "kg/sub/b",
            frozenset({"pkg/sub/b.js"}),
            [LanguageRegistration(id="lang", extensions=frozenset({".js"}))],
            [],
        ),
    ],
    ids=[
        "bare-filename",
        "extensionless",
        "index-file",
        "not-found",
        "nested-suffix",
        "suffix-within-a-segment",
    ],
)
def case_a_path_style_module_names_a_file_or_a_suffix_of_one(
    module: str,
    files: frozenset[str],
    registrations: list[LanguageRegistration],
    expected: list[str],
) -> ImportScenario:
    """A path-style module names a file as written, with an extension added, as an index file, or as a whole-segment path suffix."""
    return ImportScenario(
        source_path="importer",
        language=registrations[0].id,
        module=module,
        registrations=registrations,
        files=files,
        expected=expected,
    )


@parametrize(
    "module,files,registrations,expected",
    [
        (
            "os.path",
            frozenset({"os/path.py"}),
            [
                LanguageRegistration(
                    id="lang",
                    extensions=frozenset({".py"}),
                    index_files=frozenset({"__init__.py"}),
                    module_style=ModuleStyle.DOTTED,
                )
            ],
            ["os/path.py"],
        ),
        (
            "pathlib",
            frozenset({"pathlib.py"}),
            [
                LanguageRegistration(
                    id="lang",
                    extensions=frozenset({".py"}),
                    index_files=frozenset({"__init__.py"}),
                    module_style=ModuleStyle.DOTTED,
                )
            ],
            ["pathlib.py"],
        ),
        (
            "rbtr.index",
            frozenset({"rbtr/index/__init__.py"}),
            [
                LanguageRegistration(
                    id="lang",
                    extensions=frozenset({".py"}),
                    index_files=frozenset({"__init__.py"}),
                    module_style=ModuleStyle.DOTTED,
                )
            ],
            ["rbtr/index/__init__.py"],
        ),
        (
            "nonexistent.module",
            frozenset({"other.py"}),
            [
                LanguageRegistration(
                    id="lang",
                    extensions=frozenset({".py"}),
                    index_files=frozenset({"__init__.py"}),
                    module_style=ModuleStyle.DOTTED,
                )
            ],
            [],
        ),
        (
            "rbtr.index.store",
            frozenset({"packages/rbtr/src/rbtr/index/store.py"}),
            [
                LanguageRegistration(
                    id="lang",
                    extensions=frozenset({".py"}),
                    index_files=frozenset({"__init__.py"}),
                    source_roots=("", "src"),
                    module_style=ModuleStyle.DOTTED,
                )
            ],
            ["packages/rbtr/src/rbtr/index/store.py"],
        ),
        (
            "rbtr.utils.helpers",
            frozenset({"other/helpers.py", "packages/rbtr/src/rbtr/utils/helpers.py"}),
            [
                LanguageRegistration(
                    id="lang",
                    extensions=frozenset({".py"}),
                    index_files=frozenset({"__init__.py"}),
                    source_roots=("", "src"),
                    module_style=ModuleStyle.DOTTED,
                )
            ],
            ["packages/rbtr/src/rbtr/utils/helpers.py"],
        ),
        (
            "common.io",
            frozenset({"packages/a/src/common/io.py", "packages/b/src/common/io.py"}),
            [
                LanguageRegistration(
                    id="lang",
                    extensions=frozenset({".py"}),
                    index_files=frozenset({"__init__.py"}),
                    source_roots=("", "src"),
                    module_style=ModuleStyle.DOTTED,
                )
            ],
            [],
        ),
        (
            "os",
            frozenset({"a/b/os.py"}),
            [
                LanguageRegistration(
                    id="lang",
                    extensions=frozenset({".py"}),
                    index_files=frozenset({"__init__.py"}),
                    module_style=ModuleStyle.DOTTED,
                )
            ],
            [],
        ),
        (
            "pydantic.main",
            frozenset({"src/app.py"}),
            [
                LanguageRegistration(
                    id="lang",
                    extensions=frozenset({".py"}),
                    index_files=frozenset({"__init__.py"}),
                    source_roots=("", "src"),
                    module_style=ModuleStyle.DOTTED,
                )
            ],
            [],
        ),
        (
            "app.models",
            frozenset({"app/models.py", "deep/nested/app/models.py"}),
            [
                LanguageRegistration(
                    id="lang",
                    extensions=frozenset({".py"}),
                    index_files=frozenset({"__init__.py"}),
                    module_style=ModuleStyle.DOTTED,
                )
            ],
            ["app/models.py"],
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
def case_a_dotted_module_names_its_slashed_path(
    module: str,
    files: frozenset[str],
    registrations: list[LanguageRegistration],
    expected: list[str],
) -> ImportScenario:
    """A dotted module is looked for with its dots made slashes; a suffix in two places, or a one-segment suffix, names nothing."""
    return ImportScenario(
        source_path="importer",
        language=registrations[0].id,
        module=module,
        registrations=registrations,
        files=files,
        expected=expected,
    )


@parametrize(
    "source_path,module,files,registrations,expected",
    [
        (
            "app.py",
            "pkg.models",
            frozenset({"a/pkg/models.py", "a/pkg/models.pyi"}),
            [
                LanguageRegistration(
                    id="lang",
                    extensions=frozenset({".pyi", ".py"}),
                    module_style=ModuleStyle.DOTTED,
                )
            ],
            ["a/pkg/models.py"],
        ),
        (
            "app.ts",
            "ui/Button",
            frozenset({"src/ui/Button.js", "src/ui/Button.tsx"}),
            [
                LanguageRegistration(
                    id="lang",
                    extensions=frozenset({".ts", ".tsx"}),
                    import_targets=frozenset({"lang", "js"}),
                ),
                LanguageRegistration(id="js", extensions=frozenset({".js"})),
            ],
            ["src/ui/Button.tsx"],
        ),
        (
            "app.js",
            "ui/Button",
            frozenset({"src/ui/Button.css", "src/ui/Button.js"}),
            [
                LanguageRegistration(
                    id="lang",
                    extensions=frozenset({".js", ".jsx"}),
                    import_targets=frozenset({"lang", "css"}),
                ),
                LanguageRegistration(id="css", extensions=frozenset({".css"})),
            ],
            ["src/ui/Button.js"],
        ),
        (
            "app.js",
            "ui/Button",
            frozenset({"ui/Button.js", "ui/Button.jsx"}),
            [
                LanguageRegistration(
                    id="lang",
                    extensions=frozenset({".js", ".jsx"}),
                    import_targets=frozenset({"jsx", "plain_js"}),
                ),
                LanguageRegistration(id="jsx", extensions=frozenset({".jsx"})),
                LanguageRegistration(id="plain_js", extensions=frozenset({".js"})),
            ],
            ["ui/Button.js"],
        ),
        (
            "app.js",
            "ui/Button",
            frozenset({"ui/Button.css", "ui/Button.jsx"}),
            [
                LanguageRegistration(
                    id="lang",
                    extensions=frozenset({".js", ".jsx"}),
                    import_targets=frozenset({"css", "jsx"}),
                ),
                LanguageRegistration(id="css", extensions=frozenset({".css"})),
                LanguageRegistration(id="jsx", extensions=frozenset({".jsx"})),
            ],
            ["ui/Button.jsx"],
        ),
        (
            "app.py",
            "pkg.models",
            frozenset({"a/pkg/models.py", "b/pkg/models.py"}),
            [
                LanguageRegistration(
                    id="lang", extensions=frozenset({".py"}), module_style=ModuleStyle.DOTTED
                )
            ],
            [],
        ),
    ],
    ids=[
        "py-over-pyi",
        "tsx-over-js-same-language",
        "js-over-css",
        "root-prefers-importer-extension",
        "root-prefers-importer-language",
        "same-ext-two-paths-dropped",
    ],
)
def case_a_candidate_with_the_importers_extension_wins(
    source_path: str,
    module: str,
    files: frozenset[str],
    registrations: list[LanguageRegistration],
    expected: list[str],
) -> ImportScenario:
    """Among candidates differing by extension, the importer's own wins, then its language's; two paths with one extension name nothing."""
    return ImportScenario(
        source_path=source_path,
        language=registrations[0].id,
        module=module,
        registrations=registrations,
        files=files,
        expected=expected,
    )


@parametrize(
    "module,hint,expected",
    [("app", "javascript", "src/app.js"), ("styles", "css", "assets/styles.css")],
    ids=["hint-js", "hint-css"],
)
def case_a_hint_resolves_by_the_hinted_languages_rules(
    module: str, hint: str, expected: str
) -> ImportScenario:
    """An HTML `<script src>` or `<link href>` resolves by JavaScript's or CSS's rules, not HTML's."""
    return ImportScenario(
        source_path="page.html",
        language="html",
        module=module,
        language_hint=hint,
        registrations=[
            LanguageRegistration(id="html", extensions=frozenset({".html"})),
            LanguageRegistration(
                id="javascript", extensions=frozenset({".js"}), source_roots=("src",)
            ),
            LanguageRegistration(
                id="css", extensions=frozenset({".css"}), source_roots=("assets",)
            ),
        ],
        files=frozenset({"src/app.js", "src/app.ts", "assets/styles.css"}),
        expected=[expected],
    )


@parametrize(
    "module,expected",
    [
        (".models", ["pkg/sub/models.py"]),
        ("..models", ["pkg/models.py"]),
        ("....models", []),
        (".", ["pkg/sub/__init__.py"]),
    ],
    ids=["sibling", "parent", "past-the-root", "package-itself"],
)
def case_a_prefix_climbs_from_the_importer_of_a_dotted_language(
    module: str, expected: list[str]
) -> ImportScenario:
    """A dotted module's leading dots climb, one per level from the file's directory; past the root it reaches nothing.

    The root `models.py` is what a climb stopping at the root would reach.
    """
    return ImportScenario(
        source_path="pkg/sub/app.py",
        language="lang",
        module=module,
        registrations=[
            LanguageRegistration(
                id="lang",
                extensions=frozenset({".py"}),
                index_files=frozenset({"__init__.py"}),
                module_style=ModuleStyle.DOTTED,
            ),
        ],
        files=frozenset(
            {
                "models.py",
                "pkg/models.py",
                "pkg/sub/models.py",
                "pkg/sub/__init__.py",
                "pkg/sub/app.py",
            }
        ),
        expected=expected,
    )


@parametrize(
    "module,expected",
    [
        ("./models", ["a/b/c/models.js"]),
        ("../utils", ["a/b/utils.js"]),
        ("../../shared/helpers", ["a/shared/helpers.js"]),
        ("../../../root", ["root.js"]),
        ("./x/y/z", ["a/b/c/x/y/z.js"]),
        ("./styles.css", ["a/b/c/styles.css"]),
        (".", ["a/b/c/index.js"]),
        ("..", ["a/b/index.js"]),
        ("react", []),
        ("@testing-library/react", []),
    ],
    ids=[
        "current-dir",
        "parent-dir",
        "grandparent-dir",
        "triple-parent",
        "current-dir-nested",
        "current-with-extension",
        "current-dir-alone",
        "parent-dir-alone",
        "absolute-package",
        "scoped-package",
    ],
)
def case_a_written_prefix_climbs_from_the_importer(
    module: str, expected: list[str]
) -> ImportScenario:
    """`./` or `.` is the importer's directory and each `../` or `..` one more up, for a path-style language.

    Root `models.js` and `utils.js` are what an unanchored lookup would reach.
    """
    return ImportScenario(
        source_path="a/b/c/app.js",
        language="lang",
        module=module,
        registrations=[
            LanguageRegistration(
                id="lang",
                extensions=frozenset({".js"}),
                index_files=frozenset({"index.js"}),
                import_targets=frozenset({"lang", "css"}),
            ),
            LanguageRegistration(id="css", extensions=frozenset({".css"})),
        ],
        files=frozenset(
            {
                "a/b/c/app.js",
                "a/b/c/index.js",
                "a/b/index.js",
                "a/b/c/models.js",
                "models.js",
                "a/b/utils.js",
                "utils.js",
                "a/shared/helpers.js",
                "root.js",
                "a/b/c/x/y/z.js",
                "a/b/c/styles.css",
            }
        ),
        expected=expected,
    )


@parametrize(
    "module,expected",
    [
        ("crate::models", ["src/models.rs"]),
        ("crate::shop::order", ["src/shop/order.rs"]),
    ],
    ids=["crate-rewritten", "nested"],
)
def case_a_scoped_module_names_its_slashed_path(module: str, expected: list[str]) -> ImportScenario:
    """A `::` module's separators are slashes once its substitution is applied (`crate::` is `src/`)."""
    return ImportScenario(
        source_path="src/shop/app.rs",
        language="lang",
        module=module,
        registrations=[
            LanguageRegistration(
                id="lang",
                extensions=frozenset({".rs"}),
                index_files=frozenset({"mod.rs"}),
                path_substitutions=(("crate::", "src/"),),
                module_style=ModuleStyle.SCOPED,
            ),
        ],
        files=frozenset({"src/models.rs", "src/shop/order.rs", "src/util.rs", "src/shop/app.rs"}),
        expected=expected,
    )


@parametrize(
    "source_path,module,expected",
    [
        ("src/shop/mod.rs", "super::util", ["src/util.rs"]),
        ("src/shop/order.rs", "super::util", ["src/shop/util.rs"]),
        ("src/shop/mod.rs", "super::super::models", []),
        ("src/shop/mod.rs", "superb::util", []),
        ("src/shop/mod.rs", "self::util", ["src/shop/util.rs"]),
        ("src/main.rs", "self::util", ["src/util.rs"]),
        ("src/shop/order.rs", "self::util", ["src/shop/order/util.rs"]),
        ("src/shop/order.rs", "selfish::util", []),
    ],
    ids=[
        "parent-from-a-directory-module",
        "parent-from-a-file-module",
        "past-the-root",
        "a-longer-parent-word",
        "current-from-a-directory-module",
        "current-from-a-root-module",
        "current-from-a-file-module",
        "a-longer-current-word",
    ],
)
def case_a_module_word_reads_from_the_importers_module(
    source_path: str, module: str, expected: list[str]
) -> ImportScenario:
    """A leading current word names the importer's own module; each leading parent word the parent of the module before it.

    A directory module file (`mod.rs`, `main.rs`) is its directory's
    module; any other file is a module of its own inside its directory, so
    its child modules sit in the directory named by its stem, as Rust's
    `src/shop/order.rs` is `crate::shop::order` and its `self::util` is
    `src/shop/order/util.rs`.
    """
    return ImportScenario(
        source_path=source_path,
        language="lang",
        module=module,
        registrations=[
            LanguageRegistration(
                id="lang",
                extensions=frozenset({".rs"}),
                index_files=frozenset({"mod.rs"}),
                directory_module_files=frozenset({"mod.rs", "main.rs"}),
                module_style=ModuleStyle.SCOPED,
                parent_module="super",
                current_module="self",
            ),
        ],
        files=frozenset(
            {
                "src/main.rs",
                "src/util.rs",
                "src/shop/util.rs",
                "src/shop/order.rs",
                "src/shop/order/util.rs",
                "src/shop/mod.rs",
            }
        ),
        expected=expected,
    )


@parametrize(
    "names,expected",
    [("usage", ["guide.md"]), ("", [])],
    ids=["naming-something", "naming-nothing"],
)
def case_a_file_reaches_itself_only_to_name_something(
    names: str, expected: list[str]
) -> ImportScenario:
    """An import of its own file reaches it when it names something in it, never whole."""
    return ImportScenario(
        source_path="guide.md",
        language="lang",
        module="guide.md",
        names=names,
        registrations=[
            LanguageRegistration(id="lang", extensions=frozenset({".md"})),
        ],
        files=frozenset({"guide.md"}),
        expected=expected,
    )
