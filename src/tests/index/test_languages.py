"""Tests for language detection and grammar loading."""

import pytest

from rbtr.index.languages import detect_language, get_language, load_grammar, missing_grammar
from rbtr.plugins.manager import get_manager

# ── Detection ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("src/foo/bar.py", "python"),
        ("stubs/foo.pyi", "python"),
        ("package.json", "json"),
        ("ci.yml", "yaml"),
        ("config.yaml", "yaml"),
        ("pyproject.toml", "toml"),
        ("deploy.sh", "bash"),
        ("Makefile", "bash"),
        ("Dockerfile", "bash"),
        ("app.ts", "typescript"),
        ("component.tsx", "typescript"),
        ("index.js", "javascript"),
        ("App.jsx", "javascript"),
        ("main.go", "go"),
        ("lib.rs", "rust"),
    ],
    ids=[
        "py",
        "pyi",
        "json",
        "yml",
        "yaml",
        "toml",
        "sh",
        "makefile",
        "dockerfile",
        "ts",
        "tsx",
        "js",
        "jsx",
        "go",
        "rust",
    ],
)
def test_detect_language(path: str, expected: str) -> None:
    assert detect_language(path) == expected


@pytest.mark.parametrize(
    "path",
    ["README", "data.xyz"],
    ids=["no-extension", "unknown-extension"],
)
def test_detect_unknown(path: str) -> None:
    assert detect_language(path) is None


# ── Grammar loading ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    "lang_id",
    ["python", "json", "yaml", "toml", "bash"],
)
def test_load_grammar(lang_id: str) -> None:
    assert load_grammar(lang_id) is not None


def test_missing_grammar_for_installed() -> None:
    assert not missing_grammar("python")


def test_get_language_python() -> None:
    result = get_language("foo.py")
    assert result is not None
    lang, grammar = result
    assert lang == "python"
    assert grammar is not None


def test_get_language_unknown_file() -> None:
    assert get_language("data.xyz") is None


# ── Plugin system ────────────────────────────────────────────────────


def test_manager_has_all_builtin_languages() -> None:
    manager = get_manager()
    all_ids = manager.all_language_ids()
    for expected in (
        "python",
        "javascript",
        "typescript",
        "go",
        "rust",
        "java",
        "bash",
        "c",
        "cpp",
        "ruby",
        "json",
        "yaml",
        "toml",
    ):
        assert expected in all_ids, f"missing registration for {expected}"


@pytest.mark.parametrize(
    "lang_id",
    ["python", "javascript", "typescript", "go", "rust", "java", "bash", "c", "cpp", "ruby"],
)
def test_registration_has_query(lang_id: str) -> None:
    reg = get_manager().get_registration(lang_id)
    assert reg is not None
    assert reg.query is not None, f"missing query for {lang_id}"


@pytest.mark.parametrize(
    "lang_id",
    ["python", "javascript", "typescript", "go", "rust", "java", "c", "cpp", "ruby"],
)
def test_registration_has_import_extractor(lang_id: str) -> None:
    reg = get_manager().get_registration(lang_id)
    assert reg is not None
    assert reg.import_extractor is not None, f"missing extractor for {lang_id}"
