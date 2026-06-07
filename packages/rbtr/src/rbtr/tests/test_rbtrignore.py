"""Tests for .rbtrignore parsing — gitignore semantics."""

from __future__ import annotations

from pathlib import Path

from rbtr.rbtrignore import default_ignore, load_ignore, parse_ignore

# ── parse_ignore ─────────────────────────────────────────────────────


def test_parse_excludes_matching_file() -> None:
    spec = parse_ignore("vendor/\n*.gen.go\n")
    assert spec.match_file("vendor/lib.go")
    assert spec.match_file("src/foo.gen.go")


def test_parse_negation_includes() -> None:
    spec = parse_ignore("vendor/\n!vendor/internal/\n")
    assert spec.match_file("vendor/lib.go")
    assert not spec.match_file("vendor/internal/core.go")


def test_parse_unmatched_file_not_excluded() -> None:
    spec = parse_ignore("vendor/\n")
    assert not spec.match_file("src/main.py")


def test_parse_comments_and_blanks_ignored() -> None:
    spec = parse_ignore("# comment\n\nvendor/\n")
    assert spec.match_file("vendor/x")
    assert not spec.match_file("src/x")


def test_parse_empty_string() -> None:
    spec = parse_ignore("")
    assert not spec.match_file("anything.py")


# ── default_ignore ───────────────────────────────────────────────────


def test_default_excludes_rbtr_dir() -> None:
    spec = default_ignore()
    assert spec.match_file(".rbtr/index/data")


def test_default_allows_source_files() -> None:
    spec = default_ignore()
    assert not spec.match_file("src/main.py")


# ── load_ignore ──────────────────────────────────────────────────────


def test_load_from_file(tmp_path: Path) -> None:
    (tmp_path / ".rbtrignore").write_text("data/\n")
    spec = load_ignore(tmp_path)
    assert spec.match_file("data/big.csv")
    assert not spec.match_file("src/main.py")


def test_load_missing_file_returns_default(tmp_path: Path) -> None:
    spec = load_ignore(tmp_path)
    # Should behave like default_ignore
    assert spec.match_file(".rbtr/index/data")
    assert not spec.match_file("src/main.py")
