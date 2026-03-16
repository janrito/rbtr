"""Tests for file tools — read_file, grep, list_files."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pygit2
import pytest

from rbtr.llm.tools.file import grep, list_files, read_file
from rbtr.models import BranchTarget
from rbtr.state import EngineState

from .conftest import FakeCtx

# ── File tools — shared fixture ──────────────────────────────────────
#
# A realistic two-branch git repo with semantically distinct files:
#   - src/api/handler.py   — Python handler, modified between base/head
#   - src/api/routes.py    — Python routes, head-only (new file)
#   - config/settings.toml — non-code config file
#   - docs/README.md       — prose with known headings
#   - assets/logo.png      — binary blob (null bytes)
#
# Base (main): handler v1, settings, README, logo
# Head (feature): handler v2, routes (new), settings, README, logo

_HANDLER_V1 = """\
from api.utils import validate

async def handle_request(request):
    \"\"\"Handle incoming HTTP requests.\"\"\"
    data = await request.json()
    validated = validate(data)
    return Response(status=200, body=validated)

async def health_check():
    \"\"\"Simple health endpoint.\"\"\"
    return Response(status=200, body="ok")
"""

_HANDLER_V2 = """\
from api.utils import validate
from api.auth import require_auth

@require_auth
async def handle_request(request):
    \"\"\"Handle incoming HTTP requests with auth.\"\"\"
    data = await request.json()
    validated = validate(data)
    return Response(status=200, body=validated)

async def health_check():
    \"\"\"Simple health endpoint.\"\"\"
    return Response(status=200, body="ok")

async def shutdown():
    \"\"\"Graceful shutdown handler.\"\"\"
    await cleanup_connections()
"""

_ROUTES = """\
from api.handler import handle_request, health_check, shutdown

ROUTES = [
    ("POST", "/api/handle", handle_request),
    ("GET", "/health", health_check),
    ("POST", "/shutdown", shutdown),
]
"""

_SETTINGS = """\
[server]
host = "0.0.0.0"
port = 8080
workers = 4

[database]
url = "postgresql://localhost/mydb"
pool_size = 10

[logging]
level = "INFO"
format = "json"
"""

_README = """\
# My Project

## Overview

A simple HTTP API server with authentication.

## API Endpoints

The `handle_request` function processes incoming data.
The `health_check` endpoint returns server status.

## Configuration

See `config/settings.toml` for server and database settings.

## Development

Run tests with `pytest`.  See CONTRIBUTING.md for guidelines.
"""

# 20 bytes with nulls — enough to trigger binary detection.
_BINARY_LOGO = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x10"


def _make_file_repo(tmp: str) -> tuple[pygit2.Repository, str, str]:
    """Create a two-branch repo with realistic file content.

    Returns (repo, main_sha, feature_sha).
    """
    repo = pygit2.init_repository(tmp)
    sig = pygit2.Signature("Test", "test@test.com")

    # ── Base commit (main) ───────────────────────────────────────
    blobs_base = {
        "src/api/handler.py": repo.create_blob(_HANDLER_V1.encode()),
        "config/settings.toml": repo.create_blob(_SETTINGS.encode()),
        "docs/README.md": repo.create_blob(_README.encode()),
        "assets/logo.png": repo.create_blob(_BINARY_LOGO),
    }

    # Build nested trees bottom-up.
    api_tb = repo.TreeBuilder()
    api_tb.insert("handler.py", blobs_base["src/api/handler.py"], pygit2.GIT_FILEMODE_BLOB)
    src_tb = repo.TreeBuilder()
    src_tb.insert("api", api_tb.write(), pygit2.GIT_FILEMODE_TREE)

    config_tb = repo.TreeBuilder()
    config_tb.insert("settings.toml", blobs_base["config/settings.toml"], pygit2.GIT_FILEMODE_BLOB)

    docs_tb = repo.TreeBuilder()
    docs_tb.insert("README.md", blobs_base["docs/README.md"], pygit2.GIT_FILEMODE_BLOB)

    assets_tb = repo.TreeBuilder()
    assets_tb.insert("logo.png", blobs_base["assets/logo.png"], pygit2.GIT_FILEMODE_BLOB)

    root_tb = repo.TreeBuilder()
    root_tb.insert("src", src_tb.write(), pygit2.GIT_FILEMODE_TREE)
    root_tb.insert("config", config_tb.write(), pygit2.GIT_FILEMODE_TREE)
    root_tb.insert("docs", docs_tb.write(), pygit2.GIT_FILEMODE_TREE)
    root_tb.insert("assets", assets_tb.write(), pygit2.GIT_FILEMODE_TREE)

    c1 = repo.create_commit("refs/heads/main", sig, sig, "initial", root_tb.write(), [])
    repo.set_head("refs/heads/main")

    # ── Head commit (feature) ────────────────────────────────────
    blobs_head = {
        "src/api/handler.py": repo.create_blob(_HANDLER_V2.encode()),
        "src/api/routes.py": repo.create_blob(_ROUTES.encode()),
        "config/settings.toml": blobs_base["config/settings.toml"],
        "docs/README.md": blobs_base["docs/README.md"],
        "assets/logo.png": blobs_base["assets/logo.png"],
    }

    api_tb2 = repo.TreeBuilder()
    api_tb2.insert("handler.py", blobs_head["src/api/handler.py"], pygit2.GIT_FILEMODE_BLOB)
    api_tb2.insert("routes.py", blobs_head["src/api/routes.py"], pygit2.GIT_FILEMODE_BLOB)
    src_tb2 = repo.TreeBuilder()
    src_tb2.insert("api", api_tb2.write(), pygit2.GIT_FILEMODE_TREE)

    config_tb2 = repo.TreeBuilder()
    config_tb2.insert("settings.toml", blobs_head["config/settings.toml"], pygit2.GIT_FILEMODE_BLOB)

    docs_tb2 = repo.TreeBuilder()
    docs_tb2.insert("README.md", blobs_head["docs/README.md"], pygit2.GIT_FILEMODE_BLOB)

    assets_tb2 = repo.TreeBuilder()
    assets_tb2.insert("logo.png", blobs_head["assets/logo.png"], pygit2.GIT_FILEMODE_BLOB)

    root_tb2 = repo.TreeBuilder()
    root_tb2.insert("src", src_tb2.write(), pygit2.GIT_FILEMODE_TREE)
    root_tb2.insert("config", config_tb2.write(), pygit2.GIT_FILEMODE_TREE)
    root_tb2.insert("docs", docs_tb2.write(), pygit2.GIT_FILEMODE_TREE)
    root_tb2.insert("assets", assets_tb2.write(), pygit2.GIT_FILEMODE_TREE)

    c2 = repo.create_commit(
        "refs/heads/feature", sig, sig, "add routes and auth", root_tb2.write(), [c1]
    )

    return repo, str(c1), str(c2)


def _file_state(repo: pygit2.Repository) -> EngineState:
    """EngineState with repo and review target for file tool tests."""
    state = EngineState(repo=repo, owner="o", repo_name="r")
    state.review_target = BranchTarget(
        base_branch="main",
        head_branch="feature",
        base_commit="main",
        head_commit="feature",
        updated_at=0,
    )
    return state


# ── read_file ────────────────────────────────────────────────────────


def test_read_file_full() -> None:
    """Full file read returns numbered lines with correct content."""
    with tempfile.TemporaryDirectory() as tmp:
        repo, _, _ = _make_file_repo(tmp)
        ctx = FakeCtx(_file_state(repo))
        result = read_file(ctx, "config/settings.toml")  # type: ignore[arg-type]
        assert "config/settings.toml" in result
        assert 'host = "0.0.0.0"' in result
        assert "pool_size = 10" in result
        # Line numbers present.
        assert "│" in result


def test_read_file_line_range() -> None:
    """Line range returns exact slice with correct line numbers."""
    with tempfile.TemporaryDirectory() as tmp:
        repo, _, _ = _make_file_repo(tmp)
        ctx = FakeCtx(_file_state(repo))
        result = read_file(ctx, "config/settings.toml", offset=5, max_lines=3)  # type: ignore[arg-type]
        # Lines 6-8 are the [database] section.
        assert "[database]" in result
        assert 'url = "postgresql://localhost/mydb"' in result
        assert "pool_size = 10" in result
        # Should NOT contain [server] section (lines 1-4).
        assert "[server]" not in result


def test_read_file_head_vs_base() -> None:
    """Head and base refs return different content for modified file."""
    with tempfile.TemporaryDirectory() as tmp:
        repo, _, _ = _make_file_repo(tmp)
        ctx = FakeCtx(_file_state(repo))
        head_result = read_file(ctx, "src/api/handler.py", ref="head")  # type: ignore[arg-type]
        base_result = read_file(ctx, "src/api/handler.py", ref="base")  # type: ignore[arg-type]
        # Head has @require_auth and shutdown; base does not.
        assert "require_auth" in head_result
        assert "shutdown" in head_result
        assert "require_auth" not in base_result
        assert "shutdown" not in base_result


def test_read_file_raw_ref() -> None:
    """Raw commit SHA resolves correctly."""
    with tempfile.TemporaryDirectory() as tmp:
        repo, c1, _ = _make_file_repo(tmp)
        ctx = FakeCtx(_file_state(repo))
        result = read_file(ctx, "src/api/handler.py", ref=c1[:8])  # type: ignore[arg-type]
        # c1 is the base commit — should have v1 content.
        assert "require_auth" not in result
        assert "handle_request" in result


@pytest.mark.parametrize("bad_path", ["../etc/passwd", "src/../../../secret", "foo/../../bar"])
def test_read_file_rejects_traversal(bad_path: str) -> None:
    """Paths with '..' are rejected."""
    with tempfile.TemporaryDirectory() as tmp:
        repo, _, _ = _make_file_repo(tmp)
        ctx = FakeCtx(_file_state(repo))
        result = read_file(ctx, bad_path)  # type: ignore[arg-type]
        assert "'..' " in result or "traversal" in result.lower() or "contains '..'" in result


def test_read_file_binary_rejection() -> None:
    """Binary files are rejected with a clear message."""
    with tempfile.TemporaryDirectory() as tmp:
        repo, _, _ = _make_file_repo(tmp)
        ctx = FakeCtx(_file_state(repo))
        result = read_file(ctx, "assets/logo.png")  # type: ignore[arg-type]
        assert "binary" in result.lower()


def test_read_file_not_found() -> None:
    """Nonexistent path returns not-found message."""
    with tempfile.TemporaryDirectory() as tmp:
        repo, _, _ = _make_file_repo(tmp)
        ctx = FakeCtx(_file_state(repo))
        result = read_file(ctx, "nonexistent/file.py")  # type: ignore[arg-type]
        assert "not found" in result.lower()


def test_read_file_truncation(config_path: Path) -> None:
    """Files exceeding max_lines are limited with pagination hint."""
    from rbtr.config import config as cfg

    cfg.tools.max_lines = 3  # tiny limit

    with tempfile.TemporaryDirectory() as tmp:
        repo, _, _ = _make_file_repo(tmp)
        ctx = FakeCtx(_file_state(repo))
        result = read_file(ctx, "src/api/handler.py")  # type: ignore[arg-type]
        assert "... limited" in result
        assert "offset=" in result


def test_read_file_bad_ref() -> None:
    """Unresolvable ref returns error."""
    with tempfile.TemporaryDirectory() as tmp:
        repo, _, _ = _make_file_repo(tmp)
        ctx = FakeCtx(_file_state(repo))
        result = read_file(ctx, "src/api/handler.py", ref="nonexistent_branch")  # type: ignore[arg-type]
        assert "not found" in result.lower()


# ── grep ─────────────────────────────────────────────────────────────


def test_grep_single_match() -> None:
    """Single match returns the line with context."""
    with tempfile.TemporaryDirectory() as tmp:
        repo, _, _ = _make_file_repo(tmp)
        ctx = FakeCtx(_file_state(repo))
        result = grep(ctx, "CONTRIBUTING", pattern="docs/README.md")  # type: ignore[arg-type]
        assert "1 match" in result
        assert "CONTRIBUTING" in result
        # Should include surrounding context.
        assert "Development" in result


def test_grep_multiple_matches() -> None:
    """Multiple matches each appear in output."""
    with tempfile.TemporaryDirectory() as tmp:
        repo, _, _ = _make_file_repo(tmp)
        ctx = FakeCtx(_file_state(repo))
        # "handle_request" appears in both handler.py imports and the ROUTES list.
        result = grep(ctx, "handle_request", pattern="src/api/routes.py")  # type: ignore[arg-type]
        assert "handle_request" in result
        # Should show match count.
        assert "2 matches" in result


def test_grep_overlapping_context_merge() -> None:
    """Matches close together produce merged context (no duplicate lines)."""
    with tempfile.TemporaryDirectory() as tmp:
        repo, _, _ = _make_file_repo(tmp)
        ctx = FakeCtx(_file_state(repo))
        # In README, "handle_request" and "health_check" are on adjacent lines.
        result = grep(ctx, "handle_request", pattern="docs/README.md", context_lines=3)  # type: ignore[arg-type]
        # Count line number markers — should not have duplicate line numbers.
        numbered = [line for line in result.split("\n") if "│" in line]
        line_nums = [line.split("│")[0].strip() for line in numbered]
        assert len(line_nums) == len(set(line_nums)), "duplicate line numbers in merged context"


def test_grep_custom_context_lines() -> None:
    """Custom context_lines overrides the default window size."""
    with tempfile.TemporaryDirectory() as tmp:
        repo, _, _ = _make_file_repo(tmp)
        ctx = FakeCtx(_file_state(repo))
        # Search with tiny context (1 line) vs default (10).
        small = grep(ctx, "CONTRIBUTING", pattern="docs/README.md", context_lines=1)  # type: ignore[arg-type]
        large = grep(ctx, "CONTRIBUTING", pattern="docs/README.md", context_lines=10)  # type: ignore[arg-type]
        # Small context should have fewer lines.
        small_lines = [ln for ln in small.split("\n") if "│" in ln]
        large_lines = [ln for ln in large.split("\n") if "│" in ln]
        assert len(small_lines) < len(large_lines)


def test_grep_no_match() -> None:
    """No matches returns a clear message."""
    with tempfile.TemporaryDirectory() as tmp:
        repo, _, _ = _make_file_repo(tmp)
        ctx = FakeCtx(_file_state(repo))
        result = grep(ctx, "zzz_nonexistent_zzz", pattern="docs/README.md")  # type: ignore[arg-type]
        assert "No matches" in result


def test_grep_case_insensitive() -> None:
    """Search is case-insensitive."""
    with tempfile.TemporaryDirectory() as tmp:
        repo, _, _ = _make_file_repo(tmp)
        ctx = FakeCtx(_file_state(repo))
        result = grep(ctx, "overview", pattern="docs/README.md")  # type: ignore[arg-type]
        # "## Overview" has capital O but search is lowercase.
        assert "Overview" in result
        assert "1 match" in result


@pytest.mark.parametrize("bad_path", ["../etc/passwd", "src/../../../secret"])
def test_grep_rejects_traversal(bad_path: str) -> None:
    """Paths with '..' are rejected."""
    with tempfile.TemporaryDirectory() as tmp:
        repo, _, _ = _make_file_repo(tmp)
        ctx = FakeCtx(_file_state(repo))
        result = grep(ctx, "anything", pattern=bad_path)  # type: ignore[arg-type]
        assert "'..' " in result or "contains '..'" in result


def test_grep_binary_rejection() -> None:
    """Binary files are rejected."""
    with tempfile.TemporaryDirectory() as tmp:
        repo, _, _ = _make_file_repo(tmp)
        ctx = FakeCtx(_file_state(repo))
        result = grep(ctx, "PNG", pattern="assets/logo.png")  # type: ignore[arg-type]
        assert "binary" in result.lower()


# ── grep repo-wide ──────────────────────────────────────────────────


def test_grep_repo_wide_no_path() -> None:
    """Empty path searches all files, finds matches across multiple files."""
    with tempfile.TemporaryDirectory() as tmp:
        repo, _, _ = _make_file_repo(tmp)
        ctx = FakeCtx(_file_state(repo))
        # "handle_request" appears in handler.py, routes.py, and README.md.
        result = grep(ctx, "handle_request")  # type: ignore[arg-type]
        assert "src/api/handler.py" in result
        assert "src/api/routes.py" in result
        assert "docs/README.md" in result
        assert "Found" in result


def test_grep_directory_prefix() -> None:
    """Directory prefix scopes search to subtree."""
    with tempfile.TemporaryDirectory() as tmp:
        repo, _, _ = _make_file_repo(tmp)
        ctx = FakeCtx(_file_state(repo))
        # Search only in src/ — should find handler.py and routes.py but not README.md.
        result = grep(ctx, "handle_request", pattern="src/")  # type: ignore[arg-type]
        assert "src/api/handler.py" in result
        assert "src/api/routes.py" in result
        assert "README.md" not in result


def test_grep_exact_file_still_works() -> None:
    """Exact file path behaves as single-file grep."""
    with tempfile.TemporaryDirectory() as tmp:
        repo, _, _ = _make_file_repo(tmp)
        ctx = FakeCtx(_file_state(repo))
        result = grep(ctx, "pool_size", pattern="config/settings.toml")  # type: ignore[arg-type]
        assert "1 match" in result
        assert "pool_size" in result
        assert "config/settings.toml" in result


def test_grep_repo_wide_skips_binary() -> None:
    """Binary files are silently skipped in repo-wide search."""
    with tempfile.TemporaryDirectory() as tmp:
        repo, _, _ = _make_file_repo(tmp)
        ctx = FakeCtx(_file_state(repo))
        # "PNG" is in the binary logo — should not appear.
        result = grep(ctx, "pool_size")  # type: ignore[arg-type]
        assert "logo.png" not in result
        assert "pool_size" in result


def test_grep_repo_wide_ref_base() -> None:
    """ref='base' searches the base snapshot — misses head-only content."""
    with tempfile.TemporaryDirectory() as tmp:
        repo, _, _ = _make_file_repo(tmp)
        ctx = FakeCtx(_file_state(repo))
        # "require_auth" only exists in handler v2 (head).
        head_result = grep(ctx, "require_auth", ref="head")  # type: ignore[arg-type]
        base_result = grep(ctx, "require_auth", ref="base")  # type: ignore[arg-type]
        assert "require_auth" in head_result
        assert "No matches" in base_result


def test_grep_repo_wide_no_match() -> None:
    """No matches across all files returns clear message."""
    with tempfile.TemporaryDirectory() as tmp:
        repo, _, _ = _make_file_repo(tmp)
        ctx = FakeCtx(_file_state(repo))
        result = grep(ctx, "zzz_nonexistent_zzz")  # type: ignore[arg-type]
        assert "No matches" in result


def test_grep_glob_star() -> None:
    """Glob `*.py` searches only Python files."""
    with tempfile.TemporaryDirectory() as tmp:
        repo, _, _ = _make_file_repo(tmp)
        ctx = FakeCtx(_file_state(repo))
        result = grep(ctx, "handle_request", pattern="*.py")  # type: ignore[arg-type]
        assert "handle_request" in result
        # Should not search .toml or .md files.
        assert "settings.toml" not in result


def test_grep_glob_scoped() -> None:
    """Glob `src/**/*.py` scopes search to Python files under `src/`."""
    with tempfile.TemporaryDirectory() as tmp:
        repo, _, _ = _make_file_repo(tmp)
        ctx = FakeCtx(_file_state(repo))
        result = grep(ctx, "handle_request", pattern="src/**/*.py")  # type: ignore[arg-type]
        assert "handle_request" in result


def test_grep_glob_no_match() -> None:
    """Glob pattern with no matching files returns a message."""
    with tempfile.TemporaryDirectory() as tmp:
        repo, _, _ = _make_file_repo(tmp)
        ctx = FakeCtx(_file_state(repo))
        result = grep(ctx, "anything", pattern="*.rs")  # type: ignore[arg-type]
        assert "No matches" in result


def test_grep_repo_wide_truncation(config_path: Path) -> None:
    """Repo-wide results are limited at max_grep_hits."""
    from rbtr.config import config as cfg

    cfg.tools.max_grep_hits = 1  # tiny limit — only one match group

    with tempfile.TemporaryDirectory() as tmp:
        repo, _, _ = _make_file_repo(tmp)
        ctx = FakeCtx(_file_state(repo))
        # "handle_request" matches in multiple files — should limit.
        result = grep(ctx, "handle_request")  # type: ignore[arg-type]
        assert "... limited" in result
        assert "offset=" in result


# ── list_files ───────────────────────────────────────────────────────


def test_list_files_root() -> None:
    """Root listing returns all files."""
    with tempfile.TemporaryDirectory() as tmp:
        repo, _, _ = _make_file_repo(tmp)
        ctx = FakeCtx(_file_state(repo))
        result = list_files(ctx)  # type: ignore[arg-type]
        assert "src/api/handler.py" in result
        assert "src/api/routes.py" in result
        assert "config/settings.toml" in result
        assert "docs/README.md" in result
        assert "assets/logo.png" in result


def test_list_files_directory_prefix() -> None:
    """Directory prefix filters to files under that path."""
    with tempfile.TemporaryDirectory() as tmp:
        repo, _, _ = _make_file_repo(tmp)
        ctx = FakeCtx(_file_state(repo))
        result = list_files(ctx, pattern="src/api")  # type: ignore[arg-type]
        assert "handler.py" in result
        assert "routes.py" in result
        # Should NOT include files outside src/api.
        assert "settings.toml" not in result
        assert "README.md" not in result


def test_list_files_config_prefix() -> None:
    """Prefix 'config' returns only config files."""
    with tempfile.TemporaryDirectory() as tmp:
        repo, _, _ = _make_file_repo(tmp)
        ctx = FakeCtx(_file_state(repo))
        result = list_files(ctx, pattern="config")  # type: ignore[arg-type]
        assert "settings.toml" in result
        assert "handler.py" not in result


def test_list_files_base_ref_omits_new_files() -> None:
    """Base ref doesn't include head-only files (routes.py)."""
    with tempfile.TemporaryDirectory() as tmp:
        repo, _, _ = _make_file_repo(tmp)
        ctx = FakeCtx(_file_state(repo))
        result = list_files(ctx, ref="base")  # type: ignore[arg-type]
        assert "handler.py" in result
        assert "routes.py" not in result


def test_list_files_head_ref_includes_new_files() -> None:
    """Head ref includes new files added on the feature branch."""
    with tempfile.TemporaryDirectory() as tmp:
        repo, _, _ = _make_file_repo(tmp)
        ctx = FakeCtx(_file_state(repo))
        result = list_files(ctx, ref="head")  # type: ignore[arg-type]
        assert "routes.py" in result


def test_list_files_truncation(config_path: Path) -> None:
    """More than max_results entries triggers limitation."""
    from rbtr.config import config as cfg

    cfg.tools.max_results = 2  # tiny limit

    with tempfile.TemporaryDirectory() as tmp:
        repo, _, _ = _make_file_repo(tmp)
        ctx = FakeCtx(_file_state(repo))
        result = list_files(ctx)  # type: ignore[arg-type]
        assert "... limited" in result
        assert "offset=" in result


def test_list_files_no_match() -> None:
    """Nonexistent directory returns empty message."""
    with tempfile.TemporaryDirectory() as tmp:
        repo, _, _ = _make_file_repo(tmp)
        ctx = FakeCtx(_file_state(repo))
        result = list_files(ctx, pattern="nonexistent/dir")  # type: ignore[arg-type]
        assert "No files" in result


def test_list_files_bad_ref() -> None:
    """Unresolvable ref returns error."""
    with tempfile.TemporaryDirectory() as tmp:
        repo, _, _ = _make_file_repo(tmp)
        ctx = FakeCtx(_file_state(repo))
        result = list_files(ctx, ref="nonexistent_branch")  # type: ignore[arg-type]
        assert "not found" in result.lower()


def test_list_files_glob_star() -> None:
    """Glob `*.py` matches Python files at any depth."""
    with tempfile.TemporaryDirectory() as tmp:
        repo, _, _ = _make_file_repo(tmp)
        ctx = FakeCtx(_file_state(repo))
        result = list_files(ctx, pattern="*.py")  # type: ignore[arg-type]
        assert "handler.py" in result
        assert "routes.py" in result
        assert "settings.toml" not in result
        assert "README.md" not in result
        assert "logo.png" not in result


def test_list_files_glob_scoped() -> None:
    """Glob `src/**/*.py` matches only Python files under `src/`."""
    with tempfile.TemporaryDirectory() as tmp:
        repo, _, _ = _make_file_repo(tmp)
        ctx = FakeCtx(_file_state(repo))
        result = list_files(ctx, pattern="src/**/*.py")  # type: ignore[arg-type]
        assert "src/api/handler.py" in result
        assert "src/api/routes.py" in result
        assert "settings.toml" not in result


def test_list_files_glob_no_match() -> None:
    """Glob with no matches returns a message."""
    with tempfile.TemporaryDirectory() as tmp:
        repo, _, _ = _make_file_repo(tmp)
        ctx = FakeCtx(_file_state(repo))
        result = list_files(ctx, pattern="*.rs")  # type: ignore[arg-type]
        assert "No files" in result


def test_list_files_glob_single_level() -> None:
    """Glob `src/api/*.py` matches only direct children."""
    with tempfile.TemporaryDirectory() as tmp:
        repo, _, _ = _make_file_repo(tmp)
        ctx = FakeCtx(_file_state(repo))
        result = list_files(ctx, pattern="src/api/*.py")  # type: ignore[arg-type]
        assert "handler.py" in result
        assert "routes.py" in result


# ── Workspace file access (.rbtr/) ──────────────────────────────────
#
# read_file, grep, and list_files read from the local filesystem
# when paths start with .rbtr/ — these files aren't in the git tree.


def _make_workspace(tmp_path: Path) -> None:
    """Create .rbtr/notes/ workspace with review notes for testing."""
    notes = tmp_path / ".rbtr" / "notes"
    notes.mkdir(parents=True)
    (notes / "plan.md").write_text(
        """\
# Review Plan

## Phase 1: Orientation
- Read PR description
- Check changed files

## Phase 2: Deep dive
- handler.py: check error paths
- config.py: verify defaults
"""
    )
    (notes / "findings.md").write_text(
        """\
# Findings

## blocker: Missing null check in handler
The `handle_request` function does not validate
that `data` is non-empty before processing.

## suggestion: Config defaults are stale
The `pool_size` default of 5 was appropriate for
the old architecture but should be revisited.
"""
    )


def test_read_file_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """read_file reads .rbtr/ files from the local filesystem."""
    monkeypatch.chdir(tmp_path)
    _make_workspace(tmp_path)
    with tempfile.TemporaryDirectory() as repo_tmp:
        repo, _, _ = _make_file_repo(repo_tmp)
        ctx = FakeCtx(_file_state(repo))
        result = read_file(ctx, ".rbtr/notes/plan.md")  # type: ignore[arg-type]
        assert "# Review Plan" in result
        assert "Phase 1" in result
        assert "handler.py" in result


def test_read_file_workspace_not_found(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """read_file returns error for nonexistent .rbtr/ file."""
    monkeypatch.chdir(tmp_path)
    with tempfile.TemporaryDirectory() as repo_tmp:
        repo, _, _ = _make_file_repo(repo_tmp)
        ctx = FakeCtx(_file_state(repo))
        result = read_file(ctx, ".rbtr/notes/nonexistent.md")  # type: ignore[arg-type]
        assert "not found" in result.lower()


def test_read_file_workspace_ignores_ref(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """ref parameter is ignored for .rbtr/ workspace files."""
    monkeypatch.chdir(tmp_path)
    _make_workspace(tmp_path)
    with tempfile.TemporaryDirectory() as repo_tmp:
        repo, _, _ = _make_file_repo(repo_tmp)
        ctx = FakeCtx(_file_state(repo))
        # ref="base" would fail for git files, but is ignored for .rbtr/
        result = read_file(ctx, ".rbtr/notes/plan.md", ref="base")  # type: ignore[arg-type]
        assert "# Review Plan" in result


def test_grep_workspace_single_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """grep searches a single .rbtr/ file from the filesystem."""
    monkeypatch.chdir(tmp_path)
    _make_workspace(tmp_path)
    with tempfile.TemporaryDirectory() as repo_tmp:
        repo, _, _ = _make_file_repo(repo_tmp)
        ctx = FakeCtx(_file_state(repo))
        result = grep(ctx, "null check", pattern=".rbtr/notes/findings.md")  # type: ignore[arg-type]
        assert "null check" in result.lower()
        assert "findings.md" in result


def test_grep_workspace_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """grep searches all files under .rbtr/ when given a directory prefix."""
    monkeypatch.chdir(tmp_path)
    _make_workspace(tmp_path)
    with tempfile.TemporaryDirectory() as repo_tmp:
        repo, _, _ = _make_file_repo(repo_tmp)
        ctx = FakeCtx(_file_state(repo))
        # "handler" appears in both plan and findings
        result = grep(ctx, "handler", pattern=".rbtr/")  # type: ignore[arg-type]
        assert "handler" in result.lower()
        # Should show matches from at least one file
        assert "notes" in result


def test_grep_workspace_no_match(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """grep returns no-match message for workspace files."""
    monkeypatch.chdir(tmp_path)
    _make_workspace(tmp_path)
    with tempfile.TemporaryDirectory() as repo_tmp:
        repo, _, _ = _make_file_repo(repo_tmp)
        ctx = FakeCtx(_file_state(repo))
        result = grep(ctx, "xyznonexistent", pattern=".rbtr/")  # type: ignore[arg-type]
        assert "No matches" in result


def test_list_files_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """list_files lists .rbtr/ files from the local filesystem."""
    monkeypatch.chdir(tmp_path)
    _make_workspace(tmp_path)
    with tempfile.TemporaryDirectory() as repo_tmp:
        repo, _, _ = _make_file_repo(repo_tmp)
        ctx = FakeCtx(_file_state(repo))
        result = list_files(ctx, pattern=".rbtr/")  # type: ignore[arg-type]
        assert "plan.md" in result
        assert "findings.md" in result


def test_list_files_workspace_empty(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """list_files returns empty message for nonexistent .rbtr/ directory."""
    monkeypatch.chdir(tmp_path)
    with tempfile.TemporaryDirectory() as repo_tmp:
        repo, _, _ = _make_file_repo(repo_tmp)
        ctx = FakeCtx(_file_state(repo))
        result = list_files(ctx, pattern=".rbtr/")  # type: ignore[arg-type]
        assert "No files" in result


# ── Filesystem fallback — git-first, then filesystem ─────────────────
#
# When a prefix matches git files, git wins.
# When nothing is in git, filesystem is tried.
# read_file: git blob → filesystem.
# list_files: git tree → filesystem.
# grep: git blob/tree → filesystem.


def test_list_files_git_prefix_wins_over_filesystem(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When git tree has files under the prefix, filesystem is NOT used."""
    monkeypatch.chdir(tmp_path)
    # Create a filesystem file under src/api/ (same prefix as git files).
    (tmp_path / "src" / "api").mkdir(parents=True)
    (tmp_path / "src" / "api" / "local_only.py").write_text("# local\n")
    with tempfile.TemporaryDirectory() as repo_tmp:
        repo, _, _ = _make_file_repo(repo_tmp)
        ctx = FakeCtx(_file_state(repo))
        result = list_files(ctx, pattern="src/api")  # type: ignore[arg-type]
        # Git files should be listed.
        assert "handler.py" in result
        # Filesystem-only file should NOT appear — git wins.
        assert "local_only.py" not in result


def test_list_files_falls_back_to_filesystem(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When prefix has no git files, filesystem is used as fallback."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "local_dir").mkdir()
    (tmp_path / "local_dir" / "notes.txt").write_text("hello\n")
    with tempfile.TemporaryDirectory() as repo_tmp:
        repo, _, _ = _make_file_repo(repo_tmp)
        ctx = FakeCtx(_file_state(repo))
        result = list_files(ctx, pattern="local_dir")  # type: ignore[arg-type]
        assert "notes.txt" in result


def test_read_file_git_blob_wins_over_filesystem(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When a file exists in both git and filesystem, git version is used."""
    monkeypatch.chdir(tmp_path)
    # Create a local file with different content than the git blob.
    (tmp_path / "src" / "api").mkdir(parents=True)
    (tmp_path / "src" / "api" / "handler.py").write_text("# FILESYSTEM VERSION\n")
    with tempfile.TemporaryDirectory() as repo_tmp:
        repo, _, _ = _make_file_repo(repo_tmp)
        ctx = FakeCtx(_file_state(repo))
        result = read_file(ctx, "src/api/handler.py")  # type: ignore[arg-type]
        # Git content has handle_request; filesystem has "FILESYSTEM VERSION".
        assert "handle_request" in result
        assert "FILESYSTEM VERSION" not in result


def test_read_file_falls_back_to_filesystem(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When a file is not in git, falls back to local filesystem."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "untracked.txt").write_text("local content here\n")
    with tempfile.TemporaryDirectory() as repo_tmp:
        repo, _, _ = _make_file_repo(repo_tmp)
        ctx = FakeCtx(_file_state(repo))
        result = read_file(ctx, "untracked.txt")  # type: ignore[arg-type]
        assert "local content here" in result


def test_read_file_not_in_git_or_filesystem() -> None:
    """When a file is in neither git nor filesystem, returns git error."""
    with tempfile.TemporaryDirectory() as tmp:
        repo, _, _ = _make_file_repo(tmp)
        ctx = FakeCtx(_file_state(repo))
        result = read_file(ctx, "totally_missing.txt")  # type: ignore[arg-type]
        assert "not found" in result.lower()


def test_grep_git_prefix_wins_over_filesystem(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When git tree has files under prefix, grep searches git only."""
    monkeypatch.chdir(tmp_path)
    # Create filesystem file with unique content under the same prefix.
    (tmp_path / "src" / "api").mkdir(parents=True)
    (tmp_path / "src" / "api" / "local.py").write_text("UNIQUE_LOCAL_MARKER\n")
    with tempfile.TemporaryDirectory() as repo_tmp:
        repo, _, _ = _make_file_repo(repo_tmp)
        ctx = FakeCtx(_file_state(repo))
        result = grep(ctx, "UNIQUE_LOCAL_MARKER", pattern="src/api")  # type: ignore[arg-type]
        # Git has files under src/api but none contain this marker.
        assert "No matches" in result


def test_grep_falls_back_to_filesystem(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When prefix has no git files, grep searches filesystem."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "local_dir").mkdir()
    (tmp_path / "local_dir" / "notes.txt").write_text("important finding\n")
    with tempfile.TemporaryDirectory() as repo_tmp:
        repo, _, _ = _make_file_repo(repo_tmp)
        ctx = FakeCtx(_file_state(repo))
        result = grep(ctx, "important", pattern="local_dir")  # type: ignore[arg-type]
        assert "important finding" in result
        assert "notes.txt" in result


def test_grep_single_file_falls_back_to_filesystem(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When an exact file path isn't in git, grep searches filesystem."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "local.txt").write_text("needle in haystack\n")
    with tempfile.TemporaryDirectory() as repo_tmp:
        repo, _, _ = _make_file_repo(repo_tmp)
        ctx = FakeCtx(_file_state(repo))
        result = grep(ctx, "needle", pattern="local.txt")  # type: ignore[arg-type]
        assert "needle in haystack" in result
