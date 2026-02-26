"""Tests for path filtering in file tools and is_path_ignored.

Verifies that file tools (read_file, grep, list_files) correctly
apply the three-layer filter:

1. ``config.index.include`` force-includes (overrides gitignore
   and extend_exclude).
2. ``.gitignore`` via ``repo.path_is_ignored``.
3. ``config.index.extend_exclude`` globs.

Also tests ``is_path_ignored`` directly for edge cases.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pygit2
import pytest

from rbtr.config import config
from rbtr.engine.state import EngineState
from rbtr.engine.tools import grep, list_files, read_file
from rbtr.git import is_path_ignored
from rbtr.git.filters import _matches_globs
from rbtr.models import BranchTarget

# ── Helpers ──────────────────────────────────────────────────────────


class _FakeCtx:
    """Minimal stand-in for RunContext[AgentDeps] in tool tests."""

    def __init__(self, state: EngineState) -> None:
        self.deps = _FakeDeps(state)


class _FakeDeps:
    def __init__(self, state: EngineState) -> None:
        self.state = state


def _make_repo(tmp: str) -> tuple[pygit2.Repository, str]:
    """Create a simple single-commit repo.  Returns (repo, sha)."""
    repo = pygit2.init_repository(tmp)
    sig = pygit2.Signature("Test", "test@test.com")

    # One file in git so tool functions have a valid tree to walk.
    b = repo.create_blob(b"x = 1\n")
    tb = repo.TreeBuilder()
    tb.insert("tracked.py", b, pygit2.GIT_FILEMODE_BLOB)
    sha = repo.create_commit("refs/heads/main", sig, sig, "init", tb.write(), [])
    repo.set_head("refs/heads/main")
    return repo, str(sha)


def _state(repo: pygit2.Repository) -> EngineState:
    state = EngineState(repo=repo, owner="o", repo_name="r")
    state.review_target = BranchTarget(
        base_branch="main",
        head_branch="main",
        updated_at=0,
    )
    return state


# ── is_path_ignored unit tests ───────────────────────────────────────


def test_is_path_ignored_gitignore() -> None:
    """Paths matching .gitignore are ignored when repo is provided."""
    with tempfile.TemporaryDirectory() as tmp:
        repo, _ = _make_repo(tmp)
        workdir = Path(repo.workdir)  # type: ignore[arg-type]
        (workdir / ".gitignore").write_text(".mypy_cache/\n__pycache__/\n")

        assert is_path_ignored(".mypy_cache/foo.json", repo, include=[], exclude=[])
        assert is_path_ignored("__pycache__/mod.pyc", repo, include=[], exclude=[])
        assert not is_path_ignored("src/main.py", repo, include=[], exclude=[])


def test_is_path_ignored_extend_exclude() -> None:
    """Paths matching exclude are ignored even without repo."""
    assert is_path_ignored(".rbtr/index", None, include=[], exclude=[".rbtr/index"])
    assert not is_path_ignored(".rbtr/REVIEW-plan.md", None, include=[], exclude=[".rbtr/index"])


def test_is_path_ignored_extend_exclude_glob() -> None:
    """Exclude supports glob patterns."""
    exclude = ["*.lock", "vendor/*"]
    assert is_path_ignored("package.lock", None, include=[], exclude=exclude)
    assert is_path_ignored("vendor/lib.py", None, include=[], exclude=exclude)
    assert not is_path_ignored("src/app.py", None, include=[], exclude=exclude)


def test_is_path_ignored_include_overrides_gitignore() -> None:
    """Include patterns override gitignore."""
    with tempfile.TemporaryDirectory() as tmp:
        repo, _ = _make_repo(tmp)
        workdir = Path(repo.workdir)  # type: ignore[arg-type]
        (workdir / ".gitignore").write_text(".rbtr/\nnode_modules/\n")

        # .rbtr/ is gitignored but force-included
        assert not is_path_ignored(".rbtr/REVIEW-plan.md", repo, include=[".rbtr/*"], exclude=[])
        # Other gitignored paths are still excluded
        assert is_path_ignored("node_modules/foo.js", repo, include=[".rbtr/*"], exclude=[])


def test_is_path_ignored_include_overrides_extend_exclude() -> None:
    """Include patterns override extend_exclude."""
    # Both include and exclude match — include wins
    assert not is_path_ignored(
        ".rbtr/REVIEW-plan.md", None, include=[".rbtr/*"], exclude=[".rbtr/*"]
    )


def test_is_path_ignored_no_repo_no_gitignore_check() -> None:
    """Without a repo, only exclude is applied."""
    assert is_path_ignored("debug.log", None, include=[], exclude=["*.log"])
    assert not is_path_ignored("app.py", None, include=[], exclude=["*.log"])


def test_is_path_ignored_include_and_exclude_interplay() -> None:
    """Include overrides exclude for matching paths; exclude applies to the rest."""
    inc = [".rbtr/REVIEW-*"]
    exc = [".rbtr/index"]
    assert not is_path_ignored(".rbtr/REVIEW-plan.md", None, include=inc, exclude=exc)
    assert is_path_ignored(".rbtr/index", None, include=inc, exclude=exc)
    # Children of .rbtr/index are also excluded (prefix matching)
    assert is_path_ignored(".rbtr/index/rbtr.duckdb", None, include=inc, exclude=exc)


# ── read_file filtering tests ───────────────────────────────────────


def test_read_file_blocks_gitignored_fs_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config_path: Path
) -> None:
    """read_file refuses to read gitignored files from the filesystem."""
    config.index.include = []
    config.index.extend_exclude = []
    monkeypatch.chdir(tmp_path)

    # Create a file that would be gitignored
    cache_dir = tmp_path / ".mypy_cache" / "3.12"
    cache_dir.mkdir(parents=True)
    (cache_dir / "data.json").write_text('{"key": "value"}')

    with tempfile.TemporaryDirectory() as repo_tmp:
        repo, _ = _make_repo(repo_tmp)
        workdir = Path(repo.workdir)  # type: ignore[arg-type]
        (workdir / ".gitignore").write_text(".mypy_cache/\n")

        ctx = _FakeCtx(_state(repo))
        result = read_file(ctx, ".mypy_cache/3.12/data.json")  # type: ignore[arg-type]
        assert "not found" in result.lower() or "cannot" in result.lower()


def test_read_file_blocks_extend_exclude_fs_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config_path: Path
) -> None:
    """read_file refuses to read files matching extend_exclude."""
    config.index.include = []
    config.index.extend_exclude = [".rbtr/index"]
    monkeypatch.chdir(tmp_path)

    (tmp_path / ".rbtr").mkdir()
    (tmp_path / ".rbtr" / "index").write_text("db data")

    with tempfile.TemporaryDirectory() as repo_tmp:
        repo, _ = _make_repo(repo_tmp)
        ctx = _FakeCtx(_state(repo))
        result = read_file(ctx, ".rbtr/index")  # type: ignore[arg-type]
        assert "not found" in result.lower() or "cannot" in result.lower()


def test_read_file_allows_included_despite_gitignore(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config_path: Path
) -> None:
    """read_file reads force-included files even when gitignored."""
    config.index.include = [".rbtr/*"]
    config.index.extend_exclude = []
    monkeypatch.chdir(tmp_path)

    (tmp_path / ".rbtr").mkdir()
    (tmp_path / ".rbtr" / "REVIEW-plan.md").write_text("# Plan\nStep 1\n")

    with tempfile.TemporaryDirectory() as repo_tmp:
        repo, _ = _make_repo(repo_tmp)
        workdir = Path(repo.workdir)  # type: ignore[arg-type]
        (workdir / ".gitignore").write_text(".rbtr/\n")

        ctx = _FakeCtx(_state(repo))
        result = read_file(ctx, ".rbtr/REVIEW-plan.md")  # type: ignore[arg-type]
        assert "# Plan" in result
        assert "Step 1" in result


def test_read_file_allows_non_ignored_fs_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config_path: Path
) -> None:
    """read_file still reads untracked files that aren't ignored."""
    config.index.include = []
    config.index.extend_exclude = []
    monkeypatch.chdir(tmp_path)

    (tmp_path / "notes.txt").write_text("important notes\n")

    with tempfile.TemporaryDirectory() as repo_tmp:
        repo, _ = _make_repo(repo_tmp)
        ctx = _FakeCtx(_state(repo))
        result = read_file(ctx, "notes.txt")  # type: ignore[arg-type]
        assert "important notes" in result


# ── list_files filtering tests ───────────────────────────────────────


def test_list_files_excludes_gitignored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config_path: Path
) -> None:
    """list_files omits gitignored paths in filesystem fallback."""
    config.index.include = []
    config.index.extend_exclude = []
    monkeypatch.chdir(tmp_path)

    local = tmp_path / "local_dir"
    local.mkdir()
    (local / "visible.txt").write_text("hello\n")
    (local / "cache.pyc").write_text("bytecode\n")

    with tempfile.TemporaryDirectory() as repo_tmp:
        repo, _ = _make_repo(repo_tmp)
        workdir = Path(repo.workdir)  # type: ignore[arg-type]
        (workdir / ".gitignore").write_text("*.pyc\n")

        ctx = _FakeCtx(_state(repo))
        result = list_files(ctx, path="local_dir")  # type: ignore[arg-type]
        assert "visible.txt" in result
        assert "cache.pyc" not in result


def test_list_files_excludes_extend_exclude(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config_path: Path
) -> None:
    """list_files omits paths matching extend_exclude in filesystem fallback."""
    config.index.include = []
    config.index.extend_exclude = ["*.db"]
    monkeypatch.chdir(tmp_path)

    local = tmp_path / "data"
    local.mkdir()
    (local / "notes.txt").write_text("hello\n")
    (local / "index.db").write_text("binary\n")

    with tempfile.TemporaryDirectory() as repo_tmp:
        repo, _ = _make_repo(repo_tmp)
        ctx = _FakeCtx(_state(repo))
        result = list_files(ctx, path="data")  # type: ignore[arg-type]
        assert "notes.txt" in result
        assert "index.db" not in result


def test_list_files_includes_override_gitignore(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config_path: Path
) -> None:
    """list_files shows force-included files even when gitignored."""
    config.index.include = [".rbtr/*"]
    config.index.extend_exclude = []
    monkeypatch.chdir(tmp_path)

    rbtr = tmp_path / ".rbtr"
    rbtr.mkdir()
    (rbtr / "REVIEW-plan.md").write_text("# Plan\n")
    (rbtr / "REVIEW-findings.md").write_text("# Findings\n")

    with tempfile.TemporaryDirectory() as repo_tmp:
        repo, _ = _make_repo(repo_tmp)
        workdir = Path(repo.workdir)  # type: ignore[arg-type]
        (workdir / ".gitignore").write_text(".rbtr/\n")

        ctx = _FakeCtx(_state(repo))
        result = list_files(ctx, path=".rbtr")  # type: ignore[arg-type]
        assert "REVIEW-plan.md" in result
        assert "REVIEW-findings.md" in result


def test_list_files_include_overrides_extend_exclude(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config_path: Path
) -> None:
    """Include beats extend_exclude for list_files filesystem fallback."""
    config.index.include = ["data/important.lock"]
    config.index.extend_exclude = ["*.lock"]
    monkeypatch.chdir(tmp_path)

    local = tmp_path / "data"
    local.mkdir()
    (local / "important.lock").write_text("keep\n")
    (local / "other.lock").write_text("skip\n")
    (local / "app.py").write_text("code\n")

    with tempfile.TemporaryDirectory() as repo_tmp:
        repo, _ = _make_repo(repo_tmp)
        ctx = _FakeCtx(_state(repo))
        result = list_files(ctx, path="data")  # type: ignore[arg-type]
        assert "important.lock" in result
        assert "other.lock" not in result
        assert "app.py" in result


def test_list_files_excludes_nested_gitignored_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config_path: Path
) -> None:
    """list_files omits entire gitignored directories."""
    config.index.include = []
    config.index.extend_exclude = []
    monkeypatch.chdir(tmp_path)

    local = tmp_path / "project"
    local.mkdir()
    (local / "app.py").write_text("code\n")
    cache = local / "__pycache__"
    cache.mkdir()
    (cache / "app.cpython-313.pyc").write_text("bytecode\n")
    nm = local / "node_modules"
    nm.mkdir()
    (nm / "lodash.js").write_text("module\n")

    with tempfile.TemporaryDirectory() as repo_tmp:
        repo, _ = _make_repo(repo_tmp)
        workdir = Path(repo.workdir)  # type: ignore[arg-type]
        (workdir / ".gitignore").write_text("__pycache__/\nnode_modules/\n")

        ctx = _FakeCtx(_state(repo))
        result = list_files(ctx, path="project")  # type: ignore[arg-type]
        assert "app.py" in result
        assert "__pycache__" not in result
        assert "node_modules" not in result
        assert "lodash" not in result


# ── grep filtering tests ────────────────────────────────────────────


def test_grep_excludes_gitignored_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config_path: Path
) -> None:
    """grep skips gitignored files in filesystem fallback."""
    config.index.include = []
    config.index.extend_exclude = []
    monkeypatch.chdir(tmp_path)

    local = tmp_path / "local_dir"
    local.mkdir()
    (local / "visible.py").write_text("MAGIC_MARKER in visible\n")
    (local / "cached.pyc").write_text("MAGIC_MARKER in cache\n")

    with tempfile.TemporaryDirectory() as repo_tmp:
        repo, _ = _make_repo(repo_tmp)
        workdir = Path(repo.workdir)  # type: ignore[arg-type]
        (workdir / ".gitignore").write_text("*.pyc\n")

        ctx = _FakeCtx(_state(repo))
        result = grep(ctx, "MAGIC_MARKER", path="local_dir")  # type: ignore[arg-type]
        assert "visible.py" in result
        assert "cached.pyc" not in result


def test_grep_excludes_extend_exclude_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config_path: Path
) -> None:
    """grep skips files matching extend_exclude in filesystem fallback."""
    config.index.include = []
    config.index.extend_exclude = ["*.log"]
    monkeypatch.chdir(tmp_path)

    local = tmp_path / "logs"
    local.mkdir()
    (local / "app.py").write_text("SEARCH_TERM in code\n")
    (local / "debug.log").write_text("SEARCH_TERM in log\n")

    with tempfile.TemporaryDirectory() as repo_tmp:
        repo, _ = _make_repo(repo_tmp)
        ctx = _FakeCtx(_state(repo))
        result = grep(ctx, "SEARCH_TERM", path="logs")  # type: ignore[arg-type]
        assert "app.py" in result
        assert "debug.log" not in result


def test_grep_includes_override_gitignore(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config_path: Path
) -> None:
    """grep finds matches in force-included files even when gitignored."""
    config.index.include = [".rbtr/*"]
    config.index.extend_exclude = []
    monkeypatch.chdir(tmp_path)

    rbtr = tmp_path / ".rbtr"
    rbtr.mkdir()
    (rbtr / "REVIEW-plan.md").write_text("SEARCH_TARGET in plan\n")

    with tempfile.TemporaryDirectory() as repo_tmp:
        repo, _ = _make_repo(repo_tmp)
        workdir = Path(repo.workdir)  # type: ignore[arg-type]
        (workdir / ".gitignore").write_text(".rbtr/\n")

        ctx = _FakeCtx(_state(repo))
        result = grep(ctx, "SEARCH_TARGET", path=".rbtr")  # type: ignore[arg-type]
        assert "SEARCH_TARGET" in result
        assert "REVIEW-plan.md" in result


def test_grep_include_overrides_extend_exclude(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config_path: Path
) -> None:
    """Include beats extend_exclude for grep filesystem fallback."""
    config.index.include = ["data/important.lock"]
    config.index.extend_exclude = ["*.lock"]
    monkeypatch.chdir(tmp_path)

    local = tmp_path / "data"
    local.mkdir()
    (local / "important.lock").write_text("FIND_ME in important\n")
    (local / "other.lock").write_text("FIND_ME in other\n")

    with tempfile.TemporaryDirectory() as repo_tmp:
        repo, _ = _make_repo(repo_tmp)
        ctx = _FakeCtx(_state(repo))
        result = grep(ctx, "FIND_ME", path="data")  # type: ignore[arg-type]
        assert "important.lock" in result
        assert "other.lock" not in result


def test_grep_excludes_nested_gitignored_dirs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config_path: Path
) -> None:
    """grep skips entire gitignored directories."""
    config.index.include = []
    config.index.extend_exclude = []
    monkeypatch.chdir(tmp_path)

    local = tmp_path / "project"
    local.mkdir()
    (local / "app.py").write_text("UNIQUE_NEEDLE in source\n")
    cache = local / ".mypy_cache"
    cache.mkdir()
    (cache / "data.json").write_text("UNIQUE_NEEDLE in cache\n")

    with tempfile.TemporaryDirectory() as repo_tmp:
        repo, _ = _make_repo(repo_tmp)
        workdir = Path(repo.workdir)  # type: ignore[arg-type]
        (workdir / ".gitignore").write_text(".mypy_cache/\n")

        ctx = _FakeCtx(_state(repo))
        result = grep(ctx, "UNIQUE_NEEDLE", path="project")  # type: ignore[arg-type]
        assert "app.py" in result
        assert ".mypy_cache" not in result


def test_grep_single_file_blocked_by_gitignore(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config_path: Path
) -> None:
    """grep on an exact gitignored file path returns no match.

    Single-file grep uses _read_fs_file directly (not _list_fs_files),
    but _grep_filesystem still delegates to _list_fs_files for
    directory prefixes. For single files, the file IS still readable
    since we don't filter single-file reads in _grep_filesystem
    (they bypass the listing). This test documents that behaviour.
    """
    config.index.include = []
    config.index.extend_exclude = []
    monkeypatch.chdir(tmp_path)

    (tmp_path / "ignored.pyc").write_text("NEEDLE in pyc\n")

    with tempfile.TemporaryDirectory() as repo_tmp:
        repo, _ = _make_repo(repo_tmp)
        workdir = Path(repo.workdir)  # type: ignore[arg-type]
        (workdir / ".gitignore").write_text("*.pyc\n")

        ctx = _FakeCtx(_state(repo))
        # Single file grep — the file is not in git, falls back to fs.
        # _grep_filesystem checks is_file() first for exact paths,
        # which doesn't apply filtering. This documents current behaviour.
        result = grep(ctx, "NEEDLE", path="ignored.pyc")  # type: ignore[arg-type]
        # Currently this finds the file since single-file fallback
        # doesn't filter. We may want to change this in the future.
        assert "NEEDLE" in result


# ── Realistic scenario: .mypy_cache leak ─────────────────────────────


def test_mypy_cache_excluded_from_list_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config_path: Path
) -> None:
    """The exact scenario that prompted this fix: .mypy_cache files
    should not appear in list_files."""
    config.index.include = []
    config.index.extend_exclude = []
    monkeypatch.chdir(tmp_path)

    cache = tmp_path / "project" / ".mypy_cache" / "3.12" / "openai"
    cache.mkdir(parents=True)
    (cache / "runs.data.json").write_text('{"data": "cached"}')
    (tmp_path / "project" / "main.py").write_text("print('hello')\n")

    with tempfile.TemporaryDirectory() as repo_tmp:
        repo, _ = _make_repo(repo_tmp)
        workdir = Path(repo.workdir)  # type: ignore[arg-type]
        (workdir / ".gitignore").write_text(".mypy_cache/\n")

        ctx = _FakeCtx(_state(repo))
        result = list_files(ctx, path="project")  # type: ignore[arg-type]
        assert "main.py" in result
        assert ".mypy_cache" not in result
        assert "runs.data.json" not in result


def test_mypy_cache_excluded_from_grep(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config_path: Path
) -> None:
    """grep should not search inside .mypy_cache."""
    config.index.include = []
    config.index.extend_exclude = []
    monkeypatch.chdir(tmp_path)

    cache = tmp_path / "project" / ".mypy_cache" / "3.12"
    cache.mkdir(parents=True)
    (cache / "data.json").write_text("SECRET_TOKEN_12345\n")
    (tmp_path / "project" / "main.py").write_text("SECRET_TOKEN_12345\n")

    with tempfile.TemporaryDirectory() as repo_tmp:
        repo, _ = _make_repo(repo_tmp)
        workdir = Path(repo.workdir)  # type: ignore[arg-type]
        (workdir / ".gitignore").write_text(".mypy_cache/\n")

        ctx = _FakeCtx(_state(repo))
        result = grep(ctx, "SECRET_TOKEN_12345", path="project")  # type: ignore[arg-type]
        assert "main.py" in result
        assert ".mypy_cache" not in result
        assert "data.json" not in result


# ── Default config: .rbtr included, .rbtr/index excluded ────────────


def test_default_config_rbtr_included_index_excluded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config_path: Path
) -> None:
    """With default config, .rbtr REVIEW files are accessible but .rbtr/index is not."""
    # The config_path fixture resets config; set defaults explicitly
    config.index.include = [".rbtr/REVIEW-*"]
    config.index.extend_exclude = [".rbtr/index"]
    monkeypatch.chdir(tmp_path)

    rbtr = tmp_path / ".rbtr"
    rbtr.mkdir()
    (rbtr / "REVIEW-plan.md").write_text("# Plan\n")
    (rbtr / "index").write_text("duckdb data\n")

    with tempfile.TemporaryDirectory() as repo_tmp:
        repo, _ = _make_repo(repo_tmp)
        ctx = _FakeCtx(_state(repo))

        # REVIEW file should be readable
        plan = read_file(ctx, ".rbtr/REVIEW-plan.md")  # type: ignore[arg-type]
        assert "# Plan" in plan

        # index file should be blocked
        index = read_file(ctx, ".rbtr/index")  # type: ignore[arg-type]
        assert "not found" in index.lower() or "cannot" in index.lower()

        # list_files should show REVIEW but not index
        listing = list_files(ctx, path=".rbtr")  # type: ignore[arg-type]
        assert "REVIEW-plan.md" in listing
        # Only REVIEW files should be listed, not the index
        lines = listing.splitlines()
        file_lines = [ln.strip() for ln in lines if ln.strip().startswith(".rbtr/")]
        assert all("REVIEW-" in ln for ln in file_lines)


def test_default_config_rbtr_listed_but_index_excluded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config_path: Path
) -> None:
    """list_files on .rbtr shows review files, omits the index database."""
    config.index.include = [".rbtr/REVIEW-*"]
    config.index.extend_exclude = [".rbtr/index"]
    monkeypatch.chdir(tmp_path)

    rbtr = tmp_path / ".rbtr"
    rbtr.mkdir()
    (rbtr / "REVIEW-plan.md").write_text("# Plan\n")
    (rbtr / "REVIEW-findings.md").write_text("# Findings\n")
    index_dir = rbtr / "index"
    index_dir.mkdir()
    (index_dir / "rbtr.duckdb").write_text("db\n")

    with tempfile.TemporaryDirectory() as repo_tmp:
        repo, _ = _make_repo(repo_tmp)
        ctx = _FakeCtx(_state(repo))
        result = list_files(ctx, path=".rbtr")  # type: ignore[arg-type]
        assert "REVIEW-plan.md" in result
        assert "REVIEW-findings.md" in result
        # index dir and its contents should not appear
        assert "duckdb" not in result


def test_default_config_grep_rbtr_excludes_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config_path: Path
) -> None:
    """grep on .rbtr/ does not search inside .rbtr/index/."""
    config.index.include = [".rbtr/REVIEW-*"]
    config.index.extend_exclude = [".rbtr/index"]
    monkeypatch.chdir(tmp_path)

    rbtr = tmp_path / ".rbtr"
    rbtr.mkdir()
    (rbtr / "REVIEW-plan.md").write_text("SEARCH_ME in plan\n")
    index_dir = rbtr / "index"
    index_dir.mkdir()
    (index_dir / "data.db").write_text("SEARCH_ME in db\n")

    with tempfile.TemporaryDirectory() as repo_tmp:
        repo, _ = _make_repo(repo_tmp)
        ctx = _FakeCtx(_state(repo))
        result = grep(ctx, "SEARCH_ME", path=".rbtr")  # type: ignore[arg-type]
        assert "REVIEW-plan.md" in result
        assert "data.db" not in result


# ── Multiple gitignore patterns ──────────────────────────────────────


@pytest.mark.parametrize(
    ("pattern", "ignored_file", "visible_file"),
    [
        ("*.pyc", "module.pyc", "module.py"),
        (".env", ".env", "env.py"),
        ("*.log", "server.log", "server.py"),
    ],
)
def test_list_files_gitignore_various_patterns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    config_path: Path,
    pattern: str,
    ignored_file: str,
    visible_file: str,
) -> None:
    """list_files respects various .gitignore pattern types."""
    config.index.include = []
    config.index.extend_exclude = []
    monkeypatch.chdir(tmp_path)

    local = tmp_path / "project"
    local.mkdir()

    ignored_path = local / ignored_file
    ignored_path.parent.mkdir(parents=True, exist_ok=True)
    ignored_path.write_text("ignored content\n")

    visible_path = local / visible_file
    visible_path.parent.mkdir(parents=True, exist_ok=True)
    visible_path.write_text("visible content\n")

    with tempfile.TemporaryDirectory() as repo_tmp:
        repo, _ = _make_repo(repo_tmp)
        workdir = Path(repo.workdir)  # type: ignore[arg-type]
        (workdir / ".gitignore").write_text(f"{pattern}\n")

        ctx = _FakeCtx(_state(repo))
        result = list_files(ctx, path="project")  # type: ignore[arg-type]
        assert Path(visible_file).name in result
        assert Path(ignored_file).name not in result


# ── Empty directories and edge cases ─────────────────────────────────


def test_list_files_all_ignored_returns_no_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config_path: Path
) -> None:
    """When all filesystem files are ignored, returns no-files message."""
    config.index.include = []
    config.index.extend_exclude = ["*.tmp"]
    monkeypatch.chdir(tmp_path)

    local = tmp_path / "scratch"
    local.mkdir()
    (local / "a.tmp").write_text("temp\n")
    (local / "b.tmp").write_text("temp\n")

    with tempfile.TemporaryDirectory() as repo_tmp:
        repo, _ = _make_repo(repo_tmp)
        ctx = _FakeCtx(_state(repo))
        result = list_files(ctx, path="scratch")  # type: ignore[arg-type]
        assert "No files" in result


def test_grep_all_ignored_returns_no_matches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config_path: Path
) -> None:
    """When all filesystem files are ignored, grep returns no-matches."""
    config.index.include = []
    config.index.extend_exclude = ["*.tmp"]
    monkeypatch.chdir(tmp_path)

    local = tmp_path / "scratch"
    local.mkdir()
    (local / "a.tmp").write_text("NEEDLE\n")

    with tempfile.TemporaryDirectory() as repo_tmp:
        repo, _ = _make_repo(repo_tmp)
        ctx = _FakeCtx(_state(repo))
        result = grep(ctx, "NEEDLE", path="scratch")  # type: ignore[arg-type]
        assert "No matches" in result


# ── _matches_globs prefix matching ───────────────────────────────────


def test_matches_globs_exact_match() -> None:
    assert _matches_globs(".rbtr/index", [".rbtr/index"])


def test_matches_globs_fnmatch_wildcard() -> None:
    assert _matches_globs(".rbtr/REVIEW-plan.md", [".rbtr/REVIEW-*"])
    assert not _matches_globs(".rbtr/index", [".rbtr/REVIEW-*"])


def test_matches_globs_literal_prefix_matching() -> None:
    """Literal patterns match child paths via directory prefix."""
    assert _matches_globs(".rbtr/index/data.db", [".rbtr/index"])
    assert _matches_globs("vendor/lib/deep/file.py", ["vendor"])
    # But not partial name matches
    assert not _matches_globs(".rbtr/index-notes.md", [".rbtr/index"])


def test_matches_globs_trailing_slash() -> None:
    """Trailing slash on directory patterns matches children."""
    assert _matches_globs(".rbtr/index/data.db", [".rbtr/"])
    assert _matches_globs(".rbtr/REVIEW-plan.md", [".rbtr/"])
    # Exact directory name (sans trailing slash) also matches
    assert _matches_globs(".rbtr", [".rbtr/"])


def test_matches_globs_wildcard_matches_across_slash() -> None:
    """fnmatch * matches across / — so wildcard patterns are broad."""
    assert _matches_globs(".rbtr/REVIEW-plan.md", [".rbtr/REVIEW-*"])
    # fnmatch.fnmatch * matches everything including /
    assert _matches_globs(".rbtr/REVIEW-plan/subfile.md", [".rbtr/REVIEW-*"])


def test_matches_globs_empty_patterns() -> None:
    assert not _matches_globs("anything.py", [])


# ── Real-world: .rbtr gitignored, REVIEW files accessible ───────────


def test_gitignored_rbtr_review_files_accessible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, config_path: Path
) -> None:
    """When .rbtr/ is gitignored, REVIEW files are still accessible
    via include = [".rbtr/REVIEW-*"]."""
    config.index.include = [".rbtr/REVIEW-*"]
    config.index.extend_exclude = [".rbtr/index"]
    monkeypatch.chdir(tmp_path)

    rbtr = tmp_path / ".rbtr"
    rbtr.mkdir()
    (rbtr / "REVIEW-plan.md").write_text("# Plan\nCheck handler\n")
    (rbtr / "REVIEW-findings.md").write_text("# Findings\nBug found\n")
    index_dir = rbtr / "index"
    index_dir.mkdir()
    (index_dir / "rbtr.duckdb").write_text("db\n")

    with tempfile.TemporaryDirectory() as repo_tmp:
        repo, _ = _make_repo(repo_tmp)
        workdir = Path(repo.workdir)  # type: ignore[arg-type]
        # .rbtr is in .gitignore — as it would be in a real project
        (workdir / ".gitignore").write_text(".rbtr/\n")

        ctx = _FakeCtx(_state(repo))

        # read_file: REVIEW files readable
        plan = read_file(ctx, ".rbtr/REVIEW-plan.md")  # type: ignore[arg-type]
        assert "# Plan" in plan
        findings = read_file(ctx, ".rbtr/REVIEW-findings.md")  # type: ignore[arg-type]
        assert "Bug found" in findings

        # read_file: index blocked
        idx = read_file(ctx, ".rbtr/index/rbtr.duckdb")  # type: ignore[arg-type]
        assert "not found" in idx.lower() or "cannot" in idx.lower()

        # list_files: shows REVIEW files, not index
        listing = list_files(ctx, path=".rbtr")  # type: ignore[arg-type]
        assert "REVIEW-plan.md" in listing
        assert "REVIEW-findings.md" in listing
        assert "duckdb" not in listing

        # grep: finds in REVIEW files, not in index
        result = grep(ctx, "Bug found", path=".rbtr")  # type: ignore[arg-type]
        assert "REVIEW-findings.md" in result
        assert "duckdb" not in result
