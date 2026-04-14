"""Tests for git object-store operations.

Tests are organised around a shared multi-commit repository
(`sample_repo` from conftest) that has adds, modifications,
deletions, and binary files across three commits.  Individual
tests verify *behaviours* against this dataset rather than
constructing throwaway repos for each assertion.

Functions under test:
- resolve_commit
- walk_tree / list_files
- read_blob
- diff_refs / diff_single
- commit_log_between
- changed_files
- is_binary / _matches_globs
"""

from __future__ import annotations

from pathlib import Path

import pygit2
import pytest

from rbtr_legacy.config import config
from rbtr_legacy.git import FileEntry, changed_files, is_binary, list_files
from rbtr_legacy.git.filters import _matches_globs
from rbtr_legacy.git.objects import (
    DiffLineRanges,
    DiffStats,
    commit_log_between,
    diff_refs,
    diff_single,
    nearest_commentable_line,
    patch_line_ranges,
    read_blob,
    resolve_anchor,
    resolve_commit,
    translate_line,
    walk_tree,
)

from .conftest import (
    BINARY_PNG,
    HANDLER_V1,
    README,
    UTILS,
    MergeRepo,
    SampleRepo,
    make_commit,
)

_MAX = 1_000_000  # default max_file_size for tests


def _lf(
    repo: pygit2.Repository,
    ref: str,
    *,
    max_file_size: int = _MAX,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
) -> list[FileEntry]:
    """Shorthand for `list(list_files(...))` with config defaults."""
    return list(
        list_files(
            repo,
            ref,
            max_file_size=max_file_size,
            include=include if include is not None else config.index.include,
            exclude=exclude if exclude is not None else config.index.extend_exclude,
        )
    )


# ── _matches_globs ───────────────────────────────────────────────────


def test_matches_globs_hit() -> None:
    assert _matches_globs("style.min.css", ["*.min.css", "*.map"])


def test_matches_globs_miss() -> None:
    assert not _matches_globs("style.css", ["*.min.css", "*.map"])


def test_matches_globs_empty_patterns() -> None:
    assert not _matches_globs("anything.py", [])


# ── is_binary ───────────────────────────────────────────────────────


def test_is_binary_with_null_byte() -> None:
    assert is_binary(b"hello\x00world")


def test_is_binary_text() -> None:
    assert not is_binary(b"hello world\n")


def test_is_binary_empty() -> None:
    assert not is_binary(b"")


def test_is_binary_null_beyond_sample() -> None:
    data = b"a" * 100 + b"\x00"
    assert not is_binary(data, sample_size=50)


# ── resolve_commit ───────────────────────────────────────────────────


def test_resolve_by_sha(sample_repo: SampleRepo) -> None:
    commit = resolve_commit(sample_repo.repo, str(sample_repo.base))
    assert commit.id == sample_repo.base


def test_resolve_by_branch_name(sample_repo: SampleRepo) -> None:
    commit = resolve_commit(sample_repo.repo, "feature")
    assert commit.id == sample_repo.head


def test_resolve_falls_back_to_origin(sample_repo: SampleRepo) -> None:
    """When a local branch doesn't exist, origin/<name> is tried."""
    repo = sample_repo.repo
    # Create a remote-only ref.
    repo.references.create("refs/remotes/origin/remote-only", sample_repo.mid)

    commit = resolve_commit(repo, "remote-only")
    assert commit.id == sample_repo.mid


def test_resolve_prefers_local_over_remote(sample_repo: SampleRepo) -> None:
    repo = sample_repo.repo
    repo.references.create("refs/remotes/origin/main", sample_repo.head)

    # "main" resolves to local (base), not origin (head).
    commit = resolve_commit(repo, "main")
    assert commit.id == sample_repo.base


def test_resolve_unknown_ref_raises(sample_repo: SampleRepo) -> None:
    with pytest.raises(KeyError, match="Cannot resolve ref"):
        resolve_commit(sample_repo.repo, "nonexistent")


# ── read_blob ────────────────────────────────────────────────────────


def test_read_blob_existing_file(sample_repo: SampleRepo) -> None:
    blob = read_blob(sample_repo.repo, str(sample_repo.base), "src/handler.py")
    assert blob is not None
    assert blob.data == HANDLER_V1


def test_read_blob_nested_path(sample_repo: SampleRepo) -> None:
    blob = read_blob(sample_repo.repo, str(sample_repo.base), "src/utils.py")
    assert blob is not None
    assert blob.data == UTILS


def test_read_blob_missing_file(sample_repo: SampleRepo) -> None:
    blob = read_blob(sample_repo.repo, str(sample_repo.base), "nonexistent.py")
    assert blob is None


def test_read_blob_missing_ref(sample_repo: SampleRepo) -> None:
    blob = read_blob(sample_repo.repo, "nonexistent-branch", "src/handler.py")
    assert blob is None


def test_read_blob_binary_file(sample_repo: SampleRepo) -> None:
    blob = read_blob(sample_repo.repo, "feature", "binary.png")
    assert blob is not None
    assert blob.data == BINARY_PNG


def test_read_blob_deleted_file(sample_repo: SampleRepo) -> None:
    """readme.md exists at base but is deleted at head."""
    assert read_blob(sample_repo.repo, str(sample_repo.base), "readme.md") is not None
    assert read_blob(sample_repo.repo, "feature", "readme.md") is None


def test_read_blob_by_branch_name(sample_repo: SampleRepo) -> None:
    blob = read_blob(sample_repo.repo, "main", "readme.md")
    assert blob is not None
    assert blob.data == README


# ── diff_refs ────────────────────────────────────────────────────────


def test_diff_refs_detects_changes(sample_repo: SampleRepo) -> None:
    """Diff base→head should show handler modified, config added, readme deleted."""
    result = diff_refs(sample_repo.repo, str(sample_repo.base), "feature")
    # 4 files: handler.py (mod), readme.md (del), config.yaml (add), binary.png (add)
    assert result.stats.files_changed == 4
    assert result.stats.insertions > 0 or result.stats.deletions > 0
    assert len(result.patch_lines) > 0


def test_diff_refs_stats_types(sample_repo: SampleRepo) -> None:
    result = diff_refs(sample_repo.repo, str(sample_repo.base), "feature")
    assert isinstance(result.stats, DiffStats)
    assert isinstance(result.stats.files_changed, int)
    assert isinstance(result.stats.insertions, int)
    assert isinstance(result.stats.deletions, int)


def test_diff_refs_identical(sample_repo: SampleRepo) -> None:
    """Diffing a ref against itself produces an empty diff."""
    result = diff_refs(sample_repo.repo, "main", "main")
    assert result.stats.files_changed == 0
    assert result.patch_lines == ["(empty diff)"]


def test_diff_refs_path_filter(sample_repo: SampleRepo) -> None:
    """Restrict diff to a single file."""
    result = diff_refs(sample_repo.repo, str(sample_repo.base), "feature", pattern="src/handler.py")
    assert result.stats.files_changed == 1
    # Patch should mention handler.py
    patch = "\n".join(result.patch_lines)
    assert "handler.py" in patch


def test_diff_refs_path_no_changes(sample_repo: SampleRepo) -> None:
    """Path filter for an unchanged file returns empty stats."""
    result = diff_refs(sample_repo.repo, str(sample_repo.base), "feature", pattern="src/utils.py")
    assert result.stats.files_changed == 0
    assert result.patch_lines == []


def test_diff_refs_path_nonexistent(sample_repo: SampleRepo) -> None:
    """Path filter for a file that doesn't exist in either ref."""
    result = diff_refs(sample_repo.repo, str(sample_repo.base), "feature", pattern="nope.py")
    assert result.stats.files_changed == 0
    assert result.patch_lines == []


def test_diff_refs_bad_ref_raises(sample_repo: SampleRepo) -> None:
    with pytest.raises(KeyError, match="Cannot resolve ref"):
        diff_refs(sample_repo.repo, "main", "nonexistent")


# ── diff_single ──────────────────────────────────────────────────────


def test_diff_single_against_parent(sample_repo: SampleRepo) -> None:
    """Mid commit modifies handler.py and adds config.yaml."""
    result = diff_single(sample_repo.repo, str(sample_repo.mid))
    assert result.stats.files_changed == 2
    patch = "\n".join(result.patch_lines)
    assert "handler.py" in patch
    assert "config.yaml" in patch


def test_diff_single_head_commit(sample_repo: SampleRepo) -> None:
    """Head commit deletes readme.md and adds binary.png."""
    result = diff_single(sample_repo.repo, str(sample_repo.head))
    assert result.stats.files_changed == 2
    patch = "\n".join(result.patch_lines)
    assert "readme.md" in patch


def test_diff_single_path_filter(sample_repo: SampleRepo) -> None:
    result = diff_single(sample_repo.repo, str(sample_repo.mid), pattern="src/handler.py")
    assert result.stats.files_changed == 1
    patch = "\n".join(result.patch_lines)
    assert "validate" in patch  # handler_v2 adds validate call


def test_diff_single_initial_commit_raises(sample_repo: SampleRepo) -> None:
    """The base commit has no parent — should raise ValueError."""
    with pytest.raises(ValueError, match="no parent"):
        diff_single(sample_repo.repo, str(sample_repo.base))


def test_diff_single_bad_ref_raises(sample_repo: SampleRepo) -> None:
    with pytest.raises(KeyError, match="Cannot resolve ref"):
        diff_single(sample_repo.repo, "nonexistent")


# ── commit_log_between ───────────────────────────────────────────────


def test_commit_log_returns_entries(sample_repo: SampleRepo) -> None:
    """Log from base to head should have 2 commits (mid + head)."""
    entries = commit_log_between(sample_repo.repo, "main", "feature")
    assert len(entries) == 2


def test_commit_log_order(sample_repo: SampleRepo) -> None:
    """Entries are in reverse chronological (topological) order."""
    entries = commit_log_between(sample_repo.repo, "main", "feature")
    # head is newer, so it comes first
    assert entries[0].sha == str(sample_repo.head)
    assert entries[1].sha == str(sample_repo.mid)


def test_commit_log_full_sha(sample_repo: SampleRepo) -> None:
    """sha field contains the full 40-char hex SHA."""
    entries = commit_log_between(sample_repo.repo, "main", "feature")
    for entry in entries:
        assert len(entry.sha) == 40
        int(entry.sha, 16)  # valid hex


def test_commit_log_authors(sample_repo: SampleRepo) -> None:
    entries = commit_log_between(sample_repo.repo, "main", "feature")
    authors = {e.author for e in entries}
    assert authors == {"Alice", "Bob"}


def test_commit_log_messages(sample_repo: SampleRepo) -> None:
    entries = commit_log_between(sample_repo.repo, "main", "feature")
    messages = {e.message for e in entries}
    assert "Remove readme, add image" in messages
    assert "Add validation and config" in messages


def test_commit_log_identical_refs(sample_repo: SampleRepo) -> None:
    """Same ref on both sides → empty list."""
    entries = commit_log_between(sample_repo.repo, "main", "main")
    assert entries == []


def test_commit_log_bad_ref_raises(sample_repo: SampleRepo) -> None:
    with pytest.raises(KeyError, match="Cannot resolve ref"):
        commit_log_between(sample_repo.repo, "main", "nonexistent")


# ── Merge topology — all two-ref functions ───────────────────────────
#
# The merge_repo fixture (see conftest) has this graph:
#
#     A ─── B ─── C  (base)
#            \
#             D ─── E ── F  (head, merge of E + C)
#
# These tests verify that every function operating on a
# (base, head) pair produces correct results when the head
# branch contains merge commits.  The commit_log_between
# regression (manual-break walk) was caught by this suite.


def test_merge_commit_log_excludes_base_history(merge_repo: MergeRepo) -> None:
    """commit_log base..head returns only D, E, F — not A, B, C."""
    entries = commit_log_between(merge_repo.repo, "base", "head")
    shas = {e.sha for e in entries}
    messages = {e.message for e in entries}

    assert len(entries) == 3
    assert messages == {"D", "E", "F"}
    for shared_oid in merge_repo.shared:
        assert str(shared_oid) not in shas


def test_merge_changed_files_only_pr_changes(merge_repo: MergeRepo) -> None:
    """changed_files base..head returns only `feature.py`."""
    paths = changed_files(merge_repo.repo, "base", "head")
    assert paths == {"feature.py"}


def test_merge_diff_refs_only_pr_changes(merge_repo: MergeRepo) -> None:
    """diff_refs base..head shows only the added `feature.py`."""
    result = diff_refs(merge_repo.repo, "base", "head")
    assert result.stats.files_changed == 1
    patch = "\n".join(result.patch_lines)
    assert "feature.py" in patch


def test_merge_diff_refs_path_unchanged_file(merge_repo: MergeRepo) -> None:
    """Diffing a file that exists on both sides but is identical."""
    result = diff_refs(merge_repo.repo, "base", "head", pattern="app.py")
    assert result.stats.files_changed == 0
    assert result.patch_lines == []


# ── list_files ───────────────────────────────────────────────────────


def test_list_files_at_base(sample_repo: SampleRepo, config_path: Path) -> None:
    entries = _lf(sample_repo.repo, "main")
    paths = {e.path for e in entries}
    assert paths == {"src/handler.py", "src/utils.py", "readme.md"}


def test_list_files_at_head(sample_repo: SampleRepo, config_path: Path) -> None:
    """Head has no readme.md but has config.yaml.  binary.png is skipped."""
    entries = _lf(sample_repo.repo, "feature")
    paths = {e.path for e in entries}
    assert paths == {"src/handler.py", "src/utils.py", "config.yaml"}
    assert "binary.png" not in paths  # binary → skipped
    assert "readme.md" not in paths  # deleted


def test_list_files_returns_file_entries(sample_repo: SampleRepo, config_path: Path) -> None:
    entries = _lf(sample_repo.repo, "main")
    assert all(isinstance(e, FileEntry) for e in entries)
    handler = next(e for e in entries if e.path == "src/handler.py")
    assert handler.content == HANDLER_V1
    assert handler.blob_sha  # non-empty


def test_list_files_skips_binary(sample_repo: SampleRepo, config_path: Path) -> None:
    entries = _lf(sample_repo.repo, "feature")
    paths = {e.path for e in entries}
    assert "binary.png" not in paths


def test_list_files_max_file_size(sample_repo: SampleRepo, config_path: Path) -> None:
    entries = _lf(sample_repo.repo, "main", max_file_size=10)
    # All sample files are > 10 bytes
    assert entries == []


def test_list_files_extend_exclude(sample_repo: SampleRepo, config_path: Path) -> None:
    entries = _lf(sample_repo.repo, "feature", exclude=["*.yaml"])
    paths = {e.path for e in entries}
    assert "config.yaml" not in paths


def test_list_files_include_overrides_extend_exclude(
    sample_repo: SampleRepo, config_path: Path
) -> None:
    entries = _lf(sample_repo.repo, "feature", include=["config.yaml"], exclude=["*.yaml"])
    paths = {e.path for e in entries}
    assert "config.yaml" in paths


def test_list_files_gitignore(sample_repo: SampleRepo, config_path: Path) -> None:
    workdir = Path(sample_repo.repo.workdir)  # type: ignore[arg-type]  # pygit2 stubs
    (workdir / ".gitignore").write_text("config.yaml\n")

    entries = _lf(sample_repo.repo, "feature", exclude=[])
    paths = {e.path for e in entries}
    assert "config.yaml" not in paths


def test_list_files_unresolvable_ref_raises(sample_repo: SampleRepo, config_path: Path) -> None:
    with pytest.raises(KeyError, match="Cannot resolve ref"):
        _lf(sample_repo.repo, "nonexistent-branch")


def test_list_files_remote_branch(sample_repo: SampleRepo, config_path: Path) -> None:
    """Resolves origin/<name> when local branch doesn't exist."""
    repo = sample_repo.repo
    repo.references.create("refs/remotes/origin/remote-only", sample_repo.mid)

    entries = _lf(repo, "remote-only")
    paths = {e.path for e in entries}
    assert "config.yaml" in paths  # added at mid


# ── changed_files ────────────────────────────────────────────────────


def test_changed_files_base_to_head(sample_repo: SampleRepo) -> None:
    paths = changed_files(sample_repo.repo, "main", "feature")
    # handler.py (modified), readme.md (deleted), config.yaml (added), binary.png (added)
    assert paths == {"src/handler.py", "readme.md", "config.yaml", "binary.png"}


def test_changed_files_base_to_mid(sample_repo: SampleRepo) -> None:
    paths = changed_files(sample_repo.repo, str(sample_repo.base), str(sample_repo.mid))
    assert paths == {"src/handler.py", "config.yaml"}


def test_changed_files_identical(sample_repo: SampleRepo) -> None:
    paths = changed_files(sample_repo.repo, "main", "main")
    assert paths == set()


def test_changed_files_with_remote_branch(sample_repo: SampleRepo) -> None:
    repo = sample_repo.repo
    repo.references.create("refs/remotes/origin/pr-head", sample_repo.head)
    paths = changed_files(repo, "main", "pr-head")
    assert "src/handler.py" in paths
    assert "readme.md" in paths


# ── walk_tree ────────────────────────────────────────────────────────


def test_walk_tree_yields_all_blobs(sample_repo: SampleRepo) -> None:
    commit = resolve_commit(sample_repo.repo, "main")
    pairs = list(walk_tree(sample_repo.repo, commit.tree, ""))
    paths = {p for p, _ in pairs}
    assert paths == {"src/handler.py", "src/utils.py", "readme.md"}


def test_walk_tree_blobs_are_readable(sample_repo: SampleRepo) -> None:
    commit = resolve_commit(sample_repo.repo, "main")
    for _path, blob in walk_tree(sample_repo.repo, commit.tree, ""):
        assert isinstance(blob, pygit2.Blob)
        assert len(blob.data) > 0


# ── Edge cases with standalone repos ─────────────────────────────────
# Some scenarios are hard to represent in the shared dataset
# (e.g. include overriding gitignore for nested dirs).


@pytest.fixture
def git_repo(tmp_path: Path) -> pygit2.Repository:
    """A bare repo for one-off edge case tests."""
    return pygit2.init_repository(str(tmp_path / "edge"))


def test_list_files_include_overrides_gitignore_nested(
    git_repo: pygit2.Repository, config_path: Path
) -> None:
    workdir = Path(git_repo.workdir)  # type: ignore[arg-type]  # pygit2 stubs
    (workdir / ".gitignore").write_text("vendor/\n")

    oid = make_commit(
        git_repo,
        {
            "app.py": b"x = 1\n",
            "vendor/important.py": b"keep\n",
            "vendor/junk.py": b"skip\n",
        },
    )
    paths = {e.path for e in _lf(git_repo, str(oid), include=["vendor/important.py"], exclude=[])}
    assert paths == {"app.py", "vendor/important.py"}


def test_list_files_include_still_skips_binary(
    git_repo: pygit2.Repository, config_path: Path
) -> None:
    oid = make_commit(git_repo, {"image.png": b"\x89PNG\r\n\x1a\n\x00\x00"})
    paths = {e.path for e in _lf(git_repo, str(oid), include=["image.png"])}
    assert paths == set()


def test_list_files_include_still_skips_oversized(
    git_repo: pygit2.Repository, config_path: Path
) -> None:
    oid = make_commit(git_repo, {"big.py": b"x" * 200})
    paths = {e.path for e in _lf(git_repo, str(oid), max_file_size=50, include=["big.py"])}
    assert paths == set()


# ── resolve_anchor ───────────────────────────────────────────────────

# sample_repo handler.py at head (HANDLER_V2):
#   line 1: def handle(request):
#   line 2:     validate(request)
#   line 3:     return "ok"


class TestResolveAnchor:
    """Tests for resolve_anchor using the shared sample_repo fixture."""

    @pytest.mark.parametrize(
        ("anchor", "expected_line"),
        [
            ("def handle(request):", 1),
            ("validate(request)", 2),
            ('return "ok"', 3),
            # Multi-line anchor — last line of match.
            ("def handle(request):\n    validate(request)", 2),
            ("validate(request)\n    return", 3),
        ],
    )
    def test_found(self, sample_repo: SampleRepo, anchor: str, expected_line: int) -> None:
        result = resolve_anchor(sample_repo.repo, str(sample_repo.head), "src/handler.py", anchor)
        assert result == expected_line

    @pytest.mark.parametrize(
        ("anchor", "error_fragment"),
        [
            ("nonexistent_function", "not found"),
            ("", "empty"),
        ],
    )
    def test_error(self, sample_repo: SampleRepo, anchor: str, error_fragment: str) -> None:
        result = resolve_anchor(sample_repo.repo, str(sample_repo.head), "src/handler.py", anchor)
        assert isinstance(result, str)
        assert error_fragment in result.lower()

    def test_multiple_matches(self, sample_repo: SampleRepo) -> None:
        # "request" appears on multiple lines in handler.py.
        result = resolve_anchor(
            sample_repo.repo, str(sample_repo.head), "src/handler.py", "request"
        )
        assert isinstance(result, str)
        assert "matches" in result.lower()

    def test_file_not_found(self, sample_repo: SampleRepo) -> None:
        result = resolve_anchor(sample_repo.repo, str(sample_repo.head), "no/such/file.py", "x")
        assert isinstance(result, str)
        assert "not found" in result.lower()

    def test_binary_file(self, sample_repo: SampleRepo) -> None:
        result = resolve_anchor(sample_repo.repo, str(sample_repo.head), "binary.png", "PNG")
        assert isinstance(result, str)
        assert "binary" in result.lower()

    def test_base_ref(self, sample_repo: SampleRepo) -> None:
        # readme.md exists at base but not at head.
        result = resolve_anchor(sample_repo.repo, str(sample_repo.base), "readme.md", "# Project")
        assert result == 1

    def test_base_ref_deleted_at_head(self, sample_repo: SampleRepo) -> None:
        # readme.md was deleted at head.
        result = resolve_anchor(sample_repo.repo, str(sample_repo.head), "readme.md", "# Project")
        assert isinstance(result, str)
        assert "not found" in result.lower()


# ── translate_line ───────────────────────────────────────────────────

# sample_repo handler.py base→mid:
#   base: line 1: def handle(request):  line 2: return "ok"
#   mid:  line 1: def handle(request):  line 2: validate(request)  line 3: return "ok"
#   → line 2 inserted; old line 1 stays, old line 2 shifts to 3.


class TestTranslateLine:
    """Tests for translate_line using the shared sample_repo fixture."""

    @pytest.mark.parametrize(
        ("old_line", "expected"),
        [
            (1, 1),  # before insertion — unchanged
            (2, 3),  # after insertion — shifted down by 1
        ],
    )
    def test_insertion(self, sample_repo: SampleRepo, old_line: int, expected: int) -> None:
        result = translate_line(
            sample_repo.repo,
            "src/handler.py",
            str(sample_repo.base),
            str(sample_repo.mid),
            old_line,
        )
        assert result == expected

    def test_file_unchanged(self, sample_repo: SampleRepo) -> None:
        # utils.py is identical across all commits.
        result = translate_line(
            sample_repo.repo,
            "src/utils.py",
            str(sample_repo.base),
            str(sample_repo.head),
            1,
        )
        assert result == 1

    def test_file_deleted(self, sample_repo: SampleRepo) -> None:
        # readme.md exists at base, deleted at head.
        result = translate_line(
            sample_repo.repo,
            "readme.md",
            str(sample_repo.base),
            str(sample_repo.head),
            1,
        )
        assert result is None

    def test_file_not_in_diff(self, sample_repo: SampleRepo) -> None:
        # File doesn't exist in either ref — identity (no patch).
        result = translate_line(
            sample_repo.repo,
            "no/such/file.py",
            str(sample_repo.base),
            str(sample_repo.mid),
            5,
        )
        assert result == 5

    def test_reverse_direction(self, sample_repo: SampleRepo) -> None:
        # mid→base: the insertion becomes a deletion.
        # mid line 2 (validate) was added, so translating mid line 2
        # backwards: it's a new line that doesn't exist in base → None.
        result = translate_line(
            sample_repo.repo,
            "src/handler.py",
            str(sample_repo.mid),
            str(sample_repo.base),
            2,
        )
        assert result is None

    def test_reverse_shifted_line(self, sample_repo: SampleRepo) -> None:
        # mid→base: mid line 3 ("return ok") was old line 2.
        result = translate_line(
            sample_repo.repo,
            "src/handler.py",
            str(sample_repo.mid),
            str(sample_repo.base),
            3,
        )
        assert result == 2


# ── nearest_commentable_line ─────────────────────────────────────────


_RANGES: DiffLineRanges = {"a.py": {5, 10, 20}, "empty.py": set()}


@pytest.mark.parametrize(
    ("path", "line", "expected"),
    [
        # Exact match.
        ("a.py", 10, 10),
        # Nearest below.
        ("a.py", 7, 5),
        # Nearest above.
        ("a.py", 12, 10),
        # Equidistant — both 10 and 20 are distance 5.
        ("a.py", 15, {10, 20}),
        # Far away — still returns nearest.
        ("a.py", 999, 20),
        # File not in ranges.
        ("b.py", 5, None),
        # File with empty line set.
        ("empty.py", 1, None),
    ],
    ids=[
        "exact",
        "nearest-below",
        "nearest-above",
        "equidistant",
        "far",
        "missing-file",
        "empty-lines",
    ],
)
def test_nearest_commentable_line(path: str, line: int, expected: int | set[int] | None) -> None:
    result = nearest_commentable_line(_RANGES, path, line)
    if isinstance(expected, set):
        assert result in expected
    else:
        assert result == expected


def test_nearest_commentable_line_empty_ranges() -> None:
    """Empty ranges dict returns `None` for any file."""
    assert nearest_commentable_line({}, "a.py", 5) is None


# ── patch_line_ranges ────────────────────────────────────────────────


def test_patch_line_ranges_includes_context() -> None:
    """Context lines are commentable on both sides, just like changed lines."""
    patch = "\n".join(
        [
            "@@ -10,3 +10,3 @@",
            " context",
            "-old_call()",
            "+new_call()",
            " context2",
        ]
    )

    right, left = patch_line_ranges(patch)

    # Changed lines.
    assert 11 in right
    assert 11 in left
    # Context lines — commentable on both sides.
    assert 10 in right
    assert 10 in left
    assert 12 in right
    assert 12 in left


def test_patch_line_ranges_empty_patch() -> None:
    assert patch_line_ranges("") == (set(), set())


def test_patch_line_ranges_addition_only() -> None:
    """Pure addition — only right-side lines."""
    patch = "@@ -5,0 +6,2 @@\n+new1\n+new2"
    right, left = patch_line_ranges(patch)
    assert right == {6, 7}
    assert left == set()


def test_patch_line_ranges_deletion_only() -> None:
    """Pure deletion — only left-side lines."""
    patch = "@@ -10,2 +9,0 @@\n-old1\n-old2"
    right, left = patch_line_ranges(patch)
    assert right == set()
    assert left == {10, 11}
