"""File tools — read_file, list_files, grep."""

from __future__ import annotations

from pathlib import Path, PurePosixPath

import pygit2
from pydantic_ai import RunContext

from rbtr.config import config
from rbtr.git import is_binary, is_path_ignored, resolve_commit, walk_tree
from rbtr.git.objects import read_blob
from rbtr.llm.deps import AgentDeps
from rbtr.llm.tools.common import get_repo, limited, repo_toolset, resolve_tool_ref, validate_path


def _read_fs_file(path: str) -> tuple[list[str], str | None]:
    """Read a file from the local filesystem.

    Returns `(lines, None)` on success, or `([], error_msg)`
    on failure.
    """
    p = Path(path)
    if not p.exists():
        return [], f"File '{path}' not found."
    try:
        data = p.read_bytes()
    except OSError as exc:
        return [], f"Cannot read '{path}': {exc}"
    if is_binary(data):
        return [], f"File '{path}' is binary — cannot display."
    return data.decode(errors="replace").splitlines(), None


def _list_fs_files(prefix: str, repo: pygit2.Repository | None = None) -> list[str]:
    """List files on the local filesystem matching *prefix*.

    *prefix* is treated as a directory path.  Returns sorted
    relative paths for all regular files under that directory.
    Respects `.gitignore`, `include`, and `extend_exclude`
    via :func:`is_path_ignored`.
    """
    base = Path(prefix) if prefix else Path(".")
    if not base.is_dir():
        return []
    entries: list[str] = []
    for p in sorted(base.rglob("*")):
        if not p.is_file():
            continue
        rel = str(PurePosixPath(p))
        if is_path_ignored(
            rel, repo, include=config.index.include, exclude=config.index.extend_exclude
        ):
            continue
        entries.append(rel)
    return entries


def _read_blob(repo: pygit2.Repository, ref: str, path: str) -> pygit2.Blob | str:
    """Return the blob for *path* at *ref*, or an error string."""
    blob = read_blob(repo, ref, path)
    if blob is None:
        return f"File '{path}' not found at ref '{ref}'."
    return blob


def _number_lines(lines: list[str], start: int) -> str:
    """Format *lines* with right-aligned line numbers starting at *start*."""
    end = start + len(lines)
    width = len(str(end))
    return "\n".join(f"{start + i:{width}d}│ {line}" for i, line in enumerate(lines))


def _format_file_page(path: str, all_lines: list[str], offset: int, max_lines: int) -> str:
    """Shared formatter for read_file — produces numbered output with pagination hint."""
    total = len(all_lines)
    selected = all_lines[offset : offset + max_lines]
    line_start = offset + 1  # 1-indexed display

    header = f"# {path}  (lines {line_start}-{line_start + len(selected) - 1} of {total})"
    body = _number_lines(selected, line_start)
    output = f"{header}\n{body}"
    shown_end = offset + len(selected)
    if shown_end < total:
        output += limited(shown_end, total, hint=f"offset={shown_end} to continue")
    return output


@repo_toolset.tool
def read_file(
    ctx: RunContext[AgentDeps],
    path: str,
    ref: str = "head",
    offset: int = 0,
    max_lines: int | None = None,
) -> str:
    """Read file content by path.

    Args:
        path: File path relative to repo root
            (e.g. `src/main.py`, `.rbtr/notes/plan.md`).
        ref: Which version of the codebase to read
            (defaults to `"head"`).
        offset: Number of lines to skip (0-indexed, default 0).
        max_lines: Maximum number of lines to return
            (defaults to `tools.max_lines` config value).
    """
    if err := validate_path(path):
        return err

    capped = (
        min(max_lines, config.tools.max_lines) if max_lines is not None else config.tools.max_lines
    )

    # Try git object store first.
    repo = get_repo(ctx)
    resolved = resolve_tool_ref(ctx, ref)
    blob_result = _read_blob(repo, resolved, path)
    if not isinstance(blob_result, str):
        data: bytes = blob_result.data
        if is_binary(data):
            return f"File '{path}' is binary — cannot display."
        return _format_file_page(path, data.decode(errors="replace").splitlines(), offset, capped)

    # Fall back to local filesystem.
    if is_path_ignored(
        path, repo, include=config.index.include, exclude=config.index.extend_exclude
    ):
        return blob_result  # treat ignored paths as not found
    fs_lines, fs_err = _read_fs_file(path)
    if fs_err:
        # Return the original git error — it's more informative.
        return blob_result
    return _format_file_page(path, fs_lines, offset, capped)


@repo_toolset.tool
def list_files(
    ctx: RunContext[AgentDeps],
    path: str = "",
    ref: str = "head",
    offset: int = 0,
    max_results: int | None = None,
) -> str:
    """List files in the repository or a subdirectory.

    Args:
        path: Directory prefix to scope the listing
            (e.g. `src/api`, `.rbtr/`).
            Empty string (default) lists from the repo root.
        ref: Which version of the codebase to read
            (defaults to `"head"`).
        offset: Number of entries to skip (default 0).
        max_results: Maximum entries to return per call
            (defaults to `tools.max_results` config value).
    """
    limit = (
        min(max_results, config.tools.max_results)
        if max_results is not None
        else config.tools.max_results
    )

    # Try git tree first.
    repo = get_repo(ctx)
    resolved = resolve_tool_ref(ctx, ref)
    try:
        commit = resolve_commit(repo, resolved)
    except KeyError:
        return f"Ref '{ref}' not found."

    prefix = path.rstrip("/")
    git_entries: list[str] = []
    for entry_path, _blob in walk_tree(repo, commit.tree, ""):
        if prefix and not entry_path.startswith(prefix + "/") and entry_path != prefix:
            continue
        git_entries.append(entry_path)

    if git_entries:
        git_entries.sort()
        return _format_file_list(git_entries, offset, limit)

    # Fall back to local filesystem when path is set.
    if path:
        fs_entries = _list_fs_files(path, repo=repo)
        if fs_entries:
            return _format_file_list(fs_entries, offset, limit)

    return f"No files found under '{path}'." if path else "No files in repository."


def _format_file_list(entries: list[str], offset: int, limit: int) -> str:
    """Format a paginated file listing."""
    total = len(entries)
    page = entries[offset : offset + limit]
    if not page:
        return f"Offset {offset} exceeds {total} files."
    header = f"Files ({total}):"
    listing = "\n".join(f"  {e}" for e in page)
    result = f"{header}\n{listing}"
    shown = offset + len(page)
    if shown < total:
        result += limited(shown, total, hint=f"offset={shown} to continue")
    return result


@repo_toolset.tool
def grep(
    ctx: RunContext[AgentDeps],
    search: str | int | float,
    path: str = "",
    ref: str = "head",
    offset: int = 0,
    max_hits: int | None = None,
    context_lines: int | None = None,
) -> str:
    """Search for a substring in one file or across the repository.

    Case-insensitive.  Binary files are skipped silently.

    Files not found in the git tree are looked up on the local
    filesystem as a fallback (e.g. `.rbtr/notes/`).

    Args:
        search: Substring to find.  Case-insensitive — `"config"`
            matches `Config`, `CONFIG`, `config`.
        path: File path or directory prefix relative to repo root
            (e.g. `src/api/handler.py`, `.rbtr/`, `""`).
            Empty string (default) searches all repo files.
        ref: Which version of the codebase to read
            (defaults to `"head"`).
        offset: Number of match groups to skip (default 0).
        max_hits: Maximum match groups to return per call
            (defaults to `tools.max_grep_hits` config value).
        context_lines: Number of lines to show above and below
            each match.
    """
    search = str(search)  # coerce non-string args (e.g. model sends a number)
    if path and (err := validate_path(path)):
        return err

    ctx_n = context_lines if context_lines is not None else config.tools.grep_context_lines
    needle = search.lower()
    capped_hits = (
        min(max_hits, config.tools.max_grep_hits)
        if max_hits is not None
        else config.tools.max_grep_hits
    )

    repo = get_repo(ctx)
    resolved = resolve_tool_ref(ctx, ref)

    if path:
        # Try as an exact file in git first.
        blob_result = _read_blob(repo, resolved, path)
        if not isinstance(blob_result, str):
            return _grep_blob(blob_result, path, needle, ctx_n, offset, capped_hits)

        # Not an exact git file — check whether any git files match the prefix.
        try:
            commit = resolve_commit(repo, resolved)
        except KeyError:
            return f"Ref '{ref}' not found."
        norm_prefix = path.rstrip("/")
        has_git_files = any(
            ep.startswith(norm_prefix + "/") or ep == norm_prefix
            for ep, _ in walk_tree(repo, commit.tree, "")
        )
        if has_git_files:
            return _grep_tree(repo, commit, path, needle, ctx_n, offset, capped_hits)

        # No git files under prefix — fall back to local filesystem.
        return _grep_filesystem(path, needle, ctx_n, offset, capped_hits, repo=repo)
    else:
        # Repo-wide search — git only.
        try:
            commit = resolve_commit(repo, resolved)
        except KeyError:
            return f"Ref '{ref}' not found."
        return _grep_tree(repo, commit, "", needle, ctx_n, offset, capped_hits)


def _grep_lines(
    all_lines: list[str], path: str, needle: str, ctx_n: int, offset: int, max_hits: int
) -> str:
    """Search *all_lines* for *needle*, returning formatted matches with context."""
    total = len(all_lines)
    match_indices = [i for i, line in enumerate(all_lines) if needle in line.lower()]
    if not match_indices:
        return f"No matches for '{needle}' in '{path}'."

    # Build merged context regions.
    regions: list[tuple[int, int]] = []
    for idx in match_indices:
        region_start = max(idx - ctx_n, 0)
        region_end = min(idx + ctx_n + 1, total)
        if regions and region_start <= regions[-1][1]:
            regions[-1] = (regions[-1][0], region_end)
        else:
            regions.append((region_start, region_end))

    # Paginate by match group (region).
    total_groups = len(regions)
    page = regions[offset : offset + max_hits]
    if not page:
        return f"Offset {offset} exceeds {total_groups} match groups."

    n_matches = len(match_indices)
    header = f"# {path}  ({n_matches} match{'es' if n_matches != 1 else ''})"
    sections: list[str] = [header]
    for region_start, region_end in page:
        sections.append(_number_lines(all_lines[region_start:region_end], region_start + 1))

    result = "\n\n".join(sections)
    shown = offset + len(page)
    if shown < total_groups:
        result += limited(shown, total_groups, hint=f"offset={shown} to see more match groups")
    return result


def _grep_blob(
    blob: pygit2.Blob, path: str, needle: str, ctx_n: int, offset: int, max_hits: int
) -> str:
    """Search a single blob for *needle*, returning formatted matches."""
    data: bytes = blob.data
    if is_binary(data):
        return f"File '{path}' is binary — cannot search."
    return _grep_lines(
        data.decode(errors="replace").splitlines(), path, needle, ctx_n, offset, max_hits
    )


def _grep_filesystem(
    path: str,
    needle: str,
    ctx_n: int,
    offset: int,
    max_hits: int,
    repo: pygit2.Repository | None = None,
) -> str:
    """Search local filesystem files under *path* for *needle*.

    Respects `.gitignore`, `include`, and `extend_exclude`
    via :func:`is_path_ignored`.
    """
    # Check if path is an exact file first.
    p = Path(path)
    if p.is_file():
        file_lines, err = _read_fs_file(path)
        if err:
            return err
        return _grep_lines(file_lines, path, needle, ctx_n, offset, max_hits)

    # Otherwise treat as a directory prefix.
    files = _list_fs_files(path, repo=repo)
    if not files:
        return f"No matches for '{needle}' under '{path}'."

    # Directory — search all files under prefix.
    all_groups: list[tuple[str, list[str], list[tuple[int, int]], list[int]]] = []
    total_matches = 0
    for fpath in files:
        file_lines, err = _read_fs_file(fpath)
        if err:
            continue
        match_indices = [i for i, line in enumerate(file_lines) if needle in line.lower()]
        if not match_indices:
            continue
        total_matches += len(match_indices)
        total_file = len(file_lines)
        regions: list[tuple[int, int]] = []
        for idx in match_indices:
            rs = max(idx - ctx_n, 0)
            re_ = min(idx + ctx_n + 1, total_file)
            if regions and rs <= regions[-1][1]:
                regions[-1] = (regions[-1][0], re_)
            else:
                regions.append((rs, re_))
        all_groups.append((fpath, file_lines, regions, match_indices))

    if not all_groups:
        return f"No matches for '{needle}' under '{path}'."

    # Flatten and paginate — same as _grep_tree.
    flat: list[tuple[str, list[str], tuple[int, int]]] = []
    for fpath, file_lines, regions, _indices in all_groups:
        for region in regions:
            flat.append((fpath, file_lines, region))

    total_groups = len(flat)
    page = flat[offset : offset + max_hits]
    if not page:
        return f"Offset {offset} exceeds {total_groups} match groups."

    result_header = (
        f"Found {total_matches} match{'es' if total_matches != 1 else ''}"
        f" in {len(all_groups)} file{'s' if len(all_groups) != 1 else ''}"
    )
    file_sections: list[str] = []
    current_file: str | None = None
    current_parts: list[str] = []
    for fpath, file_lines, (rstart, rend) in page:
        if fpath != current_file:
            if current_parts:
                file_sections.append("\n\n".join(current_parts))
            current_parts = [f"# {fpath}"]
            current_file = fpath
        current_parts.append(_number_lines(file_lines[rstart:rend], rstart + 1))
    if current_parts:
        file_sections.append("\n\n".join(current_parts))

    body = "\n\n".join(file_sections)
    result = f"{result_header}\n\n{body}"
    shown = offset + len(page)
    if shown < total_groups:
        result += limited(shown, total_groups, hint=f"offset={shown} to see more match groups")
    return result


def _grep_tree(
    repo: pygit2.Repository,
    commit: pygit2.Commit,
    prefix: str,
    needle: str,
    ctx_n: int,
    offset: int,
    max_hits: int,
) -> str:
    """Search all text files under *prefix* in the tree for *needle*."""
    max_file_size = config.index.max_file_size
    norm_prefix = prefix.rstrip("/")

    # Collect all match groups across all files.
    all_groups: list[tuple[str, list[str], list[tuple[int, int]], list[int]]] = []
    total_matches = 0

    for entry_path, blob in walk_tree(repo, commit.tree, ""):
        # Scope to prefix.
        if (
            norm_prefix
            and not entry_path.startswith(norm_prefix + "/")
            and entry_path != norm_prefix
        ):
            continue

        # Skip binary and oversized files.
        data: bytes = blob.data
        if len(data) > max_file_size or is_binary(data):
            continue

        file_lines = data.decode(errors="replace").splitlines()
        total_file = len(file_lines)
        match_indices = [i for i, line in enumerate(file_lines) if needle in line.lower()]
        if not match_indices:
            continue

        total_matches += len(match_indices)

        # Build merged context regions.
        regions: list[tuple[int, int]] = []
        for idx in match_indices:
            region_start = max(idx - ctx_n, 0)
            region_end = min(idx + ctx_n + 1, total_file)
            if regions and region_start <= regions[-1][1]:
                regions[-1] = (regions[-1][0], region_end)
            else:
                regions.append((region_start, region_end))

        all_groups.append((entry_path, file_lines, regions, match_indices))

    if not all_groups:
        scope = f" under '{prefix}'" if prefix else ""
        return f"No matches for '{needle}'{scope}."

    # Flatten to (file, region) pairs for pagination.
    flat: list[tuple[str, list[str], tuple[int, int]]] = []
    for entry_path, file_lines, regions, _indices in all_groups:
        for region in regions:
            flat.append((entry_path, file_lines, region))

    total_groups = len(flat)
    page = flat[offset : offset + max_hits]
    if not page:
        return f"Offset {offset} exceeds {total_groups} match groups."

    result_header = (
        f"Found {total_matches} match{'es' if total_matches != 1 else ''}"
        f" in {len(all_groups)} file{'s' if len(all_groups) != 1 else ''}"
    )

    # Group page entries by file for readable output.
    file_sections: list[str] = []
    current_file: str | None = None
    current_parts: list[str] = []
    for entry_path, file_lines, (rstart, rend) in page:
        if entry_path != current_file:
            if current_parts:
                file_sections.append("\n\n".join(current_parts))
            current_parts = [f"# {entry_path}"]
            current_file = entry_path
        current_parts.append(_number_lines(file_lines[rstart:rend], rstart + 1))
    if current_parts:
        file_sections.append("\n\n".join(current_parts))

    body = "\n\n".join(file_sections)
    result = f"{result_header}\n\n{body}"
    shown = offset + len(page)
    if shown < total_groups:
        result += limited(shown, total_groups, hint=f"offset={shown} to see more match groups")
    return result
