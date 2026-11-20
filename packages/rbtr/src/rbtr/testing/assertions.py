"""The checks language tests make the same way: what reaches a symbol, that every line is in one, and that a span's kinds are decided.

A failing check is reported at the test's line, with pytest's diff:
`rbtr.testing` registers this module for assertion rewriting, and each check
hides its own frame (`__tracebackhide__`).
"""

from __future__ import annotations

from collections.abc import Collection, Mapping

from rbtr.daemon.handlers import handle_find_refs
from rbtr.daemon.messages import FindRefsRequest
from rbtr.domain.models import EdgeKind
from rbtr.testing import IndexBuild
from rbtr.testing.reads import committed_files, outline, read, repo_path

type Target = tuple[str, str]
"""A symbol by its file and name: `("override/order.go", "Save")`."""

type Referrer = tuple[str, str, str]
"""A chunk that reaches a target, and by which edge: `("override/special.go", "Run", "calls")`."""


def assert_reached(
    build: IndexBuild,
    reached: Mapping[Target, set[Referrer]],
    edges: Collection[EdgeKind] = frozenset(EdgeKind),
) -> None:
    """Each target is reached, by *edges*, by exactly the referrers *reached* lists.

    A target no symbol of its path names fails: `find-refs` answers an
    unknown name with no refs, so it would otherwise pass as reached by
    nothing.
    """
    __tracebackhide__ = True
    root = repo_path(build)
    found = {
        (path, symbol): {
            (ref.file_path, ref.name, ref.edge.value)
            for ref in handle_find_refs(
                FindRefsRequest(repo_path=root, symbol=symbol, file_paths=[path]), build.index
            ).refs
            if ref.edge in edges
        }
        for path, symbol in reached
        if read(build, symbol, path)
    }
    assert found == reached


def assert_every_line_in_a_chunk(build: IndexBuild) -> None:
    """No non-blank line of a committed file falls outside a symbol; each one that does is listed as `path:line: text`."""
    __tracebackhide__ = True
    files = committed_files(build)
    spanned = {
        (symbol.file_path, line)
        for symbol in outline(build)
        for line in range(symbol.line_start, symbol.line_end + 1)
    }
    uncovered = [
        f"{path}:{number}: {text}"
        for path, content in sorted(files.items())
        for number, text in enumerate(content.splitlines(), start=1)
        if text.strip() and (path, number) not in spanned
    ]
    assert uncovered == []


def assert_no_undecided_kind_overlaps(build: IndexBuild) -> None:
    """No span of *build* was captured as two kinds neither narrows (`narrows`); each one that was is listed as `path:line: kinds -> kept`."""
    __tracebackhide__ = True
    undecided = [
        f"{overlap.path}:{overlap.line}: {', '.join(overlap.kinds)} -> {overlap.kept}"
        for overlap in build.result.kind_overlaps
        if not overlap.narrowed
    ]
    assert undecided == []
