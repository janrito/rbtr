#!/usr/bin/env python3
"""Benchmark: does docstring content help code search?

Answers the question with four repos and two indexes per repo:

  * rbtr (this repo, Python)
  * django/django               (Python)
  * mariozechner/pi-mono        (JavaScript / TypeScript)
  * astral-sh/uv                (Rust)

Each repo is indexed twice via the CLI:

  1. default — docstrings and leading doc comments kept in
     chunk content (rbtr's default behaviour).
  2. stripped — `rbtr index --strip-docstrings`, which blanks
     docstring bytes from every chunk.

The benchmark samples docstrings (deterministically, seeded) and
replays them as natural-language queries against both indexes.
Metrics per repo and aggregate are written to ``BENCHMARKS.md``.

This file is the harness only.  Phases 4-5 deliver:

  * repo-spec dataclass and four production specs.
  * repo provisioning: shallow-clone into a cache dir, checkout
    the configured ref, record the resolved SHA.
  * per-(repo, mode) ``RBTR_HOME`` layout so the benchmark
    DuckDB never collides with the user's real ``~/.rbtr/``.
  * docstring-to-query sampling (see ``bench_doc_extract``):
    first sentence of every symbol docstring, filtered by
    length and boilerplate markers, capped and seeded for
    determinism.
  * dry-run mode that prints the plan and exits without cloning.

Indexing, search replay, and metric / report rendering are
separate phases.
"""

from __future__ import annotations

import argparse
import dataclasses
import enum
import os
import random
import re
import subprocess
import sys
import tempfile
from collections.abc import Iterable
from pathlib import Path

from bench_doc_extract import DocSymbol, iter_doc_symbols


class BenchMode(enum.StrEnum):
    """Which variant of the index the benchmark is building.

    The `value` doubles as the directory name under
    ``--home-root``: ``<home-root>/<repo-slug>/<mode>/`` is the
    ``RBTR_HOME`` for that `(repo, mode)` pair, so every
    `(repo, mode)` gets its own fully-isolated rbtr home —
    separate DuckDB, separate models/ symlink, separate
    daemon sockets — and can never touch the user's real
    ``~/.rbtr/``.
    """

    DEFAULT = "default"
    """Index built with rbtr's production defaults: interior
    docstrings captured, leading comments attached."""

    STRIPPED = "stripped"
    """Index built with ``rbtr index --strip-docstrings``: all
    docstring bytes redacted from chunk content."""


@dataclasses.dataclass(frozen=True)
class RepoSpec:
    """How to fetch one benchmark repo.

    Attributes:
        slug:  Filesystem-safe name (``owner__repo``) used for the
               cache directory.
        url:   Git remote URL.  ``file://...`` for local repos; the
               harness uses the same clone pipeline regardless, so
               the "self" path is not a special case.
        ref:   Branch, tag, or SHA to check out.  The harness
               resolves this to a concrete SHA after clone and the
               resolved SHA is what the report pins.
        language: Primary language for reporting.  Informational.
    """

    slug: str
    url: str
    ref: str
    language: str


def _rbtr_repo_root() -> Path:
    """Locate the rbtr workdir relative to this script.

    ``scripts/`` sits inside the ``rbtr`` package; two parents up
    reaches the workspace root.  Using ``Path(__file__)`` keeps
    the script portable if someone runs it via an absolute path.
    """
    return Path(__file__).resolve().parents[3]


def production_specs() -> list[RepoSpec]:
    """The four repos measured by the committed ``BENCHMARKS.md``.

    Refs are deliberately readable (branch / tag names).  The
    harness records the concrete SHA each one resolved to at
    bench time so the report is reproducible.
    """
    rbtr_root = _rbtr_repo_root()
    return [
        RepoSpec(
            slug="rbtr__rbtr",
            url=f"file://{rbtr_root}",
            ref="HEAD",
            language="python",
        ),
        RepoSpec(
            slug="django__django",
            url="git@github.com:django/django.git",
            ref="main",
            language="python",
        ),
        RepoSpec(
            slug="badlogic__pi-mono",
            url="git@github.com:badlogic/pi-mono.git",
            ref="main",
            language="typescript",
        ),
        RepoSpec(
            slug="astral-sh__uv",
            url="git@github.com:astral-sh/uv.git",
            ref="main",
            language="rust",
        ),
    ]


def _run_git(args: list[str], cwd: Path | None = None) -> str:
    """Run ``git`` with the given arguments and return stdout.

    Raises on non-zero exit.  Captures stderr into the exception
    message so benchmarking failures are debuggable without
    hunting through pipe output.
    """
    # S603/S607: we invoke a known binary (`git`) with a fixed
    # argv shape; the benchmark runs locally with trusted specs.
    result = subprocess.run(  # noqa: S603
        ["git", *args],  # noqa: S607
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        msg = (
            f"git {' '.join(args)} (cwd={cwd}) failed: exit {result.returncode}\n"
            f"stderr:\n{result.stderr}"
        )
        raise RuntimeError(msg)
    return result.stdout


@dataclasses.dataclass(frozen=True)
class ProvisionedRepo:
    """Outcome of provisioning one spec."""

    spec: RepoSpec
    path: Path
    sha: str


def provision_repo(spec: RepoSpec, cache_dir: Path) -> ProvisionedRepo:
    """Shallow-clone *spec* into *cache_dir* (idempotent).

    The cache layout is ``<cache_dir>/<spec.slug>/``.  If the
    directory already contains a git repo, the harness fetches
    the configured ref and checks it out rather than re-cloning
    so iterations stay fast and the bench remains deterministic
    about which commit it measured.

    ``ref`` may be a branch, tag, concrete SHA, or the literal
    ``"HEAD"`` (used for the rbtr self-clone).  ``"HEAD"`` is not
    a symbolic ref the remote advertises, so it cannot be passed
    to ``--branch``; the harness falls back to a default shallow
    clone (tracks the remote's default branch) and the recorded
    SHA is what the report pins.
    """
    target = cache_dir / spec.slug
    is_head = spec.ref == "HEAD"

    if not (target / ".git").is_dir():
        target.mkdir(parents=True, exist_ok=True)
        print(f"  cloning {spec.url} -> {target}")
        clone_args = ["clone", "--depth", "1"]
        if not is_head:
            clone_args += ["--branch", spec.ref]
        clone_args += [spec.url, str(target)]
        _run_git(clone_args)
    else:
        print(f"  refresh {spec.slug}: fetch {spec.ref}")
        fetch_target = "HEAD" if is_head else spec.ref
        _run_git(["fetch", "--depth", "1", "origin", fetch_target], cwd=target)
        _run_git(["checkout", "FETCH_HEAD"], cwd=target)

    sha = _run_git(["rev-parse", "HEAD"], cwd=target).strip()
    return ProvisionedRepo(spec=spec, path=target, sha=sha)


def _default_cache_dir() -> Path:
    """Default cache location: ``$TMPDIR/rbtr-bench-cache/``.

    Using the system temp dir matches the user's direction
    ("temp folder") and sweeps on reboot.  Override with
    ``--cache-dir`` when you want an inspectable location.
    """
    return Path(tempfile.gettempdir()) / "rbtr-bench-cache"


def _default_home_root() -> Path:
    """Default `RBTR_HOME` root: ``$TMPDIR/rbtr-bench-home/``.

    Separate subtree from the repo cache so a ``rm -rf`` of the
    home root does not drop cloned repos (and vice versa).  Each
    `(repo, mode)` gets its own subdirectory and that path is
    fed into the ``RBTR_HOME`` env var when invoking
    ``rbtr index`` and ``rbtr search``.  The user's real
    ``~/.rbtr/`` is never touched.
    """
    return Path(tempfile.gettempdir()) / "rbtr-bench-home"


def home_for(home_root: Path, spec: RepoSpec, mode: BenchMode) -> Path:
    """Resolve the isolated ``RBTR_HOME`` for one `(repo, mode)`.

    Layout: ``<home_root>/<spec.slug>/<mode.value>/``.  Creating
    the directory here guarantees rbtr can immediately write its
    DuckDB, daemon sockets, and model cache.  Callers should pass
    the returned path through ``os.environ | {"RBTR_HOME": str(...)}``
    when they run ``rbtr``.
    """
    home = home_root / spec.slug / mode.value
    home.mkdir(parents=True, exist_ok=True)
    return home


# ── Query extraction ────────────────────────────────────────────────────────────

# A sentence ends at `.`, `!`, or `?` followed by whitespace
# or end-of-string, and a docstring worth benchmarking sits in
# a fairly tight length band — anything shorter than a dozen
# characters is usually a stub comment ("TODO", "stub", a
# single word), anything longer than a full paragraph makes a
# bad query.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n\s*\n")
_QUERY_MIN_LEN = 15
_QUERY_MAX_LEN = 200

# Leading non-prose noise that the boilerplate check skips over
# before comparing the candidate sentence against reject
# prefixes.  Includes the universal comment markers
# (`//`, `///`, `#`, `--`, triple-quotes, `*` gutters) plus
# surrounding whitespace.  Doing this in a single class rather
# than per-language keeps the bench language-agnostic — the
# indexer already told us where the docstring starts; we just
# need to look past any character that obviously isn't prose.
_REJECT_LOOKAHEAD_RE = re.compile(r"^[\s/#*\-!\"\'`]+")

# Block-comment line gutters (`/** `, ` * `, ` */`) and
# triple-quote openers/closers; removed from query text so the
# sampled first-sentence is pure prose.  These tokens are
# universal across block-comment-style languages (C / C++ /
# Java / JS / Rust `/** */`; Python / Ruby / shell triple-quote
# and `#`).  The indexer still sees them in chunk content; this
# projection only runs on the query text the bench replays.
_QUERY_NOISE_RE = re.compile(
    r"""
    (?:^|(?<=\n))[\ \t]*/\*+[\ \t]*           # block-comment opener `/**`
  | [\ \t]*\*+/[\ \t]*(?=$|\n)              # block-comment closer ` */`
  | (?:^|(?<=\n))[\ \t]*\*+[\ \t]?           # block-comment gutter ` * `
  | (?:^|(?<=\n))[\ \t]*///?!?[\ \t]?        # line-comment `//` / `///` / `//!`
  | (?:^|(?<=\n))[\ \t]*\#+!?[\ \t]?         # line-comment `#` / `#!`
  | (?:^|(?<=\n))[\ \t]*--[\ \t]?            # line-comment `--` (SQL / Lua)
  | [rRbBuU]{0,2}\"{3}                       # triple-quote `\"\"\"`
  | [rRbBuU]{0,2}\'{3}                       # triple-quote `'''`
  """,
    re.VERBOSE,
)

# Markers that indicate the docstring is boilerplate /
# scaffolding rather than a real description of the symbol.
# Queries starting with these are skipped.
_REJECT_PREFIXES = (
    "todo",
    "fixme",
    "xxx",
    "hack",
    "deprecated",
    "@param",
    "@return",
    "@returns",
    "@throws",
    "@type",
    "@see",
    "@link",
)


@dataclasses.dataclass(frozen=True)
class Query:
    """One `(symbol, first-sentence-query)` sampling point.

    `file_path`, `name`, and `line_start` together identify the
    chunk the search must retrieve for this query to count as a
    hit.
    """

    repo_slug: str
    language: str
    file_path: str
    name: str
    line_start: int
    text: str


def first_sentence(doc_text: str) -> str | None:
    """Return the first sentence of *doc_text*, or None if nothing usable.

    Splits on a sentence terminator followed by whitespace or on
    a blank line (paragraph break).  Truncates at
    `_QUERY_MAX_LEN` when no terminator appears within the first
    paragraph.  Returns None when the result is shorter than
    `_QUERY_MIN_LEN` or starts with a boilerplate marker.
    """
    # Strip universal comment markers (block-comment gutters,
    # triple-quote delimiters, line-comment openers) from the raw
    # docstring before sentence-splitting.  The indexer's span
    # told us where the docstring is; this projection only
    # normalises fixed tokens that every block-comment language
    # uses the same way, so the first sentence comes out as pure
    # prose rather than `/**\n *  Foo bar`.
    pruned = _QUERY_NOISE_RE.sub(" ", doc_text)
    pruned = re.sub(r"\s+", " ", pruned).strip()
    if not pruned:
        return None
    parts = _SENTENCE_SPLIT_RE.split(pruned, maxsplit=1)
    candidate = parts[0].strip() if parts else pruned
    if len(candidate) > _QUERY_MAX_LEN:
        candidate = candidate[:_QUERY_MAX_LEN].rstrip()
    if len(candidate) < _QUERY_MIN_LEN:
        return None
    post_noise = _REJECT_LOOKAHEAD_RE.sub("", candidate).lower()
    if any(post_noise.startswith(p) for p in _REJECT_PREFIXES):
        return None
    return candidate


def sample_queries(
    symbols: Iterable[DocSymbol],
    *,
    seed: int,
    cap: int,
) -> list[Query]:
    """Turn documented symbols into a deterministic query sample.

    Each input symbol yields at most one `Query` (its first
    sentence).  Symbols whose doc text does not pass the
    `first_sentence` filter are dropped silently.  The remaining
    list is sorted by `(file_path, line_start, name)` for
    determinism and then sampled down to `cap` via
    `random.Random(seed).sample`.
    """
    candidates: list[Query] = []
    for sym in symbols:
        text = first_sentence(sym.doc_text)
        if text is None:
            continue
        candidates.append(
            Query(
                repo_slug=sym.repo_slug,
                language=sym.language,
                file_path=sym.file_path,
                name=sym.name,
                line_start=sym.line_start,
                text=text,
            )
        )
    candidates.sort(key=lambda q: (q.file_path, q.line_start, q.name))
    if len(candidates) <= cap:
        return candidates
    # S311: we want deterministic sampling, not cryptographic.
    rng = random.Random(seed)  # noqa: S311
    sampled = rng.sample(candidates, cap)
    # Re-sort so the output is deterministic and readable regardless
    # of sampling order.
    sampled.sort(key=lambda q: (q.file_path, q.line_start, q.name))
    return sampled


def _guard_home_root(home_root: Path) -> None:
    """Refuse to run if ``home_root`` overlaps the user's real rbtr home.

    Running the benchmark with ``--home-root=~/.rbtr`` would
    silently overwrite the user's production index.  A cheap
    startup check catches the obvious slip:

    * resolve the provided path,
    * resolve whatever ``RBTR_HOME`` (or the rbtr default) points
      at,
    * refuse if they are identical or one contains the other.
    """
    resolved = home_root.expanduser().resolve()
    real_home_env = os.environ.get("RBTR_HOME")
    real_home = (
        Path(real_home_env).expanduser().resolve()
        if real_home_env
        else (Path.home() / ".rbtr").resolve()
    )
    if resolved == real_home or resolved in real_home.parents or real_home in resolved.parents:
        msg = (
            f"refusing to use --home-root={resolved}: overlaps user's rbtr home "
            f"({real_home}). Pick a disjoint path."
        )
        raise SystemExit(msg)


def _print_plan(specs: list[RepoSpec], cache_dir: Path, home_root: Path) -> None:
    """Print the harness plan without doing any work."""
    print(f"cache dir:       {cache_dir}")
    print(f"home root:       {home_root}")
    print(f"repos to bench:  {len(specs)}")
    for spec in specs:
        target = cache_dir / spec.slug
        exists = "[cached]" if (target / ".git").is_dir() else "[clone ]"
        print(
            f"  {exists} {spec.slug:30s} lang={spec.language:10s} ref={spec.ref:20s} url={spec.url}"
        )
        for mode in BenchMode:
            home = home_root / spec.slug / mode.value
            print(f"      RBTR_HOME[{mode.value:8s}] = {home}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark: docstring contribution to code search quality.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=_default_cache_dir(),
        help="Directory where repos are cloned (default: $TMPDIR/rbtr-bench-cache/).",
    )
    parser.add_argument(
        "--home-root",
        type=Path,
        default=_default_home_root(),
        help=(
            "Root for per-(repo, mode) RBTR_HOME directories (default: "
            "$TMPDIR/rbtr-bench-home/).  Each subdir is passed to rbtr "
            "via the RBTR_HOME env var so benchmark DuckDBs, daemon "
            "sockets, and model caches stay fully isolated from the "
            "user's real ~/.rbtr/ home."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for deterministic query sampling (default: 0).",
    )
    parser.add_argument(
        "--sample-cap",
        type=int,
        default=300,
        help=(
            "Maximum number of queries per repo (default: 300).  "
            "Smaller repos contribute all of their eligible queries."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan and exit without cloning, indexing, or searching.",
    )
    args = parser.parse_args(argv)

    specs = production_specs()
    cache_dir: Path = args.cache_dir
    home_root: Path = args.home_root
    _guard_home_root(home_root)
    cache_dir.mkdir(parents=True, exist_ok=True)
    home_root.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        _print_plan(specs, cache_dir, home_root)
        return 0

    _print_plan(specs, cache_dir, home_root)
    print()
    print("provisioning repos …")
    provisioned: list[ProvisionedRepo] = []
    for spec in specs:
        provisioned.append(provision_repo(spec, cache_dir))

    print()
    print("provisioned:")
    for pr in provisioned:
        print(f"  {pr.spec.slug:30s} sha={pr.sha[:12]}  path={pr.path}")
        for mode in BenchMode:
            home = home_for(home_root, pr.spec, mode)
            print(f"      RBTR_HOME[{mode.value:8s}] = {home}")

    print()
    print("sampling queries …")
    queries_by_repo: dict[str, list[Query]] = {}
    for pr in provisioned:
        symbols = list(iter_doc_symbols(pr.path, pr.spec.slug))
        queries = sample_queries(symbols, seed=args.seed, cap=args.sample_cap)
        queries_by_repo[pr.spec.slug] = queries
        print(
            f"  {pr.spec.slug:30s} documented_symbols={len(symbols):5d}  queries={len(queries):4d}"
        )

    print()
    print("phases 6-8 not yet implemented — exiting after query sampling.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] if len(sys.argv) > 1 else None))
