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

This file is the harness only.  Phase 4 delivers:

  * repo-spec dataclass and four production specs.
  * repo provisioning: shallow-clone into a cache dir, checkout
    the configured ref, record the resolved SHA.
  * dry-run mode that prints the plan and exits without cloning.

Query extraction, indexing, search replay, and metric / report
rendering are separate phases.
"""

from __future__ import annotations

import argparse
import dataclasses
import subprocess
import sys
import tempfile
from pathlib import Path


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
            url="https://github.com/django/django.git",
            ref="main",
            language="python",
        ),
        RepoSpec(
            slug="mariozechner__pi-mono",
            url="https://github.com/mariozechner/pi-mono.git",
            ref="main",
            language="typescript",
        ),
        RepoSpec(
            slug="astral-sh__uv",
            url="https://github.com/astral-sh/uv.git",
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


def _print_plan(specs: list[RepoSpec], cache_dir: Path) -> None:
    """Print the harness plan without doing any work."""
    print(f"cache dir:       {cache_dir}")
    print(f"repos to bench:  {len(specs)}")
    for spec in specs:
        target = cache_dir / spec.slug
        exists = "[cached]" if (target / ".git").is_dir() else "[clone ]"
        print(
            f"  {exists} {spec.slug:30s} lang={spec.language:10s} ref={spec.ref:20s} url={spec.url}"
        )


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
        "--dry-run",
        action="store_true",
        help="Print the plan and exit without cloning, indexing, or searching.",
    )
    args = parser.parse_args(argv)

    specs = production_specs()
    cache_dir: Path = args.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        _print_plan(specs, cache_dir)
        return 0

    _print_plan(specs, cache_dir)
    print()
    print("provisioning repos …")
    provisioned: list[ProvisionedRepo] = []
    for spec in specs:
        provisioned.append(provision_repo(spec, cache_dir))

    print()
    print("provisioned:")
    for pr in provisioned:
        print(f"  {pr.spec.slug:30s} sha={pr.sha[:12]}  path={pr.path}")

    print()
    print("phases 5-8 not yet implemented — exiting after provisioning.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] if len(sys.argv) > 1 else None))
