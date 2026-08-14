"""Scenarios for `corpus_refs` — the state of one repo's index.

Each case describes which snapshots are marked indexed for a repo
whose worktree has one commit, and what `corpus_refs` owes in
return.  The cases carry data only; `test_corpus.py` builds the git
repo and seeds the store from the description.
"""

from __future__ import annotations

from dataclasses import dataclass

from pytest_cases import case

# Symbolic stand-in the test substitutes for the repo's real HEAD SHA.
HEAD = "HEAD"


@dataclass(frozen=True)
class CorpusScenario:
    """Snapshots to mark indexed, and what `corpus_refs` should do.

    `indexed` names snapshots symbolically: `HEAD` is replaced with
    the repo's real HEAD SHA at setup; any other entry is a literal
    SHA standing for a snapshot that is not HEAD (a worktree tree,
    or an older commit).  `error_match` is `None` when the call
    should succeed, otherwise a regex the raised message must match.
    """

    indexed: list[str]
    error_match: str | None


@case(tags=["valid"])
def case_head_only() -> CorpusScenario:
    """The state the eval's pipeline maintains: HEAD indexed, nothing else."""
    return CorpusScenario(indexed=[HEAD], error_match=None)


@case(tags=["invalid"])
def case_head_not_indexed() -> CorpusScenario:
    """A repo registered but never built.

    The report would otherwise describe whatever else is present, or
    silently omit the repo.
    """
    return CorpusScenario(indexed=[], error_match="not indexed")


@case(tags=["invalid"])
def case_extra_snapshot_beside_head() -> CorpusScenario:
    """A daemon indexed a dirty worktree beside HEAD.

    This is the condition that doubled `astral-sh__uv`'s edge count:
    both snapshots are indexed, and an unscoped `COUNT(*)` sums them.
    """
    return CorpusScenario(
        indexed=[HEAD, "f87f5c2d7df48283bff6a70fc5dc32b154cb855b"],
        error_match="f87f5c2d7df48283bff6a70fc5dc32b154cb855b",
    )
