"""Cases for which job the daemon picks up next.

Scenarios describe what is indexed and what is embedded; the test
asserts which embed job `_find_next_job` hands back, naming `None`
when the answer is that no job is due.
"""

from __future__ import annotations

from dataclasses import dataclass

from pytest_cases import case

# ── Scenario dataclasses ────────────────────────────────────────────


@dataclass(frozen=True)
class SeededSnapshot:
    """One snapshot to seed, and how far its build and embed got.

    Seeded in list order, each in its own transaction, so `indexed_at`
    increases down the list — which is how a scenario expresses "this
    one was indexed more recently".
    """

    repo_path: str
    snapshot_sha: str
    indexed: bool = True
    embedded: bool = False


@dataclass(frozen=True)
class NextJobScenario:
    """Snapshots to seed, and the ref the embed job should name.

    `expected_ref` of `None` means no job at all is due.
    """

    snapshots: list[SeededSnapshot]
    expected_ref: str | None


# ── next_job cases ──────────────────────────────────────────────────


@case(tags=["next_job"])
def case_an_unembedded_snapshot_yields_an_embed_job() -> NextJobScenario:
    """Indexed, chunks written, no vectors yet — the plain case."""
    return NextJobScenario(
        snapshots=[SeededSnapshot("/repo/a", "sha1")],
        expected_ref="sha1",
    )


@case(tags=["next_job"])
def case_an_unfinished_build_yields_no_job() -> NextJobScenario:
    """Chunks exist, but the build stopped before `mark_indexed`.

    `mark_indexed` is what marks a build complete.  Rows written before
    it belong to a build still in flight or one that died, and `sweep`
    clears them; the embedder leaves them alone.
    """
    return NextJobScenario(
        snapshots=[SeededSnapshot("/repo/a", "sha1", indexed=False)],
        expected_ref=None,
    )


@case(tags=["next_job"])
def case_a_fully_embedded_snapshot_yields_no_job() -> NextJobScenario:
    """Every chunk already carries a vector."""
    return NextJobScenario(
        snapshots=[SeededSnapshot("/repo/a", "sha1", embedded=True)],
        expected_ref=None,
    )


@case(tags=["next_job"])
def case_the_most_recently_indexed_wins_across_repos() -> NextJobScenario:
    """The most recently indexed snapshot wins, whichever repo holds it.

    So a build you just ran is embedded next, ahead of an older backlog
    in a repo that happens to have been registered first.
    """
    return NextJobScenario(
        snapshots=[
            SeededSnapshot("/repo/a", "older"),
            SeededSnapshot("/repo/b", "newer"),
        ],
        expected_ref="newer",
    )
