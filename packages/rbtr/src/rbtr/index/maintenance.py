"""Index maintenance: the housekeeping the daemon runs when it is idle.

`run_maintenance` stops watching refs git can no longer resolve, forgets
checkouts missing past `forget_after_s`, collects every repo in watched
mode, and compacts project files grown past `compact_growth_ratio`, in
that order.  Each step only removes what the user would remove by hand
with `rbtr unwatch --stale`, `rbtr forget --stale`, `rbtr gc --scope all`
or `rbtr gc --compact`.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import structlog

from rbtr.config import config
from rbtr.domain.models import GcMode
from rbtr.index.gc import run_gc_all
from rbtr.index.index import Index
from rbtr.index.progress import RepoProgressCallback
from rbtr.index.watch import forget_missing_repos, remove_stale_refs

log = structlog.get_logger(__name__)


def run_maintenance(index: Index, *, now: datetime, on_progress: RepoProgressCallback) -> None:
    """Run one maintenance pass over the whole index; log what it removed.

    *now* is when the pass runs, the time a checkout found missing is
    recorded against.  *on_progress* is called with each repo as gc
    collects it.
    """
    unwatched = remove_stale_refs(index, index.repos(), dry_run=False)
    forgotten = forget_missing_repos(index, now=now, after=timedelta(seconds=config.forget_after_s))
    counts, collected = run_gc_all(
        index, mode=GcMode.WATCHED, dry_run=False, on_progress=on_progress
    )
    compacted = index.compact_grown(config.compact_growth_ratio)
    log.info(
        "maintenance_finished",
        unwatched=sum(len(refs) for refs in unwatched.values()),
        forgotten=len(forgotten),
        collected=collected,
        snapshots=counts.snapshots,
        chunks=counts.chunks,
        compacted=compacted,
    )
