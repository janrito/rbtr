"""Keep watched repositories indexed using caller-supplied data and models."""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import threading
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from enum import Enum, auto
from uuid import uuid4

import structlog

from rbtr.config import config
from rbtr.domain.models import IndexResult, SnapshotCounts, SnapshotRef
from rbtr.errors import BuildStopped, RbtrError
from rbtr.git import HEAD_REF, resolve_ref
from rbtr.index import watcher
from rbtr.index.build import build_index, drop_stale_worktree_shas
from rbtr.index.embed import write_embedding_batch
from rbtr.index.embeddings import Embedder, embedding_text
from rbtr.index.index import Index
from rbtr.index.maintenance import run_maintenance
from rbtr.index.progress import EmbedOutcome
from rbtr.logging import elapsed_ms

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class BuildJob:
    """Refs or snapshot SHAs selected for a build in one repository."""

    repo_path: str
    refs: tuple[str, ...]


@dataclass(frozen=True)
class EmbedJob:
    """A stored snapshot selected for embedding its chunks with missing vectors."""

    at: SnapshotRef

    @property
    def repo_path(self) -> str:
        """The checkout containing this snapshot."""
        return self.at.repo.repo_path


@dataclass(frozen=True)
class RunningJob[Work: BuildJob | EmbedJob]:
    """Immutable progress for one selected background run.

    `started_at` uses the monotonic clock. An active run may still be waiting
    for its execution resource, before native work starts.
    """

    work: Work
    job_id: str
    started_at: float
    phase: str
    current: int = 0
    total: int = 0


@dataclass(frozen=True)
class BuildReady:
    """Report a completed snapshot build with its current counts and build result."""

    at: SnapshotRef
    counts: SnapshotCounts
    result: IndexResult


@dataclass(frozen=True)
class EmbeddingEnded:
    """Report an embedding run's final counts and finished/stood-aside/stopped outcome."""

    at: SnapshotRef
    counts: SnapshotCounts
    outcome: EmbedOutcome


@dataclass(frozen=True)
class IndexingError:
    """Report an indexing failure for one repository to the output consumer."""

    repo_path: str
    message: str


class CollectionPhase(Enum):
    """Mark the start or end of a maintenance collection pass."""

    STARTED = auto()
    ENDED = auto()


@dataclass(frozen=True)
class CollectionProgress:
    """Report current and total progress counts during maintenance collection."""

    current: int
    total: int


type IndexerOutput = (
    RunningJob[BuildJob | EmbedJob]
    | BuildReady
    | EmbeddingEnded
    | IndexingError
    | CollectionPhase
    | CollectionProgress
    | watcher.WatchedTarget
    | watcher.DirtyWorktree
)


class Indexer:
    """Build, embed, poll and maintain using caller-owned data, models and controls.

    Construction synchronously seeds HEAD watches. Give it exclusive writable
    Index access, before sharing or while holding the writer semaphore, off a
    UI loop.

    The caller starts the chosen `run`, `watch` and `maintain` coroutines, at most
    once each on the same loop. Stop through `request_shutdown`, not cancellation,
    and await all started operations before disposing of supplied resources.
    Their completion includes native work and awaited output.
    """

    def __init__(
        self,
        index: Index,
        *,
        embedder: Embedder | None,
        gpu_lock: asyncio.Lock,
        write_sem: asyncio.Semaphore,
        stop: threading.Event,
        on_output: Callable[[IndexerOutput], Awaitable[None]],
        idle_poll_interval: float | None = None,
        busy_poll_interval: float | None = None,
    ) -> None:
        """Configure shared resources and async output delivery.

        `embedder=None` selects builds only. Share `gpu_lock` and `write_sem` with
        other users of those resources. Poll intervals override central config.

        `on_output` is awaited on the owning loop after applying run-progress
        state; it must keep the loop responsive. Callback errors follow the
        failure handling of the operation that emitted the value.
        """
        self._index = index
        self._embedder = embedder
        self._gpu_lock = gpu_lock
        self._write_sem = write_sem
        self._stopping = stop
        self._on_output = on_output
        self._idle_poll_interval = (
            config.idle_poll_interval if idle_poll_interval is None else idle_poll_interval
        )
        self._busy_poll_interval = (
            config.busy_poll_interval if busy_poll_interval is None else busy_poll_interval
        )
        self._wake = asyncio.Event()
        self._stopped = asyncio.Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._state_lock = threading.Lock()
        self._running_job: RunningJob[BuildJob | EmbedJob] | None = None
        self._failed_builds: frozenset[watcher.FailedBuild] = frozenset()
        self._idle_since = time.monotonic()
        self._backfill_head_watches(index)

    def snapshot(self) -> RunningJob[BuildJob | EmbedJob] | None:
        """Return an immutable active-run snapshot, or `None` when idle, from any thread."""
        with self._state_lock:
            return self._running_job

    def wake(self) -> None:
        """Signal reselection without waiting, from any thread; repeated wakes coalesce.

        Before a loop is attached, `run`'s initial selection covers stored work.
        """
        loop = self._loop
        if loop is not None and not loop.is_closed():
            loop.call_soon_threadsafe(self._signal)

    def _signal(self) -> None:
        self._wake.set()
        if self._stopping.is_set():
            self._stopped.set()

    def request_shutdown(self) -> None:
        """Set the shared stop signal; thread-safe, non-blocking and idempotent."""
        self._stopping.set()
        self.wake()

    async def _pause(self, delay: float) -> None:
        self._loop = asyncio.get_running_loop()
        if self._stopping.is_set():
            return
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self._stopped.wait(), timeout=delay)

    async def _emit(self, output: IndexerOutput) -> None:
        """Apply progress for the active run, then await the output callback."""
        if isinstance(output, RunningJob):
            with self._state_lock:
                active = self._running_job
                if active is not None and active.job_id == output.job_id:
                    self._running_job = output
        await self._on_output(output)

    def _emit_native(self, loop: asyncio.AbstractEventLoop, output: IndexerOutput) -> None:
        """Submit output to its loop and block this thread until delivery finishes."""
        # asyncio requires its thread-safe entry point. The caller keeps
        # the loop alive until native producers and their output finish.
        sent = asyncio.run_coroutine_threadsafe(self._emit(output), loop)
        sent.result()

    def _backfill_head_watches(self, index: Index) -> None:
        """Seed a HEAD watch for each registered repo, retaining existing watches."""
        repos = index.repos()
        if not repos:
            return
        for repo in repos:
            with index.session(repo) as ws:
                ws.add_watched_refs([HEAD_REF])
        log.info("head_watch_backfill", repos=len(repos))

    def _run_build(self, run: RunningJob[BuildJob], loop: asyncio.AbstractEventLoop) -> None:
        """Run one build on a native thread, reporting through its owning loop."""
        job = run.work
        index = self._index

        def report(phase: str, done: int, total: int) -> None:
            self._emit_native(loop, replace(run, phase=phase, current=done, total=total))

        try:
            repo = index.register(job.repo_path)
            try:
                resolved = [
                    resolve_ref(job.repo_path, ref, max_untracked_files=config.max_untracked_files)
                    for ref in job.refs
                ]
            except RbtrError as exc:
                self._emit_native(loop, IndexingError(job.repo_path, str(exc)))
                return

            for sha in resolved:
                result = build_index(
                    job.repo_path,
                    sha,
                    index,
                    on_progress=report,
                    stop=self._stopping,
                )
                counts = index.chunk_counts_for_snapshot(at=repo.at(sha))
                self._emit_native(loop, BuildReady(repo.at(sha), counts, result))
                # Clean up stale worktree tree SHAs.  After a
                # worktree build, old tree SHAs from previous edits
                # linger in indexed_snapshots.  After a commit build,
                # the old worktree was built against the previous
                # HEAD's tree.  Either way, drop any tree-type SHA
                # that isn't the one we just built.
                drop_stale_worktree_shas(index, repo, keep=sha)
        except BuildStopped:
            log.info("build_stopped_shutdown", repo=job.repo_path)
        except Exception as exc:
            self._emit_native(loop, IndexingError(job.repo_path, str(exc)))
            raise

    async def _run_embed_async(self, run: RunningJob[EmbedJob]) -> None:
        """Embed outstanding chunks with per-batch GPU and writer ownership."""
        job = run.work
        index = self._index
        embedder = self._embedder
        if embedder is None:
            return

        pending = await asyncio.to_thread(index.unembedded_chunk_ids, at=job.at)
        if not pending:
            return
        outstanding = len(pending)
        outcome = EmbedOutcome.FINISHED

        await self._emit(replace(run, phase="loading_model"))
        done = 0
        t0 = time.perf_counter()

        try:
            for page_ids in itertools.batched(pending, config.embedding_page_size, strict=False):
                page = await asyncio.to_thread(index.chunks_by_id, list(page_ids), at=job.at)
                for batch in itertools.batched(page, config.embedding_batch_size, strict=False):
                    texts = [embedding_text(c.name, c.content) for c in batch]
                    try:
                        async with self._gpu_lock:
                            results = await asyncio.to_thread(embedder.embed, texts)
                    except (RuntimeError, ValueError):
                        # A shutdown aborts the batch in progress: no failure.
                        if self._stopping.is_set():
                            outcome = EmbedOutcome.STOPPED
                            log.info("embedding_stopped_shutdown", done=done, total=outstanding)
                            return
                        log.warning("embedding_batch_failed", exc_info=True)
                        continue
                    # Under `_write_sem` like every other write: gc
                    # compacts a project by copying its file and switching
                    # to the copy, so a batch that commits to the old file
                    # during the copy is lost.  gc waits one batch.
                    async with self._write_sem:
                        await asyncio.to_thread(
                            write_embedding_batch,
                            index,
                            job.at.repo,
                            [
                                (chunk.id, result)
                                for chunk, result in zip(batch, results, strict=True)
                            ],
                        )
                    done += len(batch)
                    await self._emit(
                        replace(run, phase="embedding", current=done, total=outstanding)
                    )
                    if self._stopping.is_set():
                        outcome = EmbedOutcome.STOPPED
                        log.info("embedding_stopped_shutdown", done=done, total=outstanding)
                        return
                    # Stand aside for a build: it wants the worker, which
                    # runs one job at a time, so only returning frees it.
                    if await asyncio.to_thread(
                        watcher.pending_builds, index, frozenset(self._failed_builds)
                    ):
                        outcome = EmbedOutcome.STOOD_ASIDE
                        log.info("embedding_preempted", done=done, total=outstanding)
                        return
        finally:
            if done == 0 and outcome is not EmbedOutcome.STOPPED:
                # Every batch was rejected by the model.  Without this the
                # run reports an end with nothing embedded and no reason,
                # and the worker picks the same job up again.
                log.warning("embedding_made_no_progress", total=outstanding)
            final = await asyncio.to_thread(index.chunk_counts_for_snapshot, at=job.at)
            await self._emit(EmbeddingEnded(job.at, final, outcome))
        log.info("embedded_chunks", done=done, total=outstanding, elapsed_ms=elapsed_ms(t0))

    def next_job(self) -> BuildJob | EmbedJob | None:
        """Return a due build, else a missing-vector job, else `None`.

        This synchronous query reads Git and stored data; call it off a UI loop.
        It selects without reserving work or changing active-run state. Builds
        suppressed after failure are excluded. Builds take priority; embedding
        chooses the most recently indexed incomplete snapshot across repositories
        and is considered only when an embedder was supplied.
        """
        index = self._index
        with self._state_lock:
            failed_builds = self._failed_builds
        for target in watcher.pending_builds(index, failed_builds):
            return BuildJob(repo_path=target.repo_path, refs=(target.snapshot_sha,))

        if self._embedder is None:
            return None

        # Embeds: indexed snapshots with un-embedded chunks, newest first.
        for ref, counts in index.chunk_counts_by_snapshot().items():
            if counts.is_fully_embedded:
                continue
            return EmbedJob(at=ref)

        return None

    def _set_active[Work: BuildJob | EmbedJob](self, job: Work, job_id: str) -> RunningJob[Work]:
        """Start one run on the owning loop and return its immutable identity."""
        run = RunningJob(
            work=job,
            job_id=job_id,
            started_at=time.monotonic(),
            phase="starting" if isinstance(job, BuildJob) else "embedding",
        )
        with self._state_lock:
            self._running_job = run
        return run

    def _clear_active(self) -> None:
        """Clear the completed run on the owning loop."""
        with self._state_lock:
            self._running_job = None
        self._idle_since = time.monotonic()

    async def run(self) -> None:
        """Process jobs sequentially until stop, selecting immediately and after wakes.

        Start `watch` separately to detect later Git changes. Job failures are
        logged and selection continues; failed build targets stay suppressed in
        this instance. Selection errors are raised to the caller.
        """
        self._loop = asyncio.get_running_loop()
        self._wake.set()
        while not self._stopping.is_set():
            await self._wake.wait()
            self._wake.clear()
            while not self._stopping.is_set():
                # Off the loop: `next_job` walks git per repo and
                # reads the embedding column.  Safe in a thread because
                # it only reads, and every store read goes through a
                # cursor of the reading thread's own (`Index`'s
                # class docstring).
                job = await asyncio.to_thread(self.next_job)
                if job is None:
                    break
                job_ctx: dict[str, str] = {
                    "job_id": uuid4().hex[:8],
                    "repo": job.repo_path,
                    "job_kind": "build" if isinstance(job, BuildJob) else "embed",
                }
                if isinstance(job, EmbedJob):
                    job_ctx["ref"] = job.at.snapshot_sha
                else:
                    job_ctx["ref"] = job.refs[0][:12]
                # Bound for the job's lifetime; `to_thread` copies the
                # context, so build/embed logs in worker threads carry it.
                with structlog.contextvars.bound_contextvars(**job_ctx):
                    try:
                        if isinstance(job, EmbedJob):
                            await self._run_embed_async(self._set_active(job, job_ctx["job_id"]))
                        else:
                            run = self._set_active(job, job_ctx["job_id"])
                            async with self._write_sem:
                                await asyncio.to_thread(
                                    self._run_build, run, asyncio.get_running_loop()
                                )
                    except Exception:
                        log.exception("job_failed", job=str(job))
                        if isinstance(job, BuildJob):
                            with self._state_lock:
                                self._failed_builds |= frozenset(
                                    watcher.FailedBuild(job.repo_path, sha) for sha in job.refs
                                )
                    finally:
                        self._clear_active()

    def poll_interval(self) -> float:
        """Return the configured busy interval for an active run, else idle interval."""
        if self.snapshot() is not None:
            return self._busy_poll_interval
        return self._idle_poll_interval

    async def watch(self) -> None:
        """Poll Git at the current interval, report due targets and wake selection.

        Start `run` separately to perform the work. Poll/output errors are raised
        to the caller.
        """
        while not self._stopping.is_set():
            await self._pause(self.poll_interval())
            if self._stopping.is_set():
                break
            with structlog.contextvars.bound_contextvars(watch_cycle=uuid4().hex[:8]):
                for target in await asyncio.to_thread(
                    watcher.pending_builds, self._index, frozenset(self._failed_builds)
                ):
                    match target:
                        case watcher.WatchedTarget():
                            log.info(
                                "watched_ref_stale",
                                repo=target.repo_path,
                                ref=target.ref,
                                sha=target.snapshot_sha[:12],
                            )
                        case watcher.DirtyWorktree():
                            log.info(
                                "dirty_worktree",
                                repo=target.repo_path,
                                tree=target.snapshot_sha[:12],
                            )
                    await self._emit(target)
                    self._wake.set()

    async def maintain(self) -> None:
        """Retire eligible files periodically and run maintenance when idle.

        Retirement uses each idle-interval poll. Maintenance requires configured
        `maintenance_idle_s` without a job; later passes wait `maintenance_interval_s`
        after the previous pass. Both hold the writer semaphore, and a started
        pass finishes before stopping. Collection reports use `on_output`;
        errors are raised to the caller.
        """
        last_pass: float | None = None
        while not self._stopping.is_set():
            await self._pause(self._idle_poll_interval)
            index = self._index
            if self._stopping.is_set():
                continue
            async with self._write_sem:
                if self._stopping.is_set():
                    return
                await asyncio.to_thread(index.retire_due)
            if self._stopping.is_set():
                return
            now = time.monotonic()
            idle = self._running_job is None
            due = last_pass is None or now - last_pass >= config.maintenance_interval_s
            if idle and due and now - self._idle_since >= config.maintenance_idle_s:
                async with self._write_sem:
                    if self._stopping.is_set():
                        return
                    await asyncio.to_thread(self._maintain, index, asyncio.get_running_loop())
                last_pass = time.monotonic()

    def _maintain(self, index: Index, loop: asyncio.AbstractEventLoop) -> None:
        """Run a native maintenance pass and report its collection progress."""
        log.info("maintenance_started")
        self._emit_native(loop, CollectionPhase.STARTED)
        try:
            run_maintenance(
                index,
                now=datetime.now(UTC),
                on_progress=lambda _repo_path, done, total: self._emit_native(
                    loop, CollectionProgress(done, total)
                ),
            )
        finally:
            self._emit_native(loop, CollectionPhase.ENDED)
