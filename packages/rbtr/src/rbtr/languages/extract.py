"""Extraction of a batch of files: the adapter reads them, and the rules make their chunks.

`extract_batch` reads every file of a batch, and every block it embeds,
through tree-sitter and the language plugins (`rbtr.languages.treesitter`),
then folds comments and assembles chunks over all of them at once
(`rbtr.domain.rules`). The indexer extracts its files `BATCH_FILES` at
a time through `extract_files`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import dataframely as dy
import polars as pl
import structlog

from rbtr.config import config
from rbtr.domain.models import KindOverlap
from rbtr.domain.rules import (
    FileRow,
    KindOverlapRow,
    PlacedChunkRow,
    assemble,
    capture_chunks,
    levelled,
    line_windows,
    with_metadata,
)
from rbtr.errors import QueryCaptureError
from rbtr.git import FileEntry
from rbtr.languages.plaintext import PLAINTEXT
from rbtr.languages.treesitter import read

log = structlog.get_logger(__name__)

BATCH_FILES = 100
"""Files extracted together: a run of the rules costs about 9 ms, so a run
per file costs five times what it costs a file in a batch of 100."""


@dataclass(frozen=True)
class Extraction:
    """A batch's chunks, the spans it captured as more than one kind, and the paths of the files whose extraction raised."""

    chunks: dy.DataFrame[PlacedChunkRow]
    overlaps: list[KindOverlap]
    failed: list[str]


def kind_overlaps(rows: dy.DataFrame[KindOverlapRow]) -> list[KindOverlap]:
    """*rows* as `KindOverlap`s, each logged: one `KIND_NARROWS` left undecided as a warning."""
    overlaps = [KindOverlap.model_validate(row) for row in rows.iter_rows(named=True)]
    for overlap in overlaps:
        (log.debug if overlap.narrowed else log.warning)(
            "kind_overlap",
            path=overlap.path,
            line=overlap.line,
            kinds=[kind.value for kind in overlap.kinds],
            kept=overlap.kept.value,
        )
    return overlaps


def file_rows(files: Sequence[tuple[FileEntry, str]]) -> dy.DataFrame[FileRow]:
    """*files*, each a file and the language it is read as, as a batch's rows."""
    return pl.DataFrame(
        [
            (
                file,
                entry.path,
                entry.blob_sha,
                language,
                # Each byte that is not UTF-8 becomes one `?`: offsets stay,
                # and every slice of the content is text.
                entry.content.decode("utf-8", "surrogateescape").encode("utf-8", "replace"),
            )
            for file, (entry, language) in enumerate(files)
        ],
        schema=FileRow.create_empty().schema,
        orient="row",
    ).pipe(FileRow.validate, cast=True)


def extract_batch(files: dy.DataFrame[FileRow]) -> Extraction:
    """*files*' chunks, each file's in order, every chunk stamped with its file's language, and the spans captured as more than one kind.

    The adapter reads each file and every block it embeds (`read`); the
    rules make chunks of the captures, cut the files nothing reads into
    `PLAINTEXT` windows of lines, and assemble each file's, one chunk per
    span, over all the files at once.
    """
    captures, metadata_captures, chunked, plain = read(files)
    # Each rule is one lazy plan, its result checked within it, collected
    # before the next rule reads it: a plan copies a frame it reads twice into
    # each use, and `with_metadata` reads its inputs many times.
    settled = levelled(captures.lazy(), files).collect()
    described = with_metadata(settled.lazy(), metadata_captures.lazy()).collect()
    # Read by both of `assemble`'s plans, so collected once here.
    placed = capture_chunks(files, described.lazy()).collect()
    chunks, overlaps = assemble(
        files,
        placed.lazy(),
        chunked.lazy(),
        line_windows(files, plain, config.chunk_lines, config.chunk_overlap, PLAINTEXT),
    )
    collected, overlap_rows = pl.collect_all([chunks, overlaps])
    return Extraction(
        chunks=collected.pipe(PlacedChunkRow.validate),
        overlaps=kind_overlaps(overlap_rows.pipe(KindOverlapRow.validate)),
        failed=[],
    )


def extract_files(files: dy.DataFrame[FileRow]) -> Extraction:
    """*files*' chunks and overlapping kinds, as `extract_batch`, and the paths of the files whose extraction raised.

    A batch whose extraction raises is halved, and each half extracted
    again, down to the file that raises: it is logged, and has only its
    presence chunk, so its snapshot row still reaches a chunk.
    """
    try:
        return extract_batch(files)
    except QueryCaptureError:
        # A language's query, not a file, is at fault: no file extracts.
        raise
    except Exception:
        if files.height == 1:
            log.exception("index_file_failed", path=files["path"].item())
            presence, _ = assemble(files, PlacedChunkRow.create_empty(lazy=True))
            return Extraction(
                chunks=presence.collect(), overlaps=[], failed=files["path"].to_list()
            )
        middle = files.height // 2
        head = extract_files(files.head(middle).pipe(FileRow.validate))
        tail = extract_files(files.slice(middle).pipe(FileRow.validate))
        return Extraction(
            chunks=pl.concat([head.chunks, tail.chunks]).pipe(PlacedChunkRow.validate),
            overlaps=[*head.overlaps, *tail.overlaps],
            failed=[*head.failed, *tail.failed],
        )


def docstrings(chunks: Sequence[tuple[str, str]]) -> list[str | None]:
    """The documentation of the first documented symbol in each chunk's text, as written.

    Each of *chunks* is a text and its language, extracted as a file is;
    its documentation is the `docs` of its first chunk in its own language
    that has any. None for a chunk with no documented symbol.
    """
    files = file_rows(
        [(FileEntry("", "", content.encode()), language) for content, language in chunks]
    )
    documented = (
        extract_batch(files)
        .chunks.filter(pl.col("language") == pl.col("file_language"), pl.col("docs").is_not_null())
        .group_by("file", maintain_order=True)
        .agg(pl.col("docs").first())
    )
    found = dict(documented.iter_rows())
    return [found.get(file) for file in range(len(chunks))]
