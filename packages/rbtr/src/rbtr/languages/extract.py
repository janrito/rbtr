"""Extraction of a batch of files: the adapter reads them, and the rules make their chunks.

`extract_batch` reads every file of a batch, and every block it embeds,
through tree-sitter and the language plugins (`rbtr.languages.treesitter`),
then folds comments and assembles chunks over all of them at once
(`rbtr.domain.rules`). The indexer extracts its files `BATCH_FILES` at
a time through `extract_files`; `extract_file` runs one file the same
way, for plugin tests and third-party language packages.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import dataframely as dy
import polars as pl
import structlog

from rbtr.config import config
from rbtr.domain.models import Chunk
from rbtr.domain.rules import (
    FileRow,
    PlacedChunkRow,
    assemble,
    capture_chunks,
    line_windows,
    with_metadata,
)
from rbtr.git import FileEntry
from rbtr.languages.chunks import chunk_models
from rbtr.languages.plaintext import PLAINTEXT
from rbtr.languages.treesitter import Languages, read, registered

if TYPE_CHECKING:
    from tree_sitter import Language

    from rbtr.languages.registration import LanguageRegistration

log = structlog.get_logger(__name__)

BATCH_FILES = 100
"""Files extracted together: a run of the rules costs about 9 ms, so a run
per file costs five times what it costs a file in a batch of 100."""


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


def extract_batch(
    files: dy.DataFrame[FileRow], languages: Languages = registered
) -> dy.DataFrame[PlacedChunkRow]:
    """*files*' chunks, each file's in order, every chunk stamped with its file's language.

    The adapter reads each file and every block it embeds (`read`); the
    rules make chunks of the captures, cut the files nothing reads into
    `PLAINTEXT` windows of lines, and assemble each file's, over all the
    files at once.
    """
    captures, metadata_captures, chunked, plain = read(files, languages)
    # Each rule is one lazy plan, its result checked within it, collected
    # before the next rule reads it: a plan copies a frame it reads twice into
    # each use, and `with_metadata` reads its inputs many times.
    described = with_metadata(captures.lazy(), metadata_captures.lazy()).collect()
    return assemble(
        files,
        capture_chunks(files, described.lazy()),
        chunked.lazy(),
        line_windows(files, plain, config.chunk_lines, config.chunk_overlap, PLAINTEXT),
    ).collect()


def extract_files(
    files: dy.DataFrame[FileRow],
) -> tuple[dy.DataFrame[PlacedChunkRow], list[str]]:
    """*files*' chunks, and the paths of the files whose extraction raised.

    A batch whose extraction raises is halved, and each half extracted
    again, down to the file that raises: it is logged, and has only its
    presence chunk, so its snapshot row still reaches a chunk.
    """
    try:
        return extract_batch(files), []
    except Exception:
        if files.height == 1:
            log.exception("index_file_failed", path=files["path"].item())
            return assemble(files, PlacedChunkRow.create_empty(lazy=True)).collect(), files[
                "path"
            ].to_list()
        middle = files.height // 2
        head, head_failed = extract_files(files.head(middle).pipe(FileRow.validate))
        tail, tail_failed = extract_files(files.slice(middle).pipe(FileRow.validate))
        return pl.concat([head, tail]).pipe(PlacedChunkRow.validate), [*head_failed, *tail_failed]


def extract_file(entry: FileEntry, language: str) -> list[Chunk]:
    """Extract a file's chunks, always including one in its own language.

    Public so plugin tests (and third-party language packages) drive the
    *real* pipeline rather than a copy. Given a file's
    content and language it reads the file as its language's chunker,
    tree-sitter query, or plaintext line chunks, and any embedded-language
    injections (an SFC's `<script>`/`<style>`) on top. If none of that
    produced a chunk in the file's own *language*, a content-less
    host-presence chunk is appended so the dedup gate can skip the file on
    later builds instead of re-parsing it. The caller handles blob dedup
    and deletes stale chunks first.

    Every chunk leaves here stamped with *language* as its
    `file_language`. It is part of a chunk's identity, so the id follows
    from the stamp.
    """
    return chunk_models(extract_batch(file_rows([(entry, language)])), entry.path)


def extract_symbols(
    registered_language: LanguageRegistration,
    file_path: str,
    blob_sha: str,
    content: bytes,
    grammar: Language,
) -> list[Chunk]:
    """*content*'s chunks, read as *registered_language* with *grammar* and no other language.

    For a registration no plugin installs: a block it hands another
    language is left unread.
    """
    files = file_rows([(FileEntry(file_path, blob_sha, content), registered_language.id)])
    languages = {registered_language.id: (registered_language, grammar)}.get
    return chunk_models(extract_batch(files, languages), file_path)


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
        .filter(pl.col("language") == pl.col("file_language"), pl.col("docs").is_not_null())
        .group_by("file", maintain_order=True)
        .agg(pl.col("docs").first())
    )
    found = dict(documented.iter_rows())
    return [found.get(file) for file in range(len(chunks))]
