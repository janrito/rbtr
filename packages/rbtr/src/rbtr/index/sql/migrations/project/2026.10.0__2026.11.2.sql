-- A chunk's id becomes the hash of its blob, file language and byte span,
-- and its row stores the span (`start_byte`, `end_byte`).  Every stored id
-- and every edge naming one is of an earlier kind, so the project file is
-- emptied of what a build writes and rebuilt from git: each watched ref's
-- next build extracts its files again, and their chunks are embedded
-- again.  Watched refs and `meta` are kept.  It serves a file at any
-- version from 2026.10.0: the earlier migrations changed only `chunks` and
-- the chunk ids in `edges`, both deleted here.  DuckDB adds no NOT NULL
-- column or check constraint to a table, so `chunks` is made again.
DELETE FROM edges;

DELETE FROM indexed_snapshots;

DELETE FROM file_snapshots;

DROP INDEX idx_chunks_blob;

DROP TABLE chunks;

CREATE TABLE chunks (
  -- Content-addressed store: `id` is a sha256 (first 16 hex) of
  -- (blob_sha, file_language, start_byte, end_byte): one span of a blob
  -- read as one language is one row, whatever its kind and name, and is
  -- location-independent.  Identical content at several paths, or in
  -- several checkouts of the project, is one row here; where it lives is
  -- entirely `file_snapshots`, joined on blob_sha + file_language =
  -- detected_language.  See ARCHITECTURE.md "Content-addressed chunks".
  id TEXT NOT NULL,
  blob_sha TEXT NOT NULL,
  kind TEXT NOT NULL,
  name TEXT NOT NULL,
  scope TEXT NOT NULL DEFAULT '',
  -- The chunk's own language; `file_language` is the language the
  -- file was extracted as.  A python fence in a markdown document
  -- has language='python', file_language='markdown'.
  language TEXT NOT NULL DEFAULT '',
  file_language TEXT NOT NULL DEFAULT '',
  extraction_serial INTEGER NOT NULL DEFAULT 1,
  content TEXT NOT NULL,
  content_tokens TEXT NOT NULL DEFAULT '',
  name_tokens TEXT NOT NULL DEFAULT '',
  -- `content` is the blob's text from `start_byte` to `end_byte`.
  start_byte BIGINT NOT NULL,
  end_byte BIGINT NOT NULL,
  line_start INTEGER NOT NULL,
  line_end INTEGER NOT NULL,
  -- What the chunk's role carries beyond kind and name: an import's
  -- module, names, target language and bindings; a fragment's syntax node
  -- types; the calls and bases a definition, class or fragment writes
  -- (`refs`, each with the edge it makes).
  -- NULL for a role that carries nothing.
  metadata STRUCT(
    module TEXT,
    names TEXT,
    language_hint TEXT,
    -- Diagnostic: rbtr-eval's index report reads it to find syntax no
    -- query captures; search and the edge graph do not.
    node_types TEXT[],
    refs STRUCT(kind TEXT, lookup TEXT, path TEXT[], type TEXT[])[],
    bindings STRUCT(name TEXT, local TEXT)[]
  ),
  -- Variable-length embedding vector; dimension is determined by
  -- the model (see ARCHITECTURE.md "Embedding column").
  embedding FLOAT[] DEFAULT NULL,
  embedding_truncated BOOLEAN NOT NULL DEFAULT FALSE,
  PRIMARY KEY (id),
  -- A span ends on or after the byte and the line it starts on.
  CONSTRAINT chunk_span CHECK (
    end_byte >= start_byte
    AND line_end >= line_start
  ),
  -- Truncation describes an embedding that exists.
  CONSTRAINT truncation_needs_embedding CHECK (
    embedding IS NOT NULL
    OR NOT embedding_truncated
  )
);

CREATE INDEX idx_chunks_blob
ON chunks (blob_sha, file_language);
