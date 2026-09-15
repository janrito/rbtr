-- Delete chunks no longer referenced by any snapshot in ANY repo.
-- Chunks are content-addressed and shared across repos, so a chunk is
-- safe to remove iff no remaining file_snapshots row anywhere
-- references its (blob_sha, file_path).  This is the cross-repo
-- reference count that keeps shared chunks alive while any repo still
-- needs them.  The key matches prune_chunks: a blob backing chunks at
-- several paths is collected per path, so dropping one path does not
-- strand its chunk just because the blob lives on at another path.
--
-- This global sweep is correct because a chunk enters only for a blob
-- some file_snapshots row already claims: WriteSession._commit refuses
-- a session that stores chunks for an unclaimed blob.  A chunk with no
-- referencing snapshot is therefore one whose claims have since been
-- removed — genuine garbage, not a half-written build.
DELETE FROM chunks
WHERE NOT EXISTS (
  SELECT 1
  FROM file_snapshots AS fs
  WHERE
    fs.blob_sha = chunks.blob_sha
    AND chunks.file_language = fs.detected_language
)
