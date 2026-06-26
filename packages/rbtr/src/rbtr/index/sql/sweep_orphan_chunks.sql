-- Delete chunks no longer referenced by any snapshot in ANY repo.
-- Chunks are content-addressed and shared across repos, so a chunk is
-- safe to remove iff no remaining file_snapshots row anywhere
-- references its (blob_sha, file_path).  This is the cross-repo
-- reference count that keeps shared chunks alive while any repo still
-- needs them.  The key matches prune_chunks: a blob backing chunks at
-- several paths is collected per path, so dropping one path does not
-- strand its chunk just because the blob lives on at another path.
--
-- INVARIANT: this global sweep is correct only because a build
-- commits a commit's chunks and their file_snapshots in the SAME
-- transaction (see WriteSession).  No committed state ever holds a
-- chunk without its snapshot, so a chunk with no referencing snapshot
-- is genuine garbage.  Splitting chunk and snapshot writes across
-- transactions would let this sweep delete another repo's live chunks.
DELETE FROM chunks
WHERE NOT EXISTS (
  SELECT 1
  FROM file_snapshots AS fs
  WHERE
    fs.blob_sha = chunks.blob_sha
    AND fs.file_path = chunks.file_path
)
