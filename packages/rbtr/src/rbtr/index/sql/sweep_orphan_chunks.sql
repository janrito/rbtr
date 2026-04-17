-- Delete chunks whose blob is no longer referenced by any snapshot.
-- This is the invariant that keeps blob-shared chunks correct when
-- commits are dropped: a chunk is safe to remove iff no remaining
-- file_snapshots row references its blob_sha.
DELETE FROM chunks
WHERE repo_id = ?
  AND NOT EXISTS (
    SELECT 1
    FROM file_snapshots AS fs
    WHERE fs.repo_id = chunks.repo_id
      AND fs.blob_sha = chunks.blob_sha
  )
