-- Indexed snapshots that still hold a chunk without a vector, newest
-- first: the embedder's work list.  `EXISTS` stops at the first such
-- chunk and keeps no per-snapshot state, where counting every chunk
-- (`chunk_counts_by_snapshot.sql`) needed nearly 1 GiB over a 40-project,
-- 16 GB index and failed beside a concurrent search.
-- Newest first; `snapshot_sha` breaks ties, because `current_timestamp`
-- is fixed per transaction and snapshots marked together share one.
SELECT
  i.repo_id,
  i.snapshot_sha,
  i.indexed_at
FROM indexed_snapshots AS i
WHERE
  EXISTS (
    SELECT 1
    FROM file_snapshots AS fs
    INNER JOIN chunks AS c
      ON
        fs.blob_sha = c.blob_sha
        AND fs.detected_language = c.file_language
    WHERE
      fs.repo_id = i.repo_id
      AND fs.snapshot_sha = i.snapshot_sha
      AND c.embedding IS NULL
  )
ORDER BY i.indexed_at DESC, i.snapshot_sha ASC
