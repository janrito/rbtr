-- sqlfluff:templater:placeholder:repo_id:1
-- sqlfluff:templater:placeholder:snapshot_sha:'abc'
-- One row per indexed snapshot: how many chunks it holds, and how many
-- carry embeddings.  Driven from `indexed_snapshots` and joined outwards,
-- so a snapshot holding no chunks still gets a row, counting zero.
-- A chunk is content-addressed and reached once per location holding its
-- content, so the join fans out and the ids are counted distinctly.
-- Reading `chunks.embedding` is most of the work, so both figures come
-- from one grouped pass rather than a query per snapshot.
-- A NULL parameter matches every value, which is how one statement
-- serves a single snapshot, a whole repo, and every repo.
-- Newest first; `snapshot_sha` breaks ties, because `current_timestamp`
-- is fixed per transaction and snapshots marked together share one.
SELECT
  i.repo_id,
  i.snapshot_sha,
  count(DISTINCT c.id) AS total,
  count(DISTINCT c.id) FILTER (WHERE c.embedding IS NOT NULL) AS embedded
FROM indexed_snapshots AS i
LEFT JOIN file_snapshots AS fs
  ON
    i.repo_id = fs.repo_id
    AND i.snapshot_sha = fs.snapshot_sha
LEFT JOIN chunks AS c
  ON
    fs.blob_sha = c.blob_sha
    AND fs.detected_language = c.file_language
WHERE
  ($repo_id IS NULL OR i.repo_id = $repo_id)
  AND ($snapshot_sha IS NULL OR i.snapshot_sha = $snapshot_sha)
GROUP BY i.repo_id, i.snapshot_sha, i.indexed_at
ORDER BY i.indexed_at DESC, i.snapshot_sha ASC
