-- sqlfluff:templater:placeholder:repo_id:1
-- sqlfluff:templater:placeholder:snapshot_sha:'abc'
-- One row per indexed snapshot: how many chunks it holds, and how many
-- carry embeddings.  A chunk is content-addressed and reached once per
-- location holding its content, so the join fans out and the ids are
-- counted distinctly.  `embedding IS NULL` has to read the embedding
-- column, so both figures come from one grouped pass rather than a
-- query per snapshot.
-- A NULL parameter matches every value, which is how one statement
-- serves a single snapshot, a whole repo, and every repo.
SELECT
  fs.repo_id,
  fs.snapshot_sha,
  count(DISTINCT c.id) AS total,
  count(DISTINCT c.id) FILTER (WHERE c.embedding IS NOT NULL) AS embedded
FROM file_snapshots AS fs
INNER JOIN indexed_snapshots AS i
  ON
    fs.repo_id = i.repo_id
    AND fs.snapshot_sha = i.snapshot_sha
INNER JOIN chunks AS c
  ON
    fs.blob_sha = c.blob_sha
    AND fs.detected_language = c.file_language
WHERE
  ($repo_id IS NULL OR fs.repo_id = $repo_id)
  AND ($snapshot_sha IS NULL OR fs.snapshot_sha = $snapshot_sha)
GROUP BY fs.repo_id, fs.snapshot_sha
