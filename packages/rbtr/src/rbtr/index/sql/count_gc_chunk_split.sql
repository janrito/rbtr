-- sqlfluff:templater:placeholder:repo_id:1
-- sqlfluff:templater:placeholder:drop_shas:['x']
-- Split the chunks a GC drop set references into those that would be
-- freed versus those kept because another indexed ref still references
-- their (blob_sha, file_path).  Read-only: it does not simulate the
-- drop, so it serves dry-run and real runs identically.
--
-- `candidates` are the chunks referenced by a snapshot in the drop set
-- ($repo_id, $drop_shas).  A candidate is KEPT iff some snapshot
-- OUTSIDE that set (another ref of this repo, or any other repo)
-- references the same (blob_sha, file_path); the rest are DROPPED.
WITH candidates AS (
  SELECT DISTINCT
    c.id,
    c.blob_sha,
    c.file_path
  FROM chunks AS c
  INNER JOIN file_snapshots AS fs
    ON
      c.blob_sha = fs.blob_sha
      AND c.file_path = fs.file_path
  WHERE
    fs.repo_id = $repo_id
    AND fs.commit_sha IN (SELECT unnest($drop_shas::TEXT []))
),

kept AS (
  SELECT DISTINCT cand.id
  FROM candidates AS cand
  INNER JOIN file_snapshots AS keepfs
    ON
      cand.blob_sha = keepfs.blob_sha
      AND cand.file_path = keepfs.file_path
  WHERE NOT (
    keepfs.repo_id = $repo_id
    AND keepfs.commit_sha IN (SELECT unnest($drop_shas::TEXT []))
  )
)

SELECT
  (SELECT count(*) FROM candidates) - (SELECT count(*) FROM kept) AS dropped,
  (SELECT count(*) FROM kept) AS kept
