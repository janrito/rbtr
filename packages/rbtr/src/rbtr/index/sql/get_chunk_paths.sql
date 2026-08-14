-- Where the given chunks live, one row per chunk.
--
-- A chunk is content-addressed, so identical content at several paths
-- is one row in `chunks` reachable through every one of them.  The
-- caller pairs ids with a single location, so this collapses to the
-- first path in sort order; scoping to `_snapshot_refs` keeps paths
-- from unrelated snapshots out.
SELECT
  c.id,
  min(fs.file_path) AS file_path
FROM chunks AS c
INNER JOIN file_snapshots AS fs
  ON
    c.blob_sha = fs.blob_sha
    AND c.file_language = fs.detected_language
INNER JOIN _snapshot_refs AS rr
  ON fs.repo_id = rr.repo_id AND fs.snapshot_sha = rr.snapshot_sha
WHERE c.id IN (SELECT unnest($chunk_ids::text []))
GROUP BY c.id
