-- The indexed snapshots, in any repo of the project, that list one of the
-- extractions in `_stg` (blob_sha, file_language): the snapshots whose
-- chunks change when those blobs are extracted again, so whose edges are
-- inferred again.
SELECT DISTINCT
  fs.repo_id,
  fs.snapshot_sha
FROM file_snapshots AS fs
INNER JOIN indexed_snapshots AS i
  ON fs.repo_id = i.repo_id AND fs.snapshot_sha = i.snapshot_sha
INNER JOIN _stg AS s
  ON fs.blob_sha = s.blob_sha AND fs.detected_language = s.file_language
ORDER BY fs.repo_id, fs.snapshot_sha
