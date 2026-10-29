-- Every file in the snapshots `_snapshot_refs` names, with its detected
-- language: what a module-qualified symbol name is resolved against.
SELECT DISTINCT
  fs.repo_id,
  fs.file_path,
  fs.detected_language
FROM file_snapshots AS fs
INNER JOIN _snapshot_refs AS rr
  ON fs.repo_id = rr.repo_id AND fs.snapshot_sha = rr.snapshot_sha
