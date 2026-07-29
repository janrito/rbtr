SELECT detected_language
FROM file_snapshots
WHERE
  repo_id = $repo_id
  AND file_path = $file_path
ORDER BY snapshot_sha
LIMIT 1
