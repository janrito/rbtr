-- Each of the paths' language as the repo's snapshots recorded it. Any
-- recorded language will do; the lowest snapshot sha picks one, so the
-- answer does not change between calls.
SELECT
  file_path,
  arg_min(detected_language, snapshot_sha) AS detected_language
FROM file_snapshots
WHERE
  repo_id = $repo_id
  AND file_path IN (SELECT unnest($file_paths::TEXT[]))
GROUP BY file_path
