SELECT 1
FROM chunks
WHERE
  repo_id = $repo_id
  AND blob_sha = $blob_sha
  AND language = $language
  AND language_plugin_version = $language_plugin_version
LIMIT 1 -- noqa: AM09
