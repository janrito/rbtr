-- Global existence check: chunks are content-addressed and shared
-- across repos, so a blob chunked under the current plugin version
-- by any repo need not be re-parsed.
SELECT 1
FROM chunks
WHERE
  blob_sha = $blob_sha
  AND language = $language
  AND language_plugin_version = $language_plugin_version
LIMIT 1 -- noqa: AM09
