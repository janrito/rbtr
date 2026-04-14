SELECT
  id,
  scope,
  content,
  source_session_id,
  created_at,
  last_confirmed_at,
  confirm_count,
  superseded_by
FROM facts
WHERE
  content = ?
  AND scope = ?
  AND superseded_by IS NULL
ORDER BY last_confirmed_at DESC
LIMIT 1;
