SELECT
  id,
  scope,
  content,
  source_session_id,
  created_at,
  last_confirmed_at,
  confirm_count
FROM facts
WHERE
  scope = ?
  AND superseded_by IS NULL
ORDER BY last_confirmed_at DESC
LIMIT ?;
