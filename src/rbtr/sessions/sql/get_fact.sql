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
WHERE id = ?;
