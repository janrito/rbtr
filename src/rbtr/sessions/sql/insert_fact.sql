INSERT INTO facts (
  id, scope, content, source_session_id,
  created_at, last_confirmed_at, confirm_count
) VALUES (?, ?, ?, ?, ?, ?, 1);
