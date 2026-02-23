DELETE FROM messages
WHERE session_id IN (
  SELECT session_id
  FROM messages
  GROUP BY session_id
  HAVING MAX(created_at) < ?
);
