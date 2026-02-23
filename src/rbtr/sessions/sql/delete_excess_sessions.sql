DELETE FROM messages
WHERE session_id NOT IN (
  SELECT session_id
  FROM messages
  GROUP BY session_id
  ORDER BY MAX(created_at) DESC
  LIMIT ?
);
