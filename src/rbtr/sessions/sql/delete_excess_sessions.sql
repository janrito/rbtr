DELETE FROM fragments
WHERE session_id NOT IN (
  SELECT session_id
  FROM fragments
  GROUP BY session_id
  ORDER BY MAX(created_at) DESC
  LIMIT ?
);
