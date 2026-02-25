DELETE FROM fragments
WHERE session_id IN (
  SELECT session_id
  FROM fragments
  GROUP BY session_id
  HAVING MAX(created_at) < ?
);
