SELECT user_text
FROM fragments
WHERE
  session_id = ?
  AND user_text IS NOT NULL
ORDER BY created_at DESC
LIMIT ?;
