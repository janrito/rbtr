SELECT DISTINCT user_text
FROM messages
WHERE
  user_text IS NOT NULL
  AND (? IS NULL OR user_text LIKE ? || '%')
ORDER BY created_at DESC
LIMIT ?;
