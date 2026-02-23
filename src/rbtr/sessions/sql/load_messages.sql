SELECT message_json
FROM messages
WHERE
  session_id = ?
  AND compacted_by IS NULL
  AND message_json IS NOT NULL
ORDER BY created_at;
