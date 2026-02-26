SELECT
  id,
  message_id,
  fragment_index,
  fragment_kind,
  data_json
FROM fragments
WHERE
  session_id = ?
  AND compacted_by IS NULL
  AND complete = 1
ORDER BY created_at, message_id, fragment_index;
