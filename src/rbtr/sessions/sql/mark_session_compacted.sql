UPDATE messages
SET compacted_by = ?
WHERE
  session_id = ?
  AND compacted_by IS NULL;
