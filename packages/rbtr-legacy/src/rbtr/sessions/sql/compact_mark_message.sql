-- Mark a single message and all its content fragments as compacted.
-- Params: summary_id, message_id, message_id.
UPDATE fragments
SET
  compacted_by = ?
WHERE
  compacted_by IS NULL
  AND (id = ? OR message_id = ?)
