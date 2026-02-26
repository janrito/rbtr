-- Check if any messages were added after a given fragment ID.
-- Uses UUIDv7 ordering (IDs are time-sortable).
-- Checks all messages regardless of compaction status.
-- Params: session_id, reference_id.
SELECT COUNT(*) AS new_count
FROM fragments
WHERE
  session_id = ?
  AND complete = 1
  AND id = message_id
  AND fragment_kind IN ('request-message', 'response-message')
  AND id > ?;
