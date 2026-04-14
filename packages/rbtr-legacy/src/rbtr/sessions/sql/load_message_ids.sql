-- Load message-level row IDs for a session.
-- Only returns request-message / response-message rows (same
-- set as load_messages.sql) so indices align for compaction.
-- Params: session_id, before_created_at (NULL = no filter),
--         before_created_at (repeated for OR).
SELECT id
FROM fragments
WHERE
  session_id = ?
  AND message_id = id
  AND fragment_kind IN ('request-message', 'response-message')
  AND compacted_by IS NULL
  AND (? IS NULL OR created_at < ?)
ORDER BY created_at
