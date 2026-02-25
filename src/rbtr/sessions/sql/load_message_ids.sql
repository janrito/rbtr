-- Load message-level row IDs for a session.
-- Params: session_id, before_created_at (NULL = no filter),
--         before_created_at (repeated for OR).
SELECT id
FROM fragments
WHERE
  session_id = ?
  AND message_id = id
  AND compacted_by IS NULL
  AND (? IS NULL OR created_at < ?)
ORDER BY created_at
