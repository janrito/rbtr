-- Set created_at on all fragments for a given message.
-- Params: created_at, message_id.
UPDATE fragments
SET created_at = ?
WHERE message_id = ?;
