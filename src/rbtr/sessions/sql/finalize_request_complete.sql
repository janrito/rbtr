-- Mark all rows for a message as complete.
-- Params: message_id.
UPDATE fragments
SET status = 'complete'
WHERE message_id = ?;
