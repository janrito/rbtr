-- Mark a message row as complete and set final metadata.
-- Params: cost, message_id.
UPDATE fragments
SET
  complete = 1,
  cost = ?
WHERE id = ?
