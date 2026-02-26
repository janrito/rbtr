-- Delete a message-level fragment and its parts.
-- FK cascade on message_id handles part rows.
-- Params: message_id.
DELETE FROM fragments
WHERE
  id = ?;
