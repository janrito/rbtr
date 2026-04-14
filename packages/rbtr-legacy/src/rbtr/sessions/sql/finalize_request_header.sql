-- Update the header row's data_json with post-mutation request data.
-- Params: data_json, message_id.
UPDATE fragments
SET data_json = ?
WHERE id = ?;
