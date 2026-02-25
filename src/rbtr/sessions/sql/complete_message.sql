-- Mark a message row as complete and set final metadata.
-- Params: cost, input_tokens, output_tokens, cache_read_tokens,
--         cache_write_tokens, message_id.
UPDATE fragments
SET
  complete = 1,
  cost = ?,
  input_tokens = ?,
  output_tokens = ?,
  cache_read_tokens = ?,
  cache_write_tokens = ?
WHERE id = ?
