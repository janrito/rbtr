UPDATE facts
SET
  last_confirmed_at = ?,
  confirm_count = confirm_count + 1
WHERE id = ?;
