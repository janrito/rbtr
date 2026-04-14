UPDATE fragments
SET
  data_json = ?,
  status = 'complete'
WHERE id = ?;
