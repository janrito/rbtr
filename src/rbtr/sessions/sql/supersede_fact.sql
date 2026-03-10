UPDATE facts
SET superseded_by = ?
WHERE
  id = ?
  AND superseded_by IS NULL;
