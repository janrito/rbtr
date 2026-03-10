DELETE FROM facts
WHERE
  scope = ?
  AND superseded_by IS NULL
  AND id NOT IN (
    SELECT id FROM facts
    WHERE
      scope = ?
      AND superseded_by IS NULL
    ORDER BY last_confirmed_at DESC
    LIMIT ?
  );
