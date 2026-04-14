-- Un-mark all fragments compacted by a given summary.
-- Params: summary_id.
UPDATE fragments
SET
  compacted_by = NULL
WHERE
  compacted_by = ?;
