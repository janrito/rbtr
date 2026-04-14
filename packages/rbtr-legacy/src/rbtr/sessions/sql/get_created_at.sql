-- Get the created_at timestamp for a single fragment row.
-- Params: id.
SELECT created_at
FROM fragments
WHERE id = ?
