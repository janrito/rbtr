SELECT id, file_path
FROM chunks
WHERE repo_id = ? AND id IN (SELECT unnest(?::text[]))
