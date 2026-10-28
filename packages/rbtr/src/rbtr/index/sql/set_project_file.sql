UPDATE projects
SET file = $file, compacted_bytes = $compacted_bytes
WHERE id = $project_id
