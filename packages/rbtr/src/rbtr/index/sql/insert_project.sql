INSERT INTO projects (id, common_dir, file, compacted_bytes)
VALUES (nextval('projects_id_seq'), $common_dir, $file, $compacted_bytes)
RETURNING id
