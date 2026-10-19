INSERT INTO projects (id, common_dir, file)
VALUES (nextval('projects_id_seq'), $common_dir, $file)
RETURNING id
