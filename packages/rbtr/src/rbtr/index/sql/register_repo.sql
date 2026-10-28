INSERT INTO repos (id, path, project_id)
VALUES (nextval('repos_id_seq'), $path, $project_id)
RETURNING id
