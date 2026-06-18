INSERT INTO watched_refs (repo_id, ref)
SELECT
  $repo_id,
  unnest($refs::TEXT []) AS watched_ref
ON CONFLICT (repo_id, ref) DO NOTHING
