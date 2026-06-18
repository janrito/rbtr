SELECT ref
FROM watched_refs
WHERE repo_id = $repo_id
ORDER BY ref
