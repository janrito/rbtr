DELETE FROM watched_refs
WHERE
  repo_id = $repo_id
  AND ref IN (SELECT unnest($refs::TEXT []))
