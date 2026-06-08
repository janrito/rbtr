DELETE FROM chunks
WHERE
  repo_id = $repo_id
  AND blob_sha IN (SELECT unnest($blob_shas::TEXT []))
