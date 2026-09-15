-- Count this session's inserted (blob_sha, file_language) pairs that no
-- repo claims.  `_stg` holds one row per pair written by the session's
-- chunk inserts; a pair is claimed when some file_snapshots row
-- anywhere references it, which is the same key sweep_orphan_chunks
-- deletes by.  A non-zero count means the session is about to commit
-- chunks the next sweep would take.
SELECT count(*)
FROM _stg AS s
WHERE NOT EXISTS (
  SELECT 1
  FROM file_snapshots AS fs
  WHERE
    fs.blob_sha = s.blob_sha
    AND fs.detected_language = s.file_language
)
