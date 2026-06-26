-- Cross-repo prune: drop chunks not referenced by any snapshot in
-- any repo (keyed on blob_sha + file_path).  Like sweep_orphan_chunks,
-- this is safe only because a build commits chunks and their snapshots
-- in one transaction, so an unreferenced chunk is genuine garbage and
-- not a half-written build (see WriteSession's atomicity invariant).
DELETE FROM chunks
WHERE NOT EXISTS (
  SELECT 1
  FROM file_snapshots AS fs
  WHERE
    fs.blob_sha = chunks.blob_sha
    AND fs.file_path = chunks.file_path
)
