-- Content-addressed store: chunks are shared across repos, so a
-- blob's chunks are deleted globally (chunking is deterministic per
-- plugin version, so every repo sees the same chunks for a blob).
DELETE FROM chunks
WHERE blob_sha IN (SELECT unnest($blob_shas::TEXT []))
