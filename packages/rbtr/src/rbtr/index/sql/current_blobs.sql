-- The session cursor selects this project's database. Its checkouts
-- share chunks, so another checkout can reuse a current extraction.
--
-- A blob is up to date for a host language iff it has stored chunks
-- under that `file_language` and every resulting chunk's (language,
-- extraction_serial) matches a row in the registered `_serial_map`
-- view.  Extraction always leaves at least one chunk, so rows for this
-- `file_language` are the record that the blob has been read this way:
-- the same bytes read as two languages are two extractions, and each
-- must be present for its own.  The serial match catches a bumped
-- plugin; a chunk with no matching serial row leaves `v.language` NULL
-- and forces re-extraction.  Multi-language files (SFCs) list every
-- embedded language plus the host, so a bump to any contributor
-- invalidates. The bound list limits results to the requested blobs;
-- the query returns their current host-language extractions.
SELECT
  c.blob_sha,
  c.file_language
FROM chunks AS c
LEFT JOIN _serial_map AS v
  ON
    c.language = v.language
    AND c.extraction_serial = v.extraction_serial
WHERE c.blob_sha IN (SELECT unnest($blob_shas::TEXT[]))
GROUP BY c.blob_sha, c.file_language
HAVING count(*) FILTER (WHERE v.language IS NULL) = 0
