-- Global up-to-date check: chunks are content-addressed and shared
-- across repos, so a blob whose chunks are all at the current extraction
-- serials need not be re-parsed by any repo.
--
-- A blob is up to date for a language iff it was already extracted *as
-- that `language`* and every resulting chunk's (language,
-- extraction_serial) matches a row in the registered `_serial_map`
-- view.  Extraction always leaves at least one chunk, so rows for this
-- `file_language` are the record that the blob has been read this way:
-- the same bytes read as two languages are two extractions, and each
-- must be present for its own.  The serial match catches a bumped
-- plugin; a chunk with no matching serial row leaves `v.language` NULL
-- and forces re-extraction.  Multi-language files (SFCs) list every
-- embedded language plus the host, so a bump to any contributor
-- invalidates.  `_blobs` holds a batch's (blob, language) pairs; the
-- query returns those that are up to date.
SELECT
  b.blob_sha,
  b.language
FROM _blobs AS b
INNER JOIN chunks AS c
  ON
    b.blob_sha = c.blob_sha
    AND b.language = c.file_language
LEFT JOIN _serial_map AS v
  ON
    c.language = v.language
    AND c.extraction_serial = v.extraction_serial
GROUP BY b.blob_sha, b.language
HAVING count(*) FILTER (WHERE v.language IS NULL) = 0
