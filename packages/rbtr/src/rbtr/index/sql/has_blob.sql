-- sqlfluff:templater:placeholder:blob_sha:'abc'
-- sqlfluff:templater:placeholder:language:'python'
-- Global up-to-date check: chunks are content-addressed and shared
-- across repos, so a blob whose chunks are all at the current extraction
-- serials need not be re-parsed by any repo.
--
-- A blob is up to date iff it has at least one chunk in the currently
-- detected host `language` and every chunk's (language,
-- extraction_serial) matches a row in the registered `_serial_map`
-- view. The host-language requirement catches a changed detection (a
-- file indexed as plaintext, then a plugin registered for its extension)
-- since every file leaves a host-language chunk; the serial match catches
-- a bumped plugin. A chunk with no matching serial row leaves `v.language`
-- NULL and forces re-extraction. Multi-language files (SFCs) list every
-- embedded language plus the host, so a bump to any contributor invalidates.
SELECT
  count(*) FILTER (WHERE c.language = $language) > 0
  AND count(*) FILTER (WHERE v.language IS NULL) = 0 AS up_to_date
FROM chunks AS c
LEFT JOIN _serial_map AS v
  ON
    c.language = v.language
    AND c.extraction_serial = v.extraction_serial
WHERE c.blob_sha = $blob_sha
