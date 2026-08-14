-- Distinct languages present anywhere in the content-addressed store:
-- a record of the plugin set that built the current index, which the
-- daemon checks it can still serve.  Plaintext is registered like any
-- other language and appears here under its own name.  A chunk with no
-- language recorded contributes nothing to the plugin set, so the empty
-- string is excluded.  Global (no repo filter): chunks are shared
-- across repos.
SELECT DISTINCT language
FROM chunks
WHERE language <> ''
