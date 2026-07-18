-- Distinct real languages present anywhere in the content-addressed
-- store.  The raw-chunk fallback stamps `language = ''`, so excluding
-- it leaves exactly the languages some plugin has extracted -- a record
-- of the plugin set that built the current index.  Global (no repo
-- filter): chunks are shared across repos.
SELECT DISTINCT language
FROM chunks
WHERE language <> ''
