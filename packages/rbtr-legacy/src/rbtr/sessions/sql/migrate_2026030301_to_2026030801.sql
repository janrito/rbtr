-- Migration: 2026_03_03_01 → 2026_03_08_01
-- Adds: facts table, facts_fts FTS5 virtual table, sync triggers.
-- Non-destructive: fragments table is untouched.

CREATE TABLE IF NOT EXISTS facts (
  id TEXT PRIMARY KEY,
  scope TEXT NOT NULL,
  content TEXT NOT NULL,
  source_session_id TEXT NOT NULL,
  created_at TEXT NOT NULL,
  last_confirmed_at TEXT NOT NULL,
  confirm_count INTEGER NOT NULL DEFAULT 1,
  superseded_by TEXT REFERENCES facts (id)
);

CREATE INDEX IF NOT EXISTS idx_facts_scope
ON facts (scope, superseded_by)
WHERE superseded_by IS NULL;

CREATE INDEX IF NOT EXISTS idx_facts_recency
ON facts (last_confirmed_at DESC)
WHERE superseded_by IS NULL;

CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts  -- noqa: PRS
USING fts5(content, content=facts, content_rowid=rowid);  -- noqa: PRS

CREATE TRIGGER IF NOT EXISTS facts_ai AFTER INSERT ON facts BEGIN
  INSERT INTO facts_fts (rowid, content)
  VALUES (NEW.rowid, NEW.content);
END;

CREATE TRIGGER IF NOT EXISTS facts_ad AFTER DELETE ON facts BEGIN
  INSERT INTO facts_fts (facts_fts, rowid, content)
  VALUES ('delete', OLD.rowid, OLD.content);
END;

CREATE TRIGGER IF NOT EXISTS facts_au AFTER UPDATE OF content ON facts BEGIN
  INSERT INTO facts_fts (facts_fts, rowid, content)
  VALUES ('delete', OLD.rowid, OLD.content);
  INSERT INTO facts_fts (rowid, content)
  VALUES (NEW.rowid, NEW.content);
END;
