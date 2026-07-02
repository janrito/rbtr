-- SQLite dialect sample — AUTOINCREMENT, WITHOUT ROWID, IF NOT EXISTS.

CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT NOT NULL UNIQUE,
    created_at TEXT DEFAULT (datetime('now'))
) WITHOUT ROWID;

CREATE INDEX idx_users_email ON users (email);

CREATE VIEW active_users AS SELECT id, email FROM users;
