"""Schema migrations for the sessions database.

Each migration is a function that receives a `sqlite3.Connection`
and applies its changes.  The store's `_migrate` method calls them
in sequence.  SQL files live in `sql/` with a version prefix.

When versions are collapsed, remove the corresponding function
and SQL files.
"""

from __future__ import annotations

import importlib.resources
import logging
import shutil
import sqlite3
from pathlib import Path

log = logging.getLogger(__name__)

_sql_pkg = importlib.resources.files("rbtr_legacy.sessions") / "sql"


def _load_sql(name: str) -> str:
    return (_sql_pkg / name).read_text(encoding="utf-8")


# ── 2026_03_03_01 → 2026_03_08_01: add facts ────────────────────────

_2026030301_SQL = _load_sql("migrate_2026030301_to_2026030801.sql")


def migrate_2026030301(con: sqlite3.Connection) -> None:
    """Add the `facts` table, FTS index, and sync triggers."""
    log.info("sessions: migrating → v2026030801 (add facts)")
    con.executescript(_2026030301_SQL)


# ── 2026_03_08_01 → 2026_03_20_01: fix message ordering ─────────────

_2026030801_FIND_SQL = _load_sql("migrate_2026030801_find_inverted_pairs.sql")
_2026030801_SWAP_SQL = _load_sql("migrate_2026030801_swap_created_at.sql")


def migrate_2026030801(con: sqlite3.Connection) -> None:
    """Swap `created_at` for inverted response→request pairs.

    Before the fix, `begin_response` created the response row
    before the request was saved, giving responses an earlier
    `created_at`.  This swaps each inverted pair so the request
    sorts first on reload.
    """
    log.info("sessions: migrating → v2026032001 (fix message ordering)")
    rows = con.execute(_2026030801_FIND_SQL).fetchall()
    if not rows:
        return

    log.info("sessions: fixing %d inverted response→request pairs", len(rows))
    for row in rows:
        con.execute(_2026030801_SWAP_SQL, [row["req_at"], row["resp_id"]])
        con.execute(_2026030801_SWAP_SQL, [row["resp_at"], row["req_id"]])
    con.commit()


# ── Backup ───────────────────────────────────────────────────────────


def backup(db_path: Path | None) -> None:
    """Copy the DB file before a migration.

    No-op for in-memory databases.
    """
    if db_path is None:
        return
    dest = db_path.with_suffix(".db.pre-migration")
    shutil.copy2(db_path, dest)
    log.info("sessions: backed up %s → %s", db_path.name, dest.name)
