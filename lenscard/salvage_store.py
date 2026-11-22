"""Separate, small persistence layer for reverse-salvage jobs.

Salvage records never enter the card library: they are reconstructions, not
imported camera originals, so they get their own database and asset folder.
"""
import json
import sqlite3
from pathlib import Path

from .models import Missing, require_id, short_text, storage_id, utc_now
from .storage import encode_json, put_atomic

MAX_SALVAGE_JOBS = 12
SCHEMA = """
CREATE TABLE IF NOT EXISTS salvage_jobs (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    report TEXT NOT NULL,
    assets TEXT NOT NULL,
    created TEXT NOT NULL
);
"""


class SalvageStore:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "salvage.sqlite"
        self.asset_dir = self.root / "salvage-assets"
        connection = sqlite3.connect(self.db_path, timeout=15)
        try:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript(SCHEMA)
        finally:
            connection.close()

    def _database(self):
        connection = sqlite3.connect(self.db_path, timeout=15)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=5000")
        return connection

    def _asset_path(self, digest):
        return self.asset_dir / digest[:2] / digest

    def store_asset(self, payload):
        from hashlib import sha256
        digest = sha256(payload).hexdigest()
        put_atomic(self.asset_dir / digest[:2] / digest, payload)
        return {"sha256": digest, "size": len(payload), "filename": digest}

    def asset_bytes(self, reference):
        path = self._asset_path(reference["sha256"])
        if not path.is_file():
            raise Missing("This salvage asset is missing from local storage.")
        return path.read_bytes()

    def save_job(self, title, report, assets):
        title = short_text(title, "salvage title", maximum=100)
        with self._database() as db:
            count = db.execute("SELECT COUNT(*) FROM salvage_jobs").fetchone()[0]
            if count >= MAX_SALVAGE_JOBS:
                # Drop the oldest job row only. Assets are content-addressed and may
                # still be referenced by other jobs, so files are reaped separately.
                db.execute("DELETE FROM salvage_jobs WHERE id IN ("
                           "SELECT id FROM salvage_jobs ORDER BY created LIMIT ?)",
                           (count - MAX_SALVAGE_JOBS + 1,))
            identifier = storage_id()
            db.execute(
                "INSERT INTO salvage_jobs(id, title, report, assets, created) VALUES (?, ?, ?, ?, ?)",
                (identifier, title, encode_json(report), encode_json(assets), utc_now()),
            )
        return identifier

    def reap_orphan_assets(self):
        """Delete asset files no surviving job references. Returns the count."""
        with self._database() as db:
            rows = db.execute("SELECT assets FROM salvage_jobs").fetchall()
        referenced = set()
        for row in rows:
            referenced.update(ref["sha256"] for ref in json.loads(row["assets"]).values())
        removed = 0
        if not self.asset_dir.exists():
            return removed
        for path in self.asset_dir.glob("*/*"):
            if path.is_file() and path.name not in referenced:
                path.unlink()
                removed += 1
        return removed

    def list_jobs(self):
        with self._database() as db:
            rows = db.execute(
                "SELECT id, title, report, created FROM salvage_jobs ORDER BY created DESC"
            )
            return [
                {"id": row["id"], "title": row["title"], "created": row["created"],
                 "summary": json.loads(row["report"])}
                for row in rows
            ]

    def job(self, identifier):
        require_id(identifier, "salvage job")
        with self._database() as db:
            row = db.execute("SELECT * FROM salvage_jobs WHERE id=?", (identifier,)).fetchone()
        if row is None:
            raise Missing("This salvage job is no longer available.")
        return {
            "id": row["id"], "title": row["title"], "created": row["created"],
            "report": json.loads(row["report"]), "assets": json.loads(row["assets"]),
        }
