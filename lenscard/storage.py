"""Persistent card library, original bytes and immutable tested print records."""
from contextlib import contextmanager
from hashlib import sha256
import json
import os
from pathlib import Path
import sqlite3
import tempfile

from .models import (
    Capacity, Conflict, MAX_CARDS, MAX_VIEWS, Missing, Plan, Unprocessable,
    ViewSetting, require_id, require_sha, short_text, storage_id, utc_now,
)


SCHEMA = """
CREATE TABLE IF NOT EXISTS cards (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    note TEXT NOT NULL,
    synthetic INTEGER NOT NULL,
    version INTEGER NOT NULL DEFAULT 0,
    config TEXT NOT NULL,
    created TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS views (
    id TEXT PRIMARY KEY,
    card_id TEXT NOT NULL REFERENCES cards(id) ON DELETE CASCADE,
    sha TEXT NOT NULL,
    name TEXT NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    bytes INTEGER NOT NULL,
    information TEXT NOT NULL,
    title TEXT NOT NULL,
    position REAL NOT NULL,
    enabled INTEGER NOT NULL,
    exposure_ev REAL NOT NULL,
    created TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS views_card ON views(card_id, position, id);
CREATE TABLE IF NOT EXISTS proofs (
    id TEXT PRIMARY KEY,
    card_id TEXT NOT NULL REFERENCES cards(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    state TEXT NOT NULL,
    base_version INTEGER NOT NULL,
    fingerprint TEXT NOT NULL,
    snapshot TEXT NOT NULL,
    assets TEXT NOT NULL,
    created TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS proofs_card ON proofs(card_id, created);
CREATE TABLE IF NOT EXISTS approval (
    id TEXT PRIMARY KEY,
    card_id TEXT NOT NULL REFERENCES cards(id) ON DELETE CASCADE,
    proof_id TEXT NOT NULL REFERENCES proofs(id),
    note TEXT NOT NULL,
    created TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS approval_card ON approval(card_id, created);
CREATE TABLE IF NOT EXISTS measurements (
    id TEXT PRIMARY KEY,
    card_id TEXT NOT NULL REFERENCES cards(id) ON DELETE CASCADE,
    proof_id TEXT REFERENCES proofs(id),
    kind TEXT NOT NULL,
    title TEXT NOT NULL,
    report TEXT NOT NULL,
    assets TEXT NOT NULL,
    created TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS measurements_card ON measurements(card_id, created);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY,
    card_id TEXT NOT NULL REFERENCES cards(id) ON DELETE CASCADE,
    entity TEXT NOT NULL,
    action TEXT NOT NULL,
    message TEXT NOT NULL,
    details TEXT NOT NULL,
    created TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS event_card ON events(card_id, id);
"""

MAX_PROOFS = 36
MAX_MEASUREMENTS = 60
MAX_CARD_SOURCE_BYTES = 230 * 1024**2
VIEW_FIELDS = "id, card_id, sha, name, width, height, bytes, information, title, position, enabled, exposure_ev, created"
CARD_FIELDS = "id, title, note, synthetic, version, config, created"


def encode_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(",", ":"))


def digest_object(value):
    return sha256(encode_json(value).encode()).hexdigest()


def decode_row(row, *fields):
    item = dict(row)
    for key in fields:
        item[key] = json.loads(item[key])
    for key in ("synthetic", "enabled"):
        if key in item:
            item[key] = bool(item[key])
    return item


def put_atomic(path, content):
    if path.is_file():
        if path.stat().st_size != len(content):
            raise Conflict("Stored bytes at this digest have an unexpected size.")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, staged = tempfile.mkstemp(dir=path.parent, prefix=".pending-")
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(staged, path)
    finally:
        if os.path.exists(staged):
            os.unlink(staged)


class Library:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "lenscard.sqlite"
        self.original_dir = self.root / "originals"
        self.preview_dir = self.root / "previews"
        self.evidence_dir = self.root / "rendered"
        connection = sqlite3.connect(self.db_path, timeout=15)
        try:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript(SCHEMA)
        finally:
            connection.close()

    @contextmanager
    def database(self, *, write=False):
        connection = sqlite3.connect(self.db_path, timeout=15)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("PRAGMA busy_timeout=5000")
            connection.execute("BEGIN IMMEDIATE" if write else "BEGIN")
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def source_path(self, digest):
        key = require_sha(digest, "original")
        return self.original_dir / key[:2] / key

    def preview_path(self, digest):
        key = require_sha(digest, "preview")
        return self.preview_dir / key[:2] / f"{key}.webp"

    def artifact_path(self, digest):
        key = require_sha(digest, "render")
        return self.evidence_dir / key[:2] / key

    def store_source(self, original, preview):
        digest = sha256(original).hexdigest()
        put_atomic(self.source_path(digest), original)
        put_atomic(self.preview_path(digest), preview)
        return digest

    def store_asset(self, data, mime, filename):
        digest = sha256(data).hexdigest()
        put_atomic(self.artifact_path(digest), data)
        return {"sha256": digest, "size": len(data), "mime": mime, "filename": filename}

    def asset_bytes(self, reference):
        key = reference["sha256"]
        path = self.artifact_path(key)
        if not path.is_file():
            raise Unprocessable("A retained print asset is missing.")
        result = path.read_bytes()
        if len(result) != reference["size"] or sha256(result).hexdigest() != key:
            raise Conflict("Retained print bytes fail integrity verification.")
        return result

    @staticmethod
    def _event(db, card_id, entity, action, message, details=None):
        db.execute(
            "INSERT INTO events(card_id, entity, action, message, details, created) VALUES (?, ?, ?, ?, ?, ?)",
            (card_id, entity, action, message, encode_json(details or {}), utc_now()),
        )

    @staticmethod
    def _card(db, card_id):
        row = db.execute(f"SELECT {CARD_FIELDS} FROM cards WHERE id=?", (require_id(card_id, "card"),)).fetchone()
        if row is None:
            raise Missing("This card is no longer in the local library.")
        return decode_row(row, "config")

    @staticmethod
    def _views(db, card_id):
        rows = db.execute(
            f"SELECT {VIEW_FIELDS} FROM views WHERE card_id=? ORDER BY position, created, id",
            (card_id,),
        )
        return [decode_row(row, "information") for row in rows]

    @staticmethod
    def _proof(db, card_id, proof_id):
        row = db.execute(
            "SELECT * FROM proofs WHERE card_id=? AND id=?",
            (card_id, require_id(proof_id, "proof")),
        ).fetchone()
        if row is None:
            raise Missing("This print proof does not belong to the selected card.")
        return decode_row(row, "assets", "snapshot")

    @staticmethod
    def _latest_approval(db, card_id):
        row = db.execute(
            "SELECT * FROM approval WHERE card_id=? ORDER BY created DESC, id DESC LIMIT 1",
            (card_id,),
        ).fetchone()
        return dict(row) if row is not None else None

    @staticmethod
    def _version(db, card_id, expected, *, increment=True):
        card = Library._card(db, card_id)
        if expected is not None and card["version"] != expected:
            raise Conflict("The card changed. Refresh before editing this candidate.", details={"current": card["version"]})
        if increment:
            result = db.execute("UPDATE cards SET version=version+1 WHERE id=? AND version=?", (card_id, card["version"]))
            if result.rowcount != 1:
                raise Conflict("Another edit changed this card.")
        return card["version"]

    def list_cards(self):
        with self.database() as db:
            rows = db.execute(
                "SELECT c.id, c.title, c.synthetic, c.version, c.created, "
                "(SELECT COUNT(*) FROM views v WHERE v.card_id=c.id) AS view_count, "
                "(SELECT MAX(p.created) FROM approval p WHERE p.card_id=c.id) AS approved_at "
                "FROM cards c ORDER BY c.created DESC, c.id",
            )
            return [decode_row(row) for row in rows]

    def create_card(self, title, note="", *, plan=None, synthetic=False):
        identifier = storage_id()
        title = short_text(title, "card title")
        note = short_text(note, "working note", maximum=500, allow_blank=True)
        config = Plan.parse(plan or {}).public()
        with self.database(write=True) as db:
            if db.execute("SELECT COUNT(*) FROM cards").fetchone()[0] >= MAX_CARDS:
                raise Capacity("The local library is full. Archive work from a separate library first.")
            db.execute(
                "INSERT INTO cards(id,title,note,synthetic,config,created) VALUES (?, ?, ?, ?, ?, ?)",
                (identifier, title, note, int(synthetic), encode_json(config), utc_now()),
            )
            self._event(db, identifier, identifier, "card-created", "Started a print card.")
            return self._card(db, identifier)

    def get_card(self, card_id):
        with self.database() as db:
            card = self._card(db, card_id)
            active = self._latest_approval(db, card_id)
            return {**card, "views": self._views(db, card_id), "approval": active}

    def update_card(self, card_id, *, version, title, note, config):
        settings = Plan.parse(config).public()
        title = short_text(title, "card title")
        note = short_text(note, "working note", maximum=500, allow_blank=True)
        with self.database(write=True) as db:
            self._version(db, card_id, version)
            db.execute(
                "UPDATE cards SET title=?, note=?, config=? WHERE id=?",
                (title, note, encode_json(settings), card_id),
            )
            self._event(db, card_id, card_id, "settings-saved", "Saved a new working draft.", {"based_on": version})
        return self.get_card(card_id)

    def locate_demo(self, title):
        with self.database() as db:
            row = db.execute(
                "SELECT id FROM cards WHERE synthetic=1 AND title=? ORDER BY created LIMIT 1", (title,)
            ).fetchone()
            return row["id"] if row is not None else None

    def add_view(self, card_id, frame, settings, *, version=None):
        return self.add_views(card_id, [(frame, settings)], version=version)[0]

    def add_views(self, card_id, entries, *, version=None):
        with self.database(write=True) as db:
            self._version(db, card_id, version)
            existing = self._views(db, card_id)
            if len(existing) + len(entries) > MAX_VIEWS:
                raise Capacity(f"Reserve at most {MAX_VIEWS} views in one card.")
            if sum(v["bytes"] for v in existing) + sum(frame["bytes"] for frame, _ in entries) > MAX_CARD_SOURCE_BYTES:
                raise Capacity("The original view bytes exceed this card's local budget.")
            new = []
            for frame, settings in entries:
                view_id = storage_id()
                setting = ViewSetting.parse(settings, {"title": frame["name"], "position": frame["position"]})
                db.execute(
                    "INSERT INTO views(" + VIEW_FIELDS + ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        view_id, card_id, frame["sha"], frame["name"], frame["width"], frame["height"],
                        frame["bytes"], encode_json(frame["information"]), setting.title, setting.position,
                        int(setting.enabled), setting.exposure_ev, utc_now(),
                    ),
                )
                self._event(db, card_id, view_id, "view-imported", "Preserved an original angle view.", {"sha": frame["sha"]})
                new.append(view_id)
            return [self._view(db, card_id, view_id) for view_id in new]

    @staticmethod
    def _view(db, card_id, view_id):
        row = db.execute(
            f"SELECT {VIEW_FIELDS} FROM views WHERE card_id=? AND id=?",
            (card_id, require_id(view_id, "view")),
        ).fetchone()
        if row is None:
            raise Missing("This original is not in this card.")
        return decode_row(row, "information")

    def view(self, card_id, view_id):
        with self.database() as db:
            return self._view(db, card_id, view_id)

    def edit_view(self, card_id, view_id, changes, version):
        with self.database(write=True) as db:
            self._version(db, card_id, version)
            current = self._view(db, card_id, view_id)
            spec = ViewSetting.parse(changes, current)
            db.execute(
                "UPDATE views SET title=?, position=?, enabled=?, exposure_ev=? WHERE card_id=? AND id=?",
                (spec.title, spec.position, int(spec.enabled), spec.exposure_ev, card_id, view_id),
            )
            self._event(
                db, card_id, view_id, "view-adjusted", "Edited view metadata; original bytes retained.",
                {"old_position": current["position"], "new_position": spec.position},
            )
            return self._view(db, card_id, view_id)

    def material_snapshot(self, card_id):
        with self.database() as db:
            return self._card(db, card_id), self._views(db, card_id)

    def list_proofs(self, card_id):
        with self.database() as db:
            self._card(db, card_id)
            rows = db.execute(
                "SELECT id,title,state,base_version,fingerprint,assets,created FROM proofs "
                "WHERE card_id=? ORDER BY created DESC,id DESC",
                (card_id,),
            )
            return [decode_row(row, "assets") for row in rows]

    def proof(self, card_id, proof_id):
        with self.database() as db:
            return self._proof(db, card_id, proof_id)

    def retain_proof(self, card_id, title, version, fingerprint, snapshot, assets):
        title = short_text(title, "proof name")
        require_sha(fingerprint, "candidate")
        with self.database(write=True) as db:
            self._version(db, card_id, version, increment=False)
            previous = db.execute(
                "SELECT id FROM proofs WHERE card_id=? AND fingerprint=? AND base_version=?",
                (card_id, fingerprint, version),
            ).fetchone()
            if previous is not None:
                return self._proof(db, card_id, previous["id"])
            if db.execute("SELECT COUNT(*) FROM proofs WHERE card_id=?", (card_id,)).fetchone()[0] >= MAX_PROOFS:
                raise Capacity("This card already has the maximum number of retained proofs.")
            key = storage_id()
            db.execute(
                "INSERT INTO proofs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (key, card_id, title, "draft", version, fingerprint, encode_json(snapshot), encode_json(assets), utc_now()),
            )
            self._event(db, card_id, key, "proof-rendered", "Retained an inspected print candidate.", {"fingerprint": fingerprint})
            return self._proof(db, card_id, key)

    def release_proof(self, card_id, proof_id, fingerprint, note):
        note = short_text(note, "approval note", maximum=500, allow_blank=True)
        with self.database(write=True) as db:
            proof = self._proof(db, card_id, proof_id)
            current = self._card(db, card_id)
            if proof["fingerprint"] != require_sha(fingerprint, "proof fingerprint"):
                raise Conflict("This is not the proof displayed for review.")
            for asset in proof["assets"].values():
                self.asset_bytes(asset)
            if "plate" not in proof["assets"] or "pdf" not in proof["assets"]:
                raise Conflict("The print candidate is incomplete.")
            active = self._latest_approval(db, card_id)
            if active is not None and active["proof_id"] == proof_id:
                return active
            if current["version"] != proof["base_version"]:
                raise Conflict("The draft changed since this proof. Render and check a new candidate.")
            key = storage_id()
            db.execute(
                "INSERT INTO approval VALUES (?, ?, ?, ?, ?)",
                (key, card_id, proof_id, note, utc_now()),
            )
            db.execute("UPDATE proofs SET state='approved' WHERE id=?", (proof_id,))
            self._event(db, card_id, proof_id, "proof-approved", "Released a fixed print proof.", {"approval": key})
            return self._latest_approval(db, card_id)

    def approved(self, card_id):
        with self.database() as db:
            self._card(db, card_id)
            active = self._latest_approval(db, card_id)
            if active is None:
                raise Missing("Approve a rendered proof before downloading the current print.")
            return active, self._proof(db, card_id, active["proof_id"])

    def measurements(self, card_id):
        with self.database() as db:
            self._card(db, card_id)
            return [
                decode_row(row, "assets", "report")
                for row in db.execute("SELECT * FROM measurements WHERE card_id=? ORDER BY created DESC LIMIT 60", (card_id,))
            ]

    def measurement(self, card_id, key):
        with self.database() as db:
            self._card(db, card_id)
            row = db.execute(
                "SELECT * FROM measurements WHERE card_id=? AND id=?", (card_id, require_id(key, "measurement"))
            ).fetchone()
            if row is None:
                raise Missing("This measurement does not belong to this card.")
            return decode_row(row, "report", "assets")

    def retain_measurement(self, card_id, title, kind, report, assets, proof_id=None):
        title = short_text(title, "measurement title")
        with self.database(write=True) as db:
            self._card(db, card_id)
            if proof_id is not None:
                self._proof(db, card_id, proof_id)
            count = db.execute("SELECT COUNT(*) FROM measurements WHERE card_id=?", (card_id,)).fetchone()[0]
            if count >= MAX_MEASUREMENTS:
                raise Capacity("This card already has the maximum number of evidence measurements.")
            key = storage_id()
            db.execute(
                "INSERT INTO measurements VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (key, card_id, proof_id, kind, title, encode_json(report), encode_json(assets), utc_now()),
            )
            self._event(db, card_id, key, "measurement-saved", "Retained the measurements and their source reference.", {"kind": kind})
        return self.measurement(card_id, key)

    def event_log(self, card_id, limit=30):
        with self.database() as db:
            self._card(db, card_id)
            records = db.execute(
                "SELECT * FROM events WHERE card_id=? ORDER BY id DESC LIMIT ?", (card_id, limit)
            )
            return [decode_row(row, "details") for row in records]
