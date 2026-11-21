"""Portable original view sets and browser-checkable print handoffs."""
from dataclasses import dataclass, field
from hashlib import sha256
from io import BytesIO, StringIO
from pathlib import PurePosixPath
from zipfile import BadZipFile, ZIP_DEFLATED, ZIP_STORED, ZipFile, ZipInfo
import csv
import json
import re
import stat

from .imaging import frame_input, read_capture
from .models import Capacity, Conflict, MAX_VIEWS, Plan, Unprocessable, ViewSetting, require_sha, short_text
from .storage import encode_json


MAX_MEMBERS = 90
MAX_ARCHIVE_BYTES = 138 * 1024**2
SAFE_NAME = re.compile(r"^[a-z0-9._/-]+$")
SCHEMA = "lenscard-portable-1"
KINDS = ("view-set", "print-proof", "measurement")
ZIP_STAMP = (2026, 1, 1, 0, 0, 0)
EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}
ASSET_CONTENTS = frozenset(("image/png", "image/webp", "application/pdf", "application/json", "text/csv", "application/octet-stream"))


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def safe_path(value):
    if not isinstance(value, str):
        raise Unprocessable("The archive name is not text.")
    path = PurePosixPath(value)
    if (
        not path.parts or path.is_absolute() or ".." in path.parts
        or str(path) != value or not SAFE_NAME.fullmatch(value) or len(value) > 160
    ):
        raise Unprocessable("The archive includes an unsupported member name.")
    return value


def slug(value, fallback):
    word = re.sub(r"[^a-zA-Z0-9_-]+", "-", value.strip()).strip("-").lower()
    return (word or fallback)[:58]


def guarded_cell(value):
    if isinstance(value, (list, dict)):
        value = encode_json(value)
    if isinstance(value, str) and value.lstrip("\t\r\n ").startswith(("@", "+", "=", "-")):
        return "'" + value
    return value


def report_csv(rows):
    output = StringIO(newline="")
    columns = list(dict.fromkeys(key for row in rows for key in row))
    writer = csv.DictWriter(output, columns)
    writer.writeheader()
    for row in rows:
        writer.writerow({key: guarded_cell(row.get(key)) for key in columns})
    return output.getvalue().encode("utf-8-sig")


@dataclass
class MaterialPackage:
    members: dict[str, bytes] = field(default_factory=dict)
    listing: dict[str, dict] = field(default_factory=dict)
    total: int = 0

    def add(self, name, payload, mime):
        name = safe_path(name)
        if name == "manifest.json":
            raise Unprocessable("The package manifest name is reserved.")
        if mime not in ASSET_CONTENTS:
            raise Unprocessable("The content type cannot be preserved in a LensCard package.")
        digest = sha256(payload).hexdigest()
        if name in self.listing:
            if digest != self.listing[name]["sha256"]:
                raise Conflict("Two archived materials collide at one file name.")
            return
        if len(self.members) >= MAX_MEMBERS or self.total + len(payload) > MAX_ARCHIVE_BYTES:
            raise Capacity("The package exceeds the local archive budget.")
        self.members[name] = payload
        self.listing[name] = {"sha256": digest, "size": len(payload), "mime": mime}
        self.total += len(payload)

    def add_json(self, name, value):
        self.add(name, json_bytes(value), "application/json")

    def to_zip(self, kind, details):
        if kind not in KINDS:
            raise Unprocessable("Unknown portable package type.")
        metadata = {"schema": SCHEMA, "kind": kind, "members": self.listing, "details": details}
        output = BytesIO()
        with ZipFile(output, "w") as archive:
            for filename, value in sorted({**self.members, "manifest.json": json_bytes(metadata)}.items()):
                entry = ZipInfo(filename, ZIP_STAMP)
                entry.compress_type = ZIP_DEFLATED
                entry.external_attr = 0o100644 << 16
                archive.writestr(entry, value, compresslevel=4)
        return output.getvalue()


def _extension(filename):
    suffix = PurePosixPath(filename).suffix.lower()
    return suffix if suffix in EXTENSIONS else ".img"


def _view_member(library, package, view):
    name = "originals/" + view["sha"] + _extension(view["name"])
    body = library.source_path(view["sha"]).read_bytes()
    if sha256(body).hexdigest() != view["sha"] or len(body) != view["bytes"]:
        raise Conflict("An original view fails its retained hash.")
    package.add(name, body, "application/octet-stream")
    return {
        "name": view["name"],
        "member": name,
        "source_sha": view["sha"],
        "size": [view["width"], view["height"]],
        "settings": {key: view[key] for key in ("title", "position", "enabled", "exposure_ev")},
    }


def pack_view_set(library, card, views):
    material = MaterialPackage()
    entries = [_view_member(library, material, view) for view in views]
    info = {
        "title": card["title"], "note": card["note"], "synthetic": card["synthetic"],
        "plan": card["config"], "entries": entries,
    }
    material.add_json("views.json", info)
    payload = material.to_zip("view-set", {"title": card["title"], "sources": len(entries)})
    return payload, slug(card["title"], "card") + "-views.zip"


def pack_print(library, card, approval, proof, include_originals):
    material = MaterialPackage()
    for label, ref in proof["assets"].items():
        extension = {"image/png": ".png", "application/pdf": ".pdf", "application/json": ".json"}.get(ref["mime"], ".bin")
        material.add(f"verified/{safe_path(label)}{extension}", library.asset_bytes(ref), ref["mime"])
    frozen = proof["snapshot"]
    material.add_json("receipts/approval.json", approval)
    material.add_json("receipts/proof.json", {
        "card_id": card["id"], "proof_id": proof["id"], "state": proof["state"],
        "candidate": proof["fingerprint"], "base_version": proof["base_version"],
        "assets": proof["assets"], "snapshot": frozen,
    })
    if include_originals:
        for item in frozen["identity"]["views"]:
            digest = item["sha"]
            body = library.source_path(digest).read_bytes()
            if sha256(body).hexdigest() != digest:
                raise Conflict("A print source changed; the package cannot be constructed.")
            material.add(f"originals/{digest}.img", body, "application/octet-stream")
    details = {
        "card": card["title"], "proof": proof["title"], "released_sha": proof["fingerprint"],
        "include_originals": include_originals,
    }
    return material.to_zip("print-proof", details), slug(card["title"], "card") + "-approved-print.zip"


def pack_measurement(library, record):
    material = MaterialPackage()
    material.add_json("inspection.json", record)
    for label, ref in record["assets"].items():
        payload = library.asset_bytes(ref)
        suffix = ".png" if ref["mime"] == "image/png" else ".bin"
        material.add(f"diagrams/{safe_path(label)}{suffix}", payload, ref["mime"])
    return material.to_zip("measurement", {"title": record["title"], "kind": record["kind"]})


def _members(archive):
    seen, size = {}, 0
    records = archive.infolist()
    if not 1 <= len(records) <= MAX_MEMBERS + 1:
        raise Unprocessable("Unexpected member count in the package.")
    for info in records:
        name = safe_path(info.filename)
        mode = info.external_attr >> 16
        if name in seen or info.is_dir() or stat.S_ISLNK(mode) or info.flag_bits & 1:
            raise Unprocessable("The package contains a repeated or unsupported archive member.")
        if info.compress_type not in (ZIP_STORED, ZIP_DEFLATED):
            raise Unprocessable("The archive uses an unsupported compression type.")
        size += info.file_size
        if size > MAX_ARCHIVE_BYTES:
            raise Capacity("The package would expand beyond the safe local budget.")
        seen[name] = info
    if "manifest.json" not in seen or seen["manifest.json"].file_size > 1_000_000:
        raise Unprocessable("A small LensCard manifest is required.")
    return seen


def _metadata(archive):
    manifest = json.loads(archive.read("manifest.json"))
    if not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA or manifest.get("kind") not in KINDS:
        raise Unprocessable("This archive does not contain a supported LensCard manifest.")
    listing = manifest.get("members")
    if not isinstance(listing, dict) or not isinstance(manifest.get("details"), dict):
        raise Unprocessable("The archive manifest has no member records.")
    return manifest


def package_read(payload, *, keep_bytes=False):
    if not payload or len(payload) > MAX_ARCHIVE_BYTES:
        raise Capacity("The package is empty or exceeds the archive upload budget.")
    validation = []
    selected = {}
    try:
        with ZipFile(BytesIO(payload)) as archive:
            members = _members(archive)
            manifest = _metadata(archive)
            listing = manifest["members"]
            undisclosed = set(members) - {"manifest.json"} - set(listing)
            if undisclosed:
                raise Unprocessable("The archive contains files absent from its manifest.")
            for name, declared in listing.items():
                safe_path(name)
                if not isinstance(declared, dict) or declared.get("mime") not in ASSET_CONTENTS:
                    raise Unprocessable("The archive contains unsupported declared media.")
                require_sha(declared.get("sha256"))
                if name not in members:
                    validation.append({"name": name, "status": "missing"})
                    continue
                actual = archive.read(name)
                hashed = sha256(actual).hexdigest()
                matched = hashed == declared["sha256"] and len(actual) == declared.get("size")
                validation.append({"name": name, "status": "verified" if matched else "difference",
                                   "bytes": len(actual), "sha256": hashed})
                if keep_bytes and matched:
                    selected[name] = actual
    except (BadZipFile, OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        if isinstance(error, Unprocessable):
            raise
        raise Unprocessable("Could not read the portable LensCard package.") from error
    return {
        "schema": manifest["schema"], "kind": manifest["kind"], "details": manifest["details"],
        "digest": sha256(payload).hexdigest(), "verified": all(row["status"] == "verified" for row in validation),
        "members": validation,
    }, selected


def preview_view_upload(payload):
    report, assets = package_read(payload, keep_bytes=True)
    if not report["verified"] or report["kind"] != "view-set":
        raise Unprocessable("Use a fully verified original-view set to create a new card.")
    if "views.json" not in assets:
        raise Unprocessable("The view-set has no source settings file.")
    try:
        data = json.loads(assets["views.json"])
        plan = Plan.parse(data["plan"]).public()
        entries = data["entries"]
        if not isinstance(entries, list) or len(entries) > MAX_VIEWS:
            raise Unprocessable("The package has too many views.")
        decoded = []
        for item in entries:
            verified = assets.get(safe_path(item["member"]))
            if verified is None or require_sha(item["source_sha"]) != sha256(verified).hexdigest():
                raise Unprocessable("An original is absent or fails its assigned hash.")
            setting = ViewSetting.parse(item["settings"]).public()
            raw_name = short_text(item["name"], "source filename", maximum=220)
            shape = read_capture(verified, raw_name)
            if [shape.width, shape.height] != item["size"]:
                raise Unprocessable("Decoded pixels do not match the archived view size.")
            decoded.append((raw_name, setting, verified))
        destination = {
            "title": short_text(data["title"], "previous card title"),
            "note": short_text(data.get("note", ""), "card note", maximum=500, allow_blank=True),
            "synthetic": data.get("synthetic") is True,
            "plan": plan,
        }
    except (KeyError, ValueError, AttributeError, TypeError) as error:
        if isinstance(error, Unprocessable):
            raise
        raise Unprocessable("The portable view settings are incomplete or malformed.") from error
    return report, destination, decoded


def accept_view_upload(library, payload, expected_digest):
    report, destination, decoded = preview_view_upload(payload)
    if require_sha(expected_digest, "reviewed archive") != report["digest"]:
        raise Conflict("Import the same file you inspected.")
    ready = [(frame_input(binary, name, config["position"], library), config)
             for name, config, binary in decoded]
    title = (destination["title"][:80] + " copy") if destination["title"] else "Imported view set"
    card = library.create_card(
        title, destination["note"], plan=destination["plan"], synthetic=destination["synthetic"],
    )
    if ready:
        library.add_views(card["id"], ready, version=0)
    return library.get_card(card["id"])
