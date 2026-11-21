"""Independent readback of a frozen proof and its source-view lineage."""
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from .imaging import read_capture
from .models import Plan, WorkbenchError
from .printing import physical_dimensions
from .storage import digest_object


@dataclass(frozen=True)
class ProofInputs:
    identity: dict
    version: int
    proof_id: str
    card_id: str

    @classmethod
    def from_proof(cls, proof):
        return cls(proof["snapshot"]["identity"], proof["base_version"], proof["id"], proof["card_id"])

    @property
    def stored_source_ids(self):
        return [view["id"] for view in self.identity["views"]]

    @property
    def sheet_limits(self):
        return Plan.parse(self.identity["plan"]).sheet


def check_view_bytes(library, original, frozen):
    path = library.source_path(frozen["sha"])
    if not path.is_file():
        return {
            "source_id": frozen["id"], "binding": frozen["sha"], "state": "missing",
            "current_source": original, "expected_size": [frozen["width"], frozen["height"]],
        }
    content = path.read_bytes()
    filename = original["name"] if original else Path(path).name + ".png"
    digest = sha256(content).hexdigest()
    if digest != frozen["sha"]:
        return {"source_id": frozen["id"], "binding": frozen["sha"], "state": "different-bytes",
                "current_source": original, "calculated": digest}
    try:
        capture = read_capture(content, filename)
    except WorkbenchError:
        return {"source_id": frozen["id"], "binding": frozen["sha"], "state": "decode-failed"}
    expected = [frozen["width"], frozen["height"]]
    observed = [capture.width, capture.height]
    same = expected == observed
    record = {"source_id": frozen["id"], "binding": digest, "state": "same-original" if same else "different-dimensions",
              "original_encoding": capture.information, "observed_size": observed, "expected_size": expected,
              "source_bytes": len(content), "label": frozen["title"], "angle": frozen["position"],
              "enabled_in_proof": frozen["enabled"]}
    if original is None:
        record["working_draft"] = "source record not in current draft"
    else:
        changes = _view_changes(frozen, original)
        record["working_draft"] = "same metadata" if not changes else "metadata edited since proof"
        record["current_field_differences"] = changes
    return record


def _view_changes(first, second):
    change_fields = ("sha", "width", "height", "title", "position", "enabled", "exposure_ev")
    return [name for name in change_fields if first.get(name) != second.get(name)]


def _assets(library, proof, inputs):
    media = []
    for name, reference in proof["assets"].items():
        body = library.asset_bytes(reference)
        item = {
            "name": name, "format": reference["mime"],
            "hash": sha256(body).hexdigest(), "bytes": len(body),
            "integrity": "verified" if reference["sha256"] == sha256(body).hexdigest() else "difference",
        }
        if name == "pdf":
            size = physical_dimensions(body)
            item["observed_mm"] = size
            item["expected_mm"] = [inputs.sheet_limits.width_mm, inputs.sheet_limits.height_mm]
            item["page_size_status"] = (
                "verified" if all(abs(a - b) < .001 for a, b in zip(size, item["expected_mm"], strict=True))
                else "different"
            )
        media.append(item)
    return media


def _leaf_differences(path, original, now):
    if isinstance(original, dict) and isinstance(now, dict):
        out = []
        for key in sorted(set(original).union(now)):
            out.extend(_leaf_differences([*path, key], original.get(key), now.get(key)))
        return out
    if original == now:
        return []
    return [{"setting": ".".join(path), "retained": original, "working": now}]


def lineage(library, card_id, proof_id):
    card, records = library.material_snapshot(card_id)
    proof = library.proof(card_id, proof_id)
    frozen = ProofInputs.from_proof(proof)
    active = library.get_card(card_id)["approval"]
    current = {view["id"]: view for view in records}
    sources = [check_view_bytes(library, current.get(view["id"]), view) for view in frozen.identity["views"]]
    checked_media = _assets(library, proof, frozen)
    raster = validate_device_identity(library, proof)
    diff = _leaf_differences([], frozen.identity["plan"], card["config"])
    missing = sorted(set(frozen.stored_source_ids) - set(current))
    new = [key for key in current if key not in frozen.stored_source_ids]
    identity_digest = digest_object(frozen.identity)
    clear = (
        identity_digest == proof["fingerprint"] and card_id == frozen.identity["card_id"]
        and frozen.version == frozen.identity["version"]
        and raster["native_png_verified"] and raster["matches_planned_device_density"]
        and all(item["state"] == "same-original" for item in sources)
        and all(item["integrity"] == "verified" and item.get("page_size_status", "verified") == "verified" for item in checked_media)
    )
    status = {
        "kind": "verified-print-lineage", "verified": clear, "bound_card": frozen.card_id,
        "proof": frozen.proof_id, "released": bool(active and active["proof_id"] == proof_id),
        "retained_version": frozen.version, "working_version": card["version"],
        "original_fingerprint": proof["fingerprint"], "rechecked_fingerprint": identity_digest,
        "setting_changes": diff, "sources": sources, "added_working_sources": new, "removed_working_sources": missing,
        "media": checked_media, "raster_check": raster,
        "message": "Draft edits are listed separately from the fixed proof materials.",
    }
    return status


def source_coordinate_receipt(report):
    return [
        {"source_title": item.get("label"), "source_id": item["source_id"],
         "sha": item["binding"], "status": item["state"], "expected": item.get("expected_size"),
         "observed": item.get("observed_size"), "angle": item.get("angle")}
        for item in report["sources"]
    ]


def validate_device_identity(library, proof):
    """Check the actual native raster against retained sampling and dimension claims."""
    from io import BytesIO
    from PIL import Image
    import numpy as np
    from .optics import LensGrid, phase_map_image
    from .models import Unprocessable

    original = library.asset_bytes(proof["assets"]["plate"])
    exported = proof["snapshot"]["outputs"]
    proof_plan = Plan.parse(proof["snapshot"]["identity"]["plan"])
    count = sum(1 for view in proof["snapshot"]["identity"]["views"] if view["enabled"])
    w, h = proof_plan.sheet.size
    grid = LensGrid(proof_plan.lens, proof_plan.sheet.dpi, w, h)
    with Image.open(BytesIO(original)) as image:
        h_density = image.info.get("dpi", (0, 0))
        if image.size != (w, h):
            raise Unprocessable("Retained print device pixels differ from its fixed stock geometry.")
        if image.mode not in ("RGB", "L"):
            raise Unprocessable("Exported device raster has an unexpected transparency or colour mode.")
        sample = np.asarray(image.convert("RGB"))
    side = proof_plan.sheet.mount
    tested_rows = [0, h // 2, h - 1]
    readout = []
    for row in tested_rows:
        view_mask = phase_map_image(grid, row, row + 1, count, side)[0]
        orientation = int(view_mask[0]), int(view_mask[-1])
        coverage = np.bincount(view_mask, minlength=count)
        sample_rgb = sample[row]
        expected_changes = np.count_nonzero(view_mask[1:] != view_mask[:-1])
        observed_transitions = np.min(np.abs(np.diff(sample_rgb.astype(float), axis=0)), axis=1) >= 4
        readout.append({
            "row": row, "view_fraction": [round(float(n) / w, 8) for n in coverage],
            "first_and_last_view": orientation, "predicted_transition_count": int(expected_changes),
            "conservative_color_transition_count": int(observed_transitions.sum()),
        })
    checked_hash = sha256(original).hexdigest()
    measured = tuple(round(float(value), 2) for value in h_density)
    same_density = len(measured) == 2 and all(abs(value - proof_plan.sheet.dpi) < 1 for value in measured)
    return {
        "native_png_verified": checked_hash == exported["png_sha256"],
        "native_png_sha": checked_hash, "device_mode": str(image.mode),
        "png_embedded_density": measured,
        "matches_planned_device_density": same_density,
        "native_dimensions": [w, h],
        "observer_mount": side, "sampled_rows": readout,
        "note": "View assignments are dot-center predictions. Colour transitions can be absent in identical source areas.",
    }
