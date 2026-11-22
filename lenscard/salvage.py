"""Reverse salvage: split one surviving interlaced print back into per-view strips.

Every output is a *reconstruction* from the printed pixels the user still holds.
The lens grid only re-sorts columns that already exist; it cannot recover a
camera original.  Each output pixel is classified as:

* read      - a printed column assigned to this view was present and copied;
* estimated - no column was available here, so a value was interpolated;
* missing   - no printed column exists anywhere in this row for this view.
"""
from base64 import b64decode, b64encode
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import math

import numpy as np
from PIL import Image, ImageDraw

from .imaging import RESAMPLING, _SRGB_8, fit_size, linear_to_srgb, read_capture
from .models import Capacity, Lens, Unprocessable, choice, integer, number, object_value, validate_keys
from .optics import LensGrid, band_index, physical_slot
from .visual import png_bytes

MISSING_RGB = (24, 33, 43)
STATUS_COLORS = np.asarray([
    [64, 190, 138],   # read
    [232, 177, 78],   # estimated
    [214, 79, 92],    # missing
], np.uint8)
BAND_COLORS = np.asarray([
    [38, 142, 114], [191, 112, 82], [99, 104, 182], [232, 177, 78],
    [62, 160, 195], [185, 91, 142], [107, 145, 80], [126, 103, 83],
], np.uint8)
MAX_SALVAGE_PIXELS = 10_000_000
MAX_SALVAGE_EDGE = 4000
PREVIEW_EDGE = 900
PROVENANCE = (
    "Reconstructed by de-interlacing the surviving printed raster with a "
    "user-supplied lens grid. These are not recovered camera originals: read "
    "pixels are copied print columns, estimated pixels are interpolated, and "
    "missing pixels have no printed source."
)


@dataclass(frozen=True)
class SalvageSettings:
    view_count: int
    lpi: float
    phase: float
    shear: float
    order: str
    latitude: float
    width_mm: float
    height_mm: float
    dpi: float
    mount: str
    crop: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)
    damage: tuple[tuple[float, float, float, float], ...] = ()

    @classmethod
    def parse(cls, body):
        obj = object_value(body, "salvage settings")
        validate_keys(
            obj,
            {"view_count", "lpi", "phase", "shear", "order", "latitude",
             "width_mm", "height_mm", "dpi", "mount", "crop", "damage"},
            "salvage settings",
        )
        crop = cls._parse_crop(obj.get("crop"))
        damage = cls._parse_damage(obj.get("damage"))
        settings = cls(
            integer(obj.get("view_count", 3), "view count", 2, 8),
            number(obj.get("lpi", 50), "pitch lines per inch", 10, 180),
            number(obj.get("phase", 0), "phase in cycles", -0.5, 0.5),
            number(obj.get("shear", 0), "side drift per millimetre", -0.25, 0.25),
            choice(obj.get("order", "forward"), "view order", ("forward", "reverse")),
            number(obj.get("latitude", 22), "half viewing cone in degrees", 5, 54),
            number(obj.get("width_mm", 54), "print width in millimetres", 18, 220),
            number(obj.get("height_mm", 68), "print height in millimetres", 18, 290),
            number(obj.get("dpi", 500), "scanned pixels per inch", 72, 2400),
            choice(obj.get("mount", "face"), "scanned side", ("face", "reverse")),
            crop,
            damage,
        )
        width, height = settings.raster_size
        if max(width, height) > MAX_SALVAGE_EDGE or width * height > MAX_SALVAGE_PIXELS:
            raise Capacity("Lower the scanned DPI or print size to fit the salvage raster budget.")
        period = settings.dpi / settings.lpi
        if settings.view_count > period:
            raise Unprocessable(
                "Each lens period holds fewer scanned pixels than views. Lower the view count or LPI.",
                details={"views": settings.view_count, "pixels_per_lens": round(period, 3)},
            )
        return settings

    @staticmethod
    def _parse_crop(value):
        if value is None:
            return (0.0, 0.0, 1.0, 1.0)
        if not isinstance(value, (list, tuple)) or len(value) != 4:
            raise Unprocessable("Crop must be four normalized bounds: left, top, right, bottom.")
        left, top, right, bottom = (
            number(part, "crop bound", 0.0, 1.0) for part in value
        )
        if right <= left or bottom <= top:
            raise Unprocessable("Choose a nonempty crop region.")
        return (left, top, right, bottom)

    @staticmethod
    def _parse_damage(value):
        if not value:
            return ()
        if not isinstance(value, (list, tuple)) or len(value) > 12:
            raise Unprocessable("Damage must be a list of up to twelve normalized rectangles.")
        regions = []
        for entry in value:
            if not isinstance(entry, (list, tuple)) or len(entry) != 4:
                raise Unprocessable("Each damage region needs left, top, right, bottom bounds.")
            left, top, right, bottom = (number(part, "damage bound", 0.0, 1.0) for part in entry)
            if right <= left or bottom <= top:
                raise Unprocessable("Choose a nonempty damage region.")
            regions.append((left, top, right, bottom))
        return tuple(regions)

    def damage_pixels(self, width, height):
        boxes = []
        for left, top, right, bottom in self.damage:
            boxes.append((
                min(width - 1, math.floor(left * width)),
                min(height - 1, math.floor(top * height)),
                min(width, max(1, math.ceil(right * width))),
                min(height, max(1, math.ceil(bottom * height))),
            ))
        return tuple(boxes)

    @property
    def raster_size(self):
        return (
            round(self.width_mm * self.dpi / 25.4),
            round(self.height_mm * self.dpi / 25.4),
        )

    def crop_pixels(self, width, height):
        left, top, right, bottom = self.crop
        return (
            min(width - 1, math.floor(left * width)),
            min(height - 1, math.floor(top * height)),
            min(width, max(1, math.ceil(right * width))),
            min(height, max(1, math.ceil(bottom * height))),
        )

    def grid(self, width, height):
        lens = Lens(self.lpi, self.phase, self.shear, self.order, self.latitude)
        return LensGrid(lens, self.dpi, width, height)

    def public(self):
        return {
            "view_count": self.view_count, "lpi": self.lpi, "phase": self.phase,
            "shear": self.shear, "order": self.order, "latitude": self.latitude,
            "width_mm": self.width_mm, "height_mm": self.height_mm,
            "dpi": self.dpi, "mount": self.mount, "crop": list(self.crop),
            "damage": [list(region) for region in self.damage],
        }


def load_print(payload, filename, settings):
    """Decode the uploaded print and resample it onto the declared print grid."""
    decoded = read_capture(payload, filename)
    width, height = settings.raster_size
    resized = decoded.image.convert("RGB").resize((width, height), RESAMPLING)
    rgb = _SRGB_8[np.asarray(resized, dtype=np.uint8)].astype(np.float32)
    return {
        "filename": filename,
        "sha256": sha256(payload).hexdigest(),
        "uploaded_size": list(decoded.image.size),
        "raster_size": [width, height],
        "resampling": "upload resized to declared print pixels; no detail is invented",
        "rgb": rgb,
    }


def _fill_row(values, known):
    """Interpolate gaps *between* reads; mark columns outside the read span missing.

    Status codes: 0 read, 1 estimated, 2 missing. Extrapolation would invent
    contour beyond what the print provides, so only internal gaps are filled.
    """
    status = np.full(values.shape[0], 2, np.uint8)
    if not known.any():
        values[:] = 0.18
        return status
    status[known] = 0
    x = np.arange(values.shape[0])
    left, right = np.flatnonzero(known)[[0, -1]]
    gap = (~known) & (x >= left) & (x <= right)
    for band in range(3):
        values[gap, band] = np.interp(x[gap], x[known], values[known, band])
    status[gap] = 1
    values[(x < left) | (x > right)] = 0.18
    return status


def _row_slot_columns(phase_row, slot, count, width):
    """Printed columns whose band equals *slot* along one row.

    The band grid is taken straight from this row's phase, so a sheared (drifting)
    lens stack is followed exactly. Only the first column of each slot run is
    kept, giving one sampled printed column per lens period.
    """
    bands = band_index(phase_row[None, :], count)[0]
    members = np.flatnonzero(bands == slot)
    if members.size == 0:
        return members, np.empty(0, np.int64)
    # Keep the first column of each contiguous run (one column per lens period).
    breaks = np.flatnonzero(np.diff(members) > 1)
    firsts = members[np.concatenate(([0], breaks + 1))]
    # Lens-period number of each run: floor of the row phase at that column.
    periods = np.floor(phase_row[firsts]).astype(np.int64)
    return firsts, periods


def reconstruct(print_data, settings):
    rgb = print_data["rgb"]
    height, width = rgb.shape[:2]
    count = settings.view_count
    grid = settings.grid(width, height)
    side = settings.mount
    phase = grid.phase_matrix(side=side)
    assigned = band_index(phase, count)
    period = grid.period_pixels
    crop_left, crop_top, crop_right, crop_bottom = settings.crop_pixels(width, height)
    damage_boxes = settings.damage_pixels(width, height)
    crop_mask = np.zeros((height, width), bool)
    crop_mask[crop_top:crop_bottom, crop_left:crop_right] = True
    damage_mask = np.zeros((height, width), bool)
    for dl, dt, dr, db in damage_boxes:
        damage_mask[dt:db, dl:dr] = True
    readable = crop_mask & ~damage_mask
    # One compact column per lens period; every view shares this compact width.
    compact_width = max(1, int(np.floor(width / period)))
    views = []
    for view_index in range(count):
        slot = physical_slot(view_index, count, settings.order)
        compact = np.zeros((height, compact_width, 3), np.float32)
        status = np.full((height, compact_width), 2, np.uint8)
        reads = np.zeros(height, np.int32)
        nominal = np.zeros(compact_width, np.int64)
        nominal_seen = np.zeros(compact_width, np.int64)
        for row in range(height):
            if not crop_top <= row < crop_bottom:
                continue
            columns, periods = _row_slot_columns(phase[row], slot, count, width)
            if columns.size == 0:
                continue
            target = periods
            in_range = (periods >= 0) & (periods < compact_width)
            columns, target = columns[in_range], target[in_range]
            inside = (columns >= crop_left) & (columns < crop_right)
            columns, target = columns[inside], target[inside]
            if columns.size == 0:
                continue
            # Columns inside a marked damage region are treated as unreadable:
            # they create internal gaps that interpolation marks estimated.
            ok = readable[row, columns]
            columns, target = columns[ok], target[ok]
            if columns.size == 0:
                continue
            good = columns[assigned[row, columns] == slot]
            good_target = target[assigned[row, columns] == slot]
            reads[row] = good.size
            compact[row, good_target] = rgb[row, good]
            known = np.zeros(compact_width, bool)
            known[good_target] = True
            status[row] = _fill_row(compact[row], known)
            nominal[good_target] += good
            nominal_seen[good_target] += 1
        empty_rows = reads == 0
        compact[empty_rows] = 0.18
        filled = nominal_seen > 0
        if filled.any():
            nominal[filled] = np.round(nominal[filled] / nominal_seen[filled]).astype(np.int64)
            if (~filled).any():
                nominal[~filled] = np.interp(
                    np.flatnonzero(~filled), np.flatnonzero(filled), nominal[filled]
                ).astype(np.int64)
        views.append({
            "index": view_index,
            "slot": slot,
            "compact_size": [compact_width, height],
            "nominal_columns": nominal,
            "status": status,
            "compact_linear": compact,
            "read_columns_per_row": reads,
            "geometry_mismatch_fraction": round(float((status == 1).mean()), 6),
            "empty_row_count": int(empty_rows.sum()),
        })
    return {
        "settings": settings,
        "grid": grid,
        "assigned": assigned,
        "print_size": [width, height],
        "compact_width": compact_width,
        "crop_pixels": [int(crop_left), int(crop_top), int(crop_right), int(crop_bottom)],
        "damage_boxes": [list(map(int, box)) for box in damage_boxes],
        "crop_mask": crop_mask,
        "damage_mask": damage_mask,
        "readable_mask": readable,
        "views": views,
    }


def _status_png(status):
    return png_bytes(Image.fromarray(STATUS_COLORS[status], "RGB"))


def _status_png_raw(status):
    buffer = BytesIO()
    Image.fromarray(status.astype(np.uint8), "L").save(buffer, "PNG", compress_level=6)
    return buffer.getvalue()


def _decode_status(payload):
    return np.asarray(Image.open(BytesIO(b64decode(payload))), dtype=np.uint8)


def _geometry_status_png(status, max_width=220):
    """Downsampled status map for click-back storage; keeps the provenance class.

    Full-resolution status images stay in the downloadable assets. This smaller
    map only has to report the class of a clicked point, so nearest-neighbour
    downsampling preserves the 0/1/2 codes while keeping the database compact.
    """
    height, width = status.shape
    if width <= max_width:
        small = status
    else:
        scale = max_width / width
        small = np.asarray(Image.fromarray(status, "L").resize(
            (max_width, max(1, round(height * scale))), Image.Resampling.NEAREST))
    return _status_png_raw(small)


def build_assets(print_data, result):
    """Render every evidence image and the per-view statistics."""
    width, height = result["print_size"]
    crop_left, crop_top, crop_right, crop_bottom = result["crop_pixels"]
    crop_mask = result["crop_mask"]
    damage_mask = result["damage_mask"]
    readable = result["readable_mask"]
    assets = {}
    preview_array = linear_to_srgb(print_data["rgb"]).copy()
    preview = Image.fromarray(preview_array)
    draw = ImageDraw.Draw(preview)
    draw.rectangle((crop_left, crop_top, crop_right - 1, crop_bottom - 1),
                   outline=(245, 224, 139), width=max(2, width // 250))
    for dl, dt, dr, db in result["damage_boxes"]:
        draw.rectangle((dl, dt, dr - 1, db - 1), outline=(214, 79, 92), width=max(2, width // 250))
    assets["print-preview"] = png_bytes(preview.resize(fit_size((width, height), PREVIEW_EDGE), RESAMPLING))
    band_map = BAND_COLORS[result["assigned"]].copy()
    band_map[~crop_mask] = (band_map[~crop_mask] // 3) + np.asarray(MISSING_RGB)
    band_map[damage_mask] = STATUS_COLORS[2]
    assets["print-band-map"] = png_bytes(Image.fromarray(band_map, "RGB"))
    records = []
    for item in result["views"]:
        index = item["index"]
        base = f"view-{index + 1}"
        evidence = np.full((height, width, 3), MISSING_RGB, np.uint8)
        mask = (result["assigned"] == item["slot"]) & readable
        evidence[mask] = linear_to_srgb(print_data["rgb"][mask])
        assets[f"{base}-evidence"] = png_bytes(Image.fromarray(evidence, "RGB"))
        compact_srgb = linear_to_srgb(item["compact_linear"])
        compact_srgb[item["status"] == 2] = MISSING_RGB
        assets[f"{base}-reconstructed"] = png_bytes(Image.fromarray(compact_srgb, "RGB"))
        assets[f"{base}-status"] = _status_png(item["status"])
        total = item["status"].size
        counts = np.bincount(item["status"].ravel(), minlength=3)
        records.append({
            "view": index + 1,
            "slot": item["slot"],
            "compact_size": item["compact_size"],
            "read_pixels": int(counts[0]),
            "estimated_pixels": int(counts[1]),
            "missing_pixels": int(counts[2]),
            "read_fraction": round(float(counts[0] / total), 6),
            "estimated_fraction": round(float(counts[1] / total), 6),
            "missing_fraction": round(float(counts[2] / total), 6),
            "geometry_mismatch_fraction": item["geometry_mismatch_fraction"],
            "empty_rows": item["empty_row_count"],
            "mean_read_columns_per_row": round(float(item["read_columns_per_row"].mean()), 4),
            "asset_names": {
                "evidence": f"{base}-evidence.png",
                "reconstructed": f"{base}-reconstructed.png",
                "status": f"{base}-status.png",
            },
        })
    return assets, records


def backmap_geometry(geometry, x01, y01):
    """Map a click on a compact reconstruction back to its printed block.

    Geometry is stored with the job so the uploaded raster does not have to be
    retained or re-uploaded to answer a click. The stored status map is
    downsampled; the click is scaled onto it to read the provenance class.
    """
    compact_width, height = geometry["compact_size"]
    width, print_height = geometry["print_size"]
    nominal = np.asarray(geometry["nominal_columns"], dtype=np.int64)
    status = _decode_status(geometry["status_png"])
    status_h, status_w = status.shape
    col = min(compact_width - 1, max(0, round(x01 * (compact_width - 1))))
    row = min(height - 1, max(0, round(y01 * (height - 1))))
    status_col = min(status_w - 1, round(x01 * (status_w - 1)))
    status_row = min(status_h - 1, round(y01 * (status_h - 1)))
    printed_col = int(nominal[col])
    period = max(1.0, float(geometry["period_pixels"]))
    half = max(1, int(np.ceil(period / 2)))
    return {
        "view": geometry["view"],
        "compact_point": [col, row],
        "printed_point": [printed_col, row],
        "printed_box": [
            max(0, printed_col - half) / width,
            max(0, row - 2) / print_height,
            min(width, printed_col + half + 1) / width,
            min(print_height, row + 3) / print_height,
        ],
        "status": ("read", "estimated", "missing")[int(status[status_row, status_col])],
        "normalized": [printed_col / (width - 1), row / (print_height - 1)],
    }


def backmap(result, view_index, x01, y01):
    item = result["views"][view_index]
    geometry = {
        "view": view_index + 1,
        "compact_size": item["compact_size"],
        "print_size": result["print_size"],
        "nominal_columns": item["nominal_columns"].tolist(),
        "period_pixels": result["grid"].period_pixels,
        "status_png": b64encode(_geometry_status_png(item["status"])).decode("ascii"),
    }
    return backmap_geometry(geometry, x01, y01)


def geometry_blob(result):
    """Compact, persistent geometry needed for later click-back answers."""
    return [
        {
            "view": item["index"] + 1,
            "compact_size": item["compact_size"],
            "print_size": result["print_size"],
            "nominal_columns": item["nominal_columns"].tolist(),
            "period_pixels": result["grid"].period_pixels,
            "status_png": b64encode(_geometry_status_png(item["status"])).decode("ascii"),
        }
        for item in result["views"]
    ]


def salvage_report(print_data, settings, records, geometry, crop_pixels, damage_boxes):
    width, height = settings.raster_size
    return {
        "kind": "lenticular-salvage",
        "provenance": PROVENANCE,
        "not_originals": True,
        "input": {
            "filename": print_data["filename"],
            "sha256": print_data["sha256"],
            "uploaded_size": print_data["uploaded_size"],
            "raster_size": print_data["raster_size"],
            "resampling_note": print_data["resampling"],
        },
        "settings": settings.public(),
        "grid": settings.grid(width, height).public(),
        "print_pixels": width * height,
        "crop_pixels": crop_pixels,
        "damage_boxes": damage_boxes,
        "views": records,
        "geometry": geometry,
        "legend": {
            "read": "copied directly from a printed column assigned to this view",
            "estimated": "interpolated between read columns on the same row across an unreadable gap",
            "missing": "no printed column for this view: outside the scan/crop span, a damaged span, or an empty row",
        },
    }
