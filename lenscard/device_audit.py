"""Evaluate printed view occupancy on the physical device grid."""
from dataclasses import dataclass
import numpy as np
from PIL import Image, ImageDraw

from .imaging import fit_size
from .models import Unprocessable, integer, number, object_value, validate_keys
from .optics import exact_slot_coverage, phase_map_image, physical_slot, solid_slot_coverage
from .printcheck import DeviceFile
from .printing import VIEW_PALETTE
from .visual import chart_png, png_bytes


MAX_SLICE = 700


@dataclass(frozen=True)
class AuditArea:
    first: int
    last: int
    mode: str
    threshold: float

    @classmethod
    def parse(cls, body, grid):
        obj = object_value(body, "raster coverage")
        validate_keys(obj, {"top", "rows", "minimum", "side"}, "band coverage")
        top = integer(obj.get("top", 0), "starting device row", 0, grid.height - 1)
        rows = integer(obj.get("rows", 140), "number of device rows", 3, MAX_SLICE)
        side = obj.get("side", "device")
        if side not in ("device", "observer"):
            raise Unprocessable("Choose device raster or the observer-facing cover.")
        return cls(top, min(grid.height, top + rows), side,
                   number(obj.get("minimum", .05), "small coverage threshold", .001, .42))


def _row_matrix(device, audit):
    plan = device.job
    count = len(device.views)
    actual_side = plan.sheet.mount if audit.mode == "device" else "face"
    assignments = phase_map_image(device.grid, audit.first, audit.last, count, actual_side)
    columns = np.arange(device.grid.width)
    x = device.grid.actual_side(columns, actual_side)
    rows = np.arange(audit.first, audit.last)
    unwrapped = device.grid.phase_at(x[None, :], rows[:, None])
    lens_cells = np.floor(unwrapped).astype(np.int32)
    return assignments, lens_cells, actual_side


def _area_coverage(device, audit, side):
    plan = device.job
    count = len(device.views)
    target = np.empty((audit.last - audit.first, count), dtype=np.float32)
    operation = exact_slot_coverage if plan.sampling.method == "coverage" else solid_slot_coverage
    for index in range(count):
        slot = physical_slot(index, count, plan.lens.order)
        coverage = operation(device.grid, audit.first, audit.last, slot, count, side=side)
        target[:, index] = coverage.mean(axis=1)
    return target


def _complete_lens_groups(assignments, cells, count):
    result = []
    for idx in range(assignments.shape[0]):
        row_values = assignments[idx]
        physical = cells[idx]
        edges = np.flatnonzero(np.diff(physical)) + 1
        # The lens prisms crossing a physical cut edge have intentionally
        # incomplete records. Exclude those two partial spans.
        interior = zip(edges[:-1], edges[1:])
        samples, incomplete = 0, 0
        min_fractions = np.ones(count)
        for start, stop in interior:
            counts = np.bincount(row_values[start:stop], minlength=count)
            if np.any(counts == 0):
                incomplete += 1
            min_fractions = np.minimum(min_fractions, counts / (stop - start))
            samples += 1
        result.append({
            "row_local": idx,
            "tested_complete_prisms": samples,
            "prisms_missing_dot_centers": incomplete,
            "min_fraction_of_each_view": [round(float(value), 8) for value in min_fractions] if samples else None,
        })
    return result


def _histogram_segments(area, audit, device):
    rows = np.arange(audit.first, audit.last, dtype=np.int32)
    series = [(view["title"], rows, area[:, index]) for index, view in enumerate(device.views)]
    return chart_png(series, x_label="Device row", y_label="Area per view (fraction)", y_bounds=(0, max(.6, float(area.max()))))


def _assignment_detail(assignments, area, audit, document):
    canvas = Image.fromarray(VIEW_PALETTE[assignments])
    width, height = fit_size(canvas.size, 940)
    canvas = canvas.resize((width, height), Image.Resampling.NEAREST)
    swatch = Image.new("RGB", (width, height + 42), (28, 42, 46))
    swatch.paste(canvas, (0, 22))
    draw = ImageDraw.Draw(swatch)
    draw.text((6, 4), f"Device rows {audit.first} to {audit.last - 1}", fill=(206, 227, 222))
    for index, record in enumerate(document.views):
        left = 7 + index * (width // len(document.views))
        color = tuple(int(c) for c in VIEW_PALETTE[index])
        draw.rectangle((left, height + 27, left + 10, height + 37), fill=color)
        draw.text((left + 15, height + 25), record["title"][:12], fill=(206, 227, 222))
    return png_bytes(swatch)


def audit_raster(library, card_id, proof_id, body):
    proof = library.proof(card_id, proof_id)
    device = DeviceFile.from_record(library, proof)
    area = AuditArea.parse(body, device.grid)
    mask, prisms, side = _row_matrix(device, area)
    fractions = _area_coverage(device, area, side)
    results = _complete_lens_groups(mask, prisms, len(device.views))
    weak = np.argwhere(fractions < area.threshold)
    warnings = [{"row": int(i) + area.first, "view": device.views[int(j)]["id"],
                 "coverage": round(float(fractions[i, j]), 7)} for i, j in weak[:45]]
    for detail in results:
        detail["row"] = area.first + detail.pop("row_local")
    inspection = {
        "kind": "frozen-view-band-audit", "fingerprint": proof["fingerprint"], "proof_id": proof_id,
        "start_row": area.first, "end_row_exclusive": area.last, "display": area.mode,
        "device_sampling": device.job.sampling.method, "minimum_coverage": area.threshold,
        "sample_rows": len(results), "prisms_examined": sum(row["tested_complete_prisms"] for row in results),
        "prisms_missing_dot_centers": sum(row["prisms_missing_dot_centers"] for row in results),
        "rendered_view_area": [
            {"view": entry["id"], "title": entry["title"],
             "average": round(float(fractions[:, k].mean()), 7), "minimum": round(float(fractions[:, k].min()), 7),
             "maximum": round(float(fractions[:, k].max()), 7)}
            for k, entry in enumerate(device.views)
        ],
        "limited_rows": warnings, "full_limited_count": int(len(weak)),
        "row_prism_counts": results,
        "note": "The colored assignment dots represent centers. The coverage trace preserves the selected production sampling mode.",
    }
    return inspection, {"assignment-map": _assignment_detail(mask, fractions, area, device),
                        "coverage-trace": _histogram_segments(fractions, area, device)}
