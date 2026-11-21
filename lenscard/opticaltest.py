"""Direct measurements for a numbered lenticular angle isolation patch."""
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import math

import numpy as np
from PIL import Image, ImageDraw

from .imaging import fit_size, linear_luminance, linear_to_srgb, read_capture, sample_capture
from .models import Position, Region, Unprocessable, integer, number, object_value, short_text, validate_keys
from .optics import angular_domains, band_index, build_grid, physical_slot
from .printing import one_raster_pdf
from .visual import chart_png, overlay_regions, png_bytes, resize_report


BLACK_LUMA = .025
WHITE_LUMA = .86
MAX_TEST_PHOTOS = 7
REFERENCE_GAP = .08
MIN_SUPPORTED = 40


@dataclass(frozen=True)
class PatchLayout:
    count: int
    left: float
    right: float
    top: float
    bottom: float

    @classmethod
    def from_frame(cls, frame, count):
        region = Region.parse(frame)
        if region.right - region.left < .24 or region.bottom - region.top < .38:
            raise Unprocessable("The angle target region needs all reference and numbered patches.")
        return cls(count, *[region.left, region.right, region.top, region.bottom])

    @property
    def rows(self):
        return self.count + 2

    def subregion(self, index):
        space = (self.bottom - self.top) / self.rows
        left = self.left + (self.right - self.left) * .05
        right = self.right - (self.right - self.left) * .05
        top = self.top + space * (index + .22)
        bottom = self.top + space * (index + .78)
        return Region(left, top, right, bottom)

    def entries(self):
        return [
            {"kind": "dark" if i == 0 else "white" if i == self.rows - 1 else "view",
             "view_index": i - 1 if 1 <= i < self.rows - 1 else None,
             "region": self.subregion(i)}
            for i in range(self.rows)
        ]


def target_definition(records, plan):
    active_views = [item for item in records if item["enabled"]]
    count = len(active_views)
    if not 2 <= count <= 8:
        raise Unprocessable("Enable between two and eight views for an angle target.")
    plan.check_sampling(count)
    width, height = plan.sheet.size
    grid = build_grid(plan, width, height)
    if width < 120 or height < 19 * (count + 2):
        raise Unprocessable("The chosen stock gives the target too few measurable rows.")
    return grid, sorted(active_views, key=lambda item: item["position"])


def _stripe_target(grid, plan, views, left, top, right, bottom, index):
    rows = np.arange(top, bottom)[:, None]
    columns = np.arange(left, right)[None, :]
    phase = grid.phase_at(grid.actual_side(columns, plan.sheet.mount), rows)
    select = band_index(phase, len(views)) == physical_slot(index, len(views), plan.lens.order)
    pixels = np.where(select, WHITE_LUMA, BLACK_LUMA)
    return Image.fromarray(linear_to_srgb(pixels))


def render_angle_target(records, plan):
    grid, views = target_definition(records, plan)
    width, height = grid.width, grid.height
    layout = PatchLayout.from_frame([.13, .17, .87, .85], len(views))
    plate = Image.new("L", (width, height), 250)
    draw = ImageDraw.Draw(plate)
    draw.text((max(5, width // 13), max(8, height // 18)), "VIEW ISOLATION / ACTUAL SIZE", fill=34)
    draw.text((max(5, width // 13), max(19, height // 11)), "Single white view per numbered band", fill=56)
    labels = []
    for region in layout.entries():
        crop = region["region"]
        left, top, right, bottom = crop.pixels(width, height)
        if region["kind"] == "view":
            index = region["view_index"]
            body = _stripe_target(grid, plan, views, left, top, right, bottom, index)
            plate.paste(body, (left, top))
            label = str(index + 1)
        else:
            ink = WHITE_LUMA if region["kind"] == "white" else BLACK_LUMA
            draw.rectangle((left, top, right - 1, bottom - 1), fill=int(linear_to_srgb(np.array(ink))))
            label = "WHITE" if region["kind"] == "white" else "DARK"
        draw.text((max(4, left - 32), (top + bottom) // 2 - 6), label, fill=40)
        labels.append({**region, "region": crop.public(), "label": label})
    frame = [layout.left, layout.top, layout.right, layout.bottom]
    frame_x1, frame_y1, frame_x2, frame_y2 = Region.parse(frame).pixels(width, height)
    draw.rectangle((frame_x1, frame_y1, frame_x2 - 1, frame_y2 - 1), outline=180, width=1)
    draw.text((max(5, width // 13), height - max(22, height // 12)), "FRAME TOP / ROW ORDER PRESERVED", fill=58)
    angle_info = angular_domains(grid, views)
    report = {
        "paper_mm": [plan.sheet.width_mm, plan.sheet.height_mm],
        "dpi": plan.sheet.dpi,
        "pixels": [width, height],
        "mount": plan.sheet.mount,
        "model": grid.public(),
        "reference_bounds": frame,
        "media_artwork": labels,
        "nominal_angle_windows": angle_info,
        "reading_note": "Photograph the whole boundary rectangle at each requested angle. Mark its edges after perspective distortion has been removed.",
    }
    pdf = one_raster_pdf(np.asarray(plate), *report["paper_mm"])
    return report, {
        "angle-target": (png_bytes(plate), "image/png", "angle-target.png"),
        "pdf": (pdf, "application/pdf", "angle-target-actual-size.pdf"),
    }


def _photo_sample(pixels, zone):
    height, width = pixels.shape
    left, top, right, bottom = zone.pixels(width, height)
    cropped = pixels.rgb[top:bottom, left:right]
    support = pixels.support[top:bottom, left:right]
    success = int(np.count_nonzero(support))
    if success < max(MIN_SUPPORTED, math.ceil(support.size * .65)):
        raise Unprocessable("One test patch has too few fully visible samples.")
    lum = linear_luminance(cropped)[support]
    p05, median, p95 = np.percentile(lum, [5, 50, 95])
    return {"luma": float(median), "p05": float(p05), "p95": float(p95),
            "sampled": success, "total": int(support.size), "source_box": [left, top, right, bottom]}


def _exposure_reference(dark, white):
    difference = white["luma"] - dark["luma"]
    if difference < REFERENCE_GAP:
        raise Unprocessable("The white and dark target patches do not provide enough contrast to normalize this capture.")
    if dark["p95"] > white["p05"] - .025:
        raise Unprocessable("Reference levels overlap. Avoid reflections or relocate the target frame.")
    return difference


def measure_photo(photo, filename, request, view_count):
    obj = object_value(request, "test photo")
    validate_keys(obj, {"angle", "frame", "expected", "name"}, "test photo")
    angle = number(obj.get("angle"), "observed angle", -85, 85)
    intended = integer(obj.get("expected"), "expected view", 1, view_count)
    name = short_text(obj.get("name", filename), "capture name", maximum=122)
    frame = PatchLayout.from_frame(obj.get("frame"), view_count)
    decoded = read_capture(photo, filename)
    pixels = sample_capture(decoded, fit_size(decoded.image.size, 1300))
    measured = [_photo_sample(pixels, patch["region"]) for patch in frame.entries()]
    dark, white = measured[0], measured[-1]
    contrast = _exposure_reference(dark, white)
    patches = []
    for i, info in enumerate(measured[1:-1], start=1):
        normalized = (info["luma"] - dark["luma"]) / contrast
        spread = (info["p95"] - info["p05"]) / contrast
        patches.append({
            "index": i, "ratio": round(normalized, 6), "robust_range": round(spread, 6),
            "luma": round(info["luma"], 7), "valid_pixels": info["sampled"], "pixel_box": info["source_box"],
        })
    projected = [max(0.0, min(1.0, patch["ratio"])) for patch in patches]
    total = sum(projected)
    expected = projected[intended - 1]
    others = [v for n, v in enumerate(projected, start=1) if n != intended]
    strongest = int(np.argmax(projected)) + 1
    comparison = {
        "expected_displayed": strongest == intended,
        "strongest_index": strongest,
        "other_light_fraction": round((total - expected) / total, 6) if total else None,
        "strongest_other_ratio": round(max(others), 6),
        "sum_normalized": round(total, 6),
        "selected_ratio": round(expected, 6),
    }
    original = Image.fromarray(linear_to_srgb(pixels.rgb))
    region_colors = []
    for patch in frame.entries():
        index = patch["view_index"]
        caption = patch["kind"] if index is None else f"V{index + 1}"
        emphasis = index is not None and index + 1 == intended
        region_colors.append((patch["region"].public(), caption,
                              (251, 197, 91) if emphasis else (105, 210, 184)))
    diagram = overlay_regions(original, region_colors)
    report = {
        "kind": "measured-angle-isolation", "angle": angle, "expected_view": intended,
        "input": {"size": [decoded.width, decoded.height], "sha256": sha256(photo).hexdigest(),
                  "label": name, "encoding": decoded.information},
        "illumination_reference": {"black": round(dark["luma"], 6), "white": round(white["luma"], 6),
                                   "contrast": round(contrast, 6)},
        "normalization": "linear luminance relative to dark and white patches from this photograph",
        "patches": patches, "result": comparison,
    }
    return report, png_bytes(resize_report(diagram, 950))


def measured_angle_set(files, settings, view_count):
    if len(files) != len(settings) or not 1 <= len(files) <= MAX_TEST_PHOTOS:
        raise Unprocessable(f"Supply matched descriptions for one to {MAX_TEST_PHOTOS} angle photographs.")
    rows = []
    diagrams = {}
    for i, ((filename, content), request) in enumerate(zip(files, settings, strict=True), start=1):
        record, image = measure_photo(content, filename, request, view_count)
        key = f"photo-{i}"
        record["inspection_plate"] = key
        rows.append(record)
        diagrams[key] = image
    rows.sort(key=lambda entry: entry["angle"])
    values = np.asarray([row["angle"] for row in rows])
    if len(set(values)) != len(values):
        raise Unprocessable("Give each photograph a unique observation angle.")
    y = np.asarray([row["result"]["strongest_other_ratio"] for row in rows])
    active = np.asarray([row["result"]["selected_ratio"] for row in rows])
    diagrams["contrast"] = chart_png([
        ("Expected band", values, active), ("Strongest competing band", values, y),
    ], x_label="Observed angle (degrees)", y_label="Normalised patch ratio", y_bounds=[0, 1])
    return {
        "kind": "measured-angle-series",
        "sources": rows, "count": len(rows), "view_count": view_count,
        "bad_views": [row["angle"] for row in rows if not row["result"]["expected_displayed"]],
        "spread_degrees": [float(values.min()), float(values.max())],
        "note": "Only measured photographs are reported. Orientation, exposure and target coverage affect the results.",
    }, diagrams


def simulated_photograph(
    records, plan, intended_view, principal=.84, neighbour=.11, angle_yield=.006,
):
    _, ordered = target_definition(records, plan)
    count = len(ordered)
    width, height = 530, 690
    layout = PatchLayout.from_frame([.13, .17, .87, .85], count)
    fixture = np.ones((height, width), np.float32) * .23
    random = np.random.default_rng(229 + intended_view)
    for index, entry in enumerate(layout.entries()):
        left, top, right, bottom = entry["region"].pixels(width, height)
        if entry["kind"] == "dark":
            relative = .06
        elif entry["kind"] == "white":
            relative = .79
        elif index == intended_view:
            relative = .06 + .73 * principal
        elif abs(index - intended_view) == 1:
            relative = .06 + .73 * neighbour
        else:
            relative = .06 + .73 * .025
        fixture[top:bottom, left:right] = relative
    fixture += random.normal(0, angle_yield, fixture.shape).astype(np.float32)
    encoded = Image.fromarray(linear_to_srgb(fixture)).convert("RGB")
    draw = ImageDraw.Draw(encoded)
    draw.text((22, 20), "SYNTHETIC TEST PHOTO / not measured material", fill=(239, 227, 179))
    return png_bytes(encoded), [layout.left, layout.top, layout.right, layout.bottom]
