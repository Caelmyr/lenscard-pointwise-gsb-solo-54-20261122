"""Direct measurements for a numbered lenticular angle isolation patch.

One observation angle may be covered by several photographs taken from the same
position, each with different specular reflections.  Every photograph keeps its
own dark/white exposure reference.  Per numbered patch the readable candidates
are compared across the photographs of that angle so the clearest source can be
selected, glare can be explained, and genuinely ambiguous patches stay blank.
"""
from dataclasses import dataclass
from hashlib import sha256
import math

import numpy as np
from PIL import Image, ImageDraw

from .imaging import fit_size, linear_luminance, linear_to_srgb, read_capture, sample_capture
from .models import Position, Region, Unprocessable, integer, number, object_value, short_text, validate_keys
from .optics import LensGrid, angular_domains, band_index, physical_slot
from .printing import one_raster_pdf
from .visual import chart_png, overlay_regions, png_bytes, resize_report


BLACK_LUMA = .025
WHITE_LUMA = .86
MAX_TEST_PHOTOS = 7
MAX_PHOTOS_PER_ANGLE = 3
REFERENCE_GAP = .08
MIN_SUPPORTED = 40
# A robust within-patch spread above this is treated as a local reflection.
GLARE_SPREAD = .30
# Normalised median readings from two captures beyond this are considered
# incompatible; the patch is then resolved from the glare evidence or left blank.
DISAGREEMENT = .18


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
    grid = LensGrid(plan.lens, plan.sheet.dpi, width, height)
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


def _excluded_indices(value, row_count, label):
    """Parse the user-marked rows that must not be read from one photograph."""
    if value is None:
        return frozenset()
    if not isinstance(value, list):
        raise Unprocessable(f"{label} must be a list of patch row indices.")
    result = set()
    for item in value:
        index = integer(item, f"{label} entry", 0, row_count - 1)
        result.add(index)
    return frozenset(result)


def _photo_sample(pixels, zone, enabled):
    height, width = pixels.shape
    left, top, right, bottom = zone.pixels(width, height)
    cropped = pixels.rgb[top:bottom, left:right]
    support = pixels.support[top:bottom, left:right].copy()
    if not enabled:
        support[:] = False
    success = int(np.count_nonzero(support))
    if not enabled:
        return {"luma": None, "p05": None, "p95": None,
                "sampled": 0, "total": int(support.size), "source_box": [left, top, right, bottom]}
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


def _glare_flags(spread):
    return spread is not None and spread > GLARE_SPREAD


def measure_photo(photo, filename, request, view_count):
    """Measure one photograph with its own dark/white reference and masks."""
    obj = object_value(request, "test photo")
    validate_keys(obj, {"angle", "frame", "expected", "name", "exclude"}, "test photo")
    angle = number(obj.get("angle"), "observed angle", -85, 85)
    intended = integer(obj.get("expected"), "expected view", 1, view_count)
    name = short_text(obj.get("name", filename), "capture name", maximum=122)
    frame = PatchLayout.from_frame(obj.get("frame"), view_count)
    excluded = _excluded_indices(obj.get("exclude"), frame.rows, "excluded patch rows")
    if 0 in excluded:
        raise Unprocessable("A photograph needs its dark reference patch; remove it from the excluded rows.")
    if frame.rows - 1 in excluded:
        raise Unprocessable("A photograph needs its white reference patch; remove it from the excluded rows.")
    decoded = read_capture(photo, filename)
    pixels = sample_capture(decoded, fit_size(decoded.image.size, 1300))
    measured = [
        _photo_sample(pixels, patch["region"], index not in excluded)
        for index, patch in enumerate(frame.entries())
    ]
    dark, white = measured[0], measured[-1]
    contrast = _exposure_reference(dark, white)
    patches = []
    for i, info in enumerate(measured[1:-1], start=1):
        if i in excluded:
            patches.append({
                "index": i, "excluded": True, "ratio": None, "robust_range": None,
                "luma": None, "valid_pixels": 0, "pixel_box": info["source_box"],
            })
            continue
        normalized = (info["luma"] - dark["luma"]) / contrast
        spread = (info["p95"] - info["p05"]) / contrast
        patches.append({
            "index": i, "excluded": False, "ratio": round(normalized, 6),
            "robust_range": round(spread, 6), "luma": round(info["luma"], 7),
            "valid_pixels": info["sampled"], "pixel_box": info["source_box"],
            "glare": _glare_flags(spread),
        })
    original = Image.fromarray(linear_to_srgb(pixels.rgb))
    region_colors = []
    for index, patch in enumerate(frame.entries()):
        view_index = patch["view_index"]
        caption = patch["kind"] if view_index is None else f"V{view_index + 1}"
        if index in excluded:
            color = (232, 96, 104)
            caption = f"x {caption}"
        elif view_index is not None and patches[view_index].get("glare"):
            color = (247, 191, 113)
        elif view_index is not None and view_index + 1 == intended:
            color = (251, 197, 91)
        else:
            color = (105, 210, 184)
        region_colors.append((patch["region"].public(), caption, color))
    diagram = overlay_regions(original, region_colors)
    draw = ImageDraw.Draw(diagram)
    draw.rectangle((4, 4, 4 + max(150, int(diagram.width * .32)), 26), fill=(21, 35, 46))
    draw.text((8, 8), f"{name}  /  {angle:g} deg", fill=(236, 241, 245))
    record = {
        "angle": angle, "expected_view": intended,
        "input": {"size": [decoded.width, decoded.height], "sha256": sha256(photo).hexdigest(),
                  "label": name, "filename": filename, "encoding": decoded.information},
        "excluded_rows": sorted(excluded),
        "illumination_reference": {"black": round(dark["luma"], 6), "white": round(white["luma"], 6),
                                   "contrast": round(contrast, 6)},
        "normalization": "linear luminance relative to dark and white patches from this photograph",
        "patches": patches,
    }
    return record, png_bytes(resize_report(diagram, 950))


def _candidate(patch, photo):
    return {
        "photo_id": photo["photo_id"], "label": photo["input"]["label"],
        "ratio": patch["ratio"], "robust_range": patch["robust_range"],
        "valid_pixels": patch["valid_pixels"], "glare": bool(patch.get("glare")),
    }


def _select_patch(index, candidates):
    """Choose a source for one numbered patch or leave it blank with a reason."""
    readable = [candidate for candidate in candidates if candidate["ratio"] is not None]
    if not readable:
        return {
            "index": index, "ratio": None, "source_photo": None, "status": "blank",
            "glare": False, "candidate_count": 0, "candidates": candidates,
            "note": "Every photograph of this angle marked this patch unreadable; no value is reported.",
        }
    if len(readable) == 1:
        only = readable[0]
        note = (f"Only {only['label']} reads this patch clearly; the value is taken from that photograph."
                if len(candidates) == 1 else
                f"The other photographs exclude this patch; the only readable value comes from {only['label']}.")
        if only["glare"]:
            note += " A wide within-patch spread still suggests a local reflection; verify before approving."
        return {
            "index": index, "ratio": only["ratio"], "source_photo": only["photo_id"],
            "source_label": only["label"], "status": "single-source",
            "glare": only["glare"], "candidate_count": len(readable),
            "candidates": candidates, "note": note,
        }
    ratios = np.asarray([candidate["ratio"] for candidate in readable], dtype=float)
    disagreement = float(ratios.max() - ratios.min())
    clean = [candidate for candidate in readable if not candidate["glare"]]
    glare_labels = [candidate["label"] for candidate in readable if candidate["glare"]]
    if not glare_labels and disagreement <= DISAGREEMENT:
        chosen = min(readable, key=lambda candidate: (candidate["robust_range"], -candidate["valid_pixels"]))
        note = (f"{len(readable)} photographs agree within {DISAGREEMENT:g}; "
                f"{chosen['label']} has the tightest within-patch spread and is used.")
        return {
            "index": index, "ratio": chosen["ratio"], "source_photo": chosen["photo_id"],
            "source_label": chosen["label"], "status": "agreed",
            "glare": False, "candidate_count": len(readable), "candidates": candidates,
            "disagreement": round(disagreement, 6), "note": note,
        }
    if glare_labels:
        if len(clean) == 1:
            chosen = clean[0]
            note = (f"{', '.join(glare_labels)} shows a local reflection (wide within-patch spread) "
                    f"and is set aside; {chosen['label']} is used. Median readings differ by "
                    f"{disagreement:.3f}.")
            return {
                "index": index, "ratio": chosen["ratio"], "source_photo": chosen["photo_id"],
                "source_label": chosen["label"], "status": "glare-rejected-other",
                "glare": False, "candidate_count": len(readable), "candidates": candidates,
                "disagreement": round(disagreement, 6), "note": note,
            }
        note = ("Every readable photograph shows a reflective patch (wide within-patch spread); "
                "no glare-free source exists, so this patch is left blank rather than guessed.")
        return {
            "index": index, "ratio": None, "source_photo": None, "status": "ambiguous",
            "glare": True, "candidate_count": len(readable), "candidates": candidates,
            "disagreement": round(disagreement, 6), "note": note,
        }
    note = (f"Readings differ by {disagreement:.3f} without a clear reflection signature in any "
            "photograph; the cause cannot be decided here, so this patch is left blank.")
    return {
        "index": index, "ratio": None, "source_photo": None, "status": "ambiguous",
        "glare": False, "candidate_count": len(readable),
        "candidates": candidates, "disagreement": round(disagreement, 6), "note": note,
    }


def _angle_summary(group, view_count):
    selections = group["patches"]
    decided = [patch for patch in selections if patch["ratio"] is not None]
    blank = [patch["index"] for patch in selections if patch["ratio"] is None]
    projected = {patch["index"]: max(0.0, min(1.0, patch["ratio"])) for patch in decided}
    intended = group["expected_view"]
    strongest = max(projected, key=projected.get) if projected else None
    expected = projected.get(intended)
    others = [value for index, value in projected.items() if index != intended]
    complete = len(decided) == view_count
    expected_displayed = complete and strongest == intended
    rejected_photos = [photo["photo_id"] for photo in group["photos"] if photo["status"] != "measured"]
    result = {
        "complete": complete,
        "blank_patches": blank,
        "expected_displayed": expected_displayed,
        "strongest_index": strongest,
        "selected_ratio": round(expected, 6) if expected is not None else None,
        "strongest_other_ratio": round(max(others), 6) if others else None,
        "sum_normalized": round(sum(projected.values()), 6) if projected else None,
        "rejected_photos": rejected_photos,
        "passed": complete and expected_displayed and not rejected_photos,
        "pass_note": (
            "All patches resolved and the intended band is dominant."
            if complete and expected_displayed and not rejected_photos
            else "This angle is not fully passed: " + "; ".join(
                ([f"patch {', '.join(map(str, blank))} could not be read"] if blank else [])
                + ([f"{len(rejected_photos)} photograph(s) failed their own reference checks"]
                   if rejected_photos else [])
                + ([] if expected_displayed or not complete else
                   ["the intended band is not the strongest readable band"])
            )
        ),
    }
    group["result"] = result
    return result


def _legacy_groups(settings):
    groups = []
    for item in settings:
        if not isinstance(item, dict):
            raise Unprocessable("Each angle entry must be an object.")
        groups.append({
            "angle": item["angle"], "expected": item["expected"], "frame": item["frame"],
            "name": item.get("name"),
            "photos": [{"name": item.get("name"), "frame": item["frame"],
                        "exclude": item.get("exclude", [])}],
        })
    return groups


def _parse_groups(settings, view_count):
    if not isinstance(settings, list) or not settings:
        raise Unprocessable("Supply at least one angle group of photographs.")
    groups = []
    if all(isinstance(item, dict) and "photos" in item for item in settings):
        for group in settings:
            validate_keys(group, {"angle", "expected", "frame", "name", "photos"}, "angle group")
            photos = group.get("photos")
            if not isinstance(photos, list) or not 1 <= len(photos) <= MAX_PHOTOS_PER_ANGLE:
                raise Unprocessable(f"Each angle needs one to {MAX_PHOTOS_PER_ANGLE} photographs.")
            parsed_photos = []
            for photo in photos:
                if not isinstance(photo, dict):
                    raise Unprocessable("Each angle photograph must be an object with name, frame and exclude.")
                validate_keys(photo, {"name", "frame", "exclude"}, "angle photograph")
                frame = photo.get("frame") if "frame" in photo else group.get("frame")
                if frame is None:
                    raise Unprocessable("Every photograph needs a target frame, either on the photo or the angle group.")
                parsed_photos.append({
                    "name": photo.get("name") or group.get("name"),
                    "frame": frame,
                    "exclude": _excluded_indices(photo.get("exclude", []),
                                                 PatchLayout.from_frame(frame, view_count).rows,
                                                 "excluded patch rows"),
                })
            groups.append({
                "angle": number(group.get("angle"), "observed angle", -85, 85),
                "expected": integer(group.get("expected"), "expected view", 1, view_count),
                "frame": group.get("frame"),
                "name": group.get("name"),
                "photos": parsed_photos,
            })
    else:
        groups = _legacy_groups(settings)
    angles = [group["angle"] for group in groups]
    if len(set(angles)) != len(angles):
        raise Unprocessable("Give each angle group a unique observation angle; repeat photographs belong in one group.")
    return groups


def measured_angle_set(files, settings, view_count):
    groups = _parse_groups(settings, view_count)
    total_photos = sum(len(group["photos"]) for group in groups)
    if len(files) != total_photos or not 1 <= total_photos <= MAX_TEST_PHOTOS:
        raise Unprocessable(
            f"Supply one image file per photograph ({total_photos} listed) and at most {MAX_TEST_PHOTOS} in total."
        )
    diagrams = {}
    angle_groups = []
    file_cursor = 0
    for group_index, group in enumerate(groups, start=1):
        photo_records = []
        for photo_index, photo in enumerate(group["photos"], start=1):
            photo_id = f"a{group_index}-photo-{photo_index}"
            filename, content = files[file_cursor]
            file_cursor += 1
            request = {
                "angle": group["angle"], "expected": group["expected"],
                "frame": photo["frame"], "exclude": sorted(photo["exclude"]),
                "name": photo["name"] or f"Angle {group['angle']:g} — photo {photo_index}",
            }
            try:
                record, image = measure_photo(content, filename, request, view_count)
                record["photo_id"] = photo_id
                record["status"] = "measured"
                diagrams[photo_id] = image
                photo_records.append(record)
            except Unprocessable as error:
                photo_records.append({
                    "photo_id": photo_id, "status": "rejected", "angle": group["angle"],
                    "expected_view": group["expected"],
                    "input": {"label": request["name"], "filename": filename,
                              "sha256": sha256(content).hexdigest()},
                    "excluded_rows": sorted(photo["exclude"]),
                    "patches": [], "rejection": str(error),
                })
        merged = _merge_group(group, photo_records, view_count)
        angle_groups.append(merged)

    angle_groups.sort(key=lambda entry: entry["angle"])
    values = np.asarray([entry["angle"] for entry in angle_groups], dtype=float)
    selected = np.asarray([entry["result"]["selected_ratio"]
                           if entry["result"]["selected_ratio"] is not None else float("nan")
                           for entry in angle_groups], dtype=float)
    competing = np.asarray([entry["result"]["strongest_other_ratio"]
                            if entry["result"]["strongest_other_ratio"] is not None else float("nan")
                            for entry in angle_groups], dtype=float)
    diagrams["contrast"] = chart_png([
        ("Expected band (selected source)", values, selected),
        ("Strongest competing band", values, competing),
    ], x_label="Observed angle (degrees)", y_label="Normalised patch ratio", y_bounds=[0, 1])
    failed_angles = [
        {"angle": entry["angle"], "reason": entry["result"]["pass_note"]}
        for entry in angle_groups if not entry["result"]["passed"]
    ]
    report = {
        "kind": "measured-angle-series",
        "angle_groups": angle_groups,
        "sources": angle_groups,
        "count": len(angle_groups),
        "photo_count": total_photos,
        "view_count": view_count,
        "bad_views": [entry["angle"] for entry in angle_groups if not entry["result"]["expected_displayed"]],
        "incomplete_angles": [entry["angle"] for entry in angle_groups if not entry["result"]["complete"]],
        "failed_angles": failed_angles,
        "spread_degrees": [float(values.min()), float(values.max())],
        "selection_policy": (
            "Each photograph is normalised against its own dark and white patches. Per numbered patch the "
            "readable photographs of one angle are compared: photographs without a reflection signature that "
            "agree use the tightest-spread source; a wide within-patch spread identifies a local reflection so "
            "that capture can be set aside when another clean capture exists; if every capture reflects, or the "
            "readings disagree without an explainable cause, the patch stays blank and the angle is not passed."
        ),
        "note": "Only measured photographs are reported. Orientation, exposure, masks and target coverage affect the results.",
    }
    return report, diagrams


def _merge_group(group, photo_records, view_count):
    measured = [record for record in photo_records if record["status"] == "measured"]
    if not measured:
        rejections = [record.get("rejection", "the capture could not be measured")
                      for record in photo_records if record["status"] != "measured"]
        merged = {
            "angle": group["angle"],
            "expected_view": group["expected"],
            "photo_count": len(photo_records),
            "photos": photo_records,
            "patches": [],
            "group_warning": "No photograph of this angle passed its own reference checks: "
                             + " | ".join(rejections),
        }
        merged["result"] = {
            "complete": False, "blank_patches": list(range(1, view_count + 1)),
            "expected_displayed": False, "strongest_index": None,
            "selected_ratio": None, "strongest_other_ratio": None, "sum_normalized": None,
            "rejected_photos": [record["photo_id"] for record in photo_records],
            "passed": False,
            "pass_note": "This angle is not fully passed: every photograph failed its own reference checks.",
        }
        return merged
    patches = []
    for index in range(1, view_count + 1):
        candidates = []
        for record in measured:
            patch = next((item for item in record["patches"] if item["index"] == index), None)
            if patch is None or patch.get("excluded"):
                continue
            candidates.append(_candidate(patch, record))
        patches.append(_select_patch(index, candidates))
    merged = {
        "angle": group["angle"],
        "expected_view": group["expected"],
        "photo_count": len(photo_records),
        "photos": photo_records,
        "patches": patches,
    }
    _angle_summary(merged, view_count)
    return merged


def simulated_photograph(
    records, plan, intended_view, principal=.84, neighbour=.11, angle_yield=.006,
    glare_rows=(), glare_strength=.5,
):
    """Build a synthetic capture; ``glare_rows`` injects local bright reflections.

    Row indices follow the target layout: 0 is the dark reference, the last row
    is the white reference and the rows between are the numbered view patches.
    """
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
        if index in glare_rows:
            band_height = max(6, (bottom - top) // 3)
            band_top = top + (bottom - top) // 4
            fixture[band_top:band_top + band_height, left + 6:right - 6] = (
                np.clip(fixture[band_top:band_top + band_height, left + 6:right - 6]
                        + glare_strength, 0, 1)
            )
    fixture += random.normal(0, angle_yield, fixture.shape).astype(np.float32)
    encoded = Image.fromarray(linear_to_srgb(np.clip(fixture, 0, 1))).convert("RGB")
    draw = ImageDraw.Draw(encoded)
    draw.text((22, 20), "SYNTHETIC TEST PHOTO / not measured material", fill=(239, 227, 179))
    if glare_rows:
        draw.text((22, 40), "INJECTED LOCAL REFLECTION", fill=(247, 191, 113))
    return png_bytes(encoded), [layout.left, layout.top, layout.right, layout.bottom]
