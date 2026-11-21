"""Checks on the frozen device raster and on separately measured physical prints."""
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import math

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage, signal

from .imaging import fit_size, linear_luminance, read_capture, sample_capture
from .models import Conflict, Plan, Region, Unprocessable, integer, number, object_value, validate_keys
from .optics import LensGrid, band_index, view_index
from .printing import one_raster_pdf
from .visual import chart_png, overlay_regions, png_bytes, resize_report


READBACK_LIMIT = 1400
MIN_SCAN_CONTRAST = .015


@dataclass(frozen=True)
class RasterBox:
    left: int
    top: int
    right: int
    bottom: int

    def list(self):
        return [self.left, self.top, self.right, self.bottom]

    def figure(self, scale):
        return [round(value * scale, 6) for value in self.list()]


@dataclass(frozen=True)
class DeviceFile:
    pixels: np.ndarray
    grid: LensGrid
    job: Plan
    proof: dict

    @classmethod
    def from_record(cls, library, proof):
        plan = Plan.parse(proof["snapshot"]["identity"]["plan"])
        width, height = plan.sheet.size
        grid = LensGrid(plan.lens, plan.sheet.dpi, width, height)
        file = library.asset_bytes(proof["assets"]["plate"])
        with Image.open(BytesIO(file)) as image:
            raster = np.asarray(image.convert("RGB"))
        if raster.shape[:2] != (height, width):
            raise Conflict("The retained raster dimensions are inconsistent with its print plan.")
        return cls(raster, grid, plan, proof)

    @property
    def views(self):
        return sorted([item for item in self.proof["snapshot"]["identity"]["views"] if item["enabled"]],
                      key=lambda item: item["position"])


def _pixel_bounds(device, center_x_mm, center_y_mm, span_x_mm, span_y_mm):
    width = device.grid.width
    height = device.grid.height
    ratio = device.grid.step
    x = center_x_mm / ratio
    y = center_y_mm / ratio
    depth = 0.5 * span_x_mm / ratio
    reach = 0.5 * span_y_mm / ratio
    left = max(0, math.floor(x - depth))
    right = min(width, math.ceil(x + depth))
    top = max(0, math.floor(y - reach))
    bottom = min(height, math.ceil(y + reach))
    if right - left < 4 or bottom - top < 4:
        raise Unprocessable("The device sampling window has too few contained print pixels.")
    return RasterBox(left, top, right, bottom)


def _enlarged_tile(values, enlarge):
    h, w = values.shape[:2]
    multiplier = max(1, min(enlarge, int(1024 / max(w, h))))
    art = Image.fromarray(values)
    if multiplier > 1:
        art = art.resize((w * multiplier, h * multiplier), Image.Resampling.NEAREST)
    return art, multiplier


def _view_assignment(device, box):
    cols = device.grid.actual_side(np.arange(box.left, box.right), device.job.sheet.mount)
    rows = np.arange(box.top, box.bottom)
    phase = device.grid.phase_at(cols[None, :], rows[:, None])
    indices = view_index(band_index(phase, len(device.views)), len(device.views), device.job.lens.order)
    return indices, np.mod(phase, 1)


def _slice_counts(device, box, device_pixels):
    assigned, phases = _view_assignment(device, box)
    rows = np.arange(box.top, box.bottom)
    columns = np.arange(box.left, box.right)
    channel_average = device_pixels.mean(axis=(0, 1))
    counts = []
    for n, view in enumerate(device.views):
        subset = device_pixels[assigned == n]
        counts.append({
            "view_id": view["id"], "title": view["title"], "index": n, "central_dot_count": int(subset.shape[0]),
            "expected_angle": view["position"],
            "mean_print_rgb": [round(float(value), 4) for value in subset.mean(axis=0)] if subset.size else None,
        })
    return {
        "views": counts, "center_phase": round(float(phases[len(rows) // 2, len(columns) // 2]), 7),
        "mean_print_rgb": [round(float(value), 4) for value in channel_average],
        "midrow": [int(value) for value in assigned[len(rows) // 2]],
    }


def read_device_region(library, card_id, proof_id, settings):
    proof = library.proof(card_id, proof_id)
    device = DeviceFile.from_record(library, proof)
    obj = object_value(settings, "device crop")
    validate_keys(obj, {"x_mm", "y_mm", "width_mm", "height_mm", "enlarge"}, "device crop")
    sheet = device.job.sheet
    coordinates = (
        number(obj.get("x_mm", sheet.width_mm / 2), "device x", 0, sheet.width_mm),
        number(obj.get("y_mm", sheet.height_mm / 2), "device y", 0, sheet.height_mm),
        number(obj.get("width_mm", 12), "window width", .3, 30),
        number(obj.get("height_mm", 8), "window height", .3, 25),
    )
    box = _pixel_bounds(device, *coordinates)
    tile = device.pixels[box.top:box.bottom, box.left:box.right]
    settings = integer(obj.get("enlarge", 4), "preview magnification", 1, 10)
    enlarged, factor = _enlarged_tile(tile, settings)
    counts = _slice_counts(device, box, tile)
    return {
        "kind": "physical-raster-region",
        "proof_id": proof_id,
        "fingerprint": proof["fingerprint"],
        "mount": sheet.mount,
        "device_window_px": box.list(),
        "device_box_mm": box.figure(device.grid.step),
        "preview_factor": factor,
        "method": device.job.sampling.method,
        "sampling_note": "Assignments describe dot centers. Boundary dots can combine light in the coverage method.",
        "readout": counts,
    }, {"device-window": png_bytes(enlarged)}


@dataclass(frozen=True)
class SizeReadback:
    width: float
    height: float
    diagonal: float | None
    instrument: str

    @classmethod
    def parse(cls, values):
        obj = object_value(values, "physical size")
        validate_keys(obj, {"width", "height", "diagonal", "instrument"}, "physical size")
        diagonal = obj.get("diagonal")
        from .models import short_text
        return cls(
            number(obj.get("width"), "width", 4, 420),
            number(obj.get("height"), "height", 4, 420),
            number(diagonal, "diagonal", 5, 520) if diagonal is not None else None,
            short_text(obj.get("instrument", "hand ruler"), "instrument", maximum=46),
        )


def verify_physical_size(library, card_id, proof_id, values):
    measurement = SizeReadback.parse(values)
    proof = library.proof(card_id, proof_id)
    stored = proof["snapshot"]["identity"]["plan"]
    plan = Plan.parse(stored)
    w = plan.sheet.width_mm
    h = plan.sheet.height_mm
    aspect = (measurement.width / measurement.height) / (w / h)
    expected = math.hypot(w, h)
    diagonal_residual = measurement.diagonal - expected if measurement.diagonal is not None else None
    drift = [measurement.width / w - 1, measurement.height / h - 1]
    tolerance = .008
    acceptable = max(abs(value) for value in drift) <= tolerance
    reasons = []
    if abs(aspect - 1) > .008:
        reasons.append("Different scaling on horizontal and vertical edges.")
    if not acceptable:
        reasons.append("Check actual-size paper output and keep the approved file unchanged.")
    return {
        "kind": "physical-card-size-check", "proof": proof_id, "proof_sha256": proof["fingerprint"],
        "expected_mm": [w, h], "reported_mm": [measurement.width, measurement.height],
        "percent_vs_plan": [round(value * 100, 4) for value in drift],
        "expected_diagonal_mm": round(expected, 6),
        "diagonal_residual_mm": round(diagonal_residual, 6) if diagonal_residual is not None else None,
        "shape_ratio": round(aspect, 6), "measurement_tool": measurement.instrument,
        "status": "within-size-practice-limit" if acceptable else "outside-size-practice-limit",
        "inspection_notes": reasons,
    }


def _native_profile(pixels, region):
    h, w = pixels.shape
    left, top, right, bottom = region.pixels(w, h)
    if min(right - left, bottom - top) < 16:
        raise Unprocessable("Choose a wide, visible swatch containing multiple lens lines.")
    zone = pixels.rgb[top:bottom, left:right]
    valid = pixels.support[top:bottom, left:right]
    counts = valid.sum(axis=0)
    covered = counts > .7 * (bottom - top)
    if covered.mean() < .73:
        raise Unprocessable("Lens-stripe readback needs continuous image coverage across the target.")
    within = linear_luminance(zone)
    profile = (within * valid).sum(axis=0) / np.maximum(1, counts)
    x = np.arange(profile.size)
    return np.interp(x, x[covered], profile[covered]), [left, top, right, bottom]


def _frequency_estimate(values, pixels_per_mm, target_cycles_mm):
    if np.std(values) < MIN_SCAN_CONTRAST:
        raise Unprocessable("The ruler stripe region does not have enough contrast.")
    length = len(values)
    cleaned = signal.detrend(values) * np.hanning(length)
    output = max(2048, 2 ** math.ceil(math.log2(length * 7)))
    spectrum = np.abs(np.fft.rfft(cleaned, n=output)) ** 2
    axis = np.fft.rfftfreq(output, d=1 / pixels_per_mm)
    frequency_limit = pixels_per_mm / 2
    if target_cycles_mm * 1.4 > frequency_limit:
        raise Unprocessable("Use a larger scale photograph; the expected lens pitch would be undersampled.")
    usable = (axis >= target_cycles_mm * .62) & (axis <= target_cycles_mm * 1.42)
    if usable.sum() < 8:
        raise Unprocessable("Include more lens lines in the image to constrain the pitch.")
    indexes = np.nonzero(usable)[0]
    peaks, _ = signal.find_peaks(spectrum[indexes])
    if not len(peaks):
        raise Unprocessable("No stable pitch peak could be found near the approved lens value.")
    index = int(indexes[peaks[np.argmax(spectrum[indexes][peaks])]])
    denominator = np.median(spectrum[indexes])
    contrast = float(spectrum[index] / max(denominator, 1e-10))
    area = indexes[(axis[indexes] < axis[index] * .90) | (axis[indexes] > axis[index] * 1.1)]
    alternate = float(spectrum[area].max(initial=1e-10))
    separation = float(spectrum[index] / max(alternate, 1e-10))
    baseline = float(spectrum[index])
    slope_a, slope_c = float(spectrum[index - 1]), float(spectrum[index + 1])
    normalizer = 2 * (slope_a - 2 * baseline + slope_c)
    corrected = (slope_a - slope_c) / normalizer if normalizer else 0
    estimate = float(axis[index] + np.clip(corrected, -.5, .5) * (axis[1] - axis[0]))
    return estimate, contrast, separation, axis[indexes], spectrum[indexes]


def measure_ruler_pitch(library, card_id, proof_id, filename, image_bytes, values):
    proof = library.proof(card_id, proof_id)
    plan = Plan.parse(proof["snapshot"]["identity"]["plan"])
    obj = object_value(values, "pitch readback")
    validate_keys(obj, {"camera_px_per_mm", "region"}, "pitch readback")
    known_scale = number(obj.get("camera_px_per_mm"), "photograph pixels per millimetre", 1.5, 450)
    region = Region.parse(obj.get("region"))
    decoded = read_capture(image_bytes, filename)
    chosen = fit_size((decoded.width, decoded.height), READBACK_LIMIT)
    pixels = sample_capture(decoded, chosen)
    sample_scale = known_scale * chosen[0] / decoded.width
    target = plan.lens.lpi / 25.4
    radial, source = _native_profile(pixels, region)
    actual, contrast, separation, axis, power = _frequency_estimate(radial, sample_scale, target)
    retained = contrast > 8 and separation > 1.45
    chart = chart_png(
        [("linear luminance power", axis, 10 * np.log10(np.maximum(power, 1e-9)))],
        x_label="Cycles per millimetre in scan", y_label="Power (dB)", markers=[(target, "planned")],
    )
    scan = Image.fromarray(np.asarray(decoded.image.convert("RGB").resize(chosen)))
    marked = overlay_regions(scan, [(region.public(), "validated pitch area", (247, 204, 99))])
    return {
        "kind": "optical-stripe-pitch-readback", "proof_id": proof_id, "proof_sha": proof["fingerprint"],
        "input_sha": sha256(image_bytes).hexdigest(), "sample_region": region.public(),
        "scanned_pixels": chosen, "scanned_region_px": source, "recorded_camera_scale": known_scale,
        "planned_lpi": plan.lens.lpi, "readback_lpi": round(actual * 25.4, 5),
        "deviation_lpi": round((actual - target) * 25.4, 5),
        "peak_to_floor": round(contrast, 6), "peak_to_alternate": round(separation, 6),
        "status": "supported" if retained else "multiple-candidates",
        "interpretation": "Compare a perpendicular photograph of the actual ruled target. Separate lenticular harmonics or perspective may change the observed peak.",
    }, {"pitch-spectrum": chart, "marked-scan": png_bytes(resize_report(marked, 920))}


def device_contact_sheet(library, card_id, proof_id):
    proof = library.proof(card_id, proof_id)
    device = DeviceFile.from_record(library, proof)
    dpi = 180
    scale = dpi / 25.4
    paper_width = 210.0
    paper_height = 297.0
    w = round(paper_width * scale)
    h = round(paper_height * scale)
    live = device.job.sheet
    raster_w = round(live.width_mm * scale)
    raster_h = round(live.height_mm * scale)
    if raster_w + 108 > w or raster_h + 245 > h:
        raise Unprocessable("This card is too wide or tall to show at actual size in the A4 contact sheet.")
    top = 128
    left = (w - raster_w) // 2
    page = Image.new("RGB", (w, h), (249, 249, 244))
    draw = ImageDraw.Draw(page)
    art = Image.fromarray(device.pixels).resize((raster_w, raster_h), Image.Resampling.BOX)
    page.paste(art, (left, top))
    draw.rectangle((left - 1, top - 1, left + raster_w, top + raster_h), outline=(28, 45, 40))
    draw.text((left, 26), "LENS CARD / HAND MEASUREMENT SHEET", fill=(24, 52, 45))
    draw.text((left, 55), f"Proof {proof_id[:10]} | {live.width_mm:.2f} x {live.height_mm:.2f} mm (100% print)", fill=(35, 55, 47))
    base = top + raster_h + 24
    length = round(35 * scale)
    draw.line((left, base, left + length, base), fill=(17, 47, 39), width=2)
    for n in range(36):
        col = left + round(n * scale)
        bottom = base + (14 if n % 5 == 0 else 6)
        draw.line((col, base, col, bottom), fill=(26, 47, 41), width=1)
        if n % 10 == 0:
            draw.text((col - 4, bottom + 3), str(n), fill=(36, 56, 49))
    draw.text((left, base + 45), "35 mm ruler on A4 | check this before using cut-size readings", fill=(29, 54, 45))
    image = np.asarray(page)
    report = {
        "kind": "cut-size-contact-proof", "proof_id": proof_id, "proof_sha": proof["fingerprint"],
        "positions_mm": [round(left / scale, 6), round(top / scale, 6), live.width_mm, live.height_mm],
        "paper_mm": [paper_width, paper_height], "card_raster_mm": [raster_w / scale, raster_h / scale],
        "ruler_mm": 35, "page_dpi": dpi, "copy_artwork": "Reduced resolution for physical measurement only",
    }
    return report, {"contact-proof": png_bytes(resize_report(page, 1000))}, one_raster_pdf(image, paper_width, paper_height)
