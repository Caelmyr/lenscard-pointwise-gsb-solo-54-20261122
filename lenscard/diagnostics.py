"""Measure angle originals without changing the pixels or the frozen print."""
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import math

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage, signal

from .framing import feather_comparison, fit_views, prepare_registered, working_preview_sizes
from .imaging import (
    CapturePixels, fit_size, float_resize, linear_luminance, linear_to_srgb,
    percentile_summary, read_capture, read_pixels, valid_neighborhood,
)
from .models import Position, Region, Unprocessable, number, require_id
from .visual import (
    chart_png, code_statistics, heatmap, overlay_regions, paired_bounds, png_bytes,
    resize_report, rounding, side_by_side,
)


TONE_LIMIT = 1100
PROFILE_POINTS = 256
COMPARE_LIMIT = 850
SUPPORT_WARNING = .13


@dataclass
class InputSample:
    record: dict
    pixels: CapturePixels
    zone: Region

    @property
    def box(self):
        height, width = self.pixels.shape
        return self.zone.pixels(width, height)

    def area(self, values):
        left, top, right, bottom = self.box
        return values[top:bottom, left:right]

    def base(self):
        valid = self.area(self.pixels.support)
        if np.count_nonzero(valid) < 16:
            raise Unprocessable("Move the region to at least 16 fully supported pixels.")
        rgb = self.area(self.pixels.rgb)
        return rgb, valid

    def public(self):
        left, top, right, bottom = self.box
        px_h, px_w = self.pixels.shape
        raw_width, raw_height = self.pixels.original_size
        source_box = [
            math.floor(left * raw_width / px_w),
            math.floor(top * raw_height / px_h),
            math.ceil(right * raw_width / px_w),
            math.ceil(bottom * raw_height / px_h),
        ]
        return {
            "id": self.record["id"],
            "name": self.record["title"],
            "source_sha": self.record["sha"],
            "source_size": list(self.pixels.original_size),
            "chart_pixels": [px_w, px_h],
            "selected_source_box": source_box,
            "region": self.zone.public(),
        }


def load_sample(library, card_id, view_id, region=None, limit=TONE_LIMIT):
    record = library.view(card_id, require_id(view_id, "view"))
    target = fit_size((record["width"], record["height"]), limit)
    pixels = read_pixels(library, record, target)
    return InputSample(record, pixels, Region.parse(region))


def _fraction(values):
    return round(float(np.mean(values)), 7)


def get_original_info(library, card_id, view_id):
    record = library.view(card_id, view_id)
    body = library.source_path(record["sha"]).read_bytes()
    digest = sha256(body).hexdigest()
    frame = read_capture(body, record["name"])
    size = fit_size((frame.width, frame.height), 540)
    from .imaging import sample_capture
    px = sample_capture(frame, size)
    return {
        "original_bytes": len(body),
        "original_requested": record["sha"],
        "verified": digest == record["sha"] and len(body) == record["bytes"],
        "width": record["width"], "height": record["height"],
        "stored_fields": record["information"],
        "decoded_fields": frame.information,
        "supported_fraction": _fraction(px.support),
    }


def _compact_stats(rgb, support):
    luma = rgb @ np.array([.2126, .7152, .0722], dtype=np.float32)
    rows = {}
    for band, values in zip(("red", "green", "blue", "luma"), [*np.moveaxis(rgb, 2, 0), luma], strict=True):
        codes = values[support]
        rows[band] = {
            "percentiles": percentile_summary(values, support),
            "shadow_fraction": _fraction(codes < .003),
            "high_fraction": _fraction(codes > .96),
        }
    return rows


def exposure_report(sample):
    rgb, support = sample.base()
    v = linear_luminance(rgb)
    plots = []
    domain = np.arange(64) / 63
    bands = [*np.moveaxis(rgb, 2, 0), v]
    for name, codes in zip(("Red", "Green", "Blue", "Luma"), bands, strict=True):
        vals, _ = np.histogram(codes[support], bins=64, range=(0, 1))
        scale = max(1, np.max(vals))
        plots.append((name, domain, np.sqrt(vals / scale)))
    chart = chart_png(plots, x_label="Linear input", y_label="Histogram (root peak normalised)")
    marker = Image.fromarray(linear_to_srgb(rgb))
    clipping = np.any(rgb > .96, axis=-1) & support
    shadows = (v < .003) & support
    plate = np.asarray(marker).copy()
    plate[clipping] = [234, 108, 78]
    plate[shadows] = [63, 145, 217]
    plate[~support] = [32, 40, 48]
    return {
        "kind": "exposure",
        "source": sample.public(),
        "exposure": _compact_stats(rgb, support),
        "render_visible_count": int(np.count_nonzero(support)),
        "unsupported_fraction": round(1 - _fraction(support), 6),
        "interpretation": "Clipping and shadows are marked on the selected original; this does not modify exposure.",
    }, {"histogram": chart, "clipping": png_bytes(Image.fromarray(plate))}


def profile_line(sample, start, end):
    begin = Position.parse(start, "profile start")
    finish = Position.parse(end, "profile end")
    height, width = sample.pixels.shape
    x0, y0 = begin.pixel(width, height)
    x1, y1 = finish.pixel(width, height)
    length = math.hypot(x1 - x0, y1 - y0)
    if length < 12:
        raise Unprocessable("Draw a longer profile through the source image.")
    steps = min(PROFILE_POINTS, max(12, int(length)))
    x = np.linspace(x0, x1, steps)
    y = np.linspace(y0, y1, steps)
    positions = [y, x]
    support = ndimage.map_coordinates(sample.pixels.support.astype(float), positions, order=1) > .995
    if np.count_nonzero(support) < 10:
        raise Unprocessable("The line does not cross enough supported source pixels.")
    components = []
    for channel in range(3):
        drawn = ndimage.map_coordinates(sample.pixels.rgb[..., channel], positions, order=1)
        drawn[~support] = np.nan
        components.append(drawn)
    ref_x = np.asarray(x) * sample.pixels.original_size[0] / width
    ref_y = np.asarray(y) * sample.pixels.original_size[1] / height
    along = np.hypot(ref_x - ref_x[0], ref_y - ref_y[0])
    chart = chart_png(
        list(zip(("Red", "Green", "Blue"), [along] * 3, components)),
        x_label="Distance on the original (pixels)", y_label="Linear brightness",
    )
    values = [
        {
            "distance_px": rounding(distance, 3), "raw_x": rounding(xx, 3), "raw_y": rounding(yy, 3),
            "coverage": bool(ok), "red": rounding(r), "green": rounding(g), "blue": rounding(b),
        }
        for distance, xx, yy, ok, r, g, b in zip(along, ref_x, ref_y, support, *components, strict=True)
    ]
    picture = Image.fromarray(linear_to_srgb(sample.pixels.rgb))
    draw = ImageDraw.Draw(picture)
    draw.line((x0, y0, x1, y1), width=2, fill=(252, 219, 97))
    for (xx, yy), tag in [((x0, y0), "A"), ((x1, y1), "B")]:
        draw.ellipse((xx - 3, yy - 3, xx + 3, yy + 3), fill=(245, 196, 79))
        draw.text((xx + 5, yy + 3), tag, fill=(255, 232, 137))
    return {"kind": "line-profile", "source": sample.public(), "valid_fraction": _fraction(support),
            "samples": values}, {"profile": chart, "location": png_bytes(resize_report(picture, 740))}


def spot_values(sample, point, radius):
    x, y = Position.parse(point, "probe point").pixel(*sample.pixels.shape[::-1])
    h, w = sample.pixels.shape
    margin = max(2, int(radius))
    around_x = np.arange(max(0, round(x) - margin), min(w, round(x) + margin + 1))
    around_y = np.arange(max(0, round(y) - margin), min(h, round(y) + margin + 1))
    domain = (around_x[None, :] - x) ** 2 + (around_y[:, None] - y) ** 2 <= margin ** 2
    domain &= sample.pixels.support[around_y[:, None], around_x]
    if np.count_nonzero(domain) < 3:
        raise Unprocessable("Choose a spot with more supported image pixels.")
    rgb = sample.pixels.rgb[around_y[:, None], around_x]
    summary = _compact_stats(rgb, domain)
    scale_x = sample.pixels.original_size[0] / w
    scale_y = sample.pixels.original_size[1] / h
    picture = Image.fromarray(linear_to_srgb(sample.pixels.rgb))
    draw = ImageDraw.Draw(picture)
    draw.ellipse((x - margin, y - margin, x + margin, y + margin), outline=(247, 207, 93), width=2)
    return {
        "kind": "original-spot", "source": sample.public(), "center_source_pixels": [round(x * scale_x, 3), round(y * scale_y, 3)],
        "radius_source_pixels_approx": [round(margin * scale_x, 3), round(margin * scale_y, 3)],
        "included_samples": int(domain.sum()), "stats": summary,
    }, {"spot": png_bytes(resize_report(picture, 740))}


def sharpness_report(sample):
    tile, mask = sample.base()
    lum = linear_luminance(tile)
    safe = valid_neighborhood(mask, 3)
    if safe.sum() < 54:
        raise Unprocessable("The selected region is too small or interrupted for a sharpness reading.")
    dx = ndimage.sobel(lum, axis=1, mode="nearest") / 8
    dy = ndimage.sobel(lum, axis=0, mode="nearest") / 8
    gradient = np.hypot(dx, dy)
    lo, hi = paired_bounds(gradient, safe, 10, 95)
    high = np.maximum(hi, lo + .005)
    bands = {
        "horizontal": code_statistics(np.abs(dx[safe])),
        "vertical": code_statistics(np.abs(dy[safe])),
        "combined": code_statistics(gradient[safe]),
    }
    report = {
        "kind": "sharpness",
        "source": sample.public(),
        "gradient": bands,
        "supported_fraction": _fraction(safe),
        "range": [round(lo, 6), round(high, 6)],
        "interpretation": "A larger report gradient marks stronger local contrast; it does not measure print resolution.",
    }
    return report, {"edge-energy": png_bytes(resize_report(heatmap(gradient, safe, lo, high)))}


def pattern_report(sample, direction):
    if direction not in ("across", "down"):
        raise Unprocessable("Choose an across or down image direction.")
    rgb, mask = sample.base()
    lum = linear_luminance(rgb)
    axis = 0 if direction == "across" else 1
    line_count = mask.sum(axis=axis)
    usable = line_count > np.max(line_count) * .65
    if usable.sum() < 28:
        raise Unprocessable("Extend the selected region across more supported rows or columns.")
    values = (lum * mask).sum(axis=axis) / np.maximum(1, line_count)
    domain = np.arange(len(values))
    contiguous = np.interp(domain, domain[usable], values[usable])
    detrended = signal.detrend(contiguous) * np.hanning(len(values))
    power = np.abs(np.fft.rfft(detrended)) ** 2
    frequencies = np.fft.rfftfreq(len(values))
    selection = np.arange(2, len(power))
    peaks, props = signal.find_peaks(power[selection], prominence=max(1e-12, np.max(power) * .025))
    measured_size = sample.pixels.shape[1 if axis == 0 else 0]
    real_size = sample.pixels.original_size[0 if axis == 0 else 1]
    scaled = frequencies * measured_size / real_size
    peak_rows = []
    for peak in sorted(peaks, key=lambda idx: power[selection[idx]], reverse=True)[:6]:
        actual = selection[peak]
        peak_rows.append({
            "cycles_per_source_px": round(float(scaled[actual]), 8),
            "period_source_px": round(float(1 / scaled[actual]), 5),
            "peak_power": round(float(power[actual]), 7),
        })
    log = 10 * np.log10(np.maximum(1e-10, power))
    chart = chart_png([("Texture energy", scaled[selection], log[selection])],
                      x_label="Cycles / source pixel", y_label="Relative power (dB)")
    return {
        "kind": "texture-spectrum",
        "source": sample.public(),
        "direction": direction,
        "input_samples": int(len(values)),
        "supported_fraction": _fraction(usable),
        "periods": peak_rows,
        "interpretation": "Peaks can be scene detail or print structure. Check the image and an unprinted reference separately.",
    }, {"spectrum": chart}


def compare_views(library, records, plan, first, other, divider):
    if first == other:
        raise Unprocessable("Choose two different enabled views to compare.")
    prepared = prepare_registered(library, records, plan.framing)
    width, height = working_preview_sizes(plan.sheet, COMPARE_LIMIT)
    selected = {first, other}
    matching = {
        item.record["id"]: item
        for item in fit_views(prepared, width, height, plan.sheet, plan.framing.cover)
        if item.record["id"] in selected
    }
    if set(matching) != selected:
        raise Unprocessable("Enable both originals before comparing them.")
    left, right = matching[first], matching[other]
    joint = left.support & right.support
    if np.count_nonzero(joint) < 16:
        raise Unprocessable("The two views have too little corresponding visible area.")
    a, b = linear_luminance(left.linear), linear_luminance(right.linear)
    signed = np.where(joint, b - a, 0)
    low, high = paired_bounds(np.abs(signed), joint, low=1, high=98)
    bound = max(.015, high)
    pair, column = feather_comparison(left.linear, right.linear, divider)
    split = Image.fromarray(linear_to_srgb(pair))
    ImageDraw.Draw(split).line((column, 0, column, height - 1), fill=(250, 213, 114), width=2)
    delta = heatmap(signed, joint, -bound, bound, diverging=True)
    return {
        "kind": "registered-comparison", "left": left.public(), "right": right.public(),
        "working_image_size": [width, height], "framing": prepared.public(),
        "paired_support": _fraction(joint), "signed_delta": code_statistics(signed[joint]),
        "changed_over_008": _fraction(np.abs(signed[joint]) >= .08),
        "warning": "Small common support." if joint.mean() < SUPPORT_WARNING else None,
        "interpretation": "Signed difference shows change in supported regions. Differences can be intentional between angles.",
    }, {
        "divider": png_bytes(split), "signed": png_bytes(delta),
        "paired": png_bytes(resize_report(side_by_side(
            Image.fromarray(linear_to_srgb(left.linear)),
            Image.fromarray(linear_to_srgb(right.linear))), 1180)),
    }
