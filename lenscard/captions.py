"""Two flip captions sharing one blank paper area, interleaved into lens bands.

Each message is written into a different physical lens band, so turning the card
one way reveals one message and turning it the other way reveals the other. Both
texts always share the same fixed region; this module never relocates them. It
reports the angle span each message actually reads over, the span where both stay
quiet, and which side could retreat by one band if the bands sit too close.

Nothing here touches an original capture: masks are composited into the derived
print raster and into observation previews only.
"""
from dataclasses import dataclass
from functools import lru_cache
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

from .imaging import linear_luminance
from .models import Captions, Region, Unprocessable
from .optics import LensGrid, band_index, observation_phase, phase_to_angle, physical_slot_weights


INK_LINEAR = 0.045
GHOST_WEIGHT = 0.06
BUSY_GRADIENT = 0.012
AUTO_MAX_FONT = 240
MIN_FONT_PX = 5
LINE_SPACING = 1.16
READABLE_SHARE = 0.5
SCAN_POINTS = 1201
MIN_CHARACTER_LENSES = 3.0
MIN_STROKE_LENSES = 1.0


@dataclass(frozen=True)
class TextMask:
    lines: list[str]
    mask: np.ndarray
    fits: bool


@dataclass
class CaptionLayout:
    box: tuple[int, int, int, int]
    left: TextMask
    right: TextMask
    size_mm: float
    auto_sized: bool
    ink_fraction: float
    pixels_per_mm: float = 1.0
    left_band: int = 0
    right_band: int = 1

    @property
    def renderable(self):
        return self.left.fits and self.right.fits

    def public(self):
        left, top, right, bottom = self.box
        return {
            "box": [left, top, right, bottom],
            "size_mm": round(self.size_mm, 4),
            "auto_sized": self.auto_sized,
            "left_lines": self.left.lines,
            "right_lines": self.right.lines,
            "left_fits": self.left.fits,
            "right_fits": self.right.fits,
            "left_band": self.left_band,
            "right_band": self.right_band,
            "ink_fraction": round(self.ink_fraction, 6),
        }


@lru_cache(maxsize=48)
def caption_font(size):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except (OSError, ValueError):
        return ImageFont.load_default(size=size)


def _line_height(font):
    ascent, descent = font.getmetrics()
    return max(1, round((ascent + descent) * LINE_SPACING))


def _wrap(font, text, width):
    """Break on explicit newlines, then greedily on spaces, then on characters."""
    lines = []
    for paragraph in text.split("\n"):
        words = paragraph.split()
        if not words:
            lines.append("")
            continue
        current = ""
        for word in words:
            candidate = word if not current else current + " " + word
            if font.getlength(candidate) <= width:
                current = candidate
                continue
            if current:
                lines.append(current)
                current = ""
            if font.getlength(word) <= width:
                current = word
                continue
            for character in word:
                if font.getlength(current + character) <= width or not current:
                    current += character
                else:
                    lines.append(current)
                    current = character
        if current:
            lines.append(current)
    return lines or [""]


def _block(font, text, width):
    lines = _wrap(font, text, max(1, width))
    return lines, len(lines) * _line_height(font)


def _auto_size(texts, width, height):
    """Largest common font size whose wrapped blocks fit the shared area."""
    low, high, best = MIN_FONT_PX, max(MIN_FONT_PX, min(AUTO_MAX_FONT, height)), None
    while low <= high:
        size = (low + high) // 2
        font = caption_font(size)
        if all(_block(font, text, width)[1] <= height for text in texts):
            best = size
            low = size + 1
        else:
            high = size - 1
    return best


def _draw_mask(text, size, area_width, area_height, wrap_width):
    font = caption_font(size)
    lines, block_height = _block(font, text, wrap_width)
    canvas = Image.new("L", (area_width, area_height), 0)
    draw = ImageDraw.Draw(canvas)
    top = max(0, (area_height - block_height) // 2)
    step = _line_height(font)
    for index, line in enumerate(lines):
        if line:
            draw.text((area_width // 2, top + index * step), line, font=font,
                      fill=255, anchor="ma")
    mask = np.asarray(canvas, dtype=np.float32) / 255
    return TextMask(lines, mask, block_height <= area_height)


def build_layout(captions: Captions, width, height, pixels_per_mm):
    """Rasterize both captions into their shared observer-facing region."""
    if not captions.enabled:
        return None
    left, top, right, bottom = captions.region.pixels(width, height)
    area_width, area_height = right - left, bottom - top
    if area_width < 8 or area_height < 8:
        raise Unprocessable("The caption area is too small on this card.")
    pad_x, pad_y = max(1, round(area_width * .02)), max(1, round(area_height * .04))
    wrap_width, wrap_height = area_width - 2 * pad_x, area_height - 2 * pad_y
    texts = (captions.left.text, captions.right.text)
    if captions.size_mm > 0:
        size_px, auto = max(MIN_FONT_PX, round(captions.size_mm * pixels_per_mm)), False
    else:
        size_px, auto = _auto_size(texts, wrap_width, wrap_height), True
    if size_px is None:
        empty = np.zeros((area_height, area_width), dtype=np.float32)
        return CaptionLayout(
            (left, top, right, bottom), TextMask([], empty, False),
            TextMask([], empty, False), 0.0, auto, 0.0, pixels_per_mm=pixels_per_mm,
        )
    left_mask = _draw_mask(captions.left.text, size_px, area_width, area_height, wrap_width)
    right_mask = _draw_mask(captions.right.text, size_px, area_width, area_height, wrap_width)
    ink = float(np.mean(np.maximum(left_mask.mask, right_mask.mask)))
    return CaptionLayout(
        (left, top, right, bottom), left_mask, right_mask,
        size_px / pixels_per_mm, auto, ink, pixels_per_mm=pixels_per_mm,
    )


def band_for_angle(grid: LensGrid, angle: float, count: int) -> int:
    """Physical lens band an observer's window centres on at this angle."""
    phase = observation_phase(grid, angle)
    return int(min(count - 1, max(0, math.floor(phase * count))))


def resolve_bands(captions: Captions, grid: LensGrid, count: int):
    """Physical band for each message; identical targets share one band."""
    limit = grid.lens.latitude
    left = band_for_angle(grid, min(max(captions.left.angle, -limit * .999), limit * .999), count)
    right = band_for_angle(grid, min(max(captions.right.angle, -limit * .999), limit * .999), count)
    return left, right


def band_window(grid: LensGrid, band: int, count: int):
    return [round(phase_to_angle(grid, band / count), 5),
            round(phase_to_angle(grid, (band + 1) / count), 5)]


def _band_center_angle(grid, band, count):
    return phase_to_angle(grid, (band + 0.5) / count)


def _scan(grid, count, sampling):
    angles = np.linspace(-grid.lens.latitude, grid.lens.latitude, SCAN_POINTS)
    weights = np.stack([physical_slot_weights(grid, float(angle), count, sampling)
                        for angle in angles])
    return angles, weights


def _longest_run(angles, accepted):
    indices = np.nonzero(accepted)[0]
    if not indices.size:
        return None
    breaks = np.nonzero(np.diff(indices) > 1)[0]
    run = max(np.split(indices, breaks + 1), key=len)
    return [round(float(angles[run[0]]), 5), round(float(angles[run[-1]]), 5)]


def readable_window(angles, weights, band, rival):
    """Angular span where this caption dominates while the rival stays under the ghost floor."""
    own = weights[:, band] >= READABLE_SHARE
    clean = _longest_run(angles, own & (weights[:, rival] <= GHOST_WEIGHT))
    if clean is not None:
        return clean
    # The rival never falls below the floor while this caption dominates, which is
    # a genuine boundary mix; still show where this text itself carries.
    return _longest_run(angles, own)


def blank_window(angles, weights, first, second):
    quiet = np.maximum(weights[:, first], weights[:, second]) <= GHOST_WEIGHT
    return _longest_run(angles, quiet)


def simultaneous_fraction(angles, weights, first, second):
    """Highest share both captions hold at the same angle; the mix at the boundary."""
    both = np.minimum(weights[:, first], weights[:, second])
    index = int(np.argmax(both))
    return round(float(both[index]), 6), round(float(angles[index]), 5)


def visible_window(angles, weights, band):
    """Angular span where this caption carries at least the ghost floor of the light."""
    return _longest_run(angles, weights[:, band] > GHOST_WEIGHT)


def _message(grid, count, angles, weights, band, rival, requested):
    window = readable_window(angles, weights, band, rival)
    peak = float(weights[:, band].max())
    plateau = _longest_run(angles, weights[:, band] >= peak - 1e-9) if peak > 0 else None
    center = _band_center_angle(grid, band, count)
    return {
        "band": band,
        "nominal_window_deg": band_window(grid, band, count),
        "readable_window_deg": window,
        "readable_width_deg": round(window[1] - window[0], 5) if window else 0.0,
        "visible_window_deg": visible_window(angles, weights, band),
        "center_angle_deg": round(center, 5),
        "dominant_window_deg": plateau,
        "dominant_fraction": round(peak, 6),
        "requested_angle_deg": round(requested, 5),
        "target_offset_deg": round(center - requested, 5),
    }


def _retreat_options(grid, count, angles, weights, first, second):
    """One-band retreat offered per side; reported, never applied."""
    options = []
    for side, moving in (("left", first), ("right", second)):
        step = -1 if side == "left" else 1
        target = moving + step
        if not 0 <= target < count:
            options.append({
                "side": side, "feasible": False, "slide_bands": 1,
                "message": "No further lens band remains on that side inside the card.",
            })
            continue
        new_first, new_second = (target, second) if side == "left" else (first, target)
        if new_second == new_first:
            options.append({
                "side": side, "feasible": False, "slide_bands": 1,
                "message": "That band is already used by the other message.",
            })
            continue
        own_band = new_first if side == "left" else new_second
        rival = new_second if side == "left" else new_first
        window = readable_window(angles, weights, own_band, rival)
        mix, mix_angle = simultaneous_fraction(angles, weights, new_first, new_second)
        touches = new_second - new_first == 1
        options.append({
            "side": side, "feasible": True, "slide_bands": 1,
            "new_band": own_band,
            "blank_between": max(0, new_second - new_first - 1),
            "still_shares_boundary": touches,
            "readable_window_deg": window,
            "readable_width_deg": round(window[1] - window[0], 5) if window else 0.0,
            "max_simultaneous_fraction": mix,
            "max_simultaneous_angle_deg": mix_angle,
            "suggested_target_angle_deg": round(_band_center_angle(grid, own_band, count), 5),
            "angle_shift_deg": round(_band_center_angle(grid, own_band, count)
                                     - _band_center_angle(grid, moving, count), 5),
            "message": (
                "Move only this message one lens band outward; the bands would then touch."
                if touches else
                "Move only this message one lens band outward to free a blank band."
            ),
        })
    return options


STATUS_TEXT = {
    "off": "Flip captions are off.",
    "separated": "Each message reads in its own lens band and the rival stays below the ghost floor.",
    "shared_boundary": "At some angle both messages show at once. Retreat one side by a band; nothing moved automatically.",
    "same_band": "Both target angles resolve to the same lens band. Move one message to another band.",
    "unreadable": "A message never becomes the dominant band inside this viewing cone.",
    "overflow": "The text does not fit the shared blank area at this type size.",
    "outside_cone": "A target angle lies outside the viewing cone and cannot be read.",
}


def analyse(grid, captions: Captions, count, sampling, layout: CaptionLayout | None):
    """Angular reach of both flip messages, boundary mixing and retreat choices."""
    base = {
        "enabled": captions.enabled,
        "region": captions.region.public(),
        "requested": {
            "left": {"text": captions.left.text, "angle_deg": round(captions.left.angle, 5)},
            "right": {"text": captions.right.text, "angle_deg": round(captions.right.angle, 5)},
        },
        "viewing_cone_degrees": [-grid.lens.latitude, grid.lens.latitude],
        "ghost_threshold": GHOST_WEIGHT,
        "readable_share": READABLE_SHARE,
        "active_views": count,
        "layout": layout.public() if layout else None,
    }
    if not captions.enabled:
        return {**base, "status": "off", "rendered": False, "message": STATUS_TEXT["off"]}

    limit = grid.lens.latitude
    outside = [side for side, angle in (("left", captions.left.angle), ("right", captions.right.angle))
               if not -limit <= angle <= limit]
    first, second = resolve_bands(captions, grid, count)
    angles, weights = _scan(grid, count, sampling)
    mix, mix_angle = simultaneous_fraction(angles, weights, first, second)
    layout_ok = layout is not None and layout.renderable
    left = _message(grid, count, angles, weights, first, second, captions.left.angle)
    right = _message(grid, count, angles, weights, second, first, captions.right.angle)

    if outside:
        status = "outside_cone"
    elif not layout_ok:
        status = "overflow"
    elif second == first:
        status = "same_band"
    elif left["readable_window_deg"] is None or right["readable_window_deg"] is None:
        status = "unreadable"
    elif mix > GHOST_WEIGHT:
        status = "shared_boundary"
    else:
        status = "separated"

    return {
        **base,
        "status": status,
        "rendered": status in ("separated", "shared_boundary"),
        "bands": {"left": first, "right": second,
                  "separation": abs(second - first), "blank_between": max(0, abs(second - first) - 1)},
        "left": left,
        "right": right,
        "blank_angular_span_deg": blank_window(angles, weights, first, second),
        "max_simultaneous_fraction": mix,
        "max_simultaneous_angle_deg": mix_angle,
        "retreat_needed": status in ("shared_boundary", "same_band", "unreadable"),
        "retreat_options": _retreat_options(grid, count, angles, weights, first, second)
        if status in ("shared_boundary", "same_band", "unreadable") else [],
        "message": STATUS_TEXT[status],
        "note": (
            "Readable widths come from the lens-band model and a finite observation window. "
            "The shared area is fixed; a retreat is reported for you to choose, never applied automatically."
        ),
    }


def region_mm(region: Region, sheet) -> dict:
    left, top, right, bottom = region.public()
    return {
        "left_mm": round(left * sheet.width_mm, 4),
        "top_mm": round(top * sheet.height_mm, 4),
        "width_mm": round((right - left) * sheet.width_mm, 4),
        "height_mm": round((bottom - top) * sheet.height_mm, 4),
    }


def region_content(registered, plan, region: Region, max_edge=420):
    """Approximate how much picture detail already sits under the caption area."""
    from .framing import fit_views, working_preview_sizes

    width, height = working_preview_sizes(plan.sheet, max_edge)
    left, top, right, bottom = region.pixels(width, height)
    if right - left < 4 or bottom - top < 4:
        return {"busy_fraction_max": None, "overlaps_picture": False, "warning": None, "views": []}
    rows = []
    for image in fit_views(registered, width, height, plan.sheet, plan.framing.cover):
        luma = linear_luminance(image.linear[top:bottom, left:right])
        support = image.support[top:bottom, left:right]
        if luma.size < 16 or not support.any():
            rows.append({"title": image.record["title"], "busy_fraction": None,
                         "supported_fraction": 0.0, "luma_span": None})
            continue
        gradient = np.hypot(ndimage.sobel(luma, axis=1, mode="nearest") / 8,
                            ndimage.sobel(luma, axis=0, mode="nearest") / 8)
        values = luma[support]
        rows.append({
            "title": image.record["title"],
            "busy_fraction": round(float(np.mean((gradient > BUSY_GRADIENT) & support)), 6),
            "supported_fraction": round(float(support.mean()), 6),
            "luma_span": round(float(values.max() - values.min()), 6),
        })
    busy = [row["busy_fraction"] for row in rows if row["busy_fraction"] is not None]
    worst = max(busy) if busy else None
    overlaps = bool(worst is not None and worst > .18)
    return {
        "busy_fraction_max": worst,
        "method": "local gradient at preview resolution",
        "overlaps_picture": overlaps,
        "warning": "The caption area contains picture detail in at least one angle." if overlaps else None,
        "views": rows,
    }


def lens_sampling(layout: CaptionLayout, grid: LensGrid):
    """What survives one horizontal sample per lens, per message."""
    if layout is None or not layout.renderable:
        return {"sufficient": False, "reason": "No caption layout."}
    px_per_lens = grid.lens.mm_per_lens * layout.pixels_per_mm
    messages = {}
    for name, text in (("left", layout.left), ("right", layout.right)):
        advances = _character_advances(layout, text, px_per_lens)
        strokes = _stroke_runs(text.mask > .5, px_per_lens)
        messages[name] = {
            "characters": len(advances),
            "median_character_lenses": round(float(np.median(advances)), 3) if advances else 0.0,
            "median_stroke_lenses": round(float(np.median(strokes)), 3) if strokes else 0.0,
            "narrowest_stroke_lenses": round(float(np.min(strokes)), 3) if strokes else 0.0,
        }
    advance = min(row["median_character_lenses"] for row in messages.values())
    stroke = min(row["median_stroke_lenses"] for row in messages.values())
    return {
        "sufficient": advance >= MIN_CHARACTER_LENSES and stroke >= MIN_STROKE_LENSES,
        "lens_pitch_mm": round(grid.lens.mm_per_lens, 6),
        "samples_per_lens": 1,
        "required_character_lenses": MIN_CHARACTER_LENSES,
        "required_stroke_lenses": MIN_STROKE_LENSES,
        "median_character_lenses": round(advance, 3),
        "median_stroke_lenses": round(stroke, 3),
        "messages": messages,
        "note": (
            "The sliced card shows one horizontal sample per lens. Characters narrower than "
            "about three lenses or strokes thinner than one lens will break up; enlarge the "
            "type or use a coarser pitch. This is advisory, not a placement change."
        ),
    }


def _character_advances(layout: CaptionLayout, text: TextMask, px_per_lens):
    """Horizontal advance of each glyph, converted to lens pitches."""
    font = caption_font(max(MIN_FONT_PX, round(layout.size_mm * layout.pixels_per_mm)))
    return [font.getlength(character) / px_per_lens for character in "".join(text.lines)
            if not character.isspace()]


def _stroke_runs(mask: np.ndarray, px_per_lens: float):
    """Horizontal ink runs across all rows, converted to lens pitches."""
    runs = []
    for row in mask:
        index = 0
        while index < len(row):
            if not row[index]:
                index += 1
                continue
            start = index
            while index + 1 < len(row) and row[index + 1]:
                index += 1
            runs.append((index - start + 1) / px_per_lens)
            index += 1
    return runs


def _region_phase(grid: LensGrid, layout: CaptionLayout):
    left, top, right, bottom = layout.box
    cols = np.arange(left, right, dtype=np.float64)
    rows = np.arange(top, bottom, dtype=np.float64)
    return grid.phase_at(cols[None, :], rows[:, None])


def print_alpha(grid: LensGrid, layout: CaptionLayout, count):
    """Exact per-pixel ink coverage for the observer-facing print plate."""
    phase = _region_phase(grid, layout)
    bands = band_index(phase, count)
    alpha = np.maximum(layout.left.mask * (bands == layout.left_band),
                       layout.right.mask * (bands == layout.right_band))
    return np.clip(alpha, 0, 1)


def observed_alpha(grid: LensGrid, layout: CaptionLayout, angle, count, sampling):
    """Approximate what a viewer sees at one angle: one band sample per lens."""
    left, top, right, bottom = layout.box
    phase = _region_phase(grid, layout)
    weights = physical_slot_weights(grid, angle, count, sampling)
    cell = np.floor(phase)
    row_mm = (np.arange(top, bottom)[:, None] + .5) * grid.step
    rows = np.arange(bottom - top)[:, None]
    alpha = np.zeros(phase.shape, dtype=np.float32)
    for band, mask in ((layout.left_band, layout.left.mask),
                       (layout.right_band, layout.right.mask)):
        weight = float(weights[band])
        if weight <= 1e-6:
            continue
        target = cell + (band + .5) / count
        x_mm = (target - grid.lens.phase) * grid.lens.mm_per_lens + grid.lens.shear * row_mm
        cols = np.clip(np.rint(x_mm / grid.step - .5).astype(np.int64) - left,
                       0, mask.shape[1] - 1)
        alpha = np.maximum(alpha, mask[rows, cols] * weight)
    return np.clip(alpha, 0, 1)


def visible_messages(angle, grid: LensGrid, count, sampling, layout: CaptionLayout):
    weights = physical_slot_weights(grid, angle, count, sampling)
    return {
        "angle_deg": round(float(angle), 5),
        "left_fraction": round(float(weights[layout.left_band]), 6),
        "right_fraction": round(float(weights[layout.right_band]), 6),
        "left_visible": bool(weights[layout.left_band] > GHOST_WEIGHT),
        "right_visible": bool(weights[layout.right_band] > GHOST_WEIGHT),
        "both_visible": bool(weights[layout.left_band] > GHOST_WEIGHT
                             and weights[layout.right_band] > GHOST_WEIGHT),
    }


def composite(plate, layout: CaptionLayout, alpha, *, mirror=False):
    """Composite ink into a derived raster without touching any source image."""
    left, top, right, bottom = layout.box
    width = plate.shape[1]
    if mirror:
        alpha = alpha[:, ::-1]
        left, right = width - right, width - left
    region = plate[top:bottom, left:right]
    plate[top:bottom, left:right] = region * (1 - alpha[..., None]) + INK_LINEAR * alpha[..., None]
