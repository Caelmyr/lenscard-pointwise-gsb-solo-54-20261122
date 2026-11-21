"""Fit a global grid from registered ruler marks and prepare a print target."""
from dataclasses import dataclass
import math

import numpy as np
from PIL import Image, ImageDraw

from .models import Lens, Sheet, Unprocessable, integer, number, object_value, validate_keys
from .optics import LensGrid
from .printing import one_raster_pdf
from .visual import chart_png, png_bytes


MIN_MARKS = 5
MAX_MARKS = 30
MAX_PROOF_STEPS = 9
MARK_SIZE_MM = 1.7


@dataclass(frozen=True)
class Tick:
    x_mm: float
    y_mm: float
    stripe: int

    @classmethod
    def parse(cls, body, sheet):
        obj = object_value(body, "lens mark")
        validate_keys(obj, {"x_mm", "y_mm", "stripe"}, "lens mark")
        return cls(
            number(obj.get("x_mm"), "mark x", 0, sheet.width_mm),
            number(obj.get("y_mm"), "mark y", 0, sheet.height_mm),
            integer(obj.get("stripe"), "index", -3000, 3000),
        )

    def row(self):
        return (self.x_mm, self.y_mm, self.stripe)


@dataclass(frozen=True)
class GridFit:
    lens: Lens
    pitch_uncertainty_mm: float
    count: int
    stripe_range: tuple[int, int]
    rms_mm: float
    largest_residual_mm: float
    residual_rows: list[dict]
    predictive_limit: float

    @property
    def accepted(self):
        period = self.lens.mm_per_lens
        return self.rms_mm <= period * .09 and self.largest_residual_mm <= period * .2

    def public(self):
        period = self.lens.mm_per_lens
        return {
            "model": "x_mm = intercept + period_mm * stripe + drift_mm_per_mm * y_mm",
            "fit_status": "supported" if self.accepted else "review",
            "lens": self.lens.public(),
            "period_mm": round(period, 8),
            "rms_mm": round(self.rms_mm, 8),
            "max_residual_mm": round(self.largest_residual_mm, 8),
            "uncertainty_mm": round(self.pitch_uncertainty_mm, 8),
            "observed_points": self.count,
            "measured_stripes": list(self.stripe_range),
            "threshold_rms_mm": round(period * .09, 8),
            "threshold_max_mm": round(period * .2, 8),
            "prediction_radius_mm": round(self.predictive_limit, 6),
            "residuals": self.residual_rows,
            "warning": "Review ruler marks or lens seating." if not self.accepted else None,
        }


def index_marks(values, sheet):
    if not isinstance(values, list) or not MIN_MARKS <= len(values) <= MAX_MARKS:
        raise Unprocessable(f"Collect {MIN_MARKS} to {MAX_MARKS} ruler or alignment marks.")
    ticks = [Tick.parse(row, sheet) for row in values]
    if len(set(tick.row() for tick in ticks)) != len(ticks):
        raise Unprocessable("Remove duplicate ruler mark entries.")
    x_coords, y_coords, stripe = np.asarray([mark.row() for mark in ticks], dtype=float).T
    if np.ptp(y_coords) < max(3.0, sheet.height_mm * 0.12):
        raise Unprocessable("Collect ruler marks at two separated heights.")
    if np.ptp(stripe) < 5:
        raise Unprocessable("Mark across at least five numbered lens steps.")
    return x_coords, y_coords, stripe


def _predict_uncertainty(design, residual, covariance):
    dof = max(1, design.shape[0] - design.shape[1])
    residual_var = float((residual * residual).sum()) / dof
    p = float(math.sqrt(max(0.0, covariance[1, 1] * residual_var)))
    error_at_mark = np.sqrt(np.maximum(0, np.einsum("ij,jk,ik->i", design, covariance, design) * residual_var))
    return p, float(error_at_mark.max())


def best_global_lens(sheet, base_lens, tick_records):
    x_mm, y_mm, stripes = index_marks(tick_records, sheet)
    mean_stripe = float(stripes.mean())
    mean_y = float(y_mm.mean())
    design = np.stack([np.ones_like(x_mm), stripes - mean_stripe, y_mm - mean_y], axis=1)
    coeff, _, rank, _ = np.linalg.lstsq(design, x_mm, rcond=None)
    if rank < 3:
        raise Unprocessable("Add more marks at different rows and different stripe positions.")
    offset, pitch, drift = coeff
    if pitch <= 0:
        raise Unprocessable("Stripe index must increase from the observer's left to right.")
    intercept = offset - pitch * mean_stripe - drift * mean_y
    phase = float((-intercept / pitch + 0.5) % 1.0 - 0.5)
    lens = Lens.parse({
        "lpi": 25.4 / pitch,
        "shear": float(drift),
        "phase": phase,
        "order": base_lens.order,
        "latitude": base_lens.latitude,
    })
    predicted = design @ coeff
    errors = x_mm - predicted
    covariance = np.linalg.pinv(design.T @ design)
    uncertainty, predictive = _predict_uncertainty(design, errors, covariance)
    return GridFit(
        lens, uncertainty, len(tick_records), (int(stripes.min()), int(stripes.max())),
        float(np.sqrt(np.mean(errors ** 2))), float(np.max(np.abs(errors))),
        [
            {"x_mm": float(x), "y_mm": float(y), "stripe": int(index),
             "expected_x_mm": round(float(fit), 8), "residual_mm": round(float(error), 8)}
            for x, y, index, fit, error in zip(x_mm, y_mm, stripes, predicted, errors, strict=True)
        ],
        predictive,
    )


def fit_chart(fitted):
    x = np.asarray([row["stripe"] for row in fitted.residual_rows])
    y = np.asarray([row["residual_mm"] for row in fitted.residual_rows])
    return chart_png(
        [("Measured - fitted", x, y), ("Zero", np.asarray([x.min(), x.max()]), np.zeros(2))],
        x_label="Recorded stripe number",
        y_label="Lateral residual (mm)",
    )


def calibration_fingerprint(card, fit, ticks):
    from .storage import digest_object
    return digest_object({"card": card["id"], "version": card["version"], "fit": fit.public(), "ticks": ticks})


@dataclass(frozen=True)
class TestGrid:
    sheet: Sheet
    lens: Lens
    increments: int
    delta_lpi: float

    @classmethod
    def parse(cls, active_sheet, active_lens, request):
        obj = object_value(request, "target")
        validate_keys(obj, {"increments", "delta_lpi"}, "target")
        increments = integer(obj.get("increments", 7), "patch count", 3, MAX_PROOF_STEPS)
        delta_lpi = number(obj.get("delta_lpi", .2), "pitch interval", .02, 4.0)
        pitches = active_lens.lpi + (np.arange(increments) - (increments - 1) * .5) * delta_lpi
        if pitches.min() < 10 or pitches.max() > 180:
            raise Unprocessable("The calibration target extends outside supported material pitch.")
        return cls(active_sheet, active_lens, increments, delta_lpi)

    def pitches(self):
        return self.lens.lpi + (np.arange(self.increments) - (self.increments - 1) * .5) * self.delta_lpi


def _tick_ruler(draw, grid, y, *, from_mm=6, to_mm=None):
    width_mm = grid.width * grid.step
    end = min(round(width_mm) - 7, int(to_mm or 0)) if to_mm else round(width_mm) - 7
    x0, x1 = round(from_mm / grid.step), round(end / grid.step)
    draw.line((x0, y, x1, y), fill=(40, 48, 47), width=2)
    for mark in range(int(from_mm), end + 1):
        x = round(mark / grid.step)
        extent = 3.2 if mark % 10 == 0 else 1.7 if mark % 5 == 0 else 0.75
        draw.line((x, y, x, int(y + extent / grid.step)), width=1, fill=(40, 48, 47))
        if mark % 10 == 0:
            draw.text((x - 4, y + round(4 / grid.step)), str(mark), fill=(32, 45, 45))


def render_target(spec):
    sheet, lens = spec.sheet, spec.lens
    width, height = sheet.size
    grid = LensGrid(lens, sheet.dpi, width, height)
    pad = max(20, round(6 / grid.step))
    header = max(26, round(11 / grid.step))
    footer = max(24, round(10 / grid.step))
    label_margin = max(14, round(2.9 / grid.step))
    patch_height = (height - header - footer) // spec.increments
    if patch_height < label_margin * 2 or width < 6 * pad:
        raise Unprocessable("Choose a larger card before producing the pitch target.")
    paper = Image.new("RGB", sheet.size, (251, 250, 247))
    draw = ImageDraw.Draw(paper)
    draw.text((pad, pad // 2), "LENS PITCH TARGET / print at actual size", fill=(21, 35, 32))
    patches = []
    columns = np.arange(pad, width - pad, dtype=np.float64)[None, :]
    for index, lpi in enumerate(spec.pitches()):
        profile = Lens(float(lpi), lens.phase, lens.shear, lens.order, lens.latitude)
        local = LensGrid(profile, grid.dpi, grid.width, grid.height)
        top = header + patch_height * index
        bottom = top + patch_height - label_margin
        rows = np.arange(top, bottom, dtype=np.float64)[:, None]
        phase = np.mod(local.phase_at(columns, rows), 1)
        patch = ((phase < .5) * 202 + 24).astype(np.uint8)
        paper.paste(Image.fromarray(patch).convert("RGB"), (pad, top))
        draw.text((pad, bottom + 1), f"{index + 1}  /  {lpi:.3f} LPI", fill=(51, 67, 65))
        patches.append({
            "stripe_pitch_lpi": round(float(lpi), 6),
            "box_px": [pad, top, width - pad, bottom],
            "y_mm": [round(top * grid.step, 4), round(bottom * grid.step, 4)],
        })
    _tick_ruler(draw, grid, height - footer + 4)
    report = {
        "requested_mm": [sheet.width_mm, sheet.height_mm],
        "output_pixels": list(sheet.size), "dpi": sheet.dpi,
        "patches": patches,
        "usage": "Align a real loose lens above this chart. Identify the most stable uniform band at actual print size, then fit marks.",
    }
    return report, paper


def target_assets(spec):
    report, paper = render_target(spec)
    result = {
        "target": (png_bytes(paper), "image/png", "pitch-target.png"),
        "pdf": (one_raster_pdf(np.asarray(paper), *report["requested_mm"]), "application/pdf", "pitch-target-size.pdf"),
    }
    return report, result


def teaching_ticks(sheet, lens):
    grid = LensGrid(lens, sheet.dpi, *sheet.size)
    y_values = [sheet.height_mm * .19, sheet.height_mm * .74]
    step = int(max(3, grid.width / grid.period_pixels * .19))
    result = []
    for y in y_values:
        for number in (step, step * 2, step * 3):
            x = (number - lens.phase) * lens.mm_per_lens + lens.shear * y
            result.append({"x_mm": round(x, 6), "y_mm": round(y, 6), "stripe": number})
    return result
