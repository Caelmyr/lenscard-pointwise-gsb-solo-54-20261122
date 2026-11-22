"""One print-coordinate definition shared by line bands and observed views."""
from dataclasses import dataclass
import math

import numpy as np

from .models import Lens, Sampling, Unprocessable


@dataclass(frozen=True)
class LensGrid:
    """The x axis is the observer-facing card edge. y increases down the card."""
    lens: Lens
    dpi: int
    width: int
    height: int

    @property
    def step(self):
        return 25.4 / self.dpi

    @property
    def period_pixels(self):
        return self.lens.dot_period(self.dpi)

    def phase_at(self, x, y):
        x_mm = (np.asarray(x, dtype=np.float64) + .5) * self.step
        y_mm = (np.asarray(y, dtype=np.float64) + .5) * self.step
        return (x_mm - self.lens.shear * y_mm) / self.lens.mm_per_lens + self.lens.phase

    def actual_side(self, scan_x, mount):
        return self.width - 1 - scan_x if mount == "reverse" else scan_x

    def phase_matrix(self, start=0, stop=None, side="face"):
        finish = self.height if stop is None else stop
        rows = np.arange(start, finish, dtype=np.float64)
        columns = self.actual_side(np.arange(self.width, dtype=np.float64), side)
        return self.phase_at(columns[None, :], rows[:, None])

    def screen_angles(self, count=17):
        return np.linspace(-self.lens.latitude, self.lens.latitude, count)

    def public(self):
        return {
            "pixels": [self.width, self.height],
            "dot_mm": round(self.step, 8),
            "period_pixels": round(self.period_pixels, 7),
            "lens_pitch_mm": round(self.lens.mm_per_lens, 7),
            "viewing_cone_degrees": [-self.lens.latitude, self.lens.latitude],
            "face_zero_rule": "origin is the upper-left observer-facing card edge",
            "lens": self.lens.public(),
        }


def physical_slot(view_index, view_count, order):
    return view_count - 1 - view_index if order == "reverse" else view_index


def view_index(slot, view_count, order):
    return view_count - 1 - slot if order == "reverse" else slot


def band_index(phase, count):
    index = np.floor(np.mod(phase, 1) * count).astype(np.int16)
    # Floating-point modulo can return 1.0 at a period boundary; fold it in.
    return np.minimum(count - 1, np.maximum(0, index))


def overlap_length(left, right, interval_start, interval_stop):
    return np.clip(np.minimum(right, interval_stop) - np.maximum(left, interval_start), 0, None)


def fraction_for_slot(center, half_width, slot, count):
    left = np.mod(center, 1) - half_width
    right = left + 2 * half_width
    start, end = slot / count, (slot + 1) / count
    total = sum(overlap_length(left, right, start + offset, end + offset) for offset in (-1, 0, 1))
    return total / (2 * half_width)


def exact_slot_coverage(grid, start, stop, slot, count, *, side="face"):
    phase = grid.phase_matrix(start, stop, side)
    extent = 0.5 * (1.0 + abs(grid.lens.shear)) / grid.period_pixels
    return fraction_for_slot(phase, extent, slot, count).astype(np.float32)


def solid_slot_coverage(grid, start, stop, slot, count, *, side="face"):
    phase = grid.phase_matrix(start, stop, side)
    return (band_index(phase, count) == slot).astype(np.float32)


def phase_map_image(grid, start, stop, view_count, side="face"):
    return view_index(band_index(grid.phase_matrix(start, stop, side), view_count), view_count, grid.lens.order)


def observation_phase(grid, angle):
    lim = grid.lens.latitude
    if not -lim <= angle <= lim:
        raise Unprocessable("This angle is outside the measured viewing cone.")
    position = 0.5 + 0.5 * math.tan(math.radians(angle)) / math.tan(math.radians(lim))
    return min(1, max(0, position))


def phase_to_angle(grid, fraction):
    tan_value = (2 * fraction - 1) * math.tan(math.radians(grid.lens.latitude))
    return math.degrees(math.atan(tan_value))


def _aperture_window(grid, phase, count, sampling):
    width = (1 - sampling.aperture) / count + 1 / grid.period_pixels
    half_width = width / 2
    left = max(0, phase - half_width)
    right = min(1, phase + half_width)
    return left, right


def angular_response(grid, angle, count, sampling: Sampling):
    phase = observation_phase(grid, angle)
    left, right = _aperture_window(grid, phase, count, sampling)
    response = np.asarray([
        float(overlap_length(left, right, slot / count, (slot + 1) / count) / (right - left))
        for slot in range(count)
    ])
    if grid.lens.order == "reverse":
        response = response[::-1]
    return response, [left, right]


def response_report(grid, angles, records, sampling):
    rows = []
    count = len(records)
    for angle in angles:
        weights, window = angular_response(grid, float(angle), count, sampling)
        strongest = int(weights.argmax())
        rows.append({
            "angle_deg": round(float(angle), 5),
            "phase_window": [round(x, 6) for x in window],
            "principal_view": records[strongest]["id"],
            "principal_title": records[strongest]["title"],
            "mixed_fraction": round(float(1 - weights[strongest]), 6),
            "contributors": [
                {"view": record["id"], "weight": round(float(weight), 6)}
                for record, weight in zip(records, weights, strict=True)
            ],
        })
    return rows


def angular_domains(grid, frames):
    records = []
    for index, item in enumerate(frames):
        slot = physical_slot(index, len(frames), grid.lens.order)
        left = phase_to_angle(grid, slot / len(frames))
        right = phase_to_angle(grid, (slot + 1) / len(frames))
        center = phase_to_angle(grid, (slot + 0.5) / len(frames))
        records.append({
            "id": item["id"],
            "source_angle": item["position"],
            "printed_center_angle": round(center, 5),
            "nominal_angle_window": [round(left, 5), round(right, 5)],
            "label": item["title"],
        })
    return records


def print_quanta(grid, view_count, sampling, side):
    by_view = np.zeros(view_count, dtype=np.float64)
    sample = solid_slot_coverage if sampling.method == "solid" else exact_slot_coverage
    end = min(grid.height, 160)
    for index in range(view_count):
        slot = physical_slot(index, view_count, grid.lens.order)
        by_view[index] = sample(grid, 0, end, slot, view_count, side=side).mean()
    return {
        "device_dots_per_lens": round(grid.period_pixels, 6),
        "nominal_dots_per_view": round(grid.period_pixels / view_count, 6),
        "mean_view_coverage_near_top": [round(float(value), 6) for value in by_view],
        "sample_rows": end,
        "quality_note": "Narrow bands should be checked on the actual printer and lens stack.",
    }


def lens_measurements(grid, item_positions, desired_angle):
    if len(item_positions) < 2:
        raise Unprocessable("Enable two views before inspecting viewing angles.")
    max_angle = grid.lens.latitude
    selected = max(-max_angle, min(max_angle, desired_angle))
    centers = []
    for index, nominal in enumerate(item_positions):
        slot = physical_slot(index, len(item_positions), grid.lens.order)
        center = phase_to_angle(grid, (slot + .5) / len(item_positions))
        centers.append({
            "nominal_source_angle": nominal,
            "lens_center_angle": round(center, 4),
            "offset_degrees": round(center - nominal, 4),
            "includes_selected": abs(selected - center) <= max_angle / len(item_positions),
        })
    return centers
