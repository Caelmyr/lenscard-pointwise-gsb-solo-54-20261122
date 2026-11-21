"""One print-coordinate definition shared by line bands and observed views."""
from dataclasses import dataclass
import math

import numpy as np

from .models import Lens, Sampling, SplitLens, Unprocessable


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
    def width_mm(self):
        return self.width * self.step

    @property
    def period_pixels(self):
        return self.lens.dot_period(self.dpi)

    @property
    def split_mm(self):
        return None

    @property
    def seam_mm(self):
        return None

    @property
    def guard_mm(self):
        return 0.0

    @property
    def split_layout(self):
        return {
            "enabled": False,
            "reason": "one global lens grid",
            "top": None,
            "overlap_face_mm": None,
            "seam_mm": None,
            "guard_mm": 0.0,
        }

    def pieces(self):
        return self.lens, self.lens

    def piece_shares(self):
        return 1.0, 0.0

    def x_mm(self, x):
        return (np.asarray(x, dtype=np.float64) + .5) * self.step

    def y_mm(self, y):
        return (np.asarray(y, dtype=np.float64) + .5) * self.step

    def piece_index(self, x_mm):
        split = self.split_mm
        if split is None:
            return np.zeros(np.shape(x_mm), dtype=np.int8)
        return (np.asarray(x_mm, dtype=np.float64) >= split).astype(np.int8)

    def phase_at(self, x, y):
        x_mm = self.x_mm(x)
        y_mm = self.y_mm(y)
        pieces = self.pieces()
        left = (x_mm - pieces[0].shear * y_mm) / pieces[0].mm_per_lens + pieces[0].phase
        if pieces[0].lpi == pieces[1].lpi and pieces[0].phase == pieces[1].phase and pieces[0].shear == pieces[1].shear:
            return left
        if self.split_mm is None:
            return left
        right = (x_mm - pieces[1].shear * y_mm) / pieces[1].mm_per_lens + pieces[1].phase
        return np.where(np.broadcast_to(self.piece_index(x_mm).astype(bool), np.broadcast(x_mm, y_mm).shape), right, left)

    def actual_side(self, scan_x, mount):
        return self.width - 1 - scan_x if mount == "reverse" else scan_x

    def face_mm(self, side="face"):
        return self.x_mm(self.actual_side(np.arange(self.width, dtype=np.float64), side))

    def face_x_mm(self, x_mm, side="face"):
        """Turn paper-face millimetres into the raster column to draw them at."""
        columns = np.clip(np.rint(np.asarray(x_mm, dtype=np.float64) / self.step - .5), 0, self.width - 1)
        return self.actual_side(columns, side).astype(int)

    def phase_matrix(self, start=0, stop=None, side="face"):
        finish = self.height if stop is None else stop
        rows = np.arange(start, finish, dtype=np.float64)
        columns = self.actual_side(np.arange(self.width, dtype=np.float64), side)
        return self.phase_at(columns[None, :], rows[:, None])

    def left_columns(self, side="face"):
        split = self.split_mm
        if split is None:
            return np.ones(self.width, dtype=bool)
        return self.face_mm(side) < split

    def gap_columns(self, side="face"):
        split, guard = self.split_mm, self.guard_mm
        if split is None or guard <= 0:
            return np.zeros(self.width, dtype=bool)
        return np.abs(self.face_mm(side) - split) < guard / 2

    def gap_intervals(self, side="face"):
        """Continuous face-mm spans of the unresolved strip, for reporting."""
        split, guard = self.split_mm, self.guard_mm
        if split is None or guard <= 0:
            return []
        return [[round(max(0.0, split - guard / 2), 6), round(min(self.width_mm, split + guard / 2), 6)]]

    def coverage(self, method, start, stop, slot, count, side="face"):
        """Device weights for one view slot, zero in the unresolved cut-edge strip."""
        finish = self.height if stop is None else stop
        rows = self.y_mm(np.arange(start, finish, dtype=np.float64))[:, None]
        face = self.face_mm(side)[None, :]
        left = self.left_columns(side)
        weights = np.zeros((finish - start, self.width), dtype=np.float32)
        for index, piece in enumerate(self.pieces()):
            columns = left if index == 0 else ~left
            if not columns.any():
                continue
            phase = (face - piece.shear * rows) / piece.mm_per_lens + piece.phase
            if method == "coverage":
                period = self.dpi / piece.lpi
                extent = 0.5 * (1.0 + abs(piece.shear)) / period
                values = fraction_for_slot(phase, extent, slot, count).astype(np.float32)
            else:
                values = (band_index(phase, count) == slot).astype(np.float32)
            weights[:, columns] = values[:, columns]
        weights[:, self.gap_columns(side)] = 0
        return weights

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
            "two_piece": self.split_layout,
        }


@dataclass(frozen=True)
class SplitLensGrid(LensGrid):
    """Two independently measured lens pieces joined at a seam."""

    pair: SplitLens = None

    @property
    def split_mm(self):
        return self.pair.boundary_mm

    @property
    def seam_mm(self):
        return self.pair.seam_mm

    @property
    def guard_mm(self):
        return self.pair.guard_mm()

    def pieces(self):
        return self.pair.left.lens(self.lens), self.pair.right.lens(self.lens)

    def piece_shares(self):
        share = self.pair.boundary_mm / self.width_mm
        return share, 1.0 - share

    @property
    def split_layout(self):
        pair = self.pair
        width = self.width_mm
        boundary = pair.boundary_mm
        guard = pair.guard_mm()
        gap = self.gap_intervals()
        left_piece, right_piece = self.pieces()
        # Each piece is cut a little beyond the seam; the longer one overlaps the other.
        left_end = pair.seam_mm + pair.overlap_mm if pair.overlap_mm > 0 and pair.top == "left" else pair.seam_mm
        right_start = pair.seam_mm - pair.overlap_mm if pair.overlap_mm > 0 and pair.top == "right" else pair.seam_mm
        return {
            "enabled": True,
            "seam_mm": round(pair.seam_mm, 6),
            "overlap_mm": round(pair.overlap_mm, 6),
            "top": pair.top,
            "boundary_mm": round(boundary, 6),
            "guard_mm": round(guard, 6),
            "gap_mm": gap,
            "gap_pixels": [int(np.clip(round(value / self.step), 0, self.width)) for value in gap[0]] if gap else None,
            "overlap_face_mm": pair.overlap_face(),
            "pieces": [
                {
                    "name": "left", "lpi": left_piece.lpi, "phase": left_piece.phase,
                    "shear": left_piece.shear, "pitch_mm": round(left_piece.mm_per_lens, 7),
                    "cut_mm": [0.0, round(left_end, 6)],
                    "visible_mm": [0.0, round(boundary, 6)],
                },
                {
                    "name": "right", "lpi": right_piece.lpi, "phase": right_piece.phase,
                    "shear": right_piece.shear, "pitch_mm": round(right_piece.mm_per_lens, 7),
                    "cut_mm": [round(right_start, 6), round(width, 6)],
                    "visible_mm": [round(boundary, 6), round(width, 6)],
                },
            ],
            "rule": "Left-to-right view order is unchanged; each piece only supplies its own pitch, phase and drift.",
        }


def physical_slot(view_index, view_count, order):
    return view_count - 1 - view_index if order == "reverse" else view_index


def view_index(slot, view_count, order):
    return view_count - 1 - slot if order == "reverse" else slot


def band_index(phase, count):
    return np.minimum(count - 1, np.floor(np.mod(phase, 1) * count).astype(np.int16))


def overlap_length(left, right, interval_start, interval_stop):
    return np.clip(np.minimum(right, interval_stop) - np.maximum(left, interval_start), 0, None)


def fraction_for_slot(center, half_width, slot, count):
    left = np.mod(center, 1) - half_width
    right = left + 2 * half_width
    start, end = slot / count, (slot + 1) / count
    total = sum(overlap_length(left, right, start + offset, end + offset) for offset in (-1, 0, 1))
    return total / (2 * half_width)


def exact_slot_coverage(grid, start, stop, slot, count, *, side="face"):
    return grid.coverage("coverage", start, stop, slot, count, side)


def solid_slot_coverage(grid, start, stop, slot, count, *, side="face"):
    return grid.coverage("solid", start, stop, slot, count, side)


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


def _slot_weights(phase, count, sampling, period_pixels):
    width = (1 - sampling.aperture) / count + 1 / period_pixels
    half_width = width / 2
    left = max(0, phase - half_width)
    right = min(1, phase + half_width)
    return np.asarray([
        float(overlap_length(left, right, slot / count, (slot + 1) / count) / (right - left))
        for slot in range(count)
    ])


def angular_response(grid, angle, count, sampling: Sampling):
    phase = observation_phase(grid, angle)
    weights = np.zeros(count, dtype=np.float64)
    for piece, share in zip(grid.pieces(), grid.piece_shares()):
        if share > 0:
            weights += share * _slot_weights(phase, count, sampling, grid.dpi / piece.lpi)
    if grid.lens.order == "reverse":
        weights = weights[::-1]
    return weights, list(_aperture_window(grid, phase, count, sampling))


def build_grid(plan, width=None, height=None):
    """The physical grid for a print plan, one piece or two."""
    width = width or plan.sheet.size[0]
    height = height or plan.sheet.size[1]
    if plan.two_piece.enabled:
        return SplitLensGrid(plan.lens, plan.sheet.dpi, width, height, plan.two_piece)
    return LensGrid(plan.lens, plan.sheet.dpi, width, height)


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
    method = "solid" if sampling.method == "solid" else "coverage"
    end = min(grid.height, 160)
    for index in range(view_count):
        slot = physical_slot(index, view_count, grid.lens.order)
        by_view[index] = grid.coverage(method, 0, end, slot, view_count, side).mean()
    shares = grid.piece_shares()
    pieces = [
        {
            "piece": name,
            "lpi": piece.lpi,
            "device_dots_per_lens": round(grid.dpi / piece.lpi, 6),
            "nominal_dots_per_view": round(grid.dpi / piece.lpi / view_count, 6),
            "face_share": round(float(share), 8),
        }
        for name, piece, share in zip(("left", "right"), grid.pieces(), shares, strict=True)
        if share > 0
    ]
    return {
        "device_dots_per_lens": round(grid.period_pixels, 6),
        "nominal_dots_per_view": round(grid.period_pixels / view_count, 6),
        "pieces": pieces,
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
