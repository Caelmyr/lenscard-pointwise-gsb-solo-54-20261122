"""Rectify an observed paper photograph from four physically labelled corners."""
from dataclasses import dataclass
from hashlib import sha256
from math import sqrt

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from .imaging import fit_size, linear_to_srgb, read_capture, sample_capture
from .models import Capacity, Position, Unprocessable, integer, number, object_value, validate_keys
from .visual import png_bytes, resize_report


CANONICAL = np.asarray([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
BLOCK_ROWS = 105
MIN_QUAD_AREA = 0.035


@dataclass(frozen=True)
class MountBoard:
    width_mm: float
    height_mm: float
    resolution: float

    @classmethod
    def parse(cls, body, sheet):
        spec = object_value(body, "paper rectification")
        validate_keys(spec, {"width_mm", "height_mm", "resolution"}, "rectification")
        board = cls(
            number(spec.get("width_mm", sheet.width_mm), "document width", 15, 300),
            number(spec.get("height_mm", sheet.height_mm), "document height", 15, 320),
            number(spec.get("resolution", 12), "target pixels per millimetre", 2, 30),
        )
        w, h = board.size
        if w * h > 4_000_000 or max(w, h) > 4300:
            raise Capacity("Reduce the document rectification region or the target pixel density.")
        return board

    @property
    def size(self):
        return round(self.width_mm * self.resolution), round(self.height_mm * self.resolution)

    def public(self):
        return {
            "target_mm": [self.width_mm, self.height_mm],
            "target_px": list(self.size),
            "pixels_per_mm": self.resolution,
        }


@dataclass(frozen=True)
class Perspective:
    to_camera: np.ndarray
    original: np.ndarray

    def sample(self, x01, y01):
        homogeneous = self.to_camera
        ratio = homogeneous[2, 0] * x01 + homogeneous[2, 1] * y01 + homogeneous[2, 2]
        xs = (homogeneous[0, 0] * x01 + homogeneous[0, 1] * y01 + homogeneous[0, 2]) / ratio
        ys = (homogeneous[1, 0] * x01 + homogeneous[1, 1] * y01 + homogeneous[1, 2]) / ratio
        return np.broadcast_arrays(xs, ys)

    def manifest(self):
        return {
            "camera_normalized_per_paper_normalized": np.round(self.to_camera, 10).tolist(),
            "normalized_camera_corners": self.original.tolist(),
            "mapping_order": ["upper-left", "upper-right", "lower-right", "lower-left"],
        }


def _ordered(points):
    delta = np.roll(points, -1, axis=0) - points
    cross = delta[:, 0] * np.roll(delta[:, 1], -1) - delta[:, 1] * np.roll(delta[:, 0], -1)
    if np.min(cross) <= .0005:
        raise Unprocessable("Choose four convex corners in clockwise order from the upper left.")
    wedges = points[:, 0] * np.roll(points[:, 1], -1) - points[:, 1] * np.roll(points[:, 0], -1)
    area = sum(wedges) / 2
    if area < MIN_QUAD_AREA:
        raise Unprocessable("Choose a larger visible physical face in the photograph.")
    lengths = np.hypot(delta[:, 0], delta[:, 1])
    if np.min(lengths) < .10:
        raise Unprocessable("At least one paper edge is too short to fix a stable projection.")
    return area, lengths


def _camera_transform(dest, source):
    rows, output = [], []
    for (a, b), (x, y) in zip(dest, source, strict=True):
        rows.append([a, b, 1, 0, 0, 0, -a * x, -b * x])
        rows.append([0, 0, 0, a, b, 1, -a * y, -b * y])
        output.extend([x, y])
    system = np.asarray(rows)
    if np.linalg.cond(system) > 700:
        raise Unprocessable("The chosen image corners are too steep for a stable paper projection.")
    try:
        coeffs = np.linalg.solve(system, np.asarray(output))
    except np.linalg.LinAlgError as exc:
        raise Unprocessable("Could not fit the paper plane from these corners.") from exc
    transform = np.append(coeffs, 1).reshape((3, 3))
    denominators = CANONICAL @ transform[2, :2] + transform[2, 2]
    if np.min(denominators) < .10 or np.max(denominators) > 8.0:
        raise Unprocessable("The paper is too oblique to resample as a stable rectangular print.")
    return transform


def check_quad(positions):
    if not isinstance(positions, list) or len(positions) != 4:
        raise Unprocessable("Draw four corners of the physical print.")
    ordered = np.asarray([Position.parse(point, "paper corner").public() for point in positions], np.float64)
    area, lengths = _ordered(ordered)
    homography = _camera_transform(CANONICAL, ordered)
    return Perspective(homography, ordered), area, lengths


def _cover_stats(stats, xs, ys, support):
    if support.any():
        stats["supported"] += int(np.count_nonzero(support))
        stats["left"] = min(stats["left"], float(xs[support].min()))
        stats["top"] = min(stats["top"], float(ys[support].min()))
        stats["right"] = max(stats["right"], float(xs[support].max()))
        stats["bottom"] = max(stats["bottom"], float(ys[support].max()))


def _resample(perspective, pixels, board):
    width, height = board.size
    h0, w0 = pixels.shape
    strips = np.empty((height, width, 3), dtype=np.float32)
    overall = np.zeros((height, width), dtype=bool)
    coverage = {"supported": 0, "left": w0 + 1, "top": h0 + 1, "right": 0, "bottom": 0}
    for row in range(0, height, BLOCK_ROWS):
        bottom = min(height, row + BLOCK_ROWS)
        u = (np.arange(width)[None, :] + .5) / width
        v = (np.arange(row, bottom)[:, None] + .5) / height
        xs, ys = perspective.sample(u, v)
        xs *= w0 - 1
        ys *= h0 - 1
        coords = [ys, xs]
        support = ndimage.map_coordinates(pixels.support.astype(np.float32), coords, order=1) >= .995
        for band in range(3):
            linear = ndimage.map_coordinates(pixels.rgb[..., band], coords, order=1)
            strips[row:bottom, :, band] = np.where(support, linear, .035)
        overall[row:bottom] = support
        _cover_stats(coverage, xs, ys, support)
    return strips, overall, coverage


def _mark_photo(perspective, image):
    picture = image.convert("RGB")
    draw = ImageDraw.Draw(picture)
    pts = [(round(x * (picture.width - 1)), round(y * (picture.height - 1))) for x, y in perspective.original]
    for index, start in enumerate(pts):
        finish = pts[(index + 1) % 4]
        draw.line([start, finish], width=3, fill=(249, 204, 110))
        cx, cy = start
        draw.ellipse((cx - 5, cy - 5, cx + 5, cy + 5), fill=(245, 224, 139), outline=(52, 62, 65))
        draw.text((cx + 9, cy + 9), str(index + 1), fill=(245, 233, 164))
    return resize_report(picture, 950)


def rectify_board(raw, filename, sheet, options):
    obj = object_value(options, "document registration")
    validate_keys(obj, {"corners", "board"}, "document registration")
    board = MountBoard.parse(obj.get("board", {}), sheet)
    perspective, area, lengths = check_quad(obj.get("corners"))
    decoded = read_capture(raw, filename)
    source_size = fit_size(decoded.image.size, 2250)
    source = sample_capture(decoded, source_size)
    linear, support, bounds = _resample(perspective, source, board)
    if support.mean() < .58:
        raise Unprocessable("Too much of the rectified document is outside visible source pixels.")
    report = {
        "kind": "rectified-print-photo",
        "original": {"sha256": sha256(raw).hexdigest(), "size": list(decoded.image.size), "filename": filename},
        "document": board.public(), "source_raster": list(source_size),
        "perspective": perspective.manifest(), "observed_coverage": round(float(support.mean()), 8),
        "covered_source_samples": bounds, "camera_polygon_fraction": round(float(area), 8),
        "camera_edge_lengths": [round(float(length), 6) for length in lengths],
        "limitation": "Rectification preserves the chosen corners; it does not dewarp a curved or creased substrate.",
    }
    photo = linear_to_srgb(linear)
    photo[~support] = [43, 53, 57]
    rectified = png_bytes(photo)
    return report, {"selected": png_bytes(_mark_photo(perspective, decoded.image)), "rectified": rectified}
