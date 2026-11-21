"""Constrained rail alignment and explicit image support in reference coordinates."""

import math
from dataclasses import dataclass

import numpy as np
from scipy import fft, ndimage

from .imaging import CapturePixels, fill_nearest, fit_size, float_resize
from .models import Unprocessable

MAX_ESTIMATE_SIDE = 330
MAX_OFFSET_FRACTION = 0.17
MIN_SCALE = 0.88
MAX_SCALE = 1.14
MIN_OVERLAP = 0.66


@dataclass(frozen=True)
class Motion:
    dx: float = 0.0
    dy: float = 0.0
    scale: float = 1.0
    similarity: float | None = None
    overlap: float = 1.0
    feature_rms: float | None = None
    warning: str | None = None

    def with_scale_space(self, width_ratio: float, height_ratio: float):
        return Motion(
            dx=self.dx * width_ratio,
            dy=self.dy * height_ratio,
            scale=self.scale,
            similarity=self.similarity,
            overlap=self.overlap,
            feature_rms=self.feature_rms,
            warning=self.warning,
        )

    def public(self) -> dict:
        return {
            "dx": self.dx,
            "dy": self.dy,
            "scale": self.scale,
            "score": self.similarity,
            "overlap": self.overlap,
            "feature_rms": self.feature_rms,
            "warning": self.warning,
        }


@dataclass
class AlignedCapture:
    rgb: np.ndarray
    support: np.ndarray
    motion: Motion


@dataclass(frozen=True)
class AnalysisBox:
    x: int
    y: int
    width: int
    height: int

    def slice(self, array: np.ndarray) -> np.ndarray:
        return array[self.y:self.y + self.height, self.x:self.x + self.width]

    @property
    def pixels(self) -> int:
        return self.width * self.height

    def public(self) -> dict:
        return {"x": self.x, "y": self.y, "width": self.width, "height": self.height}


@dataclass
class AlignmentFeature:
    values: np.ndarray
    support: np.ndarray
    spectrum: np.ndarray
    texture: float


def affine_components(
    shape: tuple[int, int],
    dx: float,
    dy: float,
    scale: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    center_y, center_x = (side - 1 for side in shape)
    center_y /= 2
    center_x /= 2
    inverse = 1.0 / scale
    return (
        (inverse, inverse),
        (center_y - (center_y + dy) * inverse, center_x - (center_x + dx) * inverse),
    )


def transform_layer(
    array: np.ndarray,
    dx: float,
    dy: float,
    scale: float,
    *,
    order: int,
    cval: float = 0,
) -> np.ndarray:
    if abs(dx) + abs(dy) + abs(scale - 1.0) < 1e-9:
        return array.copy()
    coefficients, offset = affine_components(array.shape, dx, dy, scale)
    matrix = np.diag(coefficients)
    return ndimage.affine_transform(
        array, matrix, offset=offset, mode="constant", cval=cval, order=order, prefilter=False,
    )


def support_after_motion(support: np.ndarray, motion: Motion) -> np.ndarray:
    return transform_layer(
        support.astype(np.float32),
        motion.dx, motion.dy, motion.scale, order=1,
    ) > 0.999


def warp(capture: CapturePixels, motion: Motion) -> AlignedCapture:
    if abs(motion.dx) + abs(motion.dy) + abs(1 - motion.scale) < 1e-9:
        return AlignedCapture(capture.rgb, capture.support, motion)
    output = np.empty_like(capture.rgb)
    for band in range(capture.rgb.shape[2]):
        output[..., band] = transform_layer(
            capture.rgb[..., band], motion.dx, motion.dy, motion.scale, order=1,
        )
    mask = support_after_motion(capture.support, motion)
    output[~mask] = 0
    return AlignedCapture(output, mask, motion)


def frame_feature(image: np.ndarray, support: np.ndarray) -> tuple[np.ndarray, float]:
    gray = fill_nearest(image, support)
    low = ndimage.gaussian_filter(gray, 0.7, mode="nearest")
    large = ndimage.gaussian_filter(gray, 4.2, mode="nearest")
    values = (low - large).astype(np.float32)
    values[~support] = 0
    samples = values[support]
    if samples.size < 16:
        return values, 0.0
    texture = float(np.sqrt(np.mean(samples ** 2)))
    return values, texture


def window(shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    row = np.hanning(height) ** 0.7
    column = np.hanning(width) ** 0.7
    return np.asarray(row[:, None] * column[None, :], dtype=np.float32)


def _tapered_spectrum(
    values: np.ndarray,
    support: np.ndarray,
    taper: np.ndarray,
) -> np.ndarray:
    total = max(float(taper[support].sum()), 1e-8)
    mean = float(np.dot(values[support], taper[support]) / total)
    signal = np.where(support, (values - mean) * taper, 0)
    return fft.rfft2(signal, workers=1)


def make_feature(
    luminance: np.ndarray,
    support: np.ndarray,
    target_size: tuple[int, int],
) -> AlignmentFeature:
    gray = float_resize(luminance, target_size)
    mask = float_resize(support.astype(np.float32), target_size) > 0.95
    values, texture = frame_feature(gray, mask)
    spectrum = _tapered_spectrum(values, mask, window(gray.shape))
    return AlignmentFeature(values, mask, spectrum, texture)


def _centroid_peak(values: np.ndarray, axis: int, peak: tuple[int, int]) -> float:
    height, width = values.shape
    y, x = peak
    y0, y1 = ((y - 1) % height, (y + 1) % height) if axis == 0 else (y, y)
    x0, x1 = ((x - 1) % width, (x + 1) % width) if axis == 1 else (x, x)
    low, center, high = values[y0, x0], values[y, x], values[y1, x1]
    curvature = low - 2 * center + high
    if curvature >= -1e-7:
        return 0.0
    return float(np.clip(0.5 * (low - high) / curvature, -0.5, 0.5))


def phase_shift(
    reference: np.ndarray,
    moving: np.ndarray,
    shape: tuple[int, int],
    max_dx: int,
    max_dy: int,
) -> tuple[float, float]:
    cross = reference * np.conjugate(moving)
    amplitude = np.abs(cross)
    cutoff = max(float(amplitude.max()) * 1e-5, 1e-8)
    normalization = np.maximum(amplitude, cutoff) ** 0.75
    correlation = fft.irfft2(cross / normalization, s=shape, workers=1)
    height, width = shape
    row_offsets = np.arange(-max_dy, max_dy + 1, dtype=np.int32)
    column_offsets = np.arange(-max_dx, max_dx + 1, dtype=np.int32)
    scores = correlation[np.ix_(row_offsets % height, column_offsets % width)]
    row, col = np.unravel_index(int(scores.argmax()), scores.shape)
    dy, dx = int(row_offsets[row]), int(column_offsets[col])
    peak = (dy % height, dx % width)
    fractional_y = _centroid_peak(correlation, 0, peak) if abs(dy) < max_dy else 0.0
    fractional_x = _centroid_peak(correlation, 1, peak) if abs(dx) < max_dx else 0.0
    return dx + fractional_x, dy + fractional_y


def similarity(
    reference: AlignmentFeature, image: np.ndarray, support: np.ndarray,
) -> tuple[float, float, float]:
    valid = reference.support & support
    coverage = float(np.count_nonzero(valid) / max(1, np.count_nonzero(reference.support)))
    if coverage < MIN_OVERLAP:
        return -1.0, coverage, 0.0
    x = reference.values[valid].astype(np.float64)
    y = image[valid].astype(np.float64)
    x -= x.mean()
    y -= y.mean()
    norm = np.linalg.norm(x) * np.linalg.norm(y)
    corr = float(np.dot(x, y) / norm) if norm > 1e-7 else -1.0
    residual = float(np.sqrt(np.mean((x - y) ** 2)))
    return corr, coverage, residual


def evaluate_scale(
    reference: AlignmentFeature,
    moving: AlignmentFeature,
    taper: np.ndarray,
    scale: float,
) -> Motion:
    shape = moving.values.shape
    warped = transform_layer(moving.values, 0, 0, scale, order=1)
    coarse_support = support_after_motion(moving.support, Motion(scale=scale))
    spectrum = _tapered_spectrum(warped, coarse_support, taper)
    max_dx = max(1, math.floor(shape[1] * MAX_OFFSET_FRACTION))
    max_dy = max(1, math.floor(shape[0] * MAX_OFFSET_FRACTION))
    dx, dy = phase_shift(reference.spectrum, spectrum, shape, max_dx, max_dy)
    matrix = Motion(dx=dx, dy=dy, scale=scale)
    warped = transform_layer(moving.values, dx, dy, scale, order=1)
    valid = support_after_motion(moving.support, matrix)
    score, overlap, residual = similarity(reference, warped, valid)
    return Motion(dx, dy, scale, score, overlap, residual)


def _rank(candidate: Motion) -> float:
    return (
        (candidate.similarity if candidate.similarity is not None else -1)
        - 0.10 * (1.0 - candidate.overlap) - 0.025 * abs(math.log(candidate.scale))
    )


def solve_motion(
    reference: AlignmentFeature,
    current: AlignmentFeature,
    mode: str,
) -> Motion:
    if min(reference.texture, current.texture) < 0.0022:
        baseline = similarity(reference, current.values, current.support)
        return Motion(similarity=baseline[0], overlap=baseline[1], warning="Weak shared texture")
    taper = window(reference.values.shape)
    baseline = evaluate_scale(reference, current, taper, 1.0)
    best = baseline
    if mode == "breathing":
        for scale in (0.88, 0.92, 0.95, 0.975, 1.025, 1.055, 1.095, 1.14):
            candidate = evaluate_scale(reference, current, taper, scale)
            if _rank(candidate) > _rank(best):
                best = candidate
        for delta in (0.014, 0.006, 0.0025):
            center = best.scale
            for scale in (center - delta, center + delta):
                if MIN_SCALE <= scale <= MAX_SCALE:
                    candidate = evaluate_scale(reference, current, taper, scale)
                    if _rank(candidate) > _rank(best):
                        best = candidate
        if best.scale != 1 and _rank(best) - _rank(baseline) < 0.003:
            best = baseline
    warnings = [
        text for condition, text in (
            (best.similarity is not None and best.similarity < 0.28, "Weak alignment match"),
            (best.overlap < 0.79, "Small aligned overlap"),
            (best.scale < MIN_SCALE + 0.006 or best.scale > MAX_SCALE - 0.006, "Zoom at search limit"),
        )
        if condition
    ]
    return Motion(
        best.dx, best.dy, best.scale, best.similarity, best.overlap, best.feature_rms,
        ", ".join(warnings) if warnings else None,
    )


class RailAlignment:
    def __init__(self, reference: CapturePixels, mode: str):
        self.mode = mode
        self.reference = reference
        height, width = reference.shape
        self.estimation_size = fit_size((width, height), MAX_ESTIMATE_SIDE)
        self.ref_feature = (
            make_feature(reference.luminance, reference.support, self.estimation_size)
            if mode != "none" else None
        )

    def align(self, capture: CapturePixels, *, reference: bool = False) -> AlignedCapture:
        if reference or self.mode == "none":
            overlap = float(np.count_nonzero(capture.support & self.reference.support) /
                            max(1, np.count_nonzero(self.reference.support)))
            motion = Motion(similarity=1.0 if reference else None, overlap=overlap)
            return AlignedCapture(capture.rgb, capture.support, motion)
        feature = make_feature(capture.luminance, capture.support, self.estimation_size)
        motion = solve_motion(self.ref_feature, feature, self.mode)
        corrected = motion.with_scale_space(
            capture.shape[1] / self.estimation_size[0],
            capture.shape[0] / self.estimation_size[1],
        )
        return warp(capture, corrected)


def largest_rectangle(mask: np.ndarray) -> AnalysisBox:
    rows, cols = mask.shape
    heights = np.zeros(cols, dtype=np.int32)
    best = AnalysisBox(0, 0, 0, 0)
    for row in range(rows):
        heights += 1
        heights[~mask[row]] = 0
        active = []
        for col, height in enumerate(np.append(heights, 0)):
            start = col
            while active and active[-1][1] > height:
                start, candidate = active.pop()
                width = col - start
                if width * candidate > best.pixels:
                    best = AnalysisBox(int(start), int(row + 1 - candidate), int(width), int(candidate))
            if height > 0 and (not active or active[-1][1] < height):
                active.append((col if start == col else start, height))
    return best


def output_viewport(support: np.ndarray, mode: str) -> AnalysisBox:
    _, height, width = support.shape
    if mode == "reference":
        return AnalysisBox(0, 0, width, height)
    shared = np.all(support, axis=0)
    box = largest_rectangle(shared)
    if min(box.width, box.height) < 28 or box.pixels < height * width * 0.35:
        raise Unprocessable("The captures share too little uninterrupted coverage. Use the reference view or exclude a shifted frame.")
    return box


def overlap_summary(support: np.ndarray, box: AnalysisBox) -> dict:
    count = box.slice(support.sum(axis=0))
    present = count > 0
    shared = count == support.shape[0]
    return {
        "visible_fraction": round(float(present.mean()), 6),
        "shared_fraction": round(float(shared.mean()), 6),
        "uncovered_pixels": int(np.count_nonzero(~present)),
        "minimum_contributors": int(count.min()),
        "maximum_contributors": int(count.max()),
        "mean_contributors": round(float(count.mean()), 3),
    }
