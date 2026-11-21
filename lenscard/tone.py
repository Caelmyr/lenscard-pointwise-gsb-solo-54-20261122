"""Soft-proof source-view colour on explicit test paper, with inspectable regions."""
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from .imaging import linear_to_srgb, srgb_to_linear
from .models import Region, Unprocessable, choice, number, object_value, validate_keys
from .visual import chart_png, code_statistics, heatmap, overlay_regions, png_bytes, resize_report, side_by_side


RGB_TO_XYZ = np.asarray([
    [.4124564, .3575761, .1804375],
    [.2126729, .7151522, .0721750],
    [.0193339, .1191920, .9503041],
], dtype=np.float32)
D65 = np.asarray([.95047, 1, 1.08883], dtype=np.float32)
WHITE = np.asarray([1, 1, 1], dtype=np.float32)
PROOF_METHODS = ("clip", "compress", "neutral")
LAB_Y = 6 / 29


@dataclass(frozen=True)
class PaperModel:
    name: str
    white_rgb: tuple[float, float, float]
    black_floor: float
    color_factor: float
    gamma: float


PAPER_SIMULATIONS = {
    "white": PaperModel("Bright gloss", (1., .989, .977), .013, .88, 1.05),
    "warm": PaperModel("Warm stock", (.94, .889, .783), .026, .69, 1.1),
    "matte": PaperModel("Matte stock", (.916, .924, .889), .037, .76, 1.16),
}


@dataclass(frozen=True)
class SoftOptions:
    paper: str
    method: str
    strength: float
    show_loss: bool

    @classmethod
    def parse(cls, body, paper):
        obj = object_value(body, "proof simulation")
        validate_keys(obj, {"paper", "method", "strength", "show_loss"}, "soft proof")
        from .models import flag
        return cls(
            choice(obj.get("paper", paper), "paper", tuple(PAPER_SIMULATIONS)),
            choice(obj.get("method", "compress"), "tone strategy", PROOF_METHODS),
            number(obj.get("strength", 1), "simulation strength", 0, 1),
            flag(obj.get("show_loss", True), "show clipped colors"),
        )


def _lab_core(values):
    pivot = LAB_Y ** 3
    return np.where(values > pivot, np.cbrt(values), values / (3 * LAB_Y ** 2) + 4 / 29)


def linear_rgb_to_lab(linear):
    xyz = linear @ RGB_TO_XYZ.T
    mapped = _lab_core(np.maximum(xyz, 0) / D65)
    L = 116 * mapped[..., 1] - 16
    a = 500 * (mapped[..., 0] - mapped[..., 1])
    b = 200 * (mapped[..., 1] - mapped[..., 2])
    return np.stack((L, a, b), axis=-1)


def chromatic_difference(left, right):
    return np.linalg.norm(linear_rgb_to_lab(left) - linear_rgb_to_lab(right), axis=-1)


def _paper_white(model):
    return srgb_to_linear(np.asarray(model.white_rgb, np.float32))


def _linear_y(rgb):
    return rgb @ RGB_TO_XYZ[1]


def propose_compression(source, neutral, limit):
    displacement = source - neutral
    room_high = np.divide(1 - neutral, np.maximum(displacement, 1e-6))
    room_low = np.divide(neutral, np.maximum(-displacement, 1e-6))
    room = np.where(displacement >= 0, room_high, room_low)
    factor = np.clip(np.minimum(limit, np.min(room, axis=-1)), 0, 1)
    return neutral + displacement * factor[..., None]


def _apply_paper(source, model, method):
    white = _paper_white(model)
    luma = _linear_y(source)[..., None]
    if method == "neutral":
        input_color = np.broadcast_to(luma, source.shape)
    elif method == "compress":
        input_color = propose_compression(source, luma, model.color_factor)
    else:
        linear = np.maximum(source, 0)
        input_color = np.minimum(linear, model.color_factor) / model.color_factor
    out = input_color ** model.gamma
    return model.black_floor + out * (white - model.black_floor)


def _unsupported_overlay(image, support):
    coded = linear_to_srgb(image)
    coded[~support] = [40, 52, 58]
    return Image.fromarray(coded)


def soft_proof(sample, options):
    source, support = sample.base()
    model = PAPER_SIMULATIONS[options.paper]
    result = _apply_paper(source, model, options.method)
    result = source * (1 - options.strength) + result * options.strength
    diff = chromatic_difference(source, result)
    before, after = _unsupported_overlay(source, support), _unsupported_overlay(result, support)
    pictures = {"side-by-side": png_bytes(resize_report(side_by_side(before, after, 9), 1150))}
    if options.show_loss:
        maximum = max(2, float(np.percentile(diff[support], 98)))
        pictures["color-move"] = png_bytes(resize_report(heatmap(diff, support, 0, maximum)))
    neutral = np.max(source, 2) - np.min(source, 2)
    colored = support & (neutral > .09)
    y_before, y_after = _linear_y(source), _linear_y(result)
    report = {
        "kind": "material-softproof",
        "source": sample.public(),
        "paper": model.name,
        "paper_target": {"white_srgb": list(model.white_rgb), "black_floor": model.black_floor,
                         "chroma_factor": model.color_factor, "response_power": model.gamma},
        "simulation_strength": options.strength,
        "method": options.method,
        "perceptual_displacement_approx": code_statistics(diff[support]),
        "non_neutral_displacement_approx": code_statistics(diff[colored]),
        "luma_before": code_statistics(y_before[support]),
        "luma_after": code_statistics(y_after[support]),
        "big_color_moves_fraction": round(float(np.mean(diff[support] > 8)), 6),
        "note": "Approximate proofs use local sample paper whites and a simple response model; measure the printed card for colour acceptance.",
    }
    return report, pictures


def _patch_stats(image, support, region):
    h, w = support.shape
    x0, y0, x1, y1 = region.pixels(w, h)
    values = image[y0:y1, x0:x1]
    mask = support[y0:y1, x0:x1]
    if not mask.any() or mask.mean() < .72 or mask.sum() < 25:
        raise Unprocessable("Every color patch needs enough covered samples to compare.")
    return {
        "box": [x0, y0, x1, y1],
        "supported_count": int(mask.sum()),
        "rgb": np.median(values[mask], axis=0),
        "patch_luma": code_statistics(_linear_y(values)[mask]),
    }


def response_patches(sample, rows):
    if not isinstance(rows, list) or not 3 <= len(rows) <= 10:
        raise Unprocessable("Choose three to ten ordered marked density patches.")
    pixels = sample.pixels
    numbers = []
    colors = []
    for j, mapping in enumerate(rows, start=1):
        data = object_value(mapping, "response patch")
        validate_keys(data, {"region", "nominal"}, "response patch")
        target = number(data.get("nominal"), "nominal diffuse brightness", 0, 1)
        box = Region.parse(data.get("region"))
        summary = _patch_stats(pixels.rgb, pixels.support, box)
        rgb = summary.pop("rgb")
        neutral_gap = float(np.max(rgb) - np.min(rgb))
        numbers.append({
            **summary, "patch": j, "target": target, "region": box.public(),
            "luminance": round(float(_linear_y(rgb)), 7),
            "measured_rgb": [round(float(v), 6) for v in rgb],
            "neutral_gap": round(neutral_gap, 6),
        })
        colors.append((box.public(), f"{j} ({target:.2f})", (242, 197, 100)))
    sorted_rows = sorted(numbers, key=lambda row: row["target"])
    targets = [row["target"] for row in sorted_rows]
    if len(set(targets)) < len(targets):
        raise Unprocessable("Assign each measured patch a different nominal value.")
    observed = [row["luminance"] for row in sorted_rows]
    residuals = [obs - exp for exp, obs in zip(targets, observed, strict=True)]
    warnings = [
        "Two neighboring patches are out of order."
        for a, b in zip(observed, observed[1:]) if b < a - .004
    ]
    if any(row["neutral_gap"] > .09 for row in numbers):
        warnings.append("At least one density patch is visibly tinted.")
    raw = _unsupported_overlay(pixels.rgb, pixels.support)
    plate = overlay_regions(raw, colors)
    curve = chart_png([
        ("measured", np.asarray(targets), np.asarray(observed)),
        ("nominal", np.asarray([0, 1]), np.asarray([0, 1])),
    ], x_label="Nominal patch value", y_label="Measured linear luminance", x_bounds=[0, 1], y_bounds=[0, 1])
    return {
        "kind": "paper-density-readback",
        "source": sample.public(),
        "patches": sorted_rows,
        "mean_absolute_error": round(float(np.mean(np.abs(residuals))), 6),
        "warning": sorted(set(warnings)),
    }, {"response": curve, "marked-patches": png_bytes(resize_report(plate, 920))}


def analyze_neutral_noise(sample):
    rgb, support = sample.base()
    luma = _linear_y(rgb)
    cropped_support = ndimage.maximum_filter(~support, size=5) == 0
    if cropped_support.sum() < 48:
        raise Unprocessable("The noise box needs an uninterrupted interior.")
    broad = ndimage.gaussian_filter(luma, sigma=2)
    delta = (luma - broad)[cropped_support]
    dispersion = float(np.median(np.abs(delta - np.median(delta)))) * 1.4826
    xs = np.linspace(-1, 1, 101) * max(.003, dispersion * 8)
    hist, bounds = np.histogram(delta, bins=xs)
    report = {
        "kind": "source-grain",
        "source": sample.public(),
        "residual_summary": code_statistics(delta),
        "robust_noise_level": round(dispersion, 7),
        "accepted_samples": int(cropped_support.sum()),
        "note": "Use an even area. Detail texture contributes to this estimate.",
    }
    chart = chart_png([("local residual", (bounds[:-1] + bounds[1:]) * .5, hist.astype(float))],
                      x_label="Departure from local brightness", y_label="Samples")
    return report, {"grain": chart}
