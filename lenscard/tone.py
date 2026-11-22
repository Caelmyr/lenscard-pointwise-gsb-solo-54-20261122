"""Soft-proof source-view colour on explicit test paper, with inspectable regions."""
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from .imaging import linear_to_srgb, srgb_to_linear
from .models import Region, Unprocessable, choice, flag, number, object_value, validate_keys
from .visual import chart_png, code_statistics, heatmap, overlay_regions, png_bytes, resize_report, side_by_side


RGB_TO_XYZ = np.asarray([
    [.4124564, .3575761, .1804375],
    [.2126729, .7151522, .0721750],
    [.0193339, .1191920, .9503041],
], dtype=np.float32)
D65 = np.asarray([.95047, 1, 1.08883], dtype=np.float32)
WHITE = np.asarray([1, 1, 1], dtype=np.float32)
PROOF_METHODS = ("clip", "compress", "neutral")
MEASURED_KIND = "paper-density-readback"
STEP_ORDER_TOLERANCE = .004
TINT_LIMIT = .09
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
MEASURED_PREFIX = "measured:"


def _order_suspects(values):
    """Index the steps in an adjacent pair whose measured order is reversed."""
    flags = [False] * len(values)
    for index in range(1, len(values)):
        if values[index] < values[index - 1] - STEP_ORDER_TOLERANCE:
            flags[index - 1] = flags[index] = True
    return flags


def _clean_channel(value, label):
    return round(float(number(value, label, 0, 1)), 6)


@dataclass(frozen=True)
class MeasuredPaper:
    """A monotone gray-to-white response fitted from a retained step inspection."""

    label: str
    measurement_id: str
    measurement_title: str
    created: str
    source: dict
    anchors: tuple[tuple[float, float], ...]
    patches: tuple[dict, ...]
    segments: tuple[dict, ...]
    warnings: tuple[str, ...]
    white_level: float
    black_level: float
    white_rgb: tuple[float, float, float]

    def response(self, values):
        values = np.asarray(values, np.float64)
        x = np.asarray([point[0] for point in self.anchors], np.float64)
        y = np.asarray([point[1] for point in self.anchors], np.float64)
        return np.clip(np.interp(values, x, y), 0, 1)

    @property
    def white_coded(self):
        """Measured white as display-encoded RGB, for paper fills."""
        return tuple(float(value) for value in linear_to_srgb(np.asarray(self.white_rgb, np.float32)) / 255)

    def target(self):
        encoded = linear_to_srgb(np.asarray(self.white_rgb, np.float32)) / 255
        return {
            "white_srgb": [round(float(channel), 6) for channel in encoded],
            "black_floor": round(self.black_level, 6),
            "chroma_factor": None,
            "response_power": None,
            "white_linear_luma": round(self.white_level, 6),
            "response_kind": "measured monotone neutral step response",
        }

    def summary(self):
        return {
            "id": self.measurement_id,
            "title": self.measurement_title,
            "created": self.created,
            "source": self.source,
        }

    def report(self):
        return {
            "kind": "measured-paper-response",
            "measurement": self.summary(),
            "response_anchors": [list(point) for point in self.anchors],
            "steps": list(self.patches),
            "inferred_segments": list(self.segments),
            "white_linear_luma": round(self.white_level, 6),
            "black_linear_luma": round(self.black_level, 6),
            "warning": list(self.warnings),
            "note": "The neutral axis follows an ordered monotone fit. Steps marked inferred take their response "
                    "from measured neighbours, so no inverted photograph sample is copied into the proof.",
        }


def measured_paper(record):
    """Build an ordered response from a retained paper-density step inspection."""
    report = record.get("report") if isinstance(record, dict) else None
    if not isinstance(report, dict) or report.get("kind") != MEASURED_KIND:
        raise Unprocessable("Choose a retained density step inspection to describe this paper.")
    rows = report.get("patches")
    if not isinstance(rows, list) or len(rows) < 3:
        raise Unprocessable("The retained step inspection needs at least three measured patches.")
    steps = []
    for position, row in enumerate(rows, start=1):
        data = object_value(row, "measured patch")
        rgb = data.get("measured_rgb")
        if not isinstance(rgb, (list, tuple)) or len(rgb) != 3:
            raise Unprocessable("Each retained step needs a measured RGB value.")
        channels = np.asarray([_clean_channel(value, "measured channel") for value in rgb], np.float64)
        gap = data.get("neutral_gap")
        gap = float(number(gap, "neutral gap", 0, 1)) if gap is not None else float(np.max(channels) - np.min(channels))
        count = data.get("supported_count", 0)
        steps.append({
            "patch": int(data.get("patch", position)),
            "nominal": float(number(data.get("target"), "nominal patch value", 0, 1)),
            "measured_luminance": float(number(data.get("luminance"), "measured luminance", 0, 1)),
            "measured_rgb": [float(value) for value in channels],
            "neutral_gap": round(gap, 6),
            "supported_count": int(count) if isinstance(count, (int, float)) and not isinstance(count, bool) else 0,
        })
    steps.sort(key=lambda step: step["nominal"])
    if len({step["nominal"] for step in steps}) < len(steps):
        raise Unprocessable("Assign each retained step a different nominal value before using it as paper.")
    order_bad = _order_suspects([step["measured_luminance"] for step in steps])
    counts = sorted(step["supported_count"] for step in steps)
    median_count = counts[len(counts) // 2]
    for index, step in enumerate(steps):
        reasons = []
        if order_bad[index]:
            reasons.append("out of order with a neighbouring step")
        if step["neutral_gap"] > TINT_LIMIT:
            reasons.append("visibly tinted for a neutral step")
        if step["supported_count"] and step["supported_count"] < max(25, median_count * .5):
            reasons.append("fewer covered samples than the other steps")
        step["status"] = "inferred" if reasons else "measured"
        step["reasons"] = reasons
    reliable = [step for step in steps if step["status"] == "measured"]
    if len(reliable) < 2:
        raise Unprocessable("Too many retained steps disagree. Reshoot the gray wedge before using it as paper.")
    first, last = reliable[0], reliable[-1]
    black_level = min(step["measured_luminance"] for step in reliable)
    white_level = max(step["measured_luminance"] for step in reliable)
    if last["nominal"] > first["nominal"]:
        slope = (last["measured_luminance"] - first["measured_luminance"]) / (last["nominal"] - first["nominal"])
        if first["nominal"] > .02:
            black_level = max(0.0, min(black_level, first["measured_luminance"] - slope * first["nominal"]))
        if last["nominal"] < .98:
            white_level = max(white_level, min(1.0, last["measured_luminance"] + slope * (1 - last["nominal"])))
    anchors = {0.0: black_level, 1.0: white_level}
    for step in reliable:
        anchors[step["nominal"]] = min(max(step["measured_luminance"], 0.0), 1.0)
    points = sorted(anchors.items())
    separation = min(1e-5, max(1e-7, (1.0 - white_level) / max(1, len(points))))
    for index in range(1, len(points)):
        if points[index][1] <= points[index - 1][1]:
            points[index] = (points[index][0], min(1.0, points[index - 1][1] + separation))
    anchors = tuple((float(x), float(y)) for x, y in points)
    black_level, white_level = anchors[0][1], anchors[-1][1]
    curve = MeasuredPaper("", "", "", "", {}, anchors, (), (), (), white_level, black_level, (1., 1., 1.))
    runs = []
    run = []
    for index, step in enumerate(steps):
        if step["status"] == "inferred":
            step["response_luminance"] = round(float(curve.response(step["nominal"])), 7)
            run.append(index)
            continue
        step["response_luminance"] = round(step["measured_luminance"], 7)
        if run:
            runs.append((run[0], index - 1))
            run = []
    if run:
        runs.append((run[0], len(steps) - 1))
    segments = []
    for start, stop in runs:
        segments.append({
            "patches": [steps[index]["patch"] for index in range(start, stop + 1)],
            "from_nominal": steps[start - 1]["nominal"] if start else 0.0,
            "to_nominal": steps[stop + 1]["nominal"] if stop + 1 < len(steps) else 1.0,
            "method": "interpolated between neighbouring measured steps",
        })
    warnings = []
    if any(step["status"] == "inferred" for step in steps):
        warnings.append("Some retained steps are uncertain in the photograph; their response is inferred from "
                        "neighbouring measured steps and shown in the tone chart.")
    if any(step["neutral_gap"] > TINT_LIMIT for step in steps):
        warnings.append("A neutral step is visibly tinted in the photograph and was excluded from the neutral axis.")
    if segments:
        warnings.append(f"{len(segments)} segment(s) of the gray ramp are interpolated rather than copied.")
    brightest = max(reliable, key=lambda step: step["nominal"])
    measured_white = np.asarray(brightest["measured_rgb"], np.float64)
    measured_luma = float(_linear_y(measured_white))
    if measured_luma > 1e-6 and white_level > 0:
        # Keep the measured hue but match the extrapolated white level, so paper fill and ramp agree.
        scale = white_level / measured_luma
        if measured_white.max() * scale > 1:
            scale = 1 / measured_white.max()
        measured_white = measured_white * scale
    white_rgb = tuple(float(min(max(channel, 0.0), 1.0)) for channel in measured_white)
    return MeasuredPaper(
        label=f"Measured · {record.get('title') or 'retained step inspection'}",
        measurement_id=record.get("id", ""),
        measurement_title=record.get("title") or "Retained step inspection",
        created=record.get("created", ""),
        source=report.get("source") or {},
        anchors=anchors,
        patches=tuple(steps),
        segments=tuple(segments),
        warnings=tuple(warnings),
        white_level=float(white_level),
        black_level=float(black_level),
        white_rgb=white_rgb,
    )


def paper_menu(records):
    """Describe selectable built-in and measured papers for the workbench."""
    measured = [
        {
            "selection": MEASURED_PREFIX + record["id"],
            "label": f"Measured · {record['title']}",
            "title": record["title"],
            "created": record.get("created", ""),
            "kind": MEASURED_KIND,
        }
        for record in records
        if isinstance(record.get("report"), dict) and record["report"].get("kind") == MEASURED_KIND
    ]
    return {
        "builtin": [{"selection": key, "label": paper.name} for key, paper in PAPER_SIMULATIONS.items()],
        "measured": measured,
        "selected": None,
    }


@dataclass(frozen=True)
class SoftOptions:
    paper: str
    method: str
    strength: float
    show_loss: bool

    @classmethod
    def parse(cls, body, paper, measured_ids=()):
        obj = object_value(body, "proof simulation")
        validate_keys(obj, {"paper", "method", "strength", "show_loss"}, "soft proof")
        allowed = tuple(PAPER_SIMULATIONS) + tuple(MEASURED_PREFIX + key for key in measured_ids)
        if paper not in allowed:
            paper = "white"
        return cls(
            choice(obj.get("paper", paper), "paper", allowed),
            choice(obj.get("method", "compress"), "tone strategy", PROOF_METHODS),
            number(obj.get("strength", 1), "simulation strength", 0, 1),
            flag(obj.get("show_loss", True), "show clipped colors"),
        )

    @property
    def measured_id(self):
        return self.paper[len(MEASURED_PREFIX):] if self.paper.startswith(MEASURED_PREFIX) else None


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


def _apply_measured_paper(source, paper, method):
    """Map linear source through the measured neutral axis, keeping neutral input neutral."""
    luma = _linear_y(source)
    mapped = paper.response(luma)
    if method == "neutral":
        return np.broadcast_to(mapped[..., None], source.shape)
    # The measured steps define a neutral axis; keep the source's chroma offset instead of
    # scaling it by luminance, so neutral pixels are not tinted by the measured white.
    offset = source - luma[..., None]
    if method == "compress":
        offset = propose_compression(source, luma[..., None], .88) - luma[..., None]
    return np.clip(mapped[..., None] + offset, 0, 1)


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


def _tone_chart(paper, marks=()):
    nominal = np.linspace(0, 1, 256)
    fitted = paper.response(nominal)
    measured = [step for step in paper.patches if step["status"] == "measured"]
    inferred = [step for step in paper.patches if step["status"] == "inferred"]
    series = [
        ("fitted neutral axis", nominal, fitted),
        ("nominal", np.asarray([0., 1.]), np.asarray([0., 1.])),
    ]
    if measured:
        series.append(("measured steps", np.asarray([step["nominal"] for step in measured]),
                       np.asarray([step["response_luminance"] for step in measured])))
    if inferred:
        series.append(("inferred steps", np.asarray([step["nominal"] for step in inferred]),
                       np.asarray([step["response_luminance"] for step in inferred])))
    return chart_png(series, x_label="Nominal gray step", y_label="Printed linear luminance",
                     x_bounds=[0, 1], y_bounds=[0, 1], markers=list(marks))


def soft_proof(sample, options, paper=None):
    source, support = sample.base()
    if paper is not None:
        result = _apply_measured_paper(source, paper, options.method)
        paper_name = paper.label
        paper_target = paper.target()
        measured = paper.summary()
        paper_report = {
            "kind": "measured",
            "measurement": measured,
            "response_anchors": [list(point) for point in paper.anchors],
            "steps": [dict(step) for step in paper.patches],
            "inferred_segments": [dict(segment) for segment in paper.segments],
            "warning": list(paper.warnings),
        }
        pictures_extra = {"tone-response": _tone_chart(paper)}
    else:
        model = PAPER_SIMULATIONS[options.paper]
        result = _apply_paper(source, model, options.method)
        paper_name = model.name
        paper_target = {"white_srgb": list(model.white_rgb), "black_floor": model.black_floor,
                        "chroma_factor": model.color_factor, "response_power": model.gamma,
                        "response_kind": "modelled paper response"}
        paper_report = {"kind": "builtin", "selection": options.paper, "label": model.name}
        pictures_extra = {}
    result = source * (1 - options.strength) + result * options.strength
    diff = chromatic_difference(source, result)
    before, after = _unsupported_overlay(source, support), _unsupported_overlay(result, support)
    pictures = {"side-by-side": png_bytes(resize_report(side_by_side(before, after, 9), 1150))}
    pictures.update(pictures_extra)
    if options.show_loss:
        maximum = max(2, float(np.percentile(diff[support], 98)))
        pictures["color-move"] = png_bytes(resize_report(heatmap(diff, support, 0, maximum)))
    neutral = np.max(source, 2) - np.min(source, 2)
    colored = support & (neutral > .09)
    y_before, y_after = _linear_y(source), _linear_y(result)
    report = {
        "kind": "material-softproof",
        "source": sample.public(),
        "paper": paper_name,
        "paper_selection": options.paper,
        "paper_kind": paper_report["kind"],
        "paper_source": paper_report,
        "paper_target": paper_target,
        "simulation_strength": options.strength,
        "method": options.method,
        "perceptual_displacement_approx": code_statistics(diff[support]),
        "non_neutral_displacement_approx": code_statistics(diff[colored]),
        "luma_before": code_statistics(y_before[support]),
        "luma_after": code_statistics(y_after[support]),
        "big_color_moves_fraction": round(float(np.mean(diff[support] > 8)), 6),
        "note": "Approximate proofs use local sample paper whites and a simple response model; measure the printed card for colour acceptance.",
    }
    if paper is not None and paper.patches:
        bright = [step for step in paper.patches if step["nominal"] >= .6][-3:]
        separation = []
        previous = None
        for step in bright:
            level = float(step["response_luminance"])
            separation.append({
                "patch": step["patch"], "nominal": step["nominal"], "printed_luminance": level,
                "status": step["status"],
                "separation_from_previous": None if previous is None else round(level - previous, 6),
            })
            previous = level
        report["near_white_separation"] = separation
        report["note"] = ("The measured paper uses an ordered monotone neutral axis; uncertain steps are interpolated "
                          "from measured neighbours. Compare with the built-in stock on the same inspection.")
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
    order_flags = _order_suspects(observed)
    for row, suspect in zip(sorted_rows, order_flags, strict=True):
        row["order_suspect"] = suspect
        row["status"] = "inferred" if suspect or row["neutral_gap"] > TINT_LIMIT else "measured"
    warnings = [
        "Two neighboring patches are out of order."
        for a, b in zip(observed, observed[1:]) if b < a - STEP_ORDER_TOLERANCE
    ]
    if any(row["neutral_gap"] > TINT_LIMIT for row in numbers):
        warnings.append("At least one density patch is visibly tinted.")
    flagged = [row["patch"] for row in sorted_rows if row["status"] == "inferred"]
    if flagged:
        warnings.append("Stepped samples marked as inferred: " + ", ".join(str(index) for index in flagged) + ".")
    raw = _unsupported_overlay(pixels.rgb, pixels.support)
    plate = overlay_regions(raw, colors)
    curve = chart_png([
        ("measured", np.asarray(targets), np.asarray(observed)),
        ("nominal", np.asarray([0, 1]), np.asarray([0, 1])),
    ], x_label="Nominal patch value", y_label="Measured linear luminance", x_bounds=[0, 1], y_bounds=[0, 1])
    return {
        "kind": MEASURED_KIND,
        "source": sample.public(),
        "patches": sorted_rows,
        "mean_absolute_error": round(float(np.mean(np.abs(residuals))), 6),
        "order_suspect_patches": flagged,
        "warning": sorted(set(warnings)),
        "note": "Attach this inspection to the paper simulation to use the measured gray-to-white response.",
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
