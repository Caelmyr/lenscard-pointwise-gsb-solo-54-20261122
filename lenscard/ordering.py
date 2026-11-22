"""Recover capture order from two distant wall points and near-object corners.

Every participating capture marks the same two fixed wall points and, where the
object is unobstructed, a corner on one or both foreground objects. Registering
each capture so that its wall points land on a common horizontal baseline removes
hand rotation, roll and zoom. A foreground corner then keeps only parallax, which
changes monotonically with the camera's lateral position. Its sign follows the
camera direction because the object is in front of the wall. Order therefore comes
from registered parallax, never from raw image similarity, so two frames that only
look close because the phone was tilted are not treated as one angle.

A joint weighted fit gives every capture one rail coordinate. If neighbouring
captures are closer than the mark precision they are reported as one angle slot and
the operator keeps one. If they are separable but their direction is not settled by
the marks, both readings are returned for confirmation instead of being forced.
"""

from dataclasses import dataclass, replace
from hashlib import sha256
import math

import numpy as np

from .models import Conflict, Unprocessable, WorkbenchError, number, object_value, validate_keys
from .storage import encode_json

MAX_CAPTURES = 8
MAX_SIGNATURE = 40
MAX_NEAR_SIGNATURES = 2
MIN_FAR_SIGNATURES = 2
MIN_PLACED = 2
MIN_SIGNAL_SIGMAS = 3.0
WEAK_SIGNAL_SIGMAS = 5.0
SAME_ANGLE_SIGMAS = 1.5
CLEAR_ORDER_SIGMAS = 3.0
OUTLIER_IMPROVEMENT = 1.5
MAX_OUTLIERS = 3
MAX_ALTERNATIVES = 4
MODEL_FLOOR = 0.003
DEFAULT_SIGMA_FRACTION = 0.0025
MIN_BASELINE_FRACTION = 0.02
WARN_BASELINE_FRACTION = 0.15
MIN_ANGLE_STEP = 0.01
ROLL_WARNING_DEGREES = 20.0


@dataclass(frozen=True)
class Capture:
    view: str
    title: str
    width: int
    height: int

    @property
    def diagonal(self):
        return math.hypot(self.width, self.height)

    def pixel(self, x, y):
        return x * (self.width - 1), y * (self.height - 1)


@dataclass(frozen=True)
class Mark:
    view: str
    kind: str
    signature: str
    x: float
    y: float


@dataclass(frozen=True)
class Evidence:
    captures: tuple[Capture, ...]
    marks: tuple[Mark, ...]
    anchors: tuple[str | None, str | None] = (None, None)
    sigma_px: float | None = None

    @classmethod
    def parse(cls, data, views):
        obj = object_value(data, "capture order")
        validate_keys(obj, {"marks", "anchors", "sigma_px"}, "capture order")
        known = {item["id"]: item for item in views}
        raw_marks = obj.get("marks")
        if not isinstance(raw_marks, list) or not raw_marks:
            raise Unprocessable("Mark both wall points in the captures you want to order.", field="marks")
        if len(raw_marks) > MAX_CAPTURES * (MAX_NEAR_SIGNATURES + MIN_FAR_SIGNATURES):
            raise Unprocessable("There are too many marks for one card.", field="marks")
        marks, keys = [], set()
        for item in raw_marks:
            entry = object_value(item, "mark")
            validate_keys(entry, {"view", "kind", "signature", "point"}, "mark")
            view = entry.get("view")
            if view not in known:
                raise WorkbenchError("A mark names a view that is not in this card.", field="marks")
            kind = entry.get("kind")
            if kind not in ("far", "near"):
                raise WorkbenchError("Mark each point as a wall point or a near-object corner.", field="marks")
            signature = _signature(entry.get("signature"))
            x, y = _point(entry.get("point"))
            key = (view, kind, signature)
            if key in keys:
                raise WorkbenchError("A capture has the same mark twice.", field="marks")
            keys.add(key)
            marks.append(Mark(view, kind, signature, x, y))
        far_names = sorted({mark.signature for mark in marks if mark.kind == "far"})
        near_names = sorted({mark.signature for mark in marks if mark.kind == "near"})
        if len(far_names) != MIN_FAR_SIGNATURES:
            raise Unprocessable("Mark the same two distant wall points in the captures, with distinct labels.",
                                field="marks")
        if not 1 <= len(near_names) <= MAX_NEAR_SIGNATURES:
            raise Unprocessable("Mark a corner on one or two foreground objects.", field="marks")
        marked = {mark.view for mark in marks}
        if len(marked) > MAX_CAPTURES:
            raise Unprocessable(f"Mark at most {MAX_CAPTURES} captures at a time.", field="marks")
        for view in marked:
            title = known[view]["title"]
            present = [mark for mark in marks if mark.view == view]
            if not any(mark.kind == "near" for mark in present):
                raise Unprocessable(f"Mark a near-object corner in {title} as well.", field="marks")
            for name in far_names:
                if not any(mark.kind == "far" and mark.signature == name for mark in present):
                    raise Unprocessable(f"Mark both wall points in {title} so it can be registered.", field="marks")
        captures = tuple(
            Capture(view, known[view]["title"], known[view]["width"], known[view]["height"])
            for view in sorted(marked, key=lambda key: (known[key]["title"].casefold(), key))
        )
        sigma = obj.get("sigma_px")
        return cls(captures, tuple(marks), _anchors(obj.get("anchors"), known, marks),
                   None if sigma is None else number(sigma, "mark precision in pixels", 0.25, 250))


def _signature(value):
    if not isinstance(value, str) or not value.strip():
        raise WorkbenchError("Each mark needs a label, so the same point can be followed across captures.",
                             field="marks")
    clean = " ".join(value.split())
    if len(clean) > MAX_SIGNATURE:
        raise WorkbenchError(f"Mark labels must be at most {MAX_SIGNATURE} characters.", field="marks")
    return clean.casefold()


def _point(value):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise WorkbenchError("A mark needs x and y.", field="marks")
    try:
        x, y = (float(item) for item in value)
    except (TypeError, ValueError) as error:
        raise WorkbenchError("Mark coordinates must be numbers.", field="marks") from error
    if not (math.isfinite(x) and math.isfinite(y) and 0 <= x <= 1 and 0 <= y <= 1):
        raise WorkbenchError("Mark coordinates must lie between 0 and 1.", field="marks")
    return x, y


def _anchors(value, known, marks):
    if value is None:
        return (None, None)
    obj = object_value(value, "anchors")
    validate_keys(obj, {"left", "right"}, "anchors")
    left, right = obj.get("left"), obj.get("right")
    if (left is None) != (right is None):
        raise Unprocessable("Choose both the remembered left and right shots, or neither.", field="anchors")
    if left is None:
        return (None, None)
    for key in (left, right):
        if key not in known:
            raise WorkbenchError("An anchor names a view that is not in this card.", field="anchors")
    if left == right:
        raise Unprocessable("The left and right shots must be different captures.", field="anchors")
    marked = {mark.view for mark in marks}
    if not {left, right} <= marked:
        raise Unprocessable("Both remembered shots need wall points and a near corner before they can anchor "
                            "the direction.", field="anchors")
    return (left, right)


@dataclass
class Observation:
    t: float
    sigma: float


@dataclass(frozen=True)
class Line:
    intercept: float
    slope: float
    slope_sigma: float
    chi2: float
    dof: int


def _weighted_line(xs, ys, sigmas):
    weights = 1.0 / np.square(sigmas)
    total = float(weights.sum())
    sx = float(np.dot(weights, xs))
    sy = float(np.dot(weights, ys))
    sxx = float(np.dot(weights, xs * xs))
    sxy = float(np.dot(weights, xs * ys))
    denominator = total * sxx - sx * sx
    if abs(denominator) < 1e-12:
        return None
    slope = (total * sxy - sx * sy) / denominator
    intercept = (sy - slope * sx) / total
    residual = ys - intercept - slope * xs
    return Line(intercept, slope, math.sqrt(total / denominator),
                float(np.dot(weights, residual * residual)), max(0, len(xs) - 2))


def _rotate(x, y, angle):
    cosine, sine = math.cos(angle), math.sin(angle)
    return cosine * x + sine * y, -sine * x + cosine * y


def _register(capture, far_points, near_points, sigma_px):
    """Wall-registered coordinates: the two wall points on a horizontal unit baseline."""
    (first, second) = far_points
    vector = (second[0] - first[0], second[1] - first[1])
    baseline = math.hypot(*vector)
    if baseline <= 0:
        raise Unprocessable("The two wall points coincide.", field="marks")
    if baseline < MIN_BASELINE_FRACTION * capture.diagonal:
        raise Unprocessable("The two wall points are too close in frame to register the capture.", field="marks")
    middle = ((first[0] + second[0]) / 2, (first[1] + second[1]) / 2)
    angle = math.atan2(vector[1], vector[0])
    observations = {}
    for name, point in near_points.items():
        offset = (point[0] - middle[0], point[1] - middle[1])
        across, _ = _rotate(offset[0], offset[1], angle)
        lever = math.hypot(*offset) / baseline
        sigma = sigma_px / baseline * math.sqrt(1.5 + 2 * lever * lever)
        observations[name] = Observation(across / baseline, sigma)
    return observations, angle, baseline


def _signal(values, sigmas):
    return (max(values) - min(values)) / max(float(np.median(sigmas)), 1e-9)


def _normalise(values):
    centered = values - values.mean()
    spread = float(centered.std())
    return centered / spread if spread > 1e-9 else np.zeros_like(values)


def _initial_scores(observations, names, captures):
    scores = np.zeros(len(captures))
    for index, capture in enumerate(captures):
        weighted = [(obs.t, obs.sigma) for name in names
                    if (obs := observations[capture.view].get(name)) is not None]
        if weighted:
            scores[index] = (sum(value / sigma ** 2 for value, sigma in weighted)
                             / sum(1 / sigma ** 2 for _, sigma in weighted))
    return _normalise(scores)


def _fit_lines(observations, names, order, scores):
    fits = {}
    for name in names:
        rows = [(float(scores[order.index(view)]), obs.t, obs.sigma)
                for view in order if (obs := observations[view].get(name)) is not None]
        if len(rows) < 2:
            continue
        fit = _weighted_line(np.asarray([row[0] for row in rows]),
                             np.asarray([row[1] for row in rows]),
                             np.asarray([row[2] for row in rows]))
        if fit is not None and abs(fit.slope) > 1e-12:
            fits[name] = fit
    return fits


def _positions(observations, fits, order):
    positions = np.zeros(len(order))
    for index, view in enumerate(order):
        weights, values = [], []
        for name, fit in fits.items():
            obs = observations[view].get(name)
            if obs is not None and fit.slope:
                weights.append(fit.slope ** 2 / obs.sigma ** 2)
                values.append((obs.t - fit.intercept) / fit.slope)
        if weights:
            positions[index] = float(np.average(values, weights=weights))
    return _normalise(positions)


def _solve(observations, names, captures):
    order = [capture.view for capture in captures]
    scores = _initial_scores(observations, names, captures)
    for _ in range(60):
        fits = _fit_lines(observations, names, order, scores)
        if not fits:
            return None, {}, None
        candidate = _positions(observations, fits, order)
        if np.allclose(candidate, scores, atol=1e-10):
            return candidate, fits, order
        scores = candidate
    return scores, _fit_lines(observations, names, order, scores), order


def _cost(observations, names, captures):
    """Total reduced chi-square of the joint fit, or None when it cannot be solved."""
    scores, fits, _ = _solve(observations, names, captures)
    if scores is None or not fits:
        return None
    dof = sum(fit.dof for fit in fits.values())
    chi2 = sum(fit.chi2 for fit in fits.values())
    return chi2 / dof if dof else 0.0


def _removable(observations, view, name):
    """A mark can be dropped only while its point stays measurable in two captures."""
    return sum(1 for other in observations if name in observations[other]) > 2


def _titles(isolated, captures):
    return [capture.title for capture in captures if capture.view in isolated]


def recover(evidence: Evidence) -> dict:
    views = {capture.view: capture for capture in evidence.captures}
    far_names = sorted({mark.signature for mark in evidence.marks if mark.kind == "far"})
    near_names = sorted({mark.signature for mark in evidence.marks if mark.kind == "near"})
    by_capture = {capture.view: {} for capture in evidence.captures}
    for mark in evidence.marks:
        by_capture[mark.view][(mark.kind, mark.signature)] = views[mark.view].pixel(mark.x, mark.y)

    observations, rolls, baselines = {}, [], []
    for capture in evidence.captures:
        sigma = evidence.sigma_px or max(3.0, DEFAULT_SIGMA_FRACTION * capture.diagonal)
        far = [by_capture[capture.view][("far", name)] for name in far_names]
        near = {name: by_capture[capture.view][("near", name)]
                for name in near_names if ("near", name) in by_capture[capture.view]}
        frame, angle, baseline = _register(capture, far, near, sigma)
        observations[capture.view] = frame
        rolls.append(math.degrees(angle))
        baselines.append(baseline)

    warnings = []
    roll_span = _roll_span(rolls)
    if roll_span > ROLL_WARNING_DEGREES:
        warnings.append(f"Hand rotation spans {roll_span:.0f}° across the marked captures. Registration "
                        "removes it before comparing parallax.")
    baseline_ratio = max(baselines) / max(1e-9, min(baselines))
    if baseline_ratio > 1.15:
        warnings.append(f"Wall-point separation changes by {100 * (baseline_ratio - 1):.0f}% between "
                        "captures; that change is used to cancel zoom.")
    if float(np.median(baselines)) < WARN_BASELINE_FRACTION * float(np.median(
            [capture.diagonal for capture in evidence.captures])):
        warnings.append("The two wall points are close in frame, so the registration baseline is weak.")

    names, signal = [], {}
    for name in near_names:
        values = [observations[capture.view][name].t for capture in evidence.captures
                  if name in observations[capture.view]]
        sigmas = [observations[capture.view][name].sigma for capture in evidence.captures
                  if name in observations[capture.view]]
        if len(values) < 2:
            warnings.append(f"Near corner {name!r} is marked in only one capture; it cannot order the set.")
            continue
        signal[name] = _signal(values, sigmas)
        missing = _titles({capture.view for capture in evidence.captures
                           if name not in observations[capture.view]} - {None}, evidence.captures)
        if signal[name] < WEAK_SIGNAL_SIGMAS:
            warnings.append(
                f"Near corner {name!r} moves only {signal[name]:.1f} mark deviations across the set, so its "
                "ordering evidence is weak."
                + (f" It is not marked in {', '.join(missing)}." if missing else ""))
        names.append(name)
    usable = [name for name in names if signal[name] >= MIN_SIGNAL_SIGMAS]
    if not usable:
        return _insufficient(evidence, observations, signal, warnings,
                             "No marked near-object corner moves enough between the captures to order them. "
                             f"All {len(evidence.captures)} frames read as the same angle. Keep the sharpest "
                             "one and mark other angles; if you did move between shots, mark a closer corner.")

    scores, fits, order = _solve(observations, usable, evidence.captures)
    if scores is None or not fits:
        return _insufficient(evidence, observations, signal, warnings,
                             "The near-point parallax could not be fitted. Recheck the marked corners.")

    # A single mis-marked corner would otherwise pull its own capture into a false
    # slot. Remove one mark at a time only while the shared residual drops clearly.
    rejected = []
    baseline = _cost(observations, usable, evidence.captures)
    for _ in range(MAX_OUTLIERS):
        if baseline is None or baseline <= OUTLIER_IMPROVEMENT:
            break
        trials = []
        for view in [capture.view for capture in evidence.captures]:
            for name in list(observations[view]):
                if not _removable(observations, view, name):
                    continue
                saved = observations[view].pop(name)
                cost = _cost(observations, usable, evidence.captures)
                observations[view][name] = saved
                if cost is not None:
                    trials.append((cost, view, name))
        if not trials:
            break
        cost, view, name = min(trials)
        if baseline - cost < OUTLIER_IMPROVEMENT:
            break
        observations[view].pop(name)
        rejected.append((view, name))
        baseline = cost
        scores, fits, order = _solve(observations, usable, evidence.captures)
        if scores is None or not fits:
            break
    usable = [name for name in usable
              if sum(1 for view in observations if name in observations[view]) >= 2]
    unplaced = _unplaced(observations, evidence.captures)
    if len(evidence.captures) - len(unplaced) < MIN_PLACED or scores is None:
        return _insufficient(evidence, observations, signal, warnings,
                             "The marked captures disagree too much to share one rail fit. "
                             "Recheck the wall points and the near corners.", unplaced)
    if rejected:
        labels = ", ".join(f"{views[view].title} / {name}" for view, name in rejected)
        warnings.append(f"These marks disagree with the other captures and were ignored: {labels}.")
    if unplaced:
        warnings.append(f"No reliable near-corner evidence remains for {', '.join(_titles(set(unplaced), evidence.captures))}.")

    # Physical direction: a foreground corner moves opposite the camera, so with the
    # registered wall baseline pointing right, increasing rail position means a
    # decreasing parallax value. A positive joint slope therefore contradicts the
    # foreground rule and the orientation needs flipping.
    significance = sum(fit.slope / max(fit.slope_sigma, 1e-12) for fit in fits.values())
    left, right = evidence.anchors
    if left is not None and (left in unplaced or right in unplaced):
        raise Unprocessable("Both remembered anchor shots need enough marks to be placed.", field="anchors")
    direction_warning = None
    ordered_by_anchor = False
    if left is not None:
        index = {capture.view: position for position, capture in enumerate(evidence.captures)}
        if scores[index[left]] > scores[index[right]]:
            scores = -scores
            significance = -significance
        ordered_by_anchor = True
    elif significance > 0:
        scores = -scores
        significance = -significance
    physics_forward = significance < 0
    if ordered_by_anchor and not physics_forward:
        direction_warning = ("The near-point parallax runs opposite to the foreground assumption, yet your "
                             "remembered left and right shots put the order the other way. Both readings are "
                             "shown; check the two anchor photos.")
        warnings.append(direction_warning)
    fits = {name: replace(fit, slope=-abs(fit.slope) if physics_forward else abs(fit.slope))
            for name, fit in fits.items()}

    spread = float(scores.max() - scores.min())
    if spread < 1e-9:
        return _insufficient(evidence, observations, signal, warnings,
                             "Every marked capture sits at the same rail position. Keep one and import other angles.")
    display = (scores - scores.min()) / spread
    sigmas, rows = np.zeros(len(evidence.captures)), []
    for position, capture in enumerate(evidence.captures):
        weights = [fit.slope ** 2 / observations[capture.view][name].sigma ** 2
                   for name, fit in fits.items() if name in observations[capture.view]]
        base = 1.0 / math.sqrt(sum(weights)) if weights else 1.0
        inflation = max([math.sqrt(fit.chi2 / fit.dof) for fit in fits.values() if fit.dof > 0] or [1.0])
        sigmas[position] = base / spread * max(1.0, inflation)
        roll = _roll_degrees(rolls[position])
        rows.append({
            "view": capture.view, "title": capture.title,
            "position": round(float(display[position]), 5),
            "sigma": round(float(math.hypot(sigmas[position], MODEL_FLOOR)), 5),
            "roll_degrees": round(roll, 2),
            "visible": [name for name in usable if name in observations[capture.view]],
            "parallax": {name: round(obs.t, 5) for name, obs in observations[capture.view].items()
                         if name in usable},
            "placed": capture.view not in unplaced,
        })
    sigmas = np.sqrt(np.square(sigmas) + MODEL_FLOOR ** 2)

    kinds = _group(evidence, display, sigmas, warnings)
    if not kinds["slots"]:
        return _insufficient(evidence, observations, signal, warnings,
                             "Every marked capture sits at the same rail position. Keep one and import other angles.")
    if len(kinds["slots"]) < 2:
        return {
            "kind": "capture-order", "status": "same-angle",
            "reason": ("All marked captures read as one angle. Their registered near-point parallax differs by "
                       "less than the mark precision, so no left-to-right order exists. Keep one and import "
                       "other angles."),
            "captures": rows, "slots": kinds["slots"], "groups": kinds["groups"], "orders": [],
            "near_points": _near_rows(near_names, signal, fits, rows),
            "far_points": far_names, "warnings": warnings, "unplaced": unplaced,
            "direction": _direction(evidence, physics_forward, direction_warning),
            "rejected_marks": [{"view": view, "title": views[view].title, "name": name}
                               for view, name in rejected],
        }

    orders = _variants(evidence, kinds, direction_warning)
    return {
        "kind": "capture-order", "status": "ready",
        "reason": None,
        "captures": rows,
        "slots": kinds["slots"],
        "groups": kinds["groups"],
        "orders": orders,
        "uncertain_pairs": kinds["uncertain"],
        "near_points": _near_rows(near_names, signal, fits, rows),
        "far_points": far_names,
        "warnings": warnings,
        "unplaced": unplaced,
        "direction": _direction(evidence, physics_forward, direction_warning),
        "rejected_marks": [{"view": view, "title": views[view].title, "name": name}
                           for view, name in rejected],
    }


def _unplaced(observations, captures):
    return [capture.view for capture in captures if not observations[capture.view]]


def _roll_span(rolls):
    wrapped = [(roll + 90) % 180 - 90 for roll in rolls]
    return max(wrapped) - min(wrapped)


def _roll_degrees(roll):
    return (roll + 90) % 180 - 90


def _near_rows(names, signal, fits, rows):
    return [{
        "name": name,
        "signal_sigmas": round(signal[name], 2),
        "slope": round(float(fits[name].slope), 6),
        "slope_sigma": round(float(fits[name].slope_sigma), 6),
        "captures": sum(1 for row in rows if name in row["visible"]),
        "reduced_chi2": round(fits[name].chi2 / fits[name].dof, 2) if fits[name].dof > 0 else None,
    } for name in names if name in fits]


def _group(evidence, positions, sigmas, warnings):
    sequence = sorted(range(len(evidence.captures)), key=lambda index: positions[index])
    slots, uncertain = [], []
    # A gap between neighbouring frames is compared with the combined mark precision.
    # It never looks at how similar the whole images are, so a turned hand cannot pull
    # two different positions into one slot.
    for index in sequence:
        if not slots:
            slots.append([index])
            continue
        previous = slots[-1][-1]
        gap = float(positions[index] - positions[previous])
        sigma = float(math.hypot(sigmas[index], sigmas[previous]))
        ratio = gap / max(sigma, 1e-9)
        if ratio <= SAME_ANGLE_SIGMAS:
            slots[-1].append(index)
        else:
            slots.append([index])
            if ratio <= CLEAR_ORDER_SIGMAS:
                uncertain.append((len(slots) - 2, len(slots) - 1, ratio))
    if any(len(slot) > 1 for slot in slots):
        warnings.append("Captures that share a rail position read as one angle; keep one from each such group. "
                        "This uses registered parallax and ignores how similar the whole frames look.")
    rows = []
    for slot in slots:
        members = [evidence.captures[index] for index in slot]
        rows.append({
            "views": [capture.view for capture in members],
            "titles": [capture.title for capture in members],
            "position": round(float(np.mean([positions[index] for index in slot])), 5),
            "spread": round(float(np.ptp([positions[index] for index in slot])), 5),
            "same_angle": len(slot) > 1,
        })
    uncertain.sort(key=lambda item: item[2])
    return {
        "slots": rows,
        "groups": [row for row in rows if row["same_angle"]],
        "uncertain": [{
            "views": [rows[first]["views"][0], rows[second]["views"][0]],
            "titles": [rows[first]["titles"][0], rows[second]["titles"][0]],
            "separation_sigmas": round(float(ratio), 2),
        } for first, second, ratio in uncertain],
    }


def _variants(evidence, kinds, direction_warning):
    base = [slot["views"] for slot in kinds["slots"]]
    variants = [(base, "Best fit of the marked parallax.", True)]
    for pair in kinds["uncertain"][:MAX_ALTERNATIVES]:
        first = next((index for index, slot in enumerate(base) if pair["views"][0] in slot), None)
        second = next((index for index, slot in enumerate(base) if pair["views"][1] in slot), None)
        if first is None or second is None or first == second:
            continue
        swapped = [list(slot) for slot in base]
        swapped[first], swapped[second] = swapped[second], swapped[first]
        variants.append((swapped,
                         f"{pair['titles'][0]} and {pair['titles'][1]} are {pair['separation_sigmas']:.1f} mark "
                         "deviations apart. Check these two before choosing.", False))
    if direction_warning:
        variants.append(([list(slot) for slot in reversed(base)], direction_warning, False))
    elif evidence.anchors == (None, None):
        variants.append(([list(slot) for slot in reversed(base)],
                         "Opposite direction, in case the marked corners are not in front of the wall.", False))
    orders, seen = [], set()
    for slots, reason, preferred in variants:
        flattened = tuple(view for slot in slots for view in slot)
        if flattened in seen:
            continue
        seen.add(flattened)
        orders.append({
            "slots": [list(slot) for slot in slots],
            "views": list(flattened),
            "reason": reason,
            "preferred": bool(preferred and not orders),
        })
    orders[0]["preferred"] = True
    return orders


def _direction(evidence, physics_forward, warning):
    left, right = evidence.anchors
    if warning:
        source = "two readings"
    elif left is not None:
        source = "remembered left and right shots"
    else:
        source = "foreground assumption"
    if left is None:
        note = ("No anchor photos were given, so the direction comes from the foreground assumption: the marked "
                "corners sit in front of the wall and therefore move opposite the camera.")
    else:
        note = ("The anchor photos agree with the foreground rule: the marked corners sit in front of the wall."
                if physics_forward else
                "The anchor photos decide the direction, although the marked parallax looks less like the "
                "expected foreground motion. Verify the two anchor photos.")
    return {"source": source, "foreground_rule_agrees": physics_forward, "note": note}


def _insufficient(evidence, observations, signal, warnings, reason, unplaced=None):
    unplaced = unplaced if unplaced is not None else [capture.view for capture in evidence.captures]
    return {
        "kind": "capture-order", "status": "insufficient", "reason": reason,
        "captures": [{
            "view": capture.view, "title": capture.title, "position": None, "sigma": None,
            "roll_degrees": None, "visible": sorted(observations[capture.view]),
            "parallax": {name: round(obs.t, 5) for name, obs in observations[capture.view].items()},
            "placed": False,
        } for capture in evidence.captures],
        "slots": [], "groups": [], "orders": [], "uncertain_pairs": [],
        "near_points": [{"name": name, "signal_sigmas": round(signal.get(name, 0.0), 2)}
                        for name in sorted(signal)],
        "far_points": sorted({mark.signature for mark in evidence.marks if mark.kind == "far"}),
        "warnings": warnings, "unplaced": unplaced,
        "direction": {"source": "none", "foreground_rule_agrees": None, "note": None},
        "rejected_marks": [],
    }


def confirm(evidence, report, order, disabled):
    """Validate one displayed reading; returns the kept view ids in left-to-right order."""
    if report["status"] != "ready":
        raise Conflict("There is no confirmed left-to-right reading to apply yet.",
                       details={"status": report["status"]})
    known = {capture.view for capture in evidence.captures}
    if not isinstance(order, list) or not all(isinstance(view, str) for view in order):
        raise Unprocessable("Choose one of the listed readings.", field="order")
    if len(set(order)) != len(order):
        raise Conflict("The confirmed order repeats a capture.", details={"order": order})
    if not set(order) <= known:
        raise Conflict("The confirmed order names unmarked captures.", details={"order": order})
    disabled = disabled or []
    if not isinstance(disabled, list) or not all(isinstance(view, str) for view in disabled):
        raise Unprocessable("Choose which duplicate captures to leave out.", field="disabled")
    if len(set(disabled)) != len(disabled) or set(order) & set(disabled):
        raise Conflict("A capture cannot be both kept and left out.",
                       details={"order": order, "disabled": disabled})
    # Left-out captures need not have marks: an original that was never marked is
    # left out the same way. Storage checks every id against the card's views.
    if set(report["unplaced"]) - set(disabled):
        raise Conflict("Leave out the captures that could not be placed.",
                       details={"unplaced": report["unplaced"]})
    unmarked = {item["view"] for item in report.get("unmarked", [])}
    if unmarked - set(disabled):
        raise Conflict("Leave out the originals that were not marked.",
                       details={"unmarked": sorted(unmarked)})
    if len(order) != len(report["slots"]):
        raise Conflict("Choose one capture from every angle slot before applying the order.",
                       details={"expected": len(report["slots"]), "received": len(order)})
    for group in report["groups"]:
        kept = [view for view in order if view in group["views"]]
        if len(kept) != 1:
            raise Conflict("Keep exactly one capture from each group that shares one angle.",
                           details={"group": group["views"]})
        if not set(group["views"]) - set(kept) <= set(disabled):
            raise Conflict("Leave out the duplicate captures you did not keep.",
                           details={"group": group["views"]})
    # Each listed reading is one left-to-right sequence of the same angle slots; a
    # close pair may be swapped or the whole direction reversed. Validate the chosen
    # order against each reading's own slot order rather than the best-fit one.
    for variant in report["orders"]:
        slot_of_reading = {}
        for index, slot in enumerate(variant["slots"]):
            for view in slot:
                slot_of_reading[view] = index
        sequence = [slot_of_reading[view] for view in order if view in slot_of_reading]
        if len(sequence) == len(order) and all(
                later > earlier for earlier, later in zip(sequence, sequence[1:])):
            return list(order)
    raise Conflict("This order is not one of the readings supported by the marks. "
                   "Recover the order again and confirm one of the listed readings.",
                   details={"supported": report["orders"]})


def angle_list(records, positions, span):
    """Angles for the kept views, in the given left-to-right order.

    Explicit positions win. Otherwise the kept views are spread evenly over the
    requested half-span (60° when omitted), centred on zero, so the confirmed
    left-to-right sequence alone decides the list and a rerun never keeps a stale
    order from an earlier attempt.
    """
    count = len(records)
    if count < 2:
        raise Unprocessable("An angle list needs at least two kept captures.", field="order")
    if positions is not None:
        if not isinstance(positions, list) or len(positions) != count:
            raise Unprocessable("Give one angle for each kept capture.", field="positions")
        angles = [number(value, "angle", -85, 85) for value in positions]
    else:
        half = 60.0 if span is None else number(span, "angle half-span", 1, 85)
        step = 2 * half / (count - 1)
        angles = [-half + step * index for index in range(count)]
    for left, right in zip(angles, angles[1:]):
        if right - left < MIN_ANGLE_STEP:
            raise Unprocessable("Use angles at least 0.01° apart in left-to-right order.", field="positions")
    return [round(float(value), 6) for value in angles]


def order_digest(report):
    """Stable fingerprint of a recovered reading, so applying cannot drift from what was shown."""
    return sha256(encode_json(report).encode()).hexdigest()
