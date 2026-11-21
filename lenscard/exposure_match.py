"""Match angle exposure from shared matte-wall references marked on the originals.

Each view can contribute one or more rectangles of the same neutral wall.  The
captures are registered into one frame with the constrained rail model, so a
wall strip visible in two neighbouring views becomes a measurable pair even
when no single view shows the whole wall.  Every usable pair votes for the
relative exposure difference (in stops) between its two views; a robust
weighted least-squares pass combines the votes.  Pairs whose vote is
inconsistent with the others are flagged instead of trusted, and views that
only take part in flagged pairs keep their current manual exposure value.
"""
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO

import numpy as np
from PIL import Image

from .alignment import RailAlignment, Motion, support_after_motion
from .framing import reference_view
from .imaging import linear_luminance, linear_to_srgb, processing_size, read_pixels
from .models import (
    Region, Unprocessable, number, object_value, ordered_views, require_id, validate_keys,
)
from .storage import encode_json
from .visual import chart_png, overlay_regions, png_bytes, resize_report, rounding


MIN_MARK_PIXELS = 120
MIN_OVERLAP_PIXELS = 64
MAX_MARKS_PER_VIEW = 4
# A neutral matte wall must stay close to grey after the camera's white balance.
NEUTRAL_GAP_LIMIT = 0.09
CLIP_FRACTION_LIMIT = 0.04
# Reference votes disagreeing by more than this many stops fail the consistency pass.
DEFAULT_DISCORD_EV = 0.18
# A shared physical wall patch must expose nearly the same per-channel exposure
# ratio; a larger channel spread means one rectangle is not the same neutral wall.
COLOR_SPREAD_LIMIT = 0.12
# Reject wall samples that are effectively black or blown out.
LUMA_FLOOR = 0.004
LUMA_CEIL = 0.96


@dataclass(frozen=True)
class MarkSpec:
    view_id: str
    region: Region

    @classmethod
    def parse(cls, entry):
        obj = object_value(entry, "wall reference")
        validate_keys(obj, {"view", "region"}, "wall reference")
        return cls(require_id(obj.get("view"), "view"), Region.parse(obj.get("region")))


@dataclass
class RegisteredView:
    record: dict
    rgb: np.ndarray
    support: np.ndarray
    motion: Motion
    marks: list[MarkSpec]


@dataclass
class MarkReading:
    spec: MarkSpec
    mask: np.ndarray
    rgb: np.ndarray
    luma: np.ndarray
    luma_frame: np.ndarray
    rgb_frame: np.ndarray
    median_luma: float
    median_rgb: np.ndarray
    neutral_gap: float
    clip_fraction: float
    sample_count: int
    warnings: list


def parse_marks(body):
    rows = body.get("marks")
    if not isinstance(rows, list) or not rows:
        raise Unprocessable("Mark at least one matte-wall rectangle on two angle originals.")
    if len(rows) > MAX_MARKS_PER_VIEW * 8:
        raise Unprocessable("Too many wall references; keep to a few rectangles per view.")
    marks = [MarkSpec.parse(row) for row in rows]
    counts: dict[str, int] = {}
    for mark in marks:
        counts[mark.view_id] = counts.get(mark.view_id, 0) + 1
        if counts[mark.view_id] > MAX_MARKS_PER_VIEW:
            raise Unprocessable("Mark at most four wall rectangles on any single original.")
    return marks


def register_raw_views(library, records, fit_motion):
    """Align enabled originals in a common frame without applying any exposure EV."""
    views = ordered_views(records)
    size = processing_size(views)
    picked = reference_view(views)
    reference = read_pixels(library, picked, size)
    register = RailAlignment(reference, "breathing" if fit_motion else "none")
    aligned = []
    for item in views:
        capture = reference if item["id"] == picked["id"] else read_pixels(library, item, size)
        result = register.align(capture, reference=item["id"] == picked["id"])
        aligned.append(RegisteredView(item, result.rgb, result.support, result.motion, []))
    return aligned, size


def _mark_mask(shape, region, motion):
    height, width = shape
    canvas = np.zeros(shape, dtype=np.float32)
    left, top, right, bottom = region.pixels(width, height)
    canvas[top:bottom, left:right] = 1.0
    warped = support_after_motion(canvas > 0, motion)
    return warped


def read_marks(registered, marks):
    by_id = {view.record["id"]: view for view in registered}
    unknown = sorted({mark.view_id for mark in marks} - set(by_id))
    if unknown:
        raise Unprocessable("Wall references must use enabled views of this card.", details={"views": unknown})
    readings: dict[str, list[MarkReading]] = {}
    for view in registered:
        view.marks = [mark for mark in marks if mark.view_id == view.record["id"]]
    for view in registered:
        for spec in view.marks:
            region_mask = _mark_mask(view.support.shape, spec.region, view.motion)
            mask = region_mask & view.support
            count = int(mask.sum())
            warnings = []
            if count < MIN_MARK_PIXELS:
                warnings.append("Too few supported wall pixels after registration.")
                readings.setdefault(spec.view_id, []).append(
                    MarkReading(spec, mask, np.empty(0), np.empty(0), np.empty(0), np.empty(0),
                                np.nan, np.empty(0), np.nan, np.nan, count, warnings))
                continue
            luma_frame = linear_luminance(view.rgb)
            rgb = view.rgb[mask]
            luma = luma_frame[mask]
            median_rgb = np.median(rgb, axis=0)
            median_luma = float(np.median(luma))
            neutral_gap = float(np.max(median_rgb) - np.min(median_rgb))
            clip_fraction = float(np.mean(np.any(rgb > 0.96, axis=-1)))
            if neutral_gap > NEUTRAL_GAP_LIMIT:
                warnings.append("The reference looks tinted rather than neutral grey.")
            if clip_fraction > CLIP_FRACTION_LIMIT:
                warnings.append("Part of the reference is clipped in this original.")
            if not LUMA_FLOOR <= median_luma <= LUMA_CEIL:
                warnings.append("The reference is too close to black or white to compare exposure.")
            readings.setdefault(spec.view_id, []).append(
                MarkReading(spec, mask, rgb, luma, luma_frame, view.rgb, median_luma, median_rgb,
                            neutral_gap, clip_fraction, count, warnings))
    usable = [item for group in readings.values() for item in group if np.isfinite(item.median_luma)]
    if len({item.spec.view_id for item in usable}) < 2:
        raise Unprocessable(
            "At least two enabled views need a usable wall reference to compare exposure.",
            details={"usable_views": len({item.spec.view_id for item in usable})},
        )
    return readings


def _vote_weight(overlap, sigma):
    # Wide, low-noise overlaps dominate; a floor keeps small strips informative.
    quality = min(1.0, overlap / 900.0)
    spread = 1.0 / max(sigma, 0.012) ** 2
    return quality * spread


def build_edges(readings):
    items = [item for group in readings.values() for item in group if np.isfinite(item.median_luma)]
    edges = []
    for index, first in enumerate(items):
        for second in items[index + 1:]:
            if first.spec.view_id == second.spec.view_id:
                continue
            overlap_mask = first.mask & second.mask
            overlap = int(overlap_mask.sum())
            if overlap < MIN_OVERLAP_PIXELS:
                continue
            # Compare the same physical pixels through each view's warped frame;
            # every mask lives in the registered full-frame geometry.
            safe_a = np.maximum(first.luma_frame[overlap_mask], 1e-6)
            safe_b = np.maximum(second.luma_frame[overlap_mask], 1e-6)
            pair_log = np.log2(safe_a / safe_b)
            delta_ev = float(np.median(pair_log))
            sigma = float(1.4826 * np.median(np.abs(pair_log - np.median(pair_log))))
            if not np.isfinite(sigma):
                sigma = 0.0
            # Neutral wall: the per-channel exposure ratios must agree. A wide
            # channel spread reveals a tinted or non-matching reference rectangle.
            channel_log = np.log2(
                np.maximum(first.rgb_frame[overlap_mask], 1e-6) /
                np.maximum(second.rgb_frame[overlap_mask], 1e-6)
            )
            channel_medians = np.median(channel_log, axis=0)
            color_spread = float(np.max(channel_medians) - np.min(channel_medians))
            edge_warnings = sorted(set(first.warnings + second.warnings))
            if color_spread > COLOR_SPREAD_LIMIT:
                edge_warnings.append(
                    f"Channel balance differs by {color_spread:.2f} EV; the two rectangles may not be the same neutral wall."
                )
            weight = _vote_weight(overlap, sigma)
            edges.append({
                "left": first.spec.view_id,
                "right": second.spec.view_id,
                "overlap_pixels": overlap,
                "delta_ev": rounding(delta_ev),
                "scatter_ev": rounding(sigma),
                "channel_spread_ev": rounding(color_spread),
                "weight": rounding(weight),
                "left_median": rounding(first.median_luma),
                "right_median": rounding(second.median_luma),
                "warnings": edge_warnings,
                "_weight": weight,
                "_delta": delta_ev,
            })
    if not edges:
        raise Unprocessable(
            "No two marked wall rectangles share enough registered pixels. "
            "Mark the same wall strip on neighbouring views so the marks overlap after alignment."
        )
    return edges


def _connected_groups(node_ids, edges):
    adjacency = {node: set() for node in node_ids}
    for edge in edges:
        adjacency[edge["left"]].add(edge["right"])
        adjacency[edge["right"]].add(edge["left"])
    seen = set()
    groups = []
    for node in node_ids:
        if node in seen:
            continue
        stack = [node]
        group = []
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            group.append(current)
            stack.extend(adjacency[current] - seen)
        groups.append(group)
    return groups


def _solve_component(nodes, edges):
    """Weighted least squares for log-exposure offsets, gauge fixed on the first node."""
    index = {node: j for j, node in enumerate(nodes)}
    if len(nodes) == 1:
        return {nodes[0]: 0.0}
    rows = []
    targets = []
    weights = []
    for edge in edges:
        row = np.zeros(len(nodes))
        row[index[edge["left"]]] = 1.0
        row[index[edge["right"]]] = -1.0
        rows.append(row)
        targets.append(edge["_delta"])
        weights.append(edge["_weight"])
    design = np.asarray(rows)
    gauge = np.zeros(len(nodes))
    gauge[0] = 1.0
    design = np.vstack([design, gauge])
    targets = np.asarray(targets + [0.0])
    weights = np.asarray(weights + [max(weights) * 4.0])
    sqrt_w = np.sqrt(weights)
    solved, *_ = np.linalg.lstsq(design * sqrt_w[:, None], targets * sqrt_w, rcond=None)
    return {node: float(solved[index[node]]) for node in nodes}


def reconcile(registered, edges, threshold):
    """Two-pass robust solve; edges failing the consistency pass are reported."""
    marked_nodes = sorted({mark.view_id for view in registered for mark in view.marks})
    groups = _connected_groups(marked_nodes, edges)
    solution = {}
    discord = []
    for group in groups:
        group_edges = [edge for edge in edges if edge["left"] in group and edge["right"] in group]
        first_pass = _solve_component(group, group_edges)
        for edge in group_edges:
            residual = abs(first_pass[edge["left"]] - first_pass[edge["right"]] - edge["_delta"])
            edge["_residual"] = residual
            color_bad = edge["channel_spread_ev"] is not None and edge["channel_spread_ev"] > COLOR_SPREAD_LIMIT
            edge["consistent"] = bool(residual <= threshold and not color_bad)
        good = [edge for edge in group_edges if edge["consistent"]]
        discord.extend(edge for edge in group_edges if not edge["consistent"])
        if not good:
            # Every vote in this component is contradictory: the references line
            # up geometrically but disagree about the brightness correction, so
            # every member stays on its manual value rather than guessing.
            for node in group:
                solution[node] = None
            continue
        refined = _solve_component(group, good)
        confident = set()
        for edge in good:
            confident.add(edge["left"])
            confident.add(edge["right"])
        for node in group:
            solution[node] = refined[node] if node in confident else None
    connected = {node for edge in edges for node in (edge["left"], edge["right"])}
    return solution, discord, groups, connected


def marked_contact_sheet(registered, readings):
    panes = []
    for view in registered:
        if not view.marks:
            continue
        image = Image.fromarray(linear_to_srgb(view.rgb))
        entries = []
        for mark in view.marks:
            reading = next((item for item in readings.get(mark.view_id, []) if item.spec is mark), None)
            tag = "wall"
            color = (105, 193, 111)
            if reading is not None and reading.warnings:
                color = (234, 108, 78)
                tag = "review"
            entries.append((mark.region.public(), tag, color))
        panes.append(png_bytes(resize_report(overlay_regions(image, entries), 420)))
    if not panes:
        return None
    images = [Image.open(BytesIO(blob)).convert("RGB") for blob in panes]
    gap = 7
    width = sum(image.width for image in images) + gap * (len(images) - 1)
    sheet = Image.new("RGB", (width, max(image.height for image in images)), (21, 35, 46))
    x = 0
    for image in images:
        sheet.paste(image, (x, 0))
        x += image.width + gap
    return png_bytes(resize_report(sheet, 1180))


def ev_chart(view_rows):
    order = [row["index"] for row in view_rows]
    current = [row["current_ev"] for row in view_rows]
    proposed = [row["proposed_ev"] if row["confident"] else np.nan for row in view_rows]
    return chart_png(
        [("Current angle EV", np.asarray(order), np.asarray(current)),
         ("Matched proposal", np.asarray(order), np.asarray(proposed, dtype=float))],
        x_label="View order (left to right angle)", y_label="Exposure adjustment (EV)",
    )


def wall_luma_chart(view_rows, readings):
    order = [row["index"] for row in view_rows]
    raw = []
    corrected = []
    for row in view_rows:
        values = [item.median_luma for item in readings.get(row["view_id"], []) if np.isfinite(item.median_luma)]
        if not values:
            raw.append(np.nan)
            corrected.append(np.nan)
            continue
        median = float(np.median(values))
        raw.append(median)
        if row["confident"]:
            corrected.append(median * (2 ** row["proposed_ev"]))
        else:
            corrected.append(np.nan)
    return chart_png(
        [("Wall median now", np.asarray(order), np.asarray(raw)),
         ("After proposed EV", np.asarray(order), np.asarray(corrected, dtype=float))],
        x_label="View order (left to right angle)", y_label="Wall linear luminance",
    )


def proposal_fingerprint(records, marks, threshold):
    payload = {
        "views": [
            {"sha": record["sha"], "ev": record["exposure_ev"], "enabled": record["enabled"]}
            for record in sorted(records, key=lambda item: item["id"])
        ],
        "marks": [{"view": mark.view_id, "region": mark.region.public()} for mark in marks],
        "threshold": threshold,
    }
    return sha256(encode_json(payload).encode("utf-8")).hexdigest()


def exposure_match(library, records, body):
    marks = parse_marks(body)
    fit_motion = bool(body.get("fit_motion", False))
    threshold = number(body.get("discord_ev", DEFAULT_DISCORD_EV),
                       "inconsistency threshold in stops", 0.02, 1.5)
    registered, size = register_raw_views(library, records, fit_motion)
    readings = read_marks(registered, marks)
    edges = build_edges(readings)
    solution, discord, groups, connected = reconcile(registered, edges, threshold)

    view_rows = []
    for index, view in enumerate(registered, start=1):
        record = view.record
        relative = solution.get(record["id"])
        current = float(record["exposure_ev"])
        mark_items = readings.get(record["id"], [])
        marked = bool(mark_items)
        usable = any(np.isfinite(item.median_luma) for item in mark_items)
        if relative is None or not usable:
            confident = False
            proposed = current
            if not marked:
                reason = "No wall rectangle was marked on this original."
            elif not usable:
                reason = "The marked rectangle has too few usable wall pixels; kept the current value."
            elif record["id"] in connected:
                reason = ("Wall references on this view disagree about the brightness correction "
                          "(see inconsistent pairs); kept the current value.")
            else:
                reason = "Its wall rectangle does not overlap any other marked view; kept the current value."
        else:
            confident = True
            proposed = current - relative
            reason = "Matched through consistent wall references."
        clamped = not -3.0 <= proposed <= 3.0
        proposed_clamped = float(np.clip(proposed, -3.0, 3.0))
        if clamped:
            reason += " Proposal reaches the +/-3 EV limit and was clamped."
        view_rows.append({
            "index": index,
            "view_id": record["id"],
            "title": record["title"],
            "angle": record["position"],
            "current_ev": rounding(current),
            "relative_exposure_ev": None if relative is None else rounding(relative),
            "proposed_ev": rounding(proposed_clamped),
            "delta_ev": rounding(proposed_clamped - current),
            "confident": confident,
            "marked": marked,
            "usable_reference": usable,
            "reason": reason,
            "wall_median_luma": rounding(np.median([
                item.median_luma for item in mark_items if np.isfinite(item.median_luma)
            ])) if usable else None,
            "alignment_warning": view.motion.warning,
        })

    public_edges = [{key: value for key, value in edge.items() if not key.startswith("_")}
                    for edge in edges]
    confident_count = sum(1 for row in view_rows if row["confident"])
    report = {
        "kind": "angle-exposure-match",
        "registered_size": list(size),
        "fit_motion": fit_motion,
        "discord_threshold_ev": threshold,
        "proposal_sha": proposal_fingerprint(records, marks, threshold),
        "summary": {
            "views": len(view_rows),
            "references": len(edges),
            "consistent_references": sum(1 for edge in edges if edge["consistent"]),
            "inconsistent_references": len(discord),
            "confident_views": confident_count,
            "uncertain_views": len(view_rows) - confident_count,
        },
        "views": view_rows,
        "references": public_edges,
        "inconsistent": [
            {key: edge[key] for key in ("left", "right", "delta_ev", "scatter_ev",
                                        "channel_spread_ev", "overlap_pixels", "warnings")}
            for edge in discord
        ],
        "interpretation": (
            "EV proposals come only from wall references whose pairwise votes agree within "
            f"{threshold:g} stops. Uncertain views keep their current exposure_ev until you decide."
        ),
    }
    images = {
        "marked-wall": marked_contact_sheet(registered, readings),
        "ev-proposal": ev_chart(view_rows),
        "wall-luminance": wall_luma_chart(view_rows, readings),
    }
    images = {key: value for key, value in images.items() if value is not None}
    return report, images
