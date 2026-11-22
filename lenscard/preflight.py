"""Evaluate how a set of oriented angle captures will support the current card."""
from collections import Counter
from dataclasses import dataclass
import math

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from .framing import paper_rgb, prepare_registered, working_preview_sizes
from .imaging import fit_size, linear_luminance, read_pixels
from .models import Plan, Unprocessable, ordered_views
from .optics import LensGrid, angular_domains, print_quanta
from .visual import code_statistics, png_bytes


MIN_VIEW_SUPPORT = .74
MAX_SIGNATURE_EDGE = 540


@dataclass(frozen=True)
class CoverageSignature:
    sample_size: tuple[int, int]
    visible_fraction: float
    edge_support: dict
    image_bounds: tuple[int, int, int, int]
    largest_dark_block: int
    rgb_statistics: dict
    texture_score: float | None
    tone_span: float | None

    def public(self):
        return {
            "sample_pixels": list(self.sample_size), "visible_fraction": round(self.visible_fraction, 7),
            "edge_support": self.edge_support, "visible_bounds": list(self.image_bounds),
            "largest_uncovered_block_pixels": self.largest_dark_block,
            "rgb_statistics": self.rgb_statistics, "texture": self.texture_score, "tone_span": self.tone_span,
        }


def _physical_edges(support):
    margin_y = max(1, math.ceil(support.shape[0] * .08))
    margin_x = max(1, math.ceil(support.shape[1] * .08))
    sides = {
        "top": support[:margin_y],
        "bottom": support[-margin_y:],
        "left": support[:, :margin_x],
        "right": support[:, -margin_x:],
    }
    return {area: round(float(plane.mean()), 6) for area, plane in sides.items()}


def _coverage_span(support):
    y, x = np.nonzero(support)
    if not len(x):
        return 0, 0, 0, 0
    return int(x.min()), int(y.min()), int(x.max() + 1), int(y.max() + 1)


def _uncovered_scaling(support):
    components, n = ndimage.label(~support)
    if not n:
        return 0
    counts = np.bincount(components.ravel(), minlength=n + 1)
    return int(counts[1:].max(initial=0))


def source_signature(library, record):
    w, h = fit_size((record["width"], record["height"]), MAX_SIGNATURE_EDGE)
    pixels = read_pixels(library, record, (w, h))
    values = pixels.rgb
    support = pixels.support
    gradient = None
    span = None
    if int(support.sum()) >= 30:
        safe = ndimage.binary_erosion(support, structure=np.ones((3, 3), dtype=bool))
        luma = linear_luminance(values)
        dx = ndimage.sobel(luma, axis=1, mode="nearest") / 8
        dy = ndimage.sobel(luma, axis=0, mode="nearest") / 8
        if safe.any():
            gradient = round(float(np.median(np.hypot(dx, dy)[safe])), 8)
        q05, q95 = np.percentile(luma[support], [5, 95])
        span = round(float(q95 - q05), 7)
    signals = {
        name: code_statistics(values[..., j][support])
        for j, name in enumerate(("red", "green", "blue"))
    }
    return CoverageSignature(
        (w, h), float(support.mean()), _physical_edges(support), _coverage_span(support),
        _uncovered_scaling(support), signals, gradient, span,
    )


def _source_alerts(record, signals, dimensions, exposures):
    issues = []
    dims = (record["width"], record["height"])
    if dimensions and dims != dimensions:
        issues.append({"level": "blocking", "message": "Oriented pixel size differs from the active set."})
    if signals.visible_fraction < MIN_VIEW_SUPPORT:
        issues.append({"level": "review", "message": "Substantial unfilled image coverage."})
    if signals.tone_span is not None and signals.tone_span < .035:
        issues.append({"level": "review", "message": "Limited source brightness range."})
    if signals.texture_score is not None and signals.texture_score < .001:
        issues.append({"level": "review", "message": "Too little shared texture for automatic anchoring."})
    if max(signals.edge_support.values()) - min(signals.edge_support.values()) > .45:
        issues.append({"level": "review", "message": "One of the original edges has intermittent coverage."})
    if exposures is not None and abs(record["exposure_ev"] - exposures) > 1.5:
        issues.append({"level": "review", "message": "Manual exposure differs substantially from the angle set."})
    return issues


def _angle_scatter(records):
    known = [view for view in records if view["enabled"]]
    if len(known) < 2:
        return {"count": len(known), "span": None, "intervals": [], "largest_gap": None}
    ordered = sorted(known, key=lambda item: item["position"])
    differences = np.diff([item["position"] for item in ordered])
    return {
        "count": len(known), "span": [ordered[0]["position"], ordered[-1]["position"]],
        "intervals": [round(float(value), 5) for value in differences],
        "largest_gap": round(float(differences.max()), 5),
        "duplicate_angles": int(np.count_nonzero(differences < .01)),
    }


def _format_inventory(entries):
    segment_w, segment_h = 200, 100
    per_row = 3
    count = max(1, len(entries))
    height = (count + per_row - 1) // per_row * segment_h
    canvas = Image.new("RGB", (per_row * segment_w, height), (30, 46, 53))
    draw = ImageDraw.Draw(canvas)
    for index, row in enumerate(entries):
        x = index % per_row * segment_w
        y = index // per_row * segment_h
        data = row["signature"]
        title = row["title"][:23]
        draw.text((x + 10, y + 9), title, fill=(222, 237, 227))
        draw.text((x + 10, y + 30), f"{row['angle']} degrees, {data['visible_fraction']*100:.1f}% visible", fill=(178, 202, 195))
        for j, (side, value) in enumerate(data["edge_support"].items()):
            col = x + 10 + j * 42
            hue = (104, 202, 147) if value >= .80 else (242, 167, 93) if value >= .45 else (204, 101, 91)
            draw.rectangle((col, y + 62, col + 32, y + 72), fill=(45, 53, 57))
            draw.rectangle((col, y + 62, col + max(1, round(32 * value)), y + 72), fill=hue)
            draw.text((col, y + 77), side, fill=(185, 203, 195))
    return png_bytes(canvas)


def _paper_white(plan, library, card):
    if not plan.sheet.paper.startswith("measured:"):
        return None
    from .tone import measured_paper
    return measured_paper(library.measurement(card["id"], plan.sheet.paper.split(":", 1)[1])).white_coded


def draft_preflight(library, card, records):
    plan = Plan.parse(card["config"])
    enabled = [item for item in records if item["enabled"]]
    modes = Counter((item["width"], item["height"]) for item in enabled)
    common = modes.most_common(1)[0][0] if modes else None
    exposures = float(np.median([item["exposure_ev"] for item in enabled])) if enabled else None
    inventory = []
    all_issues = []
    for record in records:
        spec = source_signature(library, record)
        alerts = _source_alerts(record, spec, common, exposures) if record["enabled"] else []
        for alert in alerts:
            all_issues.append({"view": record["id"], **alert})
        inventory.append({
            "view_id": record["id"], "title": record["title"], "angle": record["position"],
            "enabled": record["enabled"], "signature": spec.public(), "findings": alerts,
        })
    predicted = None
    renderable = False
    try:
        usable = ordered_views(records)
        plan.check_sampling(len(usable))
        fitted = prepare_registered(library, records, plan.framing)
        grid = LensGrid(plan.lens, plan.sheet.dpi, *plan.sheet.size)
        predicted = {"physical": grid.public(), "aligned_coverage": fitted.public(),
                     "nominal_angles": angular_domains(grid, usable),
                     "dot_budget": print_quanta(grid, len(usable), plan.sampling, plan.sheet.mount)}
        renderable = True
    except Unprocessable as warning:
        all_issues.append({"stage": "working print", "level": "blocking", "message": warning.message})
    report = {
        "kind": "angle-source-preflight",
        "working_version": card["version"], "active_count": len(enabled),
        "dominant_oriented_dimensions": list(common) if common else None, "records": inventory,
        "angle_intervals": _angle_scatter(records), "predicted": predicted,
        "likely_renderable": renderable, "hard_findings": all_issues,
        "format": {"paper_color": paper_rgb(plan.sheet, _paper_white(plan, library, card)).tolist(),
                   "screen_size": list(working_preview_sizes(plan.sheet, 750))},
    }
    return report, {"source-coverage": _format_inventory(inventory)}
