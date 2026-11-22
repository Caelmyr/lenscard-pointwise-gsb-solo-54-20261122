"""Subject-aware common framing: keep boxed subjects inside genuinely shared coverage."""
import math
from dataclasses import dataclass, replace

import numpy as np
from PIL import Image

from .alignment import AnalysisBox
from .framing import fit_views, framing_geometry, working_preview_sizes
from .imaging import linear_to_srgb
from .models import Region, Unprocessable, object_value, require_id, validate_keys
from .visual import heatmap, overlay_regions, png_bytes, resize_report

MIN_SUBJECT_WINDOW = 28
MIN_MARK_EDGE = 0.005
MAX_MARKS = 24
SUBJECT_COLOR = (245, 212, 111)
UNION_COLOR = (238, 238, 232)
CROP_COLOR = (126, 216, 143)
BORDER_COLOR = (240, 164, 90)


@dataclass(frozen=True)
class Mark:
    view_id: str
    title: str
    region: Region
    box: tuple  # reference-frame processing pixels: left, top, right, bottom


def parse_subject_marks(data, views):
    if not isinstance(data, list) or not data:
        raise Unprocessable(
            "Box at least one subject on an original before running this check.",
            field="subjects",
        )
    if len(data) > MAX_MARKS:
        raise Unprocessable(f"Keep at most {MAX_MARKS} subject boxes per check.", field="subjects")
    enabled = {view["id"] for view in views if view["enabled"]}
    marks = []
    for entry in data:
        obj = object_value(entry, "subject")
        validate_keys(obj, {"view", "frame"}, "subject")
        view_id = require_id(obj.get("view"), "subject view")
        if view_id not in enabled:
            raise Unprocessable("Subject boxes must sit on enabled views of this card.", field="subjects")
        region = Region.parse(obj.get("frame"))
        if region.right - region.left < MIN_MARK_EDGE or region.bottom - region.top < MIN_MARK_EDGE:
            raise Unprocessable("Draw a slightly larger subject box.", field="subjects")
        marks.append((view_id, region))
    return marks


def _motion_box(region, correction, size):
    """Map a normalized source rectangle into reference-frame processing pixels."""
    width, height = size
    center_x, center_y = (width - 1) / 2, (height - 1) / 2
    scale = correction["scale"]
    dx, dy = correction["dx"], correction["dy"]

    def move(nx, ny):
        px, py = nx * width, ny * height
        return (px - center_x) * scale + center_x + dx, (py - center_y) * scale + center_y + dy

    x0, y0 = move(region.left, region.top)
    x1, y1 = move(region.right, region.bottom)
    return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)


def _clip_box(box, size):
    width, height = size
    left = min(max(math.floor(box[0]), 0), width)
    top = min(max(math.floor(box[1]), 0), height)
    right = min(max(math.ceil(box[2]), 0), width)
    bottom = min(max(math.ceil(box[3]), 0), height)
    return left, top, right, bottom


def _covered_fraction(support, box, size):
    total = max((box[2] - box[0]) * (box[3] - box[1]), 1e-9)
    left, top, right, bottom = _clip_box(box, size)
    if right <= left or bottom <= top:
        return 0.0
    return min(float(np.count_nonzero(support[top:bottom, left:right]) / total), 1.0)


def _mark_kept(support, mark_box, window, size):
    """Fraction of a mark that lands inside the window on genuinely captured pixels."""
    total = max((mark_box[2] - mark_box[0]) * (mark_box[3] - mark_box[1]), 1e-9)
    kept = (
        max(mark_box[0], window.x),
        max(mark_box[1], window.y),
        min(mark_box[2], window.x + window.width),
        min(mark_box[3], window.y + window.height),
    )
    left, top, right, bottom = _clip_box(kept, size)
    if right <= left or bottom <= top:
        return 0.0
    return min(float(np.count_nonzero(support[top:bottom, left:right]) / total), 1.0)


class _WindowSearch:
    """Largest fixed-aspect window fully inside a support mask, containing a box."""

    def __init__(self, support):
        self.width, self.height = support.shape[1], support.shape[0]
        missing = np.ascontiguousarray(~support, dtype=np.int32)
        self.integral = np.pad(missing, ((1, 0), (1, 0))).cumsum(0).cumsum(1)

    def _sums(self, w, h):
        ig = self.integral
        return ig[h:, w:] - ig[:-h, w:] - ig[h:, :-w] + ig[:-h, :-w]

    def _top_left_range(self, w, h, contain):
        x0 = max(0, math.ceil(contain[2] - w - 1e-6))
        x1 = min(self.width - w, math.floor(contain[0] + 1e-6))
        y0 = max(0, math.ceil(contain[3] - h - 1e-6))
        y1 = min(self.height - h, math.floor(contain[1] + 1e-6))
        return x0, x1, y0, y1

    def _fits(self, w, h, contain):
        if w > self.width or h > self.height or w < 1 or h < 1:
            return False
        x0, x1, y0, y1 = self._top_left_range(w, h, contain)
        if x0 > x1 or y0 > y1:
            return False
        return bool((self._sums(w, h)[y0:y1 + 1, x0:x1 + 1] == 0).any())

    def best_window(self, w, h, contain):
        x0, x1, y0, y1 = self._top_left_range(w, h, contain)
        if x0 > x1 or y0 > y1:
            x = min(max((contain[0] + contain[2] - w) / 2, 0), self.width - w)
            y = min(max((contain[1] + contain[3] - h) / 2, 0), self.height - h)
            return AnalysisBox(round(x), round(y), w, h)
        sums = self._sums(w, h)[y0:y1 + 1, x0:x1 + 1]
        best = sums.min()
        ys, xs = np.nonzero(sums == best)
        target_x = (contain[0] + contain[2] - w) / 2
        target_y = (contain[1] + contain[3] - h) / 2
        index = int(np.argmin((xs + x0 - target_x) ** 2 + (ys + y0 - target_y) ** 2))
        return AnalysisBox(int(xs[index] + x0), int(ys[index] + y0), int(w), int(h))

    def largest(self, aspect, contain, min_edge=MIN_SUBJECT_WINDOW):
        union_w = contain[2] - contain[0]
        union_h = contain[3] - contain[1]
        # Feasibility is monotone only above the smallest height that can hold the
        # union at all: if a larger window fits, a smaller one fits inside it.
        low = max(min_edge, math.ceil(union_h - 1e-9), math.ceil(union_w / aspect - 1e-9))
        high = min(self.height, int(self.width / aspect))
        if low > high or not self._fits(max(1, math.ceil(low * aspect - 1e-9)), low, contain):
            return None
        while low < high:
            mid = (low + high + 1) // 2
            if self._fits(max(1, math.ceil(mid * aspect - 1e-9)), mid, contain):
                low = mid
            else:
                high = mid - 1
        return self.best_window(max(1, math.ceil(low * aspect - 1e-9)), low, contain)


def _union_box(mapped, size):
    width, height = size
    left = max(min(mark.box[0] for mark in mapped), 0.0)
    top = max(min(mark.box[1] for mark in mapped), 0.0)
    right = min(max(mark.box[2] for mark in mapped), float(width))
    bottom = min(max(mark.box[3] for mark in mapped), float(height))
    if right - left < 1 or bottom - top < 1:
        raise Unprocessable("The marked subjects fall outside the aligned capture frame.")
    return left, top, right, bottom


def _border_window(search, union, aspect, size):
    """Smallest card-shaped window holding every mark; paper fills the uncovered rest."""
    width, height = size
    need_w = max(union[2] - union[0], (union[3] - union[1]) * aspect, MIN_SUBJECT_WINDOW * aspect)
    if need_w / aspect <= height:
        w = min(math.ceil(need_w), width)
        h = min(math.ceil(need_w / aspect), height)
    else:
        h = height
        w = min(math.ceil(height * aspect), width)
    return search.best_window(w, h, union)


def _view_limit(view, aspect, union, needed_height):
    best = _WindowSearch(view.support).largest(aspect, union)
    return {
        "id": view.record["id"],
        "title": view.record["title"],
        "keeps_all_subjects": best is not None,
        "max_window_height_px": best.height if best else 0,
        "needed_window_height_px": round(needed_height, 1),
    }


def _candidate_report(name, label, cover, box, registered, mapped, shared, size):
    if box is None:
        return {
            "name": name,
            "label": label,
            "feasible": False,
            "reason": "No card-shaped window inside genuinely shared coverage can hold every marked subject.",
        }
    width, height = size
    per_view = []
    kept_total = 0.0
    mark_total = 0.0
    for view in registered.views:
        support = view.support
        paper = 1 - float(box.slice(support).mean())
        own = [mark for mark in mapped if mark.view_id == view.record["id"]]
        kept_view = 0.0
        area_view = 0.0
        for mark in own:
            area = (mark.box[2] - mark.box[0]) * (mark.box[3] - mark.box[1])
            kept = _mark_kept(support, mark.box, box, size)
            kept_view += kept * area
            area_view += area
        kept_total += kept_view
        mark_total += area_view
        per_view.append({
            "id": view.record["id"],
            "title": view.record["title"],
            "paper_fraction": round(paper, 6),
            "subject_kept_fraction": round(kept_view / area_view, 6) if area_view else None,
        })
    return {
        "name": name,
        "label": label,
        "feasible": True,
        "window_px": box.public(),
        "apply": {
            "mode": "center",
            "cover": cover,
            "frame": [
                round(box.x / width, 6),
                round(box.y / height, 6),
                round((box.x + box.width) / width, 6),
                round((box.y + box.height) / height, 6),
            ],
        },
        "shared_cover_fraction": round(float(box.slice(shared).mean()), 6),
        "subject_kept_fraction": round(kept_total / max(mark_total, 1e-9), 6),
        "worst_view_paper_fraction": round(max(row["paper_fraction"] for row in per_view), 6),
        "views": per_view,
    }


def _card_frame(mark_box, window, geometry, out_size):
    out_w, out_h = out_size
    full_w, full_h = geometry["resize"]
    start_x, start_y = geometry["source_window"][:2]
    dest_x, dest_y = geometry["destination"][:2]

    def map_point(px, py):
        fx = (px - window.x) / window.width
        fy = (py - window.y) / window.height
        return (dest_x + fx * full_w - start_x) / out_w, (dest_y + fy * full_h - start_y) / out_h

    x0, y0 = map_point(mark_box[0], mark_box[1])
    x1, y1 = map_point(mark_box[2], mark_box[3])
    return x0, y0, x1, y1


def _candidate_preview(registered, box, sheet, cover, mapped, view_id):
    trial = replace(registered, bound=box)
    width, height = working_preview_sizes(sheet, 740)
    panes = {pane.record["id"]: pane for pane in fit_views(trial, width, height, sheet, cover)}
    chosen = panes[view_id]
    image = Image.fromarray(linear_to_srgb(chosen.linear))
    geometry = framing_geometry(box.width, box.height, width, height, cover)
    overlays = [
        (_card_frame(mark.box, box, geometry, (width, height)), mark.title, SUBJECT_COLOR)
        for mark in mapped
    ]
    return overlay_regions(image, overlays)


def _coverage_map(registered, mapped, union, boxes, size):
    width, height = size
    count = np.sum([view.support for view in registered.views], axis=0)
    image = heatmap(count / len(registered.views), np.ones(count.shape, bool), 0, 1)
    image = resize_report(image, 920)

    def norm(box):
        return box[0] / width, box[1] / height, box[2] / width, box[3] / height

    overlays = [(norm(mark.box), mark.title, SUBJECT_COLOR) for mark in mapped]
    overlays.append((norm(union), "all subjects", UNION_COLOR))
    for box, text, color in boxes:
        if box is not None:
            overlays.append((norm((box.x, box.y, box.x + box.width, box.y + box.height)), text, color))
    return overlay_regions(image, overlays)


def _message(crop, border, limiting):
    if crop["feasible"]:
        kept = crop["subject_kept_fraction"]
        if kept >= 0.999:
            return ("Cropping alone keeps every marked subject inside genuinely shared coverage; "
                    "no paper edge is needed. Compare both framings before applying one.")
        return (f"Cropping alone keeps {kept * 100:.1f}% of the marked subject area; "
                "the rest lies outside every captured frame. Compare both framings before applying one.")
    worst = limiting[0]["title"] if limiting else "one view"
    kept = round(border["subject_kept_fraction"] * 100, 1)
    paper = round(border["worst_view_paper_fraction"] * 100, 1)
    return (f"Cropping alone cannot keep every marked subject: {worst} most limits the common window. "
            f"With a paper border the framing keeps {kept}% of the marked subject area; "
            f"the weakest view shows {paper}% paper.")


def plan_subject_framing(registered, sheet, marks):
    size = registered.full_size
    sheet_width, sheet_height = sheet.size
    aspect = sheet_width / sheet_height
    views = registered.views
    shared = np.all(np.stack([view.support for view in views]), axis=0)
    by_id = {view.record["id"]: view for view in views}
    mapped = [
        Mark(view_id, by_id[view_id].record["title"], region,
             _motion_box(region, by_id[view_id].correction, size))
        for view_id, region in marks
    ]
    union = _union_box(mapped, size)
    search = _WindowSearch(shared)
    crop_box = search.largest(aspect, union)
    border_box = _border_window(search, union, aspect, size)
    needed = max(MIN_SUBJECT_WINDOW, union[3] - union[1], (union[2] - union[0]) / aspect)
    limiting = []
    if crop_box is None:
        limiting = sorted(
            (_view_limit(view, aspect, union, needed) for view in views),
            key=lambda row: row["max_window_height_px"],
        )
    crop = _candidate_report("crop", "Crop only · shared coverage", "fill", crop_box,
                             registered, mapped, shared, size)
    border = _candidate_report("border", "Crop with paper border", "fit", border_box,
                               registered, mapped, shared, size)
    candidates = [crop, border]
    images = {}
    for candidate, box in zip(candidates, (crop_box, border_box)):
        if box is None:
            continue
        if candidate["name"] == "border":
            view_id = max(candidate["views"], key=lambda row: row["paper_fraction"])["id"]
        else:
            view_id = registered.reference
        candidate["preview_view"] = by_id[view_id].record["title"]
        images[candidate["name"]] = png_bytes(
            _candidate_preview(registered, box, sheet, candidate["apply"]["cover"], mapped, view_id)
        )
    images["coverage"] = png_bytes(_coverage_map(
        registered, mapped, union,
        ((crop_box, "crop", CROP_COLOR), (border_box, "border", BORDER_COLOR)), size,
    ))
    report = {
        "registered_size": list(size),
        "card_aspect": round(aspect, 6),
        "subjects": [
            {
                "view": mark.view_id,
                "title": mark.title,
                "frame": mark.region.public(),
                "shared_cover_fraction": round(_covered_fraction(shared, mark.box, size), 6),
            }
            for mark in mapped
        ],
        "subject_union_px": [round(value, 1) for value in union],
        "candidates": candidates,
        "limiting_views": limiting,
        "message": _message(crop, border, limiting),
        "interpretation": (
            "Windows use only pixels genuinely present in every enabled view; "
            "the paper-border candidate fills uncovered locations with the sheet paper tone. "
            "Apply loads the window as reference-frame crop bounds; save the draft to use it."
        ),
    }
    return report, images
