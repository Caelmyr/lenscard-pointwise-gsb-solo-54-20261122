"""Translate oriented source images into a common print-facing frame."""
from dataclasses import dataclass
import math

import numpy as np
from PIL import Image

from .alignment import AnalysisBox, RailAlignment, output_viewport, overlap_summary
from .imaging import (
    assert_compatible_dimensions, fit_size, float_resize, linear_to_srgb,
    processing_size, read_pixels, srgb_to_linear,
)
from .models import Framing, Sheet, Unprocessable, ordered_views
from .visual import heatmap, png_bytes


PAPERS = {
    "white": (255, 255, 255),
    "warm": (247, 239, 216),
    "matte": (239, 241, 238),
}
MIN_WINDOW_EDGE = 18


@dataclass
class PreparedView:
    record: dict
    linear: np.ndarray
    support: np.ndarray
    correction: dict

    def public(self):
        return {
            "id": self.record["id"],
            "source_sha": self.record["sha"],
            "title": self.record["title"],
            "angle": self.record["position"],
            "exposure_ev": self.record["exposure_ev"],
            "support_fraction": round(float(self.support.mean()), 6),
            "alignment": self.correction,
        }


@dataclass
class Registered:
    views: list[PreparedView]
    bound: AnalysisBox
    full_size: tuple[int, int]
    overlap: dict
    reference: str

    def public(self):
        return {
            "size": list(self.full_size),
            "bound": self.bound.public(),
            "reference": self.reference,
            "overlap": self.overlap,
            "views": [view.public() for view in self.views],
        }


def reference_view(views, desired=None):
    if desired is None:
        return min(views, key=lambda view: abs(view["position"]))
    for view in views:
        if view["id"] == desired:
            return view
    raise Unprocessable("The reference view must be one of the enabled originals.")


def _cut_box(base: AnalysisBox, region):
    left, top, right, bottom = region.pixels(base.width, base.height)
    clipped = AnalysisBox(base.x + left, base.y + top, right - left, bottom - top)
    if min(clipped.width, clipped.height) < MIN_WINDOW_EDGE:
        raise Unprocessable("The selected crop is too narrow after angle registration.")
    return clipped


def prepare_registered(library, records, settings: Framing):
    views = ordered_views(records)
    assert_compatible_dimensions(views)
    size = processing_size(views)
    picked = reference_view(views, settings.reference)
    reference = read_pixels(library, picked, size)
    register = RailAlignment(reference, "breathing" if settings.fit_motion else "none")
    aligned = []
    for item in views:
        capture = reference if item["id"] == picked["id"] else read_pixels(library, item, size)
        result = register.align(capture, reference=item["id"] == picked["id"])
        corrected = np.clip(result.rgb * (2 ** item["exposure_ev"]), 0, 1)
        aligned.append(PreparedView(item, corrected, result.support, result.motion.public()))
    mask = np.stack([view.support for view in aligned])
    base = output_viewport(mask, "shared" if settings.mode == "shared" else "reference")
    bound = _cut_box(base, settings.frame)
    return Registered(aligned, bound, size, overlap_summary(mask, bound), picked["id"])


def paper_rgb(sheet: Sheet):
    return np.asarray(PAPERS[sheet.paper], np.uint8)


def paper_linear(sheet: Sheet):
    return srgb_to_linear(paper_rgb(sheet).astype(np.float32) / 255)


def framing_geometry(width, height, target_width, target_height, cover):
    width_ratio = target_width / width
    height_ratio = target_height / height
    if cover == "fill":
        factor = max(width_ratio, height_ratio)
    else:
        factor = min(width_ratio, height_ratio)
    full_width = max(1, round(width * factor))
    full_height = max(1, round(height * factor))
    x = (target_width - full_width) // 2
    y = (target_height - full_height) // 2
    start_x, start_y = max(0, -x), max(0, -y)
    end_x = min(full_width, target_width - x)
    end_y = min(full_height, target_height - y)
    return {
        "resize": [full_width, full_height],
        "source_window": [start_x, start_y, end_x, end_y],
        "destination": [max(0, x), max(0, y), max(0, x) + end_x - start_x, max(0, y) + end_y - start_y],
    }


def _view_for_geometry(view, geometry, target_width, target_height, paper):
    width, height = geometry["resize"]
    # For highly eccentric source/target shapes the filled virtual rectangle can exceed
    # the full image budget. In that case crop the source before resizing.
    if width * height > target_width * target_height * 5:
        sy, sx = view.support.shape
        left, top, right, bottom = geometry["source_window"]
        xs = (np.arange(target_width) + left + 0.5) * sx / width - 0.5
        ys = (np.arange(target_height) + top + 0.5) * sy / height - 0.5
        xs = np.clip(xs.round().astype(np.int32), 0, sx - 1)
        ys = np.clip(ys.round().astype(np.int32), 0, sy - 1)
        inside = view.support[ys[:, None], xs]
        rgb = view.linear[ys[:, None], xs]
        return np.where(inside[..., None], rgb, paper), inside
    left, top, right, bottom = geometry["source_window"]
    dest_left, dest_top, dest_right, dest_bottom = geometry["destination"]
    scaled_support = float_resize(view.support.astype(np.float32), (width, height))
    alpha = np.clip(scaled_support, 0, 1)[top:bottom, left:right]
    supported = alpha >= 0.995
    visible = np.broadcast_to(paper, (target_height, target_width, 3)).copy()
    support = np.zeros((target_height, target_width), bool)
    area = (slice(dest_top, dest_bottom), slice(dest_left, dest_right))
    support[area] = supported
    for channel in range(3):
        resized = float_resize(view.linear[..., channel] * view.support, (width, height))[top:bottom, left:right]
        base = np.clip(resized / np.maximum(alpha, 0.001), 0, 1)
        visible[area + (channel,)] = np.where(supported, base, paper[channel])
    return visible, support


def fit_views(registered, width, height, sheet, cover):
    geometry = framing_geometry(registered.bound.width, registered.bound.height, width, height, cover)
    base = paper_linear(sheet)
    for view in registered.views:
        cropped = PreparedView(view.record, registered.bound.slice(view.linear), registered.bound.slice(view.support), view.correction)
        linear, support = _view_for_geometry(cropped, geometry, width, height, base)
        yield PreparedView(view.record, linear, support, view.correction)


def working_preview_sizes(sheet, max_edge):
    width, height = sheet.size
    return fit_size((width, height), max_edge)


def inspect_framing(registered, sheet, framing):
    width, height = working_preview_sizes(sheet, 740)
    panes = list(fit_views(registered, width, height, sheet, framing.cover))
    index = next(i for i, frame in enumerate(panes) if frame.record["id"] == registered.reference)
    main = panes[index]
    support = np.sum([p.support for p in panes], axis=0)
    count = len(panes)
    preview = Image.fromarray(linear_to_srgb(main.linear))
    map_image = heatmap(support / count, np.ones(support.shape, bool), 0, 1)
    images = {
        "reference": png_bytes(preview),
        "coverage": png_bytes(map_image),
    }
    entries = [
        {
            **item.public(),
            "cover_loss": round(float(1 - item.support.mean()), 6),
            "luma": round(float(np.mean(item.linear @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32))), 5),
        }
        for item in panes
    ]
    return {
        **registered.public(),
        "output_size": [width, height],
        "configured_size": list(sheet.size),
        "output_shared_fraction": round(float(np.mean(support == count)), 6),
        "output_uncovered_fraction": round(float(np.mean(support == 0)), 6),
        "views": entries,
    }, images


def feather_comparison(first, second, position):
    split = min(first.shape[1] - 1, max(1, round(position * first.shape[1])))
    merged = first.copy()
    merged[:, split:, :] = second[:, split:, :]
    return merged, split


def rect_for_report(view, region):
    height, width = view.support.shape
    left, top, right, bottom = region.pixels(width, height)
    capture = view.linear[top:bottom, left:right]
    mask = view.support[top:bottom, left:right]
    return capture, mask, [left, top, right, bottom]
