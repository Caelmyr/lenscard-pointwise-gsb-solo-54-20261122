"""Compare two frozen raster versions in their observer-facing paper coordinates."""
from dataclasses import dataclass
import math

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from .framing import feather_comparison
from .imaging import fit_size, linear_luminance, linear_to_srgb, srgb_to_linear
from .models import Unprocessable, number, object_value, require_id, validate_keys
from .printcheck import DeviceFile
from .visual import chart_png, code_statistics, heatmap, png_bytes, resize_report


MAX_COMPARE_EDGE = 980


@dataclass
class PaperRaster:
    device: DeviceFile
    safe_pixels: np.ndarray
    width_mm: float
    height_mm: float
    original_mount: str

    @classmethod
    def for_proof(cls, library, card_id, proof_id):
        proof = library.proof(card_id, require_id(proof_id, "proof"))
        material = DeviceFile.from_record(library, proof)
        original = material.pixels
        sheet = material.job.sheet
        visible = original[:, ::-1] if sheet.mount == "reverse" else original
        return cls(material, visible, sheet.width_mm, sheet.height_mm, sheet.mount)

    @property
    def pitch(self):
        return self.device.grid.lens.mm_per_lens

    def short(self):
        proof = self.device.proof
        return {
            "id": proof["id"], "title": proof["title"], "proof_sha": proof["fingerprint"],
            "pixels": list(self.safe_pixels.shape[1::-1]), "stock_mm": [self.width_mm, self.height_mm],
            "mount": self.original_mount, "pitch_lpi": self.device.job.lens.lpi,
        }


@dataclass(frozen=True)
class CompareControls:
    divider: float
    cutoff: float

    @classmethod
    def parse(cls, body):
        obj = object_value(body, "proof comparison")
        validate_keys(obj, {"first", "other", "divider", "cutoff"}, "proof comparison")
        return cls(number(obj.get("divider", .5), "preview divider", .06, .94),
                   number(obj.get("cutoff", .04), "minimum brightness difference", .001, .7))


def _union_canvas(left, right, resolution):
    width = max(left.width_mm, right.width_mm)
    height = max(left.height_mm, right.height_mm)
    desired = [max(1, round(width * resolution)), max(1, round(height * resolution))]
    return fit_size(tuple(desired), MAX_COMPARE_EDGE), width, height


def _centered_tile(paper, screen, width_mm, height_mm):
    w, h = screen
    box_w = min(w, round(paper.width_mm / width_mm * w))
    box_h = min(h, round(paper.height_mm / height_mm * h))
    x0 = (w - box_w) // 2
    y0 = (h - box_h) // 2
    x1 = min(w, x0 + box_w)
    y1 = min(h, y0 + box_h)
    view = Image.fromarray(paper.safe_pixels).resize((x1 - x0, y1 - y0), Image.Resampling.BOX)
    output = np.broadcast_to([.08, .10, .12], (h, w, 3)).astype(np.float32).copy()
    support = np.zeros((h, w), dtype=bool)
    output[y0:y1, x0:x1] = srgb_to_linear(np.asarray(view, dtype=np.float32) / 255)
    support[y0:y1, x0:x1] = True
    return output, support, (x0, y0, x1, y1)


def _find_groups(delta, shared, cutoff, width_mm, height_mm):
    mask = shared & (np.abs(delta) > cutoff)
    minimum = max(8, math.floor(mask.size / 4200))
    result = []
    # Exclude isolated pixel differences introduced by print quantization when
    # reporting coherent zones. The unfiltered pixel map is still shown.
    candidate = ndimage.binary_closing(mask, structure=np.ones((3, 3), bool)) & shared
    identity, count = ndimage.label(candidate, structure=ndimage.generate_binary_structure(2, 1))
    if count == 0:
        return result
    shapes = ndimage.find_objects(identity)
    sizes = np.bincount(identity[identity > 0].ravel(), minlength=count + 1)
    for location in sorted(range(1, count + 1), key=lambda key: int(sizes[key]), reverse=True)[:22]:
        if sizes[location] < minimum:
            break
        item = shapes[location - 1]
        y, x = item
        base = mask[item] & (identity[item] == location)
        if not base.any():
            continue
        population = delta[item][base]
        box = [int(x.start), int(y.start), int(x.stop), int(y.stop)]
        result.append({
            "preview_box": box,
            "paper_box_mm": [round(box[0] / mask.shape[1] * width_mm, 5), round(box[1] / mask.shape[0] * height_mm, 5),
                             round(box[2] / mask.shape[1] * width_mm, 5), round(box[3] / mask.shape[0] * height_mm, 5)],
            "pixels": int(base.sum()), "signed_mean": round(float(population.mean()), 7),
            "max_abs": round(float(np.abs(population).max()), 7),
        })
    return result


def _cross_card_profile(left, right, overlap, width_mm):
    a = linear_luminance(left)
    b = linear_luminance(right)
    band = np.arange(a.shape[0] // 3, a.shape[0] * 2 // 3)
    mask = overlap[band]
    available = mask.sum(axis=0)
    divisor = np.maximum(1, available)
    baseline = (a[band] * mask).sum(axis=0) / divisor
    candidate = (b[band] * mask).sum(axis=0) / divisor
    baseline[available == 0] = np.nan
    candidate[available == 0] = np.nan
    x = (np.arange(a.shape[1]) + .5) * width_mm / a.shape[1]
    return chart_png([("Proof A", x, baseline), ("Proof B", x, candidate)],
                     x_label="Physical span from common left edge (mm)", y_label="Average linear brightness")


def compare_versions(library, card_id, data):
    obj = object_value(data, "comparison")
    controls = CompareControls.parse(obj)
    first, other = obj.get("first"), obj.get("other")
    if first == other:
        raise Unprocessable("Choose two different retained device rasters.")
    left = PaperRaster.for_proof(library, card_id, first)
    right = PaperRaster.for_proof(library, card_id, other)
    resolution = min(1 / left.device.grid.step, 1 / right.device.grid.step)
    screen, width_mm, height_mm = _union_canvas(left, right, resolution)
    source, mask_a, a_box = _centered_tile(left, screen, width_mm, height_mm)
    against, mask_b, b_box = _centered_tile(right, screen, width_mm, height_mm)
    both = mask_a & mask_b
    if both.sum() < 16:
        raise Unprocessable("The physical rasters have too little covered area to compare.")
    whole = mask_a | mask_b
    delta = linear_luminance(against) - linear_luminance(source)
    values = delta[both]
    bound = max(.025, float(np.percentile(np.abs(values), 98)))
    divider, col = feather_comparison(source, against, controls.divider)
    picture = Image.fromarray(linear_to_srgb(divider))
    draw = ImageDraw.Draw(picture)
    draw.line((col, 0, col, screen[1]), width=2, fill=(231, 180, 105))
    diff = heatmap(delta, both, -bound, bound, diverging=True)
    statistics = code_statistics(values)
    report = {
        "kind": "frozen-paper-comparison",
        "left": left.short(), "right": right.short(),
        "common_paper_area_mm": [round(width_mm, 6), round(height_mm, 6)],
        "analysis_pixels": list(screen), "left_preview_extent": list(a_box), "right_preview_extent": list(b_box),
        "left_only_fraction": round(float((mask_a & ~mask_b).sum()) / whole.sum(), 7),
        "right_only_fraction": round(float((mask_b & ~mask_a).sum()) / whole.sum(), 7),
        "paired_fraction": round(float(both.sum()) / whole.sum(), 7),
        "mean_signed_difference": statistics["mean"], "delta": statistics,
        "significant_fraction": round(float((np.abs(values) > controls.cutoff).mean()), 7),
        "minimum_diff": controls.cutoff,
        "zones": _find_groups(delta, both, controls.cutoff, width_mm, height_mm),
        "alignment": "Both observer-face rasters are centered in the same physical paper area. Back-mounted prints are mirrored for comparison.",
        "limits": "Preview comparison is resampled. It is not a byte integrity check or a direct scene difference measure.",
    }
    return report, {"print-divider": png_bytes(picture), "physical-difference": png_bytes(diff),
                    "span-brightness": _cross_card_profile(source, against, both, width_mm)}
