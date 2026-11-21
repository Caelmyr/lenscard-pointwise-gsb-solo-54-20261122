"""Two-view middle synthesis with one foreground layer and honest disocclusions.

The operator circles the foreground ornament once on each capture and marks the
same clear background corner on both. Background motion comes from the corner
pair and foreground motion from the mask pair, so the two depths move
independently. The ornament is pasted a single time; background gaps are filled
only from pixels one of the captures actually saw, and anything both captures
occluded stays transparent.
"""
from dataclasses import dataclass
from hashlib import sha256

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from .alignment import transform_layer
from .imaging import TOTAL_PROCESS_PIXELS, fit_size, linear_to_srgb, read_capture, read_pixels
from .models import Conflict, Unprocessable, number, parse_vertex
from .storage import encode_json
from .visual import png_bytes

PREVIEW_LONG_EDGE = 1000
FULL_LONG_EDGE = 1400
MAX_POLYGON_POINTS = 400
MIN_MASK_FRACTION = 0.0008
FEATHER_PIXELS = 1.1
CHECK_TILE = 18


def parse_polygon(value, label):
    if not isinstance(value, list) or not 3 <= len(value) <= MAX_POLYGON_POINTS:
        raise Unprocessable(f"{label} needs 3 to {MAX_POLYGON_POINTS} marked points.", field=label)
    return [parse_vertex(point, label) for point in value]


def parse_anchor_pair(value, label):
    if not isinstance(value, list) or len(value) != 2:
        raise Unprocessable(f"{label} needs one marked corner per capture.", field=label)
    return [parse_vertex(point, label) for point in value]


def parse_request(body):
    left_id, right_id = body.get("left_view"), body.get("right_view")
    if not isinstance(left_id, str) or not isinstance(right_id, str) or left_id == right_id:
        raise Unprocessable("Choose two different angle views.")
    mid = number(body.get("mid", 0.5), "middle position", 0.001, 0.999)
    return (
        left_id, right_id,
        parse_polygon(body.get("left_polygon"), "left ornament outline"),
        parse_polygon(body.get("right_polygon"), "right ornament outline"),
        parse_anchor_pair(body.get("anchors"), "background corner"),
        mid,
    )


def raster_polygon(points, width, height, label):
    image = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(image)
    draw.polygon([(x * (width - 1), y * (height - 1)) for x, y in points], fill=255)
    mask = np.asarray(image, dtype=bool)
    if mask.sum() < max(16, int(MIN_MASK_FRACTION * width * height)):
        raise Unprocessable(f"{label} covers too few pixels; circle the whole ornament.")
    return mask


def mask_centroid(mask):
    points = np.argwhere(mask)
    return points[:, 1].mean(), points[:, 0].mean()


def foreground_shift(left_mask, right_mask, rgb_left, rgb_right):
    """Ornament translation in right-minus-left pixels, with centroid fallback.

    Normalized cross-correlation is evaluated on the masked ornament crops in
    a zero-padded window large enough to hold the full parallax jump, so a
    small ornament is not suppressed by a full-frame taper window.
    """
    lx, ly = mask_centroid(left_mask)
    rx, ry = mask_centroid(right_mask)
    centroid_dx, centroid_dy = rx - lx, ry - ly
    height, width = left_mask.shape
    rows_l, cols_l = np.where(left_mask)
    rows_r, cols_r = np.where(right_mask)
    r0, r1 = max(rows_l.min(), rows_r.min()), min(rows_l.max(), rows_r.max())
    c0, c1 = max(cols_l.min(), cols_r.min()), min(cols_l.max(), cols_r.max())
    if r1 - r0 < 8 or c1 - c0 < 8:
        return centroid_dx, centroid_dy, "centroid"
    gray_l = rgb_left.mean(axis=2)
    gray_r = rgb_right.mean(axis=2)
    pad_y = max(int(abs(centroid_dy)) + 12, 12)
    pad_x = max(int(abs(centroid_dx)) + 12, 12)
    win_h = min(height, r1 - r0 + 1 + 2 * pad_y)
    win_w = min(width, c1 - c0 + 1 + 2 * pad_x)
    cy, cx = (r0 + r1) // 2, (c0 + c1) // 2
    y0, x0 = max(0, cy - win_h // 2), max(0, cx - win_w // 2)
    y1, x1 = min(height, y0 + win_h), min(width, x0 + win_w)
    crop_l = (gray_l[y0:y1, x0:x1] - gray_l[left_mask].mean()) * left_mask[y0:y1, x0:x1]
    crop_r = (gray_r[y0:y1, x0:x1] - gray_r[right_mask].mean()) * right_mask[y0:y1, x0:x1]
    correlation = np.fft.irfft2(np.fft.rfft2(crop_l) * np.conjugate(np.fft.rfft2(crop_r)))
    energy = np.sqrt(float((crop_l ** 2).sum() * (crop_r ** 2).sum()))
    # The cross-spectrum peak sits at left-minus-right displacement; negate it
    # to report the right-minus-left translation of the ornament.
    dy, dx = np.unravel_index(int(np.argmax(correlation)), correlation.shape)
    dy = -(dy - correlation.shape[0] if dy > correlation.shape[0] // 2 else dy)
    dx = -(dx - correlation.shape[1] if dx > correlation.shape[1] // 2 else dx)
    if energy <= 1e-12 or correlation.max() / energy < 0.12:
        return centroid_dx, centroid_dy, "centroid"
    # Reject a phase-correlation lock that disagrees with the marked-mask
    # centroid by more than the mask itself spans; thin protrusions can
    # otherwise drag the peak onto a wrong structure.
    span_x = max(cols_l.max() - cols_l.min(), cols_r.max() - cols_r.min())
    span_y = max(rows_l.max() - rows_l.min(), rows_r.max() - rows_r.min())
    if abs(dx - centroid_dx) > span_x or abs(dy - centroid_dy) > span_y:
        return centroid_dx, centroid_dy, "centroid"
    return float(dx), float(dy), "correlation"


def feather(mask):
    soft = ndimage.gaussian_filter(ndimage.binary_dilation(mask, iterations=1).astype(np.float32), FEATHER_PIXELS)
    return np.clip(soft, 0, 1)


@dataclass
class Synthesis:
    rgba: np.ndarray
    geometry: dict
    attribution: dict
    warnings: list


def _warp_rgb(rgb, dx, dy):
    out = np.empty_like(rgb)
    for band in range(3):
        out[..., band] = transform_layer(rgb[..., band], dx, dy, 1.0, order=1)
    return out


def _warp_plane(plane, dx, dy):
    return transform_layer(plane, dx, dy, 1.0, order=1)


def synthesize(rgb_left, rgb_right, mask_left, mask_right, anchor_left, anchor_right, mid):
    height, width = rgb_left.shape[:2]
    ax = (anchor_right[0] - anchor_left[0]) * (width - 1)
    ay = (anchor_right[1] - anchor_left[1]) * (height - 1)
    # Positive shift means content moves left->right. Half-shift toward the
    # middle frame moves left content right (+) and right content left (-).
    bg_left = (mid * ax, mid * ay)
    bg_right = (-(1 - mid) * ax, -(1 - mid) * ay)
    fdx, fdy, fmethod = foreground_shift(mask_left, mask_right, rgb_left, rgb_right)
    fg_left = (mid * fdx, mid * fdy)
    # Where a source's own ornament sits, that capture carries no background
    # information. Warp those unknown regions into the middle frame too, so a
    # place hidden behind the ornament in both captures stays transparent.
    unknown_l = _warp_plane(mask_left.astype(np.float32), *bg_left) > 0.5
    unknown_r = _warp_plane(mask_right.astype(np.float32), *bg_right) > 0.5

    bg_l_warp = _warp_rgb(rgb_left, *bg_left)
    bg_r_warp = _warp_rgb(rgb_right, *bg_right)
    bg_l_support = (_warp_plane(np.ones((height, width), np.float32), *bg_left) > 0.999) & ~unknown_l
    bg_r_support = (_warp_plane(np.ones((height, width), np.float32), *bg_right) > 0.999) & ~unknown_r
    fg_warp = _warp_rgb(rgb_left, *fg_left)
    fg_soft = feather(_warp_plane(mask_left.astype(np.float32), *fg_left) > 0.5)
    fg_hard = fg_soft > 0.5

    bg_cover = bg_l_support | bg_r_support
    from_right = (~bg_l_support) & bg_r_support
    transparent = ~(bg_cover | fg_hard)
    background = bg_l_warp.copy()
    background[from_right] = bg_r_warp[from_right]

    # Keep the soft ornament edge inside a tight dilated band so a long
    # Gaussian tail cannot ghost a second silhouette onto the background.
    fg_band = ndimage.binary_dilation(fg_hard, iterations=2)
    fg_soft = fg_soft * fg_band
    alpha = np.where(fg_hard, 1.0, np.where(bg_cover, 1.0, 0.0)).astype(np.float32)
    alpha = np.maximum(alpha, fg_soft * (~transparent))
    foreground_weight = fg_soft[..., None]
    rgb = background * (1 - foreground_weight) + fg_warp * foreground_weight
    rgb[transparent] = 0
    rgba = np.concatenate((linear_to_srgb(np.clip(rgb, 0, 1)),
                           np.rint(alpha * 255).astype(np.uint8)[..., None]), axis=2)

    total = height * width
    attribution = {
        "foreground_pixels": int(fg_hard.sum()),
        "background_from_left_pixels": int((bg_l_support & ~fg_hard).sum()),
        "background_from_right_pixels": int((from_right & ~fg_hard).sum()),
        "transparent_pixels": int(transparent.sum()),
        "transparent_fraction": round(float(transparent.sum() / total), 6),
    }
    warnings = []
    if abs(ax) + abs(ay) > 0.35 * max(width, height):
        warnings.append("The marked corners are far apart; check they are the same corner.")
    if abs(fdx) + abs(fdy) > 0.35 * max(width, height):
        warnings.append("The ornament moves a large amount between the two captures.")
    if attribution["transparent_fraction"] > 0.08:
        warnings.append("A substantial area is unseen in both captures and stays transparent.")
    geometry = {
        "mid": mid,
        "background_shift_px": [round(ax, 3), round(ay, 3)],
        "foreground_shift_px": [round(fdx, 3), round(fdy, 3)],
        "foreground_shift_method": fmethod,
        "anchor_left": list(anchor_left),
        "anchor_right": list(anchor_right),
        "mask_left_fraction": round(float(mask_left.mean()), 6),
        "mask_right_fraction": round(float(mask_right.mean()), 6),
    }
    return Synthesis(rgba, geometry, attribution, warnings)


def _load_pair(library, records, left_id, right_id, long_edge):
    by_id = {view["id"]: view for view in records}
    if left_id not in by_id or right_id not in by_id:
        raise Unprocessable("Both marked views must belong to this card.")
    left, right = by_id[left_id], by_id[right_id]
    if (left["width"], left["height"]) != (right["width"], right["height"]):
        raise Unprocessable("The two views need matching oriented dimensions.")
    if left["width"] * left["height"] <= TOTAL_PROCESS_PIXELS // 2:
        target = (left["width"], left["height"])
    else:
        target = fit_size((left["width"], left["height"]), long_edge)
    cap_left = read_pixels(library, left, target)
    cap_right = read_pixels(library, right, target)
    return left, right, cap_left, cap_right, target


def _fingerprint(left, right, polygons, anchors, mid):
    payload = {
        "left": left["sha"], "right": right["sha"],
        "left_polygon": polygons[0], "right_polygon": polygons[1],
        "anchors": anchors, "mid": mid, "size": [left["width"], left["height"]],
    }
    return sha256(encode_json(payload).encode("utf-8")).hexdigest()


def checkerboard(rgba):
    height, width = rgba.shape[:2]
    yy, xx = np.mgrid[0:height, 0:width]
    tiles = ((xx // CHECK_TILE + yy // CHECK_TILE) % 2).astype(np.uint8)
    board = np.where(tiles[..., None] == 0, 214, 168) * np.ones((height, width, 3), np.uint8)
    alpha = rgba[..., 3:4].astype(np.float32) / 255
    rgb = (rgba[..., :3].astype(np.float32) * alpha + board.astype(np.float32) * (1 - alpha))
    return np.rint(np.clip(rgb, 0, 255)).astype(np.uint8)


def marker_overlay(rgba, geometry):
    image = Image.fromarray(rgba).convert("RGBA")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    ax = geometry["anchor_left"][0] * (width - 1), geometry["anchor_left"][1] * (height - 1)
    for x, y in (ax,):
        draw.line((x - 8, y, x + 8, y), fill=(255, 170, 40, 255), width=2)
        draw.line((x, y - 8, x, y + 8), fill=(255, 170, 40, 255), width=2)
    mid_anchor = (
        (geometry["anchor_left"][0] + geometry["mid"] *
         (geometry["anchor_right"][0] - geometry["anchor_left"][0])) * (width - 1),
        (geometry["anchor_left"][1] + geometry["mid"] *
         (geometry["anchor_right"][1] - geometry["anchor_left"][1])) * (height - 1),
    )
    draw.ellipse((mid_anchor[0] - 6, mid_anchor[1] - 6, mid_anchor[0] + 6, mid_anchor[1] + 6),
                 outline=(120, 220, 255, 255), width=2)
    return np.asarray(image)


def _encode_png(rgba):
    return png_bytes(Image.fromarray(rgba, "RGBA"))


def preview_midview(library, records, request_body):
    left_id, right_id, poly_l, poly_r, anchors, mid = parse_request(request_body)
    left, right, cap_l, cap_r, (width, height) = _load_pair(
        library, records, left_id, right_id, PREVIEW_LONG_EDGE)
    mask_l = raster_polygon(poly_l, width, height, "left ornament outline")
    mask_r = raster_polygon(poly_r, width, height, "right ornament outline")
    result = synthesize(
        cap_l.rgb, cap_r.rgb, mask_l, mask_r, anchors[0], anchors[1], mid)
    report = {
        "kind": "mid-view-synthesis",
        "left_view": {"id": left["id"], "title": left["title"]},
        "right_view": {"id": right["id"], "title": right["title"]},
        "processing_size": [width, height],
        **result.geometry, **result.attribution,
        "warnings": result.warnings,
    }
    fingerprint = _fingerprint(left, right, (poly_l, poly_r), anchors, mid)
    images = {
        "middle": png_bytes(Image.fromarray(checkerboard(result.rgba), "RGB")),
        "markers": _encode_png(marker_overlay(result.rgba, result.geometry)),
    }
    return {"report": report, "images": images, "fingerprint": fingerprint}


def commit_midview(library, records, request_body):
    left_id, right_id, poly_l, poly_r, anchors, mid = parse_request(request_body)
    left, right, cap_l, cap_r, (width, height) = _load_pair(
        library, records, left_id, right_id, FULL_LONG_EDGE)
    mask_l = raster_polygon(poly_l, width, height, "left ornament outline")
    mask_r = raster_polygon(poly_r, width, height, "right ornament outline")
    result = synthesize(
        cap_l.rgb, cap_r.rgb, mask_l, mask_r, anchors[0], anchors[1], mid)
    fingerprint = _fingerprint(left, right, (poly_l, poly_r), anchors, mid)
    if request_body.get("fingerprint") != fingerprint:
        raise Conflict("The preview changed before adding. Re-open the synthesis and preview again.")
    native = (left["width"], left["height"])
    if result.rgba.shape[1::-1] != native:
        resized = Image.fromarray(result.rgba, "RGBA").resize(native, Image.Resampling.LANCZOS)
        rgba = np.asarray(resized)
    else:
        rgba = result.rgba
    png = png_bytes(Image.fromarray(rgba, "RGBA"))
    information = {
        "format": "PNG", "source_mode": "RGBA", "mode": "RGBA",
        "native_width": left["width"], "native_height": left["height"],
        "orientation": 1, "profile": "Synthesized mid view, unprofiled sRGB",
        "has_alpha": True, "bit_depth": 8,
        "synthesized": {
            "kind": "mid-view", "left_source_sha": left["sha"], "right_source_sha": right["sha"],
            "left_polygon": poly_l, "right_polygon": poly_r, "anchors": anchors, "mid": mid,
            "fingerprint": fingerprint, "geometry": result.geometry, "attribution": result.attribution,
        },
    }
    decoded = read_capture(png, "mid-view.png")
    if (decoded.width, decoded.height) != native:
        raise Unprocessable("The synthesized frame does not match the card dimensions.")
    return png, decoded, information, fingerprint, result
