#!/usr/bin/env python3
"""Synthetic two-view windowsill scene for mid-view synthesis tests.

Left and right camera frames share a textured sill with a clear corner
landmark and a foreground ornament that moves by a different amount. A thin
recessed strip is occluded by the ornament in both frames; a wider strip is
visible in one of the two frames.
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

W, H = 640, 480
BG_SHIFT = 26          # background corner moves 26 px between captures
FG_SHIFT = 58          # ornament moves more (parallax)
ORNAMENT = (196, 92, 64)
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)


def background(shift):
    image = Image.new("RGB", (W, H), (214, 206, 188))
    draw = ImageDraw.Draw(image)
    rng = np.random.default_rng(7)
    # Textured wall with a deterministic speckle pattern (world-fixed texture).
    for _ in range(2600):
        x = int(rng.integers(0, W))
        y = int(rng.integers(0, 320))
        shade = int(rng.integers(150, 225))
        draw.point(((x - shift) % W, y), fill=(shade, shade - 8, shade - 22))
    # Windowsill slab and its clear top-right corner landmark.
    draw.rectangle((0, 320, W, H), fill=(158, 132, 104))
    for y in range(320, H, 9):
        draw.line((0, y - shift % 9, W, y - shift % 9), fill=(138, 114, 88), width=1)
    # World-fixed corner appears at different frame positions (camera moved right).
    corner_world_x = 470
    cx = corner_world_x - shift
    draw.polygon([(cx, 300), (cx + 34, 300), (cx + 34, 352), (cx, 352)],
                 fill=(40, 70, 92))
    draw.rectangle((cx + 4, 304, cx + 12, 312), fill=(240, 220, 120))
    return np.asarray(image, dtype=np.uint8)


def ornament_pixels(center_x, center_y=250, radius_x=58, radius_y=86):
    yy, xx = np.mgrid[0:H, 0:W]
    body = ((xx - center_x) / radius_x) ** 2 + ((yy - center_y) / radius_y) ** 2 <= 1
    # A short trailing arm over the recess band. Both captures' arms cover the
    # band at its full height, while the interpolated middle arm slides away,
    # exposing background that neither capture photographed.
    arm = (xx >= 244) & (xx <= 266) & (yy >= 146) & (yy <= 356)
    return body | arm


def render(shift, center_x, path, radius_y=86):
    rgb = background(shift).astype(np.float32)
    mask = ornament_pixels(center_x, radius_y=radius_y)
    shaded = np.asarray(ORNAMENT, dtype=np.float32)
    yy, xx = np.mgrid[0:H, 0:W]
    shade = 1.0 - 0.35 * np.clip((xx - center_x) / 58, -1, 1)
    # Tall recess inside BOTH ornament footprints; the right capture's ornament
    # is slightly larger (handheld scale difference), so parts of it are never
    # covered by the interpolated middle ornament and must stay transparent.
    both_hidden = (xx >= 264) & (xx < 292) & (yy >= 146) & (yy < 356)
    # Slit occluded by the left ornament but visible on the right capture.
    slit_right_visible = (xx >= 200) & (xx < 208) & (yy >= 180) & (yy < 330)
    # Slit occluded by the right ornament but visible on the left capture.
    slit_left_visible = (xx >= 312) & (xx < 320) & (yy >= 180) & (yy < 330)
    for zone, tone in ((both_hidden, (62, 52, 46)),
                       (slit_right_visible, (86, 72, 60)),
                       (slit_left_visible, (96, 82, 68))):
        rgb[zone] = tone
    for band in range(3):
        rgb[..., band][mask] = shaded[band] * shade[mask]
    Image.fromarray(np.rint(np.clip(rgb, 0, 255)).astype(np.uint8)).save(path)
    return mask


def main():
    mask_l = render(0, 250, OUT / "left.png")
    mask_r = render(BG_SHIFT, 250 + FG_SHIFT, OUT / "right.png", radius_y=96)
    polygon = mask_to_polygon(mask_l), mask_to_polygon(mask_r)
    anchors = ([(470) / (W - 1), 300 / (H - 1)],
               [(470 - BG_SHIFT) / (W - 1), 300 / (H - 1)])
    import json
    (OUT / "marks.json").write_text(json.dumps({
        "polygons": polygon, "anchors": anchors,
        "bg_shift": BG_SHIFT, "fg_shift": FG_SHIFT, "size": [W, H],
    }, indent=2))
    print("wrote", OUT)


def mask_to_polygon(mask):
    # Tight convex hull of the mask in normalized coordinates, so the arm is
    # included rather than approximated by an ellipse bounding box.
    points = np.argwhere(mask)[:, ::-1].astype(np.float64)
    points = points[np.lexsort((points[:, 1], points[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for point in points[::max(1, len(points) // 4000)]:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper = []
    for point in points[::-1][::max(1, len(points) // 4000)]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    hull = np.vstack((lower[:-1], upper[:-1]))
    return [[round(float(x) / (W - 1), 5), round(float(y) / (H - 1), 5)] for x, y in hull]


if __name__ == "__main__":
    main()
