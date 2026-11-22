"""Procedural teaching photographs for the capture-order tool.

Two fixed wall points sit on a plane far beyond the objects, two foreground objects
stand in front of them, and each photograph is taken from a different rail position
with a hand-held roll and a slight zoom. The photos are labelled as synthetic and
carry no camera metadata. They exist so the ordering maths can be tried end to end
without a camera; they are never presented as real measurements.
"""

from dataclasses import dataclass
from functools import lru_cache
import math

from PIL import Image, ImageDraw, ImageFont

from .visual import png_bytes

WIDTH = 1280
HEIGHT = 960
FOCAL = 1150.0
BORDER = (249, 238, 207)

# Wall points (normalized, on the far plane) and foreground corners with their
# parallax gains relative to the wall baseline.
WALL_POINTS = ((-0.34, -0.08), (0.32, 0.04))
NEAR_POINTS = (("planter corner", -0.06, 0.22, 0.46),
               ("post edge", 0.07, -0.18, 0.27))


@dataclass(frozen=True)
class SyntheticShot:
    title: str
    rail: float
    roll_degrees: float
    zoom: float
    hidden: tuple[str, ...] = ()


# Rail positions are in metres. Shots 01 and 02 were taken without moving the
# feet and differ only by hand rotation, so they share one rail position.
SHOTS = (
    SyntheticShot("handheld-01.png", -1.55, 16.0, 1.040),
    SyntheticShot("handheld-02.png", -1.5, -12.0, 0.984, hidden=("post edge",)),
    SyntheticShot("handheld-03.png", -0.4, 4.0, 1.012),
    SyntheticShot("handheld-04.png", 0.7, -7.0, 0.970),
    SyntheticShot("handheld-05.png", 1.8, 13.0, 1.028),
)


def _font(size):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except (OSError, ValueError):
        return ImageFont.load_default(size=size)


def _project(x, y, roll, zoom, pan):
    """Wall-plane coordinates to pixel positions, with a lens-like zoom and roll."""
    cosine, sine = math.cos(math.radians(roll)), math.sin(math.radians(roll))
    px, py = x * FOCAL * zoom - pan, y * FOCAL * zoom
    return (WIDTH / 2 + cosine * px - sine * py, HEIGHT / 2 + sine * px + cosine * py)


def _background(far_pan):
    """Wall and window frame, shifted together with the wall points."""
    image = Image.new("RGB", (WIDTH, HEIGHT), (38, 44, 56))
    draw = ImageDraw.Draw(image)
    for row in range(HEIGHT):
        shade = 34 + round(26 * row / HEIGHT)
        draw.line((0, row, WIDTH, row), fill=(shade, shade + 6, shade + 16))
    for index in range(9):
        left = 60 + index * 148 - far_pan * 0.85
        draw.rectangle((left, 90, left + 122, HEIGHT - 90), outline=(62, 70, 86), width=2)
    return image


def _scene(shot: SyntheticShot):
    far_pan = shot.rail * 44
    image = _background(far_pan)
    draw = ImageDraw.Draw(image)
    # Far plane landmarks translate with the wall.
    for x, y in WALL_POINTS:
        px, py = _project(x, y, shot.roll_degrees, shot.zoom, far_pan)
        draw.ellipse((px - 7, py - 7, px + 7, py + 7), fill=(228, 214, 176), outline=(60, 66, 80), width=2)
    # Mid-ground: a low table edge, further parallax.
    table = [(-0.55, 0.12), (0.55, 0.10), (0.55, 0.22), (-0.55, 0.25)]
    points = [_project(x + 0.22 * shot.rail, y, shot.roll_degrees, shot.zoom, far_pan) for x, y in table]
    draw.polygon(points, fill=(96, 78, 64), outline=(64, 52, 44))
    # Foreground objects: corner plus body, with strong parallax.
    colors = {"planter corner": (196, 120, 78), "post edge": (120, 158, 146)}
    for name, x, y, gain in NEAR_POINTS:
        if name in shot.hidden:
            continue
        # The marked point is the object's visible top-left corner.
        moved = x - gain * shot.rail * 0.34
        left, top = _project(moved, y, shot.roll_degrees, shot.zoom, far_pan)
        size = 150 * shot.zoom
        draw.rounded_rectangle((left, top, left + size * 0.9, top + size), radius=14,
                               fill=colors[name], outline=(34, 30, 26), width=3)
        draw.line((left, top, left + 34, top), fill=(240, 232, 210), width=4)
        draw.line((left, top, left, top + 34), fill=(240, 232, 210), width=4)
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    caption = ImageDraw.Draw(overlay)
    caption.text((26, 22), "SYNTHETIC HANDHELD SET  /  NOT A CAMERA RECORD", font=_font(18),
                 fill=BORDER + (220,))
    caption.text((26, HEIGHT - 44), "Wall points and foreground corners are procedural", font=_font(15),
                 fill=BORDER + (190,))
    image.paste(Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB"))
    return image


@lru_cache(maxsize=8)
def shot_png(index: int) -> tuple[str, bytes]:
    shot = SHOTS[index]
    return shot.title, png_bytes(_scene(shot))


def teaching_set():
    """Marks the operator would place, plus the true left-to-right order for checking."""
    marks, order = [], []
    for index, shot in enumerate(SHOTS):
        far_pan = shot.rail * 44
        marks.append({"index": index, "title": shot.title, "kind": "far", "signature": "wall left",
                      "point": _normalized(_project(WALL_POINTS[0][0], WALL_POINTS[0][1],
                                                    shot.roll_degrees, shot.zoom, far_pan))})
        marks.append({"index": index, "title": shot.title, "kind": "far", "signature": "wall right",
                      "point": _normalized(_project(WALL_POINTS[1][0], WALL_POINTS[1][1],
                                                    shot.roll_degrees, shot.zoom, far_pan))})
        for name, x, y, gain in NEAR_POINTS:
            if name in shot.hidden:
                continue
            marks.append({"index": index, "title": shot.title, "kind": "near", "signature": name,
                          "point": _normalized(_project(x - gain * shot.rail * 0.34, y,
                                                        shot.roll_degrees, shot.zoom, far_pan))})
        order.append(shot.title)
    ranked = sorted(range(len(SHOTS)), key=lambda index: SHOTS[index].rail)
    return {
        "shots": [{"index": index, "title": shot.title, "rail": shot.rail,
                   "roll_degrees": shot.roll_degrees, "zoom": shot.zoom}
                  for index, shot in enumerate(SHOTS)],
        "marks": marks,
        "expected_order": [SHOTS[index].title for index in ranked],
        "expected_left": SHOTS[ranked[0]].title,
        "expected_right": SHOTS[ranked[-1]].title,
        "size": [WIDTH, HEIGHT],
    }


def _normalized(point):
    x, y = point
    return [round(min(1.0, max(0.0, x / WIDTH)), 6), round(min(1.0, max(0.0, y / HEIGHT)), 6)]
