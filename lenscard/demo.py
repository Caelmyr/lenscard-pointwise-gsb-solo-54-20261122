"""Procedural scenes for lens angle inspection; every synthetic capture is identified."""
from dataclasses import dataclass
from functools import lru_cache
from io import BytesIO
import math
import random
from threading import Lock

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

from .calibration import teaching_ticks
from .imaging import frame_input
from .models import Plan
from .opticaltest import simulated_photograph
from .visual import png_bytes


DEMO_LOCK = Lock()
RUNTIME_SHA = "LensCard synthetic scenes, v1"
W = 690
H = 910
LAYERS = {"far": 1.8, "mid": 4.4, "near": 7.4}


@dataclass(frozen=True)
class SampleCard:
    id: str
    title: str
    subtitle: str
    description: str
    views: tuple[float, ...]
    palette: dict
    scale: float
    lens: dict

    def menu(self):
        return {
            "id": self.id, "title": self.title, "subtitle": self.subtitle,
            "description": self.description, "view_count": len(self.views), "synthetic": True,
        }


HARBOR = SampleCard(
    "harbor", "Harbour at blue hour", "A lighthouse sequence",
    "Three simulated views with foreground prismatic buoys, a lighthouse on the middle plane and a steady horizon.",
    (-18.0, 0.0, 18.0),
    {"sky_top": (28, 53, 79), "sky_bottom": (160, 117, 107), "cloud": (231, 201, 173), "water": (27, 87, 102),
     "ground": (39, 77, 79), "accent": (243, 168, 84), "tower": (216, 210, 177), "buoy": (237, 107, 68)},
    1, {"lpi": 46.8, "phase": -0.142, "shear": .011, "order": "forward", "latitude": 24},
)
CANOPY = SampleCard(
    "canopy", "Terrace in the rain", "An illustrated depth target",
    "Four angle originals of a courtyard, with a split striped canopy and a foreground paper lantern.",
    (-20.0, -6.0, 7.0, 21.0),
    {"sky_top": (82, 77, 112), "sky_bottom": (206, 151, 113), "cloud": (240, 196, 145), "water": (80, 98, 90),
     "ground": (63, 92, 70), "accent": (243, 177, 88), "tower": (229, 199, 159), "buoy": (243, 171, 91)},
    .82, {"lpi": 54, "phase": .118, "shear": -.017, "order": "reverse", "latitude": 25},
)
SAMPLES = {sample.id: sample for sample in (HARBOR, CANOPY)}


def color_mix(left, right, fraction):
    return tuple(round(a * (1 - fraction) + b * fraction) for a, b in zip(left, right, strict=True))


@lru_cache(maxsize=16)
def text_font(size):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except (OSError, ValueError):
        return ImageFont.load_default(size=size)


def sky_finish(art, sample, seed=14):
    colors = sample.palette
    canvas = Image.new("RGBA", (W, H))
    draw = ImageDraw.Draw(canvas)
    horizon = 493
    for row in range(horizon):
        phase = min(1, row / horizon)
        rgba = color_mix(colors["sky_top"], colors["sky_bottom"], phase ** 1.6) + (255,)
        draw.line((0, row, W, row), fill=rgba)
    for row in range(horizon, H):
        deeper = color_mix(colors["water"], colors["sky_top"], (row - horizon) / (H - horizon) * .8)
        draw.line((0, row, W, row), fill=deeper + (255,))
    distance = random.Random(seed)
    ripples = Image.new("RGBA", art.size)
    translucent = ImageDraw.Draw(ripples)
    for _ in range(286):
        x = distance.randrange(-60, W)
        y = distance.randrange(horizon, H)
        reach = distance.randrange(3, 52) + int((y - horizon) / 38)
        tone = (tuple(colors["cloud"]) + (distance.randrange(7, 43),)) if distance.random() > .38 else (25, 44, 59, 51)
        translucent.line((x, y, x + reach, y), fill=tone, width=1 + (y > 760))
    canvas.alpha_composite(ripples)
    art.alpha_composite(canvas)


def atmosphere(art, sample, observer):
    colors = sample.palette
    movement = round(observer * LAYERS["far"] * .15)
    shell = Image.new("RGBA", art.size)
    draw = ImageDraw.Draw(shell)
    sun_x, sun_y, span = 538 - movement, 240, 72
    for radius in range(span * 2, 0, -1):
        age = max(0.0, 1 - radius / (span * 2))
        strength = round(10 + 190 * age ** 2)
        draw.ellipse((sun_x - radius, sun_y - radius, sun_x + radius, sun_y + radius),
                     fill=tuple(colors["accent"]) + (strength,))
    cloud = random.Random(424)
    for n in range(19):
        x = cloud.randrange(-100, W)
        y = cloud.randrange(145, 439)
        length = cloud.randrange(90, 320)
        depth = cloud.randrange(8, 25)
        fade = cloud.randrange(8, 42)
        draw.ellipse((x + movement, y, x + length + movement, y + depth), fill=tuple(colors["cloud"]) + (fade,))
    art.alpha_composite(shell.filter(ImageFilter.GaussianBlur(4.5)))


def coastline(art, sample, observer):
    palette = sample.palette
    offset = round(observer * LAYERS["far"])
    draw = ImageDraw.Draw(art)
    back = (50, 77, 80) if sample.id == "harbor" else (58, 78, 76)
    draw.polygon([(-120, 465), (3 + offset, 413), (92 + offset, 440), (228 + offset, 400),
                  (325 + offset, 430), (393 + offset, 424), (522 + offset, 471),
                  (735, 480), (775, 550), (-120, 550)], fill=back)
    generator = random.Random(239)
    coast = []
    for x in range(-100, W + 100, 7):
        y = round(528 + 13 * math.sin(x * .028) + generator.uniform(-5, 5))
        coast.append((x + offset, y))
    foreground = coast + [(W + 200, 600), (-150, 582)]
    draw.polygon(foreground, fill=palette["ground"])
    draw.line(coast, fill=color_mix(palette["ground"], palette["cloud"], .22), width=2)


def beacon(art, sample, observer):
    palette = sample.palette
    movement = round(observer * LAYERS["mid"] * sample.scale)
    x, y = 330 + movement, 498
    draw = ImageDraw.Draw(art)
    shade = color_mix(palette["tower"], (36, 60, 62), .41)
    shape = [(x - 37, y), (x - 28, y - 182), (x + 25, y - 182), (x + 35, y)]
    draw.polygon(shape, fill=palette["tower"], outline=(39, 65, 62), width=2)
    draw.polygon([(x + 7, y), (x + 7, y - 182), shape[2], shape[3]], fill=shade)
    for band in range(3):
        surface = y - 139 + band * 50
        draw.polygon([(x - 32, surface), (x + 29, surface), (x + 31, surface + 15), (x - 33, surface + 15)],
                     fill=palette["buoy"])
    draw.rectangle((x - 35, y - 199, x + 32, y - 181), fill=(45, 59, 65))
    draw.rectangle((x - 28, y - 226, x + 25, y - 200), fill=palette["accent"], outline=(73, 65, 53), width=3)
    draw.polygon([(x - 37, y - 225), (x, y - 262), (x + 35, y - 225)], fill=(54, 61, 63))
    draw.arc((x - 21, y - 248, x + 18, y - 206), 205, 334, fill=(199, 164, 113), width=2)
    draw.line((x, y - 249, x, y - 281), fill=(55, 65, 60), width=3)


def lantern(art, sample, observer):
    move = round(observer * 5.75)
    draw = ImageDraw.Draw(art)
    x, y = 422 + move, 411
    pole = (55, 68, 61)
    cloth = sample.palette["accent"]
    draw.line((x, 0, x, y - 79), fill=pole, width=7)
    draw.rounded_rectangle((x - 71, y - 83, x + 71, y + 61), radius=41, fill=cloth, outline=pole, width=3)
    draw.ellipse((x - 47, y + 43, x + 47, y + 63), fill=(142, 94, 68))
    for shift in (-49, -30, -11, 11, 31, 50):
        draw.line((x + shift, y - 69, x + shift, y + 48), fill=(215, 137, 75), width=3)
    draw.line((x, y + 61, x, y + 96), fill=cloth, width=6)


def foreground(art, sample, observer):
    draw = ImageDraw.Draw(art)
    movement = round(observer * LAYERS["near"] * sample.scale)
    rng = random.Random(813)
    for index in range(22):
        x = rng.randrange(-100, W + 100) + movement
        y = rng.randrange(663, H + 60)
        size = rng.randrange(12, 37)
        turn = rng.randrange(0, 3)
        new_x = x - 29 * turn
        stem = (20, 52, 62)
        draw.line((new_x, H + 80, x, y), fill=stem, width=max(2, size // 8))
        shape = [(x, y - size), (x + size * .67, y), (x + size * .24, y + size * .65),
                 (x - size * .41, y + size * .39)]
        tag = sample.palette["buoy"] if index % 4 == 0 else color_mix(stem, sample.palette["ground"], .44)
        draw.polygon(shape, fill=tag)
        draw.line((x, y - size, x - 4, y + size * .45), fill=color_mix(tag, (250, 225, 178), .32), width=1)


def annotate(art, sample):
    layer = Image.new("RGBA", art.size)
    draw = ImageDraw.Draw(layer)
    draw.rectangle((30, 30, W - 31, H - 31), outline=(249, 238, 207, 108), width=1)
    draw.text((59, 63), "LENS CARD   /   FIELD SERIES", fill=(249, 231, 198, 220), font=text_font(13))
    draw.text((57, H - 170), sample.title, fill=(255, 240, 209, 243), font=text_font(45))
    draw.text((59, H - 100), "SYNTHETIC MULTI-VIEW SOURCE / OPTICAL SAMPLE", fill=(250, 234, 201, 224), font=text_font(12))
    art.alpha_composite(layer)


@lru_cache(maxsize=24)
def demo_original(sample_id, angle):
    sample = SAMPLES[sample_id]
    art = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    sky_finish(art, sample)
    atmosphere(art, sample, angle)
    coastline(art, sample, angle)
    if sample_id == "harbor":
        beacon(art, sample, angle)
    else:
        lantern(art, sample, angle)
    foreground(art, sample, angle)
    annotate(art, sample)
    output = BytesIO()
    art.convert("RGB").save(output, "JPEG", quality=94, subsampling=0, optimize=True)
    return output.getvalue()


def menu(library):
    for sample in SAMPLES.values():
        yield {**sample.menu(), "saved_card": library.locate_demo(sample.title), "cover": f"/api/samples/{sample.id}/cover"}


def create_sample(library, sample):
    with DEMO_LOCK:
        existing = library.locate_demo(sample.title)
        if existing:
            return library.get_card(existing)
        plan = {
            "lens": sample.lens, "sheet": {"width_mm": 47, "height_mm": 62, "dpi": 400, "paper": "warm"},
            "framing": {"mode": "center", "cover": "fill", "fit_motion": False},
        }
        card = library.create_card(sample.title, "Procedural illustration / " + sample.description, plan=plan, synthetic=True)
        complete = []
        for n, angle in enumerate(sample.views, start=1):
            name = f"{sample.id}-view-{n}.jpg"
            raw = demo_original(sample.id, angle)
            frame = frame_input(raw, name, angle, library)
            complete.append((frame, {"title": f"Angle {angle:+g}°", "position": angle, "enabled": True}))
        library.add_views(card["id"], complete, version=0)
        return library.get_card(card["id"])


def sample_measurements(library, card_id):
    card, originals = library.material_snapshot(card_id)
    if not card["synthetic"]:
        from .models import Unprocessable
        raise Unprocessable("Procedural capture examples are only for marked synthetic cards.")
    plan = Plan.parse(card["config"])
    active = [view for view in originals if view["enabled"]]
    count = len(active)
    # The upload budget is seven photographs; keep two reflective captures per
    # angle while that fits, otherwise one deliberately masked capture per angle.
    multi_photo = 2 * count <= 7
    settings = []
    files = []
    for n, record in enumerate(active, start=1):
        image_a, frame = simulated_photograph(
            active, plan, n, glare_rows=tuple(range(n + 1, count + 1)), glare_strength=.55,
        )
        label_a = f"Synthetic {record['position']:+g}° photo A (lower glare)"
        if not multi_photo:
            files.append((f"synthetic-photograph-{n}.png", image_a))
            settings.append({
                "angle": record["position"], "expected": n, "frame": frame,
                "name": f"Synthetic at {record['position']:+g}°",
                "photos": [{"name": label_a, "frame": frame, "exclude": [count] if count > 2 else []}],
            })
            continue
        image_b, _ = simulated_photograph(
            active, plan, n, glare_rows=tuple(range(1, n)), glare_strength=.55,
        )
        # The first angle also reflects on its own numbered band in every
        # photograph, so that patch stays deliberately unresolved (blank).
        if n == 1:
            image_a, frame = simulated_photograph(
                active, plan, n, glare_rows=(1, *range(n + 1, count + 1)), glare_strength=.55,
            )
            image_b, _ = simulated_photograph(
                active, plan, n, glare_rows=(1, *range(1, n)), glare_strength=.55,
            )
        label_b = f"Synthetic {record['position']:+g}° photo B (upper glare)"
        files.append((f"synthetic-photograph-{n}-a.png", image_a))
        files.append((f"synthetic-photograph-{n}-b.png", image_b))
        settings.append({
            "angle": record["position"], "expected": n, "frame": frame,
            "name": f"Synthetic at {record['position']:+g}°",
            "photos": [
                {"name": label_a, "frame": frame, "exclude": [count] if count > 2 else []},
                {"name": label_b, "frame": frame, "exclude": []},
            ],
        })
    marks = teaching_ticks(plan.sheet, plan.lens)
    return {"files": files, "settings": settings, "count": len(files),
            "angle_count": len(active), "multi_photo": multi_photo, "ticks": marks}
