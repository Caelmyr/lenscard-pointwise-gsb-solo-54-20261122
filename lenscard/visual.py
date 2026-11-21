"""PNG evidence views and measured source charts."""
from io import BytesIO
from math import ceil

import numpy as np
from PIL import Image, ImageDraw

from .imaging import fit_size
from .models import Unprocessable


def png_bytes(image: Image.Image | np.ndarray) -> bytes:
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    output = BytesIO()
    image.save(output, "PNG", compress_level=5)
    return output.getvalue()


def rounding(value, digits: int = 6) -> float | None:
    scalar = float(value)
    return round(scalar, digits) if np.isfinite(scalar) else None


def paired_bounds(data: np.ndarray, valid: np.ndarray, low=2, high=98) -> tuple[float, float]:
    values = data[valid & np.isfinite(data)]
    if values.size < 4:
        raise Unprocessable("There are too few supported pixels.")
    minimum, maximum = np.percentile(values, [low, high])
    return float(minimum), float(maximum)


def code_statistics(data: np.ndarray) -> dict:
    values = np.asarray(data)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"samples": 0, "p05": None, "median": None, "p95": None,
                "mean": None, "stddev": None}
    p05, p50, p95 = np.percentile(values, [5, 50, 95])
    return {
        "samples": int(values.size),
        "mean": rounding(values.mean()),
        "stddev": rounding(values.std()),
        "p05": rounding(p05),
        "median": rounding(p50),
        "p95": rounding(p95),
    }


def masked_rgb(rgb: np.ndarray, valid: np.ndarray, *, invalid=(42, 53, 63)) -> Image.Image:
    array = np.clip(np.nan_to_num(rgb), 0, 1)
    encoded = np.rint(array * 255).astype(np.uint8)
    encoded[~valid] = invalid
    return Image.fromarray(encoded, "RGB")


def scalar_palette(values: np.ndarray, stops: tuple[tuple[int, int, int], ...]) -> np.ndarray:
    safe = np.clip(np.nan_to_num(values), 0, 1)
    domain = np.linspace(0, 1, len(stops))
    palette = np.asarray(stops, dtype=np.float32) / 255
    return np.stack([np.interp(safe, domain, band) for band in palette.T], axis=-1)


SEQUENTIAL = ((27, 26, 65), (38, 80, 122), (24, 140, 142), (105, 193, 111), (249, 224, 102))
SIGNED = ((31, 68, 155), (99, 157, 224), (237, 241, 239), (236, 158, 89), (174, 50, 50))


def heatmap(values: np.ndarray, valid: np.ndarray, lo: float, hi: float, *, diverging=False) -> Image.Image:
    scale = hi - lo or 1.0
    mapped = (values - lo) / scale
    palette = SIGNED if diverging else SEQUENTIAL
    return masked_rgb(scalar_palette(mapped, palette), valid)


def resize_report(image: Image.Image, max_side: int = 920) -> Image.Image:
    if max(image.size) <= max_side:
        return image
    width, height = fit_size(image.size, max_side)
    return image.resize((width, height), Image.Resampling.BILINEAR)


def overlay_regions(
    image: Image.Image,
    regions: list[tuple[tuple[float, float, float, float], str, tuple]],
) -> Image.Image:
    result = image.copy()
    draw = ImageDraw.Draw(result)
    width, height = result.size
    for box, text, color in regions:
        left, top, right, bottom = box
        x0 = max(0, min(width - 1, round(left * width)))
        y0 = max(0, min(height - 1, round(top * height)))
        x1 = max(x0, min(width - 1, round(right * width)))
        y1 = max(y0, min(height - 1, round(bottom * height)))
        draw.rectangle((x0, y0, x1, y1), outline=color, width=2)
        if text:
            tw = ceil(draw.textlength(text))
            tx = min(max(0, width - tw - 7), x0 + 4)
            ty = min(height - 15, y0 + 4)
            draw.rectangle((tx - 1, ty - 1, tx + tw + 4, ty + 12), fill=(21, 35, 46))
            draw.text((tx + 1, ty), text, fill=color)
    return result


def side_by_side(first: Image.Image, second: Image.Image, gap: int = 7) -> Image.Image:
    result = Image.new("RGB", (first.width + second.width + gap, max(first.height, second.height)), (21, 35, 46))
    result.paste(first, (0, 0))
    result.paste(second, (first.width + gap, 0))
    return result


def chart_png(
    series: list[tuple[str, np.ndarray, np.ndarray]],
    *,
    x_label: str,
    y_label: str,
    x_bounds=None,
    y_bounds=None,
    markers: list[tuple[float, str]] | None = None,
) -> bytes:
    def extent(axis, bounds):
        if bounds is not None:
            return bounds
        blocks = [entry[axis][np.isfinite(entry[axis])] for entry in series]
        available = [part for part in blocks if part.size]
        if not available:
            return 0, 1
        minimum = min(float(np.min(part)) for part in available)
        maximum = max(float(np.max(part)) for part in available)
        span = maximum - minimum
        return (minimum - span * .06, maximum + span * .06) if span else (minimum - .05, maximum + .05)

    x_min, x_max = extent(1, x_bounds)
    y_min, y_max = extent(2, y_bounds)
    width, height = 760, 356
    left, right, top, bottom = 78, 732, 45, height - 63
    image = Image.new("RGB", (width, height), (19, 29, 40))
    draw = ImageDraw.Draw(image)
    ink = (220, 229, 237)
    grid = (56, 77, 90)
    colors = [(90, 207, 180), (247, 191, 113), (133, 181, 252), (245, 133, 162),
              (191, 157, 237), (180, 219, 107)]

    def screen(x, y):
        return (round(left + (right - left) * np.clip((x - x_min) / (x_max - x_min), 0, 1)),
                round(bottom - (bottom - top) * np.clip((y - y_min) / (y_max - y_min), 0, 1)))

    for proportion in np.linspace(0, 1, 6):
        x_value = x_min + proportion * (x_max - x_min)
        y_value = y_min + proportion * (y_max - y_min)
        x, y = screen(x_value, y_value)
        draw.line((left, y, right, y), fill=grid)
        draw.line((x, top, x, bottom), fill=grid)
        draw.text((left - 8, y), f"{y_value:.3g}", fill=ink, anchor="rm")
        draw.text((x, bottom + 6), f"{x_value:.3g}", fill=ink, anchor="mt")
    draw.text((10, 15), y_label, fill=ink)
    draw.text((right, bottom + 24), x_label, fill=ink, anchor="rt")
    for x_value, caption in markers or []:
        if x_min <= x_value <= x_max:
            x, _ = screen(x_value, y_min)
            draw.line((x, top, x, bottom), fill=(173, 164, 108), width=1)
            draw.text((x + 4, top + 3), caption, fill=ink)
    for index, (name, xs, ys) in enumerate(series):
        color = colors[index % len(colors)]
        prev = None
        for x_value, y_value in zip(xs, ys, strict=True):
            if not np.isfinite(x_value) or not np.isfinite(y_value):
                prev = None
                continue
            point = screen(x_value, y_value)
            if prev == point:
                continue
            if prev is not None:
                draw.line((*prev, *point), fill=color, width=2)
            else:
                draw.ellipse((point[0] - 2, point[1] - 2, point[0] + 2, point[1] + 2), fill=color)
            prev = point
        legend_x = 14 + (index % 3) * 235
        legend_y = height - 24 - (index // 3) * 17
        draw.line((legend_x, legend_y + 5, legend_x + 24, legend_y + 5), fill=color, width=3)
        draw.text((legend_x + 30, legend_y), name, fill=ink)
    return png_bytes(image)
