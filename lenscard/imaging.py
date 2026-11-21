"""Capture decoding, calibrated linear luminance, and image I/O."""

import io
import math
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageCms, ImageOps, UnidentifiedImageError
from scipy import ndimage

from .models import Capacity, Unprocessable, capture_name

SOURCE_LIMIT = 22 * 1024 * 1024
SOURCE_PIXEL_LIMIT = 14_000_000
MIN_SIDE = 48
MAX_LONG_EDGE = 1400
TOTAL_PROCESS_PIXELS = 30_000_000
ALPHA_THRESHOLD = 0.995
SUPPORTED_MODES = {"L", "LA", "RGB", "RGBA", "P", "CMYK", "I", "I;16", "I;16B", "I;16L"}
SUPPORTED_FORMATS = {"PNG", "JPEG", "TIFF", "WEBP"}
RESAMPLING = Image.Resampling.LANCZOS
ORIENTATION = 274
TAG_NAMES = {
    271: "camera_make",
    272: "camera_model",
    33434: "exposure_seconds",
    33437: "aperture",
    34855: "iso",
    36867: "captured_at",
    37386: "focal_length_mm",
    42036: "lens_model",
}


@dataclass
class DecodedCapture:
    image: Image.Image
    information: dict

    @property
    def width(self) -> int:
        return self.image.width

    @property
    def height(self) -> int:
        return self.image.height


@dataclass
class CapturePixels:
    rgb: np.ndarray
    support: np.ndarray
    original_size: tuple[int, int]
    shape: tuple[int, int]

    @property
    def luminance(self) -> np.ndarray:
        return linear_luminance(self.rgb)


def srgb_to_linear(samples: np.ndarray) -> np.ndarray:
    return np.where(samples <= 0.04045, samples / 12.92, ((samples + 0.055) / 1.055) ** 2.4)


_SRGB_8 = srgb_to_linear(np.arange(256, dtype=np.float32) / 255)


def linear_to_srgb(samples: np.ndarray) -> np.ndarray:
    linear = np.clip(samples, 0.0, 1.0)
    display = np.where(linear <= 0.0031308, 12.92 * linear, 1.055 * linear ** (1 / 2.4) - 0.055)
    return (display * 255 + 0.5).astype(np.uint8)


def linear_luminance(rgb: np.ndarray) -> np.ndarray:
    return rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722


def _orientation_metadata(exif: dict) -> dict:
    result = {}
    for tag, name in TAG_NAMES.items():
        value = exif.get(tag)
        if value is None:
            continue
        if isinstance(value, str):
            result[name] = value[:180]
            continue
        try:
            numeric = float(value)
        except (ValueError, TypeError, ZeroDivisionError, OverflowError):
            continue
        if math.isfinite(numeric):
            result[name] = numeric
    return result


def _exif_fields(image: Image.Image) -> dict:
    exif = image.getexif()
    fields = dict(exif)
    try:
        fields.update(exif.get_ifd(34665))
    except (AttributeError, KeyError, OSError, ValueError, TypeError):
        pass
    return fields


def _display_profile(image: Image.Image, profile_bytes: bytes | None) -> tuple[Image.Image, str]:
    if image.mode in ("I", "I;16", "I;16B", "I;16L"):
        return image, "Grayscale samples"
    if image.mode == "P":
        image = image.convert("RGBA" if "transparency" in image.info else "RGB")
    if not profile_bytes:
        return image.convert("RGB") if image.mode == "CMYK" else image, "Unprofiled sRGB"
    alpha = image.getchannel("A") if "A" in image.getbands() else None
    component_mode = "L" if image.mode in ("L", "LA") else "CMYK" if image.mode == "CMYK" else "RGB"
    try:
        core = ImageCms.ImageCmsProfile(io.BytesIO(profile_bytes))
        converted = ImageCms.profileToProfile(
            image.convert(component_mode),
            core,
            ImageCms.createProfile("sRGB"),
            outputMode="RGB",
        )
    except (OSError, TypeError, ValueError, ImageCms.PyCMSError):
        fallback = image.convert("RGB") if image.mode == "CMYK" else image
        return fallback, "Unreadable ICC, sRGB assumed"
    if alpha is not None:
        converted.putalpha(alpha)
    return converted, "Converted embedded ICC to sRGB"


def read_capture(data: bytes | Path, filename: str) -> DecodedCapture:
    file = io.BytesIO(data) if isinstance(data, bytes) else data
    if isinstance(data, bytes) and not data:
        raise Unprocessable("The selected capture is empty.")
    if isinstance(data, bytes) and len(data) > SOURCE_LIMIT:
        raise Capacity("Use individual captures under 22 MiB.")
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=Image.DecompressionBombWarning)
            with Image.open(file) as image:
                image_format = str(image.format).upper()
                if image_format not in SUPPORTED_FORMATS:
                    raise Unprocessable(f"{filename} is not a supported camera image.")
                native_width, native_height = image.size
                if native_width * native_height > SOURCE_PIXEL_LIMIT:
                    raise Capacity(f"{filename} exceeds the 14-megapixel capture limit.")
                if min(native_width, native_height) < MIN_SIDE:
                    raise Unprocessable(f"{filename} needs sides of at least {MIN_SIDE} pixels.")
                if getattr(image, "n_frames", 1) != 1:
                    raise Unprocessable(f"Import each plane as a separate image. {filename} contains multiple pages.")
                encoded_mode = image.mode
                if encoded_mode not in SUPPORTED_MODES:
                    raise Unprocessable(f"{filename} uses unsupported {encoded_mode} samples.")
                fields = _exif_fields(image)
                profile = image.info.get("icc_profile")
                corrected = ImageOps.exif_transpose(image)
                corrected.load()
                if corrected.mode == "I":
                    lo, hi = corrected.getextrema()
                    if lo < 0 or hi > 65535:
                        raise Unprocessable(f"{filename} needs unsigned grayscale samples up to 16 bits.")
                pixels, profile_label = _display_profile(corrected, profile)
                info = {
                    "format": image_format,
                    "source_mode": encoded_mode,
                    "mode": pixels.mode,
                    "native_width": native_width,
                    "native_height": native_height,
                    "orientation": int(fields.get(ORIENTATION, 1)),
                    "profile": profile_label,
                    "has_alpha": "A" in pixels.getbands(),
                    "bit_depth": 16 if pixels.mode in ("I", "I;16", "I;16B", "I;16L") else 8,
                    **_orientation_metadata(fields),
                }
                return DecodedCapture(pixels.copy(), info)
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
        raise Capacity(f"{filename} exceeds the safe decoding limit.") from error
    except (UnidentifiedImageError, OSError, TypeError, SyntaxError, ValueError, OverflowError) as error:
        raise Unprocessable(f"Could not decode {filename}.") from error


def fit_size(size: tuple[int, int], long_edge: int) -> tuple[int, int]:
    width, height = size
    if max(size) <= long_edge:
        return size
    scale = long_edge / max(size)
    return round(width * scale), round(height * scale)


def processing_size(captures: list[dict]) -> tuple[int, int]:
    width, height = captures[0]["width"], captures[0]["height"]
    frame_limit = TOTAL_PROCESS_PIXELS // len(captures)
    area_cap_edge = math.floor(max(width, height) * math.sqrt(frame_limit / (width * height)))
    target_width, target_height = fit_size((width, height), min(MAX_LONG_EDGE, area_cap_edge))
    if min(target_width, target_height) < 28:
        raise Capacity("This stack is too large or narrow for reconstruction. Use fewer frames or a more compact crop.")
    return target_width, target_height


def float_resize(samples: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    if samples.shape[1::-1] == size:
        return np.asarray(samples, dtype=np.float32)
    mode_f = Image.fromarray(np.asarray(samples, dtype=np.float32))
    resized = mode_f.resize(size, RESAMPLING)
    return np.asarray(resized, dtype=np.float32)


def sample_capture(decoded: DecodedCapture, size: tuple[int, int]) -> CapturePixels:
    image = decoded.image
    if decoded.information["bit_depth"] == 16:
        gray = np.asarray(image, dtype=np.float32) / 65535
        linear = np.clip(float_resize(srgb_to_linear(gray), size), 0, 1)
        rgb = np.repeat(linear[..., None], 3, axis=2)
        alpha = np.ones(linear.shape, dtype=np.float32)
        return CapturePixels(rgb, alpha > 0.5, image.size, linear.shape)
    if "A" in image.getbands():
        rgba = np.asarray(image.convert("RGBA"))
        source = _SRGB_8[rgba[..., :3]]
        input_alpha = rgba[..., 3].astype(np.float32) / 255
        alpha = np.clip(float_resize(input_alpha, size), 0, 1)
        denominator = np.maximum(alpha, 0.001)
        rgb = np.stack([
            np.clip(float_resize(source[..., band] * input_alpha, size) / denominator, 0, 1)
            for band in range(3)
        ], axis=2)
    else:
        source = _SRGB_8[np.asarray(image.convert("RGB"))]
        alpha = np.ones(size[::-1], dtype=np.float32)
        rgb = np.stack([np.clip(float_resize(source[..., band], size), 0, 1) for band in range(3)], axis=2)
    return CapturePixels(rgb, alpha >= ALPHA_THRESHOLD, image.size, size[::-1])


def show_capture(decoded: DecodedCapture, max_side: int = 480) -> Image.Image:
    size = fit_size(decoded.image.size, max_side)
    pixels = sample_capture(decoded, size)
    rgb = linear_to_srgb(pixels.rgb)
    if decoded.information["has_alpha"]:
        canvas = np.concatenate((rgb, (pixels.support * 255).astype(np.uint8)[..., None]), axis=2)
        return Image.fromarray(canvas)
    return Image.fromarray(rgb)


def encode_preview(decoded: DecodedCapture) -> bytes:
    buf = io.BytesIO()
    show_capture(decoded).save(buf, format="WEBP", quality=88, method=3)
    return buf.getvalue()


def frame_input(raw: bytes, name: str, position: float, library) -> dict:
    filename = capture_name(name)
    decoded = read_capture(raw, filename)
    digest = library.store_source(raw, encode_preview(decoded))
    return {
        "name": filename,
        "sha": digest,
        "position": position,
        "width": decoded.width,
        "height": decoded.height,
        "bytes": len(raw),
        "information": decoded.information,
    }


def read_pixels(library, captured: dict, size: tuple[int, int]) -> CapturePixels:
    source = library.source_path(captured["sha"])
    if not source.is_file():
        raise Unprocessable(f"The source asset for {captured['name']} is missing.")
    return sample_capture(read_capture(source, captured["name"]), size)


def assert_compatible_dimensions(captures: list[dict]) -> None:
    first = captures[0]
    mismatches = [
        capture["name"]
        for capture in captures[1:]
        if (capture["width"], capture["height"]) != (first["width"], first["height"])
    ]
    if mismatches:
        raise Unprocessable(
            "Enabled captures need matching oriented dimensions.",
            details={"reference_size": [first["width"], first["height"]], "mismatched": mismatches[:12]},
        )


def percentile_summary(values: np.ndarray, valid: np.ndarray) -> dict:
    data = values[valid & np.isfinite(values)]
    if not data.size:
        return {name: None for name in ("min", "p05", "median", "p95", "max", "mean")}
    percentiles = np.percentile(data, (5, 50, 95))
    return {
        "min": float(data.min()),
        "p05": float(percentiles[0]),
        "median": float(percentiles[1]),
        "p95": float(percentiles[2]),
        "max": float(data.max()),
        "mean": float(data.mean(dtype=np.float64)),
    }


def valid_neighborhood(support: np.ndarray, pixels: int) -> np.ndarray:
    if pixels <= 0:
        return support
    return ndimage.binary_erosion(support, iterations=pixels, border_value=0)


def fill_nearest(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    if valid.all():
        return values
    if not valid.any():
        return np.zeros_like(values)
    nearest = ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True)
    return values[nearest[0], nearest[1]]
