"""Input validation and domain records shared by the HTTP and reconstruction layers."""

import math
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


class WorkbenchError(Exception):
    status = 400

    def __init__(self, message: str, *, field: str | None = None, details: Any = None):
        super().__init__(message)
        self.message = message
        self.field = field
        self.details = details

    def public(self) -> dict:
        value = {"error": self.message}
        if self.field:
            value["field"] = self.field
        if self.details is not None:
            value["details"] = self.details
        return value


class Missing(WorkbenchError):
    status = 404


class Conflict(WorkbenchError):
    status = 409


class Capacity(WorkbenchError):
    status = 413


class Unprocessable(WorkbenchError):
    status = 422


IMAGE_EXTENSIONS = frozenset((".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"))

def storage_id() -> str:
    return uuid4().hex


def require_id(value: Any, label: str = "id") -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{32}", value):
        raise Missing(f"Unknown {label}.")
    return value


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def object_value(value: Any, label: str = "request") -> dict:
    if not isinstance(value, dict):
        raise WorkbenchError(f"{label.capitalize()} must be an object.", field=label)
    return value


def validate_keys(value: dict, allowed: set, label: str = "request") -> None:
    unexpected = sorted(set(value).difference(allowed))
    if unexpected:
        raise WorkbenchError(f"Unknown {label} field.", field=unexpected[0])


def number(value: Any, label: str, low: float, high: float) -> float:
    if isinstance(value, bool):
        raise WorkbenchError(f"{label.capitalize()} must be a number.", field=label)
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise WorkbenchError(f"{label.capitalize()} must be a number.", field=label) from error
    if not math.isfinite(parsed) or not low <= parsed <= high:
        raise WorkbenchError(
            f"{label.capitalize()} must be between {low:g} and {high:g}.",
            field=label,
        )
    return parsed


def integer(value: Any, label: str, low: int, high: int) -> int:
    result = number(value, label, low, high)
    if int(result) != result:
        raise WorkbenchError(f"{label.capitalize()} must be a whole number.", field=label)
    return int(result)


def choice(value: Any, label: str, options: tuple[str, ...]) -> str:
    if not isinstance(value, str) or value not in options:
        raise WorkbenchError(
            f"{label.capitalize()} must be one of {', '.join(options)}.",
            field=label,
        )
    return value


def flag(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise WorkbenchError(f"{label.capitalize()} must be true or false.", field=label)
    return value


def short_text(
    value: Any,
    label: str,
    *,
    maximum: int = 90,
    allow_blank: bool = False,
) -> str:
    if not isinstance(value, str):
        raise WorkbenchError(f"{label.capitalize()} must be text.", field=label)
    if not allow_blank and not value.strip():
        raise WorkbenchError(f"Enter {label}.", field=label)
    value = value.strip()
    if len(value) > maximum or any(ord(char) < 32 and char != "\n" for char in value):
        raise WorkbenchError(
            f"{label.capitalize()} must be at most {maximum} characters and contain valid text.",
            field=label,
        )
    return value


def capture_name(value: Any) -> str:
    clean = short_text(value, "filename", maximum=220)
    path = Path(clean.replace("\\", "/"))
    if path.suffix.casefold() not in IMAGE_EXTENSIONS:
        raise Unprocessable("Choose PNG, JPEG, TIFF or WebP capture files.", field="filename")
    return path.name


def relative_coordinate(value: Any, label: str) -> float:
    return number(value, label, 0.0, 1.0)


def parse_vertex(value: Any, label: str) -> tuple[float, float]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise WorkbenchError(f"{label.capitalize()} must contain x and y.", field=label)
    return relative_coordinate(value[0], f"{label} x"), relative_coordinate(value[1], f"{label} y")


MAX_VIEWS = 8
MAX_CARDS = 48
MAX_PRINT_PIXELS = 8_000_000
MAX_DRAW_EDGE = 5000
MAX_ASSET_BYTES = 24 * 1024**2


@dataclass(frozen=True)
class ViewSetting:
    position: float
    title: str
    enabled: bool
    exposure_ev: float = 0.0

    @classmethod
    def parse(cls, data, current=None):
        obj = object_value(data, "view")
        validate_keys(obj, {"position", "title", "enabled", "exposure_ev"}, "view")
        current = current or {}
        return cls(
            number(obj.get("position", current.get("position", 0)), "angle", -85, 85),
            short_text(obj.get("title", current.get("title", "View")), "title"),
            flag(obj.get("enabled", current.get("enabled", True)), "enabled"),
            number(obj.get("exposure_ev", current.get("exposure_ev", 0)), "exposure adjustment", -3, 3),
        )

    def public(self):
        return asdict(self)


@dataclass(frozen=True)
class Lens:
    lpi: float = 50
    phase: float = 0
    shear: float = 0
    order: str = "forward"
    latitude: float = 22

    @classmethod
    def parse(cls, data):
        obj = object_value(data, "lens")
        validate_keys(obj, {"lpi", "phase", "shear", "order", "latitude"}, "lens")
        return cls(
            lpi=number(obj.get("lpi", 50), "pitch lines per inch", 10, 180),
            phase=number(obj.get("phase", 0), "phase in cycles", -0.5, 0.5),
            shear=number(obj.get("shear", 0), "side drift per millimetre", -0.25, 0.25),
            order=choice(obj.get("order", "forward"), "view order", ("forward", "reverse")),
            latitude=number(obj.get("latitude", 22), "half viewing cone in degrees", 5, 54),
        )

    @property
    def mm_per_lens(self):
        return 25.4 / self.lpi

    def dot_period(self, dpi):
        return dpi / self.lpi

    def public(self):
        return asdict(self)


@dataclass(frozen=True)
class Sheet:
    width_mm: float = 54
    height_mm: float = 68
    dpi: int = 500
    mount: str = "face"
    paper: str = "white"

    @classmethod
    def parse(cls, data):
        obj = object_value(data, "print format")
        validate_keys(obj, {"width_mm", "height_mm", "dpi", "mount", "paper"}, "sheet")
        sheet = cls(
            width_mm=number(obj.get("width_mm", 54), "width in millimetres", 18, 220),
            height_mm=number(obj.get("height_mm", 68), "height in millimetres", 18, 290),
            dpi=integer(obj.get("dpi", 500), "printer dpi", 150, 1200),
            mount=choice(obj.get("mount", "face"), "mount side", ("face", "reverse")),
            paper=choice(obj.get("paper", "white"), "paper tone", ("white", "warm", "matte")),
        )
        width, height = sheet.size
        if max(width, height) > MAX_DRAW_EDGE or width * height > MAX_PRINT_PIXELS:
            raise Capacity("These dimensions exceed this local print raster budget. Reduce print size or DPI.")
        return sheet

    @property
    def size(self):
        return (
            round(self.width_mm * self.dpi / 25.4),
            round(self.height_mm * self.dpi / 25.4),
        )

    def public(self):
        return asdict(self)


@dataclass(frozen=True)
class Position:
    x: float
    y: float

    @classmethod
    def parse(cls, value, label):
        x, y = parse_vertex(value, label)
        return cls(x, y)

    def pixel(self, width, height):
        return self.x * (width - 1), self.y * (height - 1)

    def public(self):
        return [self.x, self.y]


@dataclass(frozen=True)
class Region:
    left: float
    top: float
    right: float
    bottom: float

    @classmethod
    def parse(cls, data):
        if data is None:
            return cls(0, 0, 1, 1)
        if not isinstance(data, list) or len(data) != 4:
            raise Unprocessable("Choose a rectangle with two opposite corners.")
        left, top, right, bottom = [relative_coordinate(value, "rectangle coordinate") for value in data]
        if right <= left or bottom <= top:
            raise Unprocessable("Choose a nonempty rectangle.")
        return cls(left, top, right, bottom)

    def pixels(self, width, height):
        left = min(width - 1, math.floor(self.left * width))
        top = min(height - 1, math.floor(self.top * height))
        right = min(width, max(left + 1, math.ceil(self.right * width)))
        bottom = min(height, max(top + 1, math.ceil(self.bottom * height)))
        return left, top, right, bottom

    def public(self):
        return [self.left, self.top, self.right, self.bottom]


@dataclass(frozen=True)
class Framing:
    mode: str = "center"
    cover: str = "fill"
    reference: str | None = None
    frame: Region = field(default_factory=lambda: Region(0, 0, 1, 1))
    fit_motion: bool = False

    @classmethod
    def parse(cls, data):
        obj = object_value(data, "framing")
        validate_keys(obj, {"mode", "cover", "reference", "frame", "fit_motion"}, "framing")
        reference = obj.get("reference")
        return cls(
            mode=choice(obj.get("mode", "center"), "framing", ("center", "shared")),
            cover=choice(obj.get("cover", "fill"), "print fit", ("fill", "fit")),
            reference=require_id(reference, "reference view") if reference else None,
            frame=Region.parse(obj.get("frame")),
            fit_motion=flag(obj.get("fit_motion", False), "fit motion"),
        )

    def public(self):
        return {**asdict(self), "frame": self.frame.public()}


@dataclass(frozen=True)
class Sampling:
    method: str = "solid"
    aperture: float = 0.84

    @classmethod
    def parse(cls, data):
        obj = object_value(data, "sampling")
        validate_keys(obj, {"method", "aperture"}, "sampling")
        return cls(
            method=choice(obj.get("method", "solid"), "sampling", ("solid", "coverage")),
            aperture=number(obj.get("aperture", 0.84), "lens aperture", 0.3, 1),
        )


DEFAULT_CAPTION_REGION = (0.08, 0.82, 0.92, 0.95)


@dataclass(frozen=True)
class CaptionMessage:
    """One of the two flip messages, with the angle where it must read clearly."""
    text: str = ""
    angle: float = 0.0

    @classmethod
    def parse(cls, data, label, *, required=False):
        obj = object_value(data, label)
        validate_keys(obj, {"text", "angle"}, label)
        text = short_text(
            obj.get("text", ""), f"{label} text",
            maximum=140, allow_blank=not required,
        )
        if required and text.count("\n") > 3:
            raise WorkbenchError(f"{label.capitalize()} must use at most four lines.", field=label)
        return cls(text, number(obj.get("angle", 0), f"{label} angle", -85, 85))

    def public(self):
        return asdict(self)


@dataclass(frozen=True)
class Captions:
    """Two texts sharing one blank paper area, interleaved into separate lens slots."""
    enabled: bool = False
    region: Region = field(default_factory=lambda: Region(*DEFAULT_CAPTION_REGION))
    left: CaptionMessage = field(default_factory=CaptionMessage)
    right: CaptionMessage = field(default_factory=CaptionMessage)
    size_mm: float = 0.0

    @classmethod
    def parse(cls, data):
        obj = object_value(data, "captions")
        validate_keys(obj, {"enabled", "region", "left", "right", "size_mm"}, "captions")
        enabled = flag(obj.get("enabled", False), "flip captions")
        captions = cls(
            enabled=enabled,
            region=Region.parse(obj.get("region", list(DEFAULT_CAPTION_REGION))),
            left=CaptionMessage.parse(obj.get("left", {}), "left caption", required=enabled),
            right=CaptionMessage.parse(obj.get("right", {}), "right caption", required=enabled),
            size_mm=number(obj.get("size_mm", 0), "caption type size in millimetres", 0, 20),
        )
        if enabled and captions.left.angle >= captions.right.angle:
            raise WorkbenchError(
                "The left caption angle must be lower than the right caption angle.",
                field="captions",
            )
        return captions

    def public(self):
        return {
            "enabled": self.enabled,
            "region": self.region.public(),
            "left": self.left.public(),
            "right": self.right.public(),
            "size_mm": self.size_mm,
        }


@dataclass(frozen=True)
class Plan:
    lens: Lens = field(default_factory=Lens)
    sheet: Sheet = field(default_factory=Sheet)
    framing: Framing = field(default_factory=Framing)
    sampling: Sampling = field(default_factory=Sampling)
    captions: Captions = field(default_factory=Captions)

    @classmethod
    def parse(cls, data):
        obj = object_value(data, "card settings")
        validate_keys(obj, {"lens", "sheet", "framing", "sampling", "captions"}, "card settings")
        return cls(
            Lens.parse(obj.get("lens", {})),
            Sheet.parse(obj.get("sheet", {})),
            Framing.parse(obj.get("framing", {})),
            Sampling.parse(obj.get("sampling", {})),
            Captions.parse(obj.get("captions", {})),
        )

    def check_sampling(self, views):
        period = self.lens.dot_period(self.sheet.dpi)
        if views > period:
            raise Unprocessable(
                "The printer has fewer device dots per lens than active views. Raise DPI, choose a coarser lens or disable views.",
                details={"active_views": views, "dots_per_lens": round(period, 3)},
            )

    def public(self):
        return {
            **asdict(self), "framing": self.framing.public(),
            "captions": self.captions.public(),
        }


def ordered_views(views, *, minimum=2):
    active = sorted((view for view in views if view["enabled"]), key=lambda view: view["position"])
    if len(active) < minimum:
        raise Unprocessable(f"Enable at least {minimum} angle views.")
    if len(active) > MAX_VIEWS:
        raise Capacity("There are too many active views for one card.")
    for left, right in zip(active, active[1:]):
        if right["position"] - left["position"] < 0.01:
            raise Unprocessable(
                "Two active views have the same angle. Disable one or change the capture angle.",
                details={"overlap": [left["title"], right["title"]]},
            )
    return active


def parse_reference(data):
    if data is None:
        return {}
    obj = object_value(data, "revision")
    validate_keys(obj, {"version", "candidate_sha"}, "revision")
    return {
        "version": integer(obj.get("version"), "revision number", 0, 99999),
        "candidate_sha": require_sha(obj.get("candidate_sha"), "candidate"),
    }


def require_sha(value, label="asset"):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise Missing(f"Unknown {label}.")
    return value


def optional_region(data):
    if data is None:
        return Region.parse(None)
    return Region.parse(data)
