"""Generate calibrated single-card media with explicit millimetre dimensions."""
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from pathlib import PurePosixPath
import zlib

import numpy as np
from PIL import Image, ImageDraw

from .framing import fit_views, working_preview_sizes
from .imaging import fit_size, linear_to_srgb
from .models import Plan, Unprocessable
from .optics import (
    LensGrid, angular_domains, angular_response, exact_slot_coverage, physical_slot, print_quanta,
    response_report, solid_slot_coverage,
)
from .storage import digest_object
from .tone import PAPER_SIMULATIONS
from .visual import chart_png, png_bytes, scalar_palette


VIEW_PALETTE = np.asarray([
    [38, 142, 114], [191, 112, 82], [99, 104, 182], [232, 177, 78],
    [62, 160, 195], [185, 91, 142], [107, 145, 80], [126, 103, 83],
], np.uint8)
PRINT_BLOCK = 140
PNG_DPI = 96


@dataclass
class PrintOutput:
    image: np.ndarray
    display: np.ndarray
    coverage: np.ndarray
    grid: LensGrid
    metadata: dict

    def preview(self):
        size = fit_size(self.image.shape[1::-1], 950)
        plate = Image.fromarray(self.image).resize(size, Image.Resampling.LANCZOS)
        return png_bytes(plate)

    def plate_png(self):
        buf = BytesIO()
        Image.fromarray(self.image).save(buf, "PNG", dpi=(self.grid.dpi, self.grid.dpi), compress_level=4)
        return buf.getvalue()


def digest_view(record):
    return {key: record[key] for key in ("id", "sha", "position", "title", "enabled", "exposure_ev", "width", "height")}


def identify_candidate(card, records, plan: Plan):
    identity = {
        "card_id": card["id"],
        "version": card["version"],
        "plan": plan.public(),
        "views": [digest_view(item) for item in records],
        "renderer": "lenscard-print-v1",
    }
    return digest_object(identity), identity


def render_print(registered, plan, paper=None):
    count = len(registered.views)
    if count < 2:
        raise Unprocessable("Enable at least two views to build a proof.")
    plan.check_sampling(count)
    width, height = plan.sheet.size
    grid = LensGrid(plan.lens, plan.sheet.dpi, width, height)
    output = np.zeros((height, width, 3), dtype=np.float32)
    seen = np.zeros((height, width), dtype=np.float32)
    width_composite, height_composite = fit_size((width, height), 950)
    support_counts = []
    weights_by_view = []
    method = solid_slot_coverage if plan.sampling.method == "solid" else exact_slot_coverage
    reverse = plan.sheet.mount == "reverse"
    white = paper.white_coded if paper is not None else None
    fitted_views = fit_views(registered, width, height, plan.sheet, plan.framing.cover, white)
    for index, image in enumerate(fitted_views):
        slot = physical_slot(index, count, plan.lens.order)
        source = image.linear[:, ::-1] if reverse else image.linear
        support = image.support[:, ::-1] if reverse else image.support
        total = 0.0
        for start in range(0, height, PRINT_BLOCK):
            stop = min(start + PRINT_BLOCK, height)
            weights = method(grid, start, stop, slot, count, side=plan.sheet.mount)
            output[start:stop] += source[start:stop] * weights[..., None]
            seen[start:stop] += support[start:stop] * weights
            total += float(np.sum(weights, dtype=np.float64))
        weights_by_view.append(round(total / (width * height), 8))
        support_counts.append(image.public())
    plate = linear_to_srgb(output)
    # Keep full-resolution print bytes. The overview is filtered only for display.
    display = np.asarray(Image.fromarray(plate).resize((width_composite, height_composite), Image.Resampling.LANCZOS))
    metadata = {
        "registered": registered.public(),
        "grid": grid.public(),
        "mount": plan.sheet.mount,
        "paper": plan.sheet.paper,
        "paper_label": paper.label if paper is not None
        else PAPER_SIMULATIONS[plan.sheet.paper].name if plan.sheet.paper in PAPER_SIMULATIONS else plan.sheet.paper,
        "requested_mm": [plan.sheet.width_mm, plan.sheet.height_mm],
        "encoded_raster_mm": [round(width * grid.step, 5), round(height * grid.step, 5)],
        "working_image_size": [width_composite, height_composite],
        "raster_pixel_count": int(width * height),
        "print_sampling": plan.sampling.method,
        "view_coverages": weights_by_view,
        "supported_print_fraction": round(float(seen.mean()), 6),
        "views": support_counts,
        "angular_windows": angular_domains(grid, [view.record for view in registered.views]),
    }
    return PrintOutput(plate, display, seen, grid, metadata)


def inspect_angle(registered, plan, angle, paper=None):
    width, height = working_preview_sizes(plan.sheet, 800)
    true_width, true_height = plan.sheet.size
    grid = LensGrid(plan.lens, plan.sheet.dpi, true_width, true_height)
    count = len(registered.views)
    weights, phase = angular_response(grid, angle, count, plan.sampling)
    linear = np.zeros((height, width, 3), dtype=np.float32)
    visible = np.zeros((height, width), dtype=np.float32)
    white = paper.white_coded if paper is not None else None
    for index, view in enumerate(fit_views(registered, width, height, plan.sheet, plan.framing.cover, white)):
        linear += view.linear * weights[index]
        visible += view.support * weights[index]
    photo = Image.fromarray(linear_to_srgb(linear))
    model = response_report(grid, [angle], [view.record for view in registered.views], plan.sampling)[0]
    return {
        **model,
        "phase": phase,
        "pixels": [width, height],
        "visible_fraction": round(float(visible.mean()), 6),
        "proof_kind": "predicted from a global lens grid and a finite observation window",
    }, photo


def coverage_preview(grid, count, side, width=760, height=350):
    origin_y = max(0, min(grid.height - height, grid.height // 2 - height // 2))
    origin_x = max(0, min(grid.width - width, grid.width // 2 - width // 2))
    rows = np.arange(origin_y, min(origin_y + height, grid.height))
    cols = np.arange(origin_x, min(origin_x + width, grid.width))
    phase = grid.phase_at(grid.actual_side(cols[None, :], side), rows[:, None])
    indices = (np.mod(phase, 1) * count).astype(np.int16)
    if grid.lens.order == "reverse":
        indices = count - 1 - indices
    return png_bytes(Image.fromarray(VIEW_PALETTE[indices])), [int(cols[0]), int(rows[0]), int(cols[-1] + 1), int(rows[-1] + 1)]


def proof_inspection(registered, plan, output, paper=None):
    angles = output.grid.screen_angles()
    response = response_report(output.grid, angles, [item.record for item in registered.views], plan.sampling)
    x = np.asarray([row["angle_deg"] for row in response], dtype=float)
    y = np.asarray([row["mixed_fraction"] for row in response], dtype=float)
    chart = chart_png([("Mixed angular fraction", x, y)], x_label="Observer angle (degrees)", y_label="Modelled leakage")
    coverage, roi = coverage_preview(output.grid, len(registered.views), plan.sheet.mount)
    angle, picture = inspect_angle(registered, plan, 0, paper)
    return {
        "bands": output.metadata,
        "quanta": print_quanta(output.grid, len(registered.views), plan.sampling, plan.sheet.mount),
        "preview_angle": angle,
        "angular_response": response,
        "band_map_box": roi,
    }, {"response": chart, "angles": png_bytes(picture), "view-map": coverage}


def _pdf_number(value):
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _stream(data, attributes=b""):
    return b"<< /Length " + str(len(data)).encode() + b" " + attributes + b" >>\nstream\n" + data + b"\nendstream"


def _pdf_bytes(objects, root=1):
    result = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for index, data in enumerate(objects, start=1):
        offsets.append(len(result))
        result.extend(f"{index} 0 obj\n".encode())
        result.extend(data)
        result.extend(b"\nendobj\n")
    xref = len(result)
    result.extend(f"xref\n0 {len(offsets)}\n0000000000 65535 f \n".encode())
    for address in offsets[1:]:
        result.extend(f"{address:010d} 00000 n \n".encode())
    result.extend(f"trailer\n<< /Size {len(offsets)} /Root {root} 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode())
    return bytes(result)


def one_raster_pdf(plate, width_mm, height_mm):
    array = np.ascontiguousarray(plate, dtype=np.uint8)
    height, width = array.shape[:2]
    colors = b"/DeviceGray" if array.ndim == 2 else b"/DeviceRGB"
    image = _stream(zlib.compress(array.tobytes(), 5),
                    f"/Type /XObject /Subtype /Image /Width {width} /Height {height} ".encode()
                    + b"/ColorSpace " + colors + b" /BitsPerComponent 8 /Filter /FlateDecode /Interpolate false")
    points = f"{_pdf_number(width_mm * 72 / 25.4)} {_pdf_number(height_mm * 72 / 25.4)}"
    w, h = points.split()
    paper = (f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {points}] "
             f"/Resources << /XObject << /Card 4 0 R >> >> /Contents 5 0 R >>").encode()
    controls = _stream(f"q\n{w} 0 0 {h} 0 0 cm\n/Card Do\nQ\n".encode())
    return _pdf_bytes([
        b"<< /Type /Catalog /Pages 2 0 R /ViewerPreferences << /PrintScaling /None >> >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        paper, image, controls,
    ])


def physical_dimensions(pdf):
    # Report PDF metadata authored here. Input verification uses the generated digest.
    mark = b"/MediaBox [0 0 "
    start = pdf.find(mark)
    if start < 0:
        raise Unprocessable("The retained proof does not contain a supported print page.")
    numbers = pdf[start + len(mark):].split(b"]", maxsplit=1)[0].split()
    return [round(float(value) * 25.4 / 72, 6) for value in numbers]


def export_media(output, plan):
    plate = output.plate_png()
    pdf = one_raster_pdf(output.image, plan.sheet.width_mm, plan.sheet.height_mm)
    return {
        "plate": (plate, "image/png", "lenscard-raster.png"),
        "pdf": (pdf, "application/pdf", "lenscard-actual-size.pdf"),
    }, {
        "pdf_actual_mm": physical_dimensions(pdf),
        "png_sha256": sha256(plate).hexdigest(),
        "pdf_sha256": sha256(pdf).hexdigest(),
    }
