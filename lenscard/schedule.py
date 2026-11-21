"""Keep a physical observation checklist indexed to approved angle bands."""
import math

import numpy as np

from .models import Plan, Unprocessable, integer, number, object_value, ordered_views, validate_keys
from .optics import LensGrid, angular_domains, angular_response, phase_to_angle, physical_slot
from .visual import chart_png


def _desired_angles(view_count, stop, steps, direction):
    if steps < view_count:
        raise Unprocessable("The angle sweep cannot observe every view with fewer stops than views.")
    values = np.linspace(-stop, stop, steps)
    return values if direction == "left_to_right" else values[::-1]


def _time_to_move(previous, next_angle, degrees_per_second):
    if previous is None:
        return 0.0
    return abs(next_angle - previous) / degrees_per_second


def _pitch_resolution(grid, stop):
    if stop >= grid.lens.latitude:
        return grid.lens.latitude * .95
    return stop


def _inspect_sweep(grid, coordinates, angles, lens_sampling, step_rate, dwell):
    expected = []
    prev = None
    elapsed = 0.0
    count = len(coordinates)
    for i, angle in enumerate(angles, start=1):
        weights, bounds = angular_response(grid, float(angle), count, lens_sampling)
        idx = int(np.argmax(weights))
        interval = _time_to_move(prev, angle, step_rate)
        elapsed += interval
        view = coordinates[idx]
        expected.append({
            "step": i, "angle_degrees": round(float(angle), 5), "started_seconds": round(elapsed, 4),
            "duration_seconds": dwell, "motion_seconds": round(float(interval), 5),
            "strongest_view_id": view["id"], "strongest_title": view["title"], "source_angle": view["position"],
            "phase_window": [round(value, 6) for value in bounds],
            "model_main_fraction": round(float(weights[idx]), 7),
        })
        prev = angle
        elapsed += dwell
    return expected, elapsed


def _build_expected(grid, records):
    declared = angular_domains(grid, records)
    return {row["id"]: {**row, "scheduled_steps": [], "well_isolated": 0}
            for row in declared}


def _schedule_report(expected, view_rows, minimum):
    for step in expected:
        view = view_rows[step["strongest_view_id"]]
        view["scheduled_steps"].append(step["step"])
        if step["model_main_fraction"] >= minimum:
            view["well_isolated"] += 1
    missing = [row["label"] for row in view_rows.values() if not row["scheduled_steps"]]
    weak = [row["label"] for row in view_rows.values() if row["scheduled_steps"] and not row["well_isolated"]]
    return list(view_rows.values()), missing, weak


def _target_reference(grid, records, minimum):
    phases = []
    for n, view in enumerate(records):
        slot = physical_slot(n, len(records), grid.lens.order)
        nominal = (slot + .5) / len(records)
        angle = phase_to_angle(grid, nominal)
        phases.append({
            "name": view["title"],
            "target_deg": round(angle, 6),
            "nominal_phase": round(nominal, 6),
            "capture_threshold": minimum,
        })
    return phases


def sweep_from_settings(card, records, config):
    obj = object_value(config, "observation sweep")
    validate_keys(obj, {"steps", "side", "dwell", "travel", "angle_limit", "dominance"}, "sweep")
    views = ordered_views(records)
    plan = Plan.parse(card["config"])
    width, height = plan.sheet.size
    grid = LensGrid(plan.lens, plan.sheet.dpi, width, height)
    steps = integer(obj.get("steps", 11), "angle stops", 4, 31)
    side = obj.get("side", "left_to_right")
    if side not in ("left_to_right", "right_to_left"):
        raise Unprocessable("Use one sweep direction.")
    dwell = number(obj.get("dwell", .55), "hold time in seconds", .15, 12)
    travel = number(obj.get("travel", 13), "device speed in degrees per second", 1, 150)
    stop = number(obj.get("angle_limit", grid.lens.latitude * .88), "sweep bounds", 3, 54)
    dominant = number(obj.get("dominance", .68), "minimum isolated ratio", .52, .96)
    actual_stop = _pitch_resolution(grid, stop)
    angles = _desired_angles(len(views), actual_stop, steps, side)
    rows, time = _inspect_sweep(grid, views, angles, plan.sampling, travel, dwell)
    seen, missing, weak = _schedule_report(rows, _build_expected(grid, views), dominant)
    values = [row["angle_degrees"] for row in rows]
    height = [row["model_main_fraction"] for row in rows]
    target = _target_reference(grid, views, dominant)
    picture = chart_png(
        [("Dominant view", np.asarray(values), np.asarray(height)),
         ("Capture threshold", np.asarray([min(values), max(values)]), np.asarray([dominant, dominant]))],
        x_label="Reading angle (degrees)", y_label="Main window (fraction)", y_bounds=[0, 1],
        markers=[(item["target_deg"], item["name"][:14]) for item in target],
    )
    return {
        "kind": "angle-capture-checklist", "card_id": card["id"], "draft_version": card["version"],
        "bounds_requested_degrees": stop, "bounds_used_degrees": actual_stop,
        "direction": side, "rows": rows, "view_windows": seen, "recommended_centers": target,
        "unvisited_views": missing, "no_isolated_views": weak,
        "operator_timing_seconds": round(float(time), 4),
        "limit_note": "A draft checklist is an optical model. Confirm physical angles independently.",
    }, {"observation-window": picture}


def _draw_angle_title(draw, report, offset):
    draw.text((45, 36), "LENS CARD  /  CAMERA ANGLE CHECKLIST", fill=(22, 58, 48))
    draw.text((45, 64), f"Working card {report['card_id'][:12]} | revision {report['draft_version']}", fill=(48, 65, 68))
    draw.text((45, 91),
              f"{report['bounds_used_degrees']:.2f} degrees from center / {report['operator_timing_seconds']:.2f} seconds",
              fill=(60, 72, 77))
    draw.line((45, offset - 13, 933, offset - 13), fill=(110, 136, 130), width=1)


def _draw_checklist(draw, report):
    rows = report["rows"]
    top = 136
    _draw_angle_title(draw, report, top)
    columns = [48, 102, 220, 342, 560, 714, 839]
    for x, text in zip(columns, ["#", "Angle", "Start (s)", "Expected original", "Main fraction", "Hold (s)", "Checked"], strict=True):
        draw.text((x, top), text, fill=(54, 70, 67))
    for index, record in enumerate(rows):
        row_top = top + 25 + index * 29
        start = float(record["started_seconds"])
        data = [
            str(record["step"]), f"{record['angle_degrees']:+.2f}", f"{start:.2f}", record["strongest_title"][:29],
            f"{record['model_main_fraction']:.3f}", f"{record['duration_seconds']:.2f}",
        ]
        draw.rectangle((44, row_top, 933, row_top + 27), fill=(239, 241, 233) if index % 2 else (229, 236, 229))
        for x, value in zip(columns, data):
            draw.text((x, row_top + 8), value, fill=(25, 49, 44))
        draw.rectangle((857, row_top + 6, 872, row_top + 21), outline=(65, 82, 83), width=1)
    y = top + 35 + len(rows) * 29
    if report["unvisited_views"]:
        draw.text((45, y), "Review missing view windows before capture: " + ", ".join(report["unvisited_views"][:4]), fill=(112, 67, 55))
        y += 27
    if report["no_isolated_views"]:
        draw.text((45, y), "Model didn't fully isolate: " + ", ".join(report["no_isolated_views"][:4]), fill=(112, 67, 55))
        y += 27
    draw.text((45, y + 13), "Confirm angles independently. This card uses draft geometry and a predicted viewing window.", fill=(43, 66, 58))


def observation_sheet(report):
    # The checklist contains no optical artwork and can be printed at office DPI.
    from PIL import Image, ImageDraw
    from .printing import one_raster_pdf
    from .transfer import report_csv

    page = Image.new("RGB", (980, 1386), (248, 248, 241))
    _draw_checklist(ImageDraw.Draw(page), report)
    csv_rows = [
        {"step": row["step"], "angle_deg": row["angle_degrees"], "wait_s": row["motion_seconds"],
         "timestamp_s": row["started_seconds"], "hold_s": row["duration_seconds"],
         "main_source": row["strongest_title"], "principal_fraction": row["model_main_fraction"]}
        for row in report["rows"]
    ]
    return one_raster_pdf(np.asarray(page), 210, 297), report_csv(csv_rows)


def observed_intervals(checklist, timings):
    if not isinstance(timings, list) or len(timings) != len(checklist["rows"]):
        raise Unprocessable("Provide a read time for each scheduled camera point.")
    trace = [
        number(seconds, "camera reading time", 0, checklist["operator_timing_seconds"] + 100)
        for seconds in timings
    ]
    if any(right <= left for left, right in zip(trace, trace[1:])):
        raise Unprocessable("Readings need strict order along the planned camera direction.")
    rows = []
    for expected, actual in zip(checklist["rows"], trace, strict=True):
        offset = actual - expected["started_seconds"]
        status = "within planned stop" if 0 <= offset <= expected["duration_seconds"] else "outside stop"
        rows.append({
            "step": expected["step"], "angle": expected["angle_degrees"], "observed_at": actual,
            "expected_start": expected["started_seconds"], "hold": expected["duration_seconds"], "status": status,
        })
    return {"total": len(rows), "matched": sum(row["status"] == "within planned stop" for row in rows),
            "operator_check": rows}


def observation_review(checklist, timings):
    traced = observed_intervals(checklist, timings)
    planned = checklist["rows"]
    entries = traced["operator_check"]
    lookups = {row["step"]: row for row in planned}
    retention = []
    for row in entries:
        spec = lookups[row["step"]]
        retention.append({
            **row, "expected_view": spec["strongest_view_id"], "source_label": spec["strongest_title"],
            "relative_to_stop_seconds": round(row["observed_at"] - row["expected_start"], 6),
            "start_after_motion_seconds": spec["motion_seconds"],
            "dominant_model_fraction": spec["model_main_fraction"],
        })
    first_missed = next((row["step"] for row in entries if row["status"] != "within planned stop"), None)
    considered = np.arange(1, len(entries) + 1, dtype=float)
    actual = np.asarray([row["observed_at"] for row in entries], dtype=float)
    baseline = np.asarray([row["expected_start"] for row in entries], dtype=float)
    early = np.maximum(0, baseline - actual)
    end = np.asarray([row["expected_start"] + row["hold"] for row in entries], dtype=float)
    late = np.maximum(0, actual - end)
    charts = {
        "observations": chart_png(
            [("Recorded start", considered, actual), ("Planned start", considered, baseline),
             ("Planned hold end", considered, end)], x_label="Camera reading number", y_label="Seconds from first stop",
        ),
        "outside-stop": chart_png(
            [("Too early", considered, early), ("Too late", considered, late)],
            x_label="Camera reading number", y_label="Seconds outside planned hold",
        ),
    }
    return {
        "kind": "actual-camera-timing-review", "eligible_vantage_count": traced["matched"],
        "sample_count": traced["total"], "first_missed": first_missed,
        "first_seconds": round(float(actual[0]), 6), "last_seconds": round(float(actual[-1]), 6),
        "max_early_seconds": round(float(early.max()), 6), "max_late_seconds": round(float(late.max()), 6),
        "draft_version": checklist["draft_version"], "observations": retention,
        "note": "An accepted timestamp falls within its planned hold; verify the actual bracket and angle target.",
    }, charts
