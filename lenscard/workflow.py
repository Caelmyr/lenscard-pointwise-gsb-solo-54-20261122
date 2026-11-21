"""Card operations kept within one version, or a pinned print record."""
from base64 import b64encode
from dataclasses import dataclass
from hashlib import sha256
import math

import numpy as np

from .calibration import TestGrid, best_global_lens, calibration_fingerprint, fit_chart, target_assets
from .diagnostics import compare_views, exposure_report, load_sample, pattern_report, profile_line, sharpness_report, spot_values
from .exposure_match import exposure_match, parse_marks, proposal_fingerprint
from .framing import inspect_framing, prepare_registered
from .models import (
    Conflict, Missing, Plan, Unprocessable, integer, number, object_value, require_sha, validate_keys,
)
from .opticaltest import measured_angle_set, render_angle_target
from .optics import LensGrid
from .printing import (
    export_media, identify_candidate, inspect_angle, physical_dimensions, proof_inspection, render_print,
)
from .tone import SoftOptions, analyze_neutral_noise, response_patches, soft_proof
from .transfer import json_bytes, report_csv
from .visual import png_bytes


def data_image(raw):
    return "data:image/png;base64," + b64encode(raw).decode("ascii")


@dataclass
class Result:
    report: dict
    images: dict[str, bytes]
    csv_rows: list[dict] | None = None

    def public(self):
        return {
            "report": self.report,
            "images": {name: data_image(payload) for name, payload in self.images.items()},
            "has_csv": bool(self.csv_rows),
        }


class CardJob:
    def __init__(self, library, card_id, override=None):
        self.library = library
        self.card, self.views = library.material_snapshot(card_id)
        plan_data = override if override is not None else self.card["config"]
        self.plan = Plan.parse(plan_data)
        self.candidate, self.identity = identify_candidate(self.card, self.views, self.plan)
        self._prepared = None

    @property
    def card_id(self):
        return self.card["id"]

    @property
    def version(self):
        return self.card["version"]

    def read_job(self):
        return {
            "card_id": self.card_id,
            "draft_version": self.version,
            "candidate_sha": self.candidate,
        }

    def check_receipt(self, expected_version, candidate):
        if expected_version != self.version or require_sha(candidate, "preview") != self.candidate:
            raise Conflict("The draft, confirmed settings or originals differ from the displayed preview.")

    def prepared(self):
        if self._prepared is None:
            self._prepared = prepare_registered(self.library, self.views, self.plan.framing)
        return self._prepared

    def layout(self):
        report, images = inspect_framing(self.prepared(), self.plan.sheet, self.plan.framing)
        return Result({**report, **self.read_job(), "kind": "framing"}, images, report["views"])

    def viewing(self, angle):
        report, image = inspect_angle(self.prepared(), self.plan, angle)
        return Result({**report, **self.read_job(), "kind": "view-angle"}, {"angle": png_bytes(image)}, report["contributors"])

    def preview(self, angle=0):
        output = render_print(self.prepared(), self.plan)
        note, images = proof_inspection(self.prepared(), self.plan, output)
        if abs(angle) > 1e-9:
            detail, shown = inspect_angle(self.prepared(), self.plan, angle)
            note["preview_angle"] = detail
            images["angles"] = png_bytes(shown)
        images["print-preview"] = output.preview()
        return Result({
            **note, **self.read_job(), "kind": "print-preview",
            "preflight": {"file_dimensions": output.metadata["raster_pixel_count"],
                          "mount": self.plan.sheet.mount,
                          "color_paper": self.plan.sheet.paper},
        }, images, note["angular_response"])

    def archive(self, title, after_version, displayed):
        self.check_receipt(after_version, displayed)
        prepared = self.prepared()
        out = render_print(prepared, self.plan)
        summary, previews = proof_inspection(prepared, self.plan, out)
        media, exported = export_media(out, self.plan)
        archive_report = {
            "identity": self.identity,
            "summary": summary,
            "outputs": exported,
            "archived_snapshot": self.read_job(),
            "projection_notes": {
                "view_order": self.plan.lens.order,
                "orientation": self.plan.sheet.mount,
                "framing": self.plan.framing.public(),
                "source_count": len(prepared.views),
                "report_owner": "This record is rendered from the listed draft and originals.",
            },
        }
        media["receipt"] = (json_bytes(archive_report), "application/json", "lenscard-print-receipt.json")
        media["print-preview"] = (out.preview(), "image/png", "screen-overview.png")
        media.update({name: (payload, "image/png", name + ".png") for name, payload in previews.items()})
        references = {name: self.library.store_asset(*asset) for name, asset in media.items()}
        return self.library.retain_proof(
            self.card_id, title, self.version, self.candidate, archive_report, references,
        )

    def target(self, options):
        spec = TestGrid.parse(self.plan.sheet, self.plan.lens, options)
        return target_assets(spec)

    def angle_target(self):
        return render_angle_target(self.views, self.plan)

    def estimate_grid(self, ticks):
        fit = best_global_lens(self.plan.sheet, self.plan.lens, ticks)
        proof = calibration_fingerprint(self.card, fit, ticks)
        return Result({
            **fit.public(), **self.read_job(), "kind": "global-lens-fit", "fit_sha": proof,
            "source_marks": ticks,
        }, {"residuals": fit_chart(fit)}, fit.residual_rows)

    def apply_grid(self, ticks, expected_version, expected_fit):
        if self.version != expected_version:
            raise Conflict("The card changed. Refit against its current stock before applying calibration.")
        fit = best_global_lens(self.plan.sheet, self.plan.lens, ticks)
        receipt = calibration_fingerprint(self.card, fit, ticks)
        if require_sha(expected_fit, "fitted marks") != receipt:
            raise Conflict("Apply the same mark set that was inspected.")
        if not fit.accepted:
            raise Unprocessable("Review the mark residuals before using this grid for printing.")
        config = {**self.plan.public(), "lens": fit.lens.public()}
        card = self.library.update_card(
            self.card_id, version=self.version, title=self.card["title"], note=self.card["note"], config=config,
        )
        return {"card": card, "fit": fit.public(), "fit_sha": receipt}

    def _source(self, body):
        identifier = body.get("view")
        if not identifier:
            enabled = [view for view in self.views if view["enabled"]]
            if not enabled:
                raise Unprocessable("Import and enable an original view before starting this check.")
            identifier = enabled[0]["id"]
        return load_sample(self.library, self.card_id, identifier, body.get("region"))

    def tool(self, name, values):
        body = object_value(values, "inspection")
        if name == "framing":
            return self.layout()
        if name == "registered-comparison":
            first = body.get("view")
            second = body.get("other")
            split = number(body.get("divider", .5), "comparison divider", .04, .96)
            report, images = compare_views(self.library, self.views, self.plan, first, second, split)
            return Result({**report, **self.read_job()}, images)
        if name == "view-angle":
            angle = number(body.get("angle", 0), "view angle", -85, 85)
            return self.viewing(angle)
        if name == "lens-fit":
            return self.estimate_grid(body.get("ticks"))
        sample = self._source(body)
        if name == "exposure":
            report, image = exposure_report(sample)
        elif name == "line":
            report, image = profile_line(sample, body.get("start", [.04, .55]), body.get("end", [.96, .55]))
            return Result({**report, **self.read_job()}, image, report["samples"])
        elif name == "spot":
            radius = number(body.get("radius", 7), "sample radius", 2, 80)
            report, image = spot_values(sample, body.get("point", [.50, .50]), radius)
        elif name == "sharpness":
            report, image = sharpness_report(sample)
        elif name == "texture":
            report, image = pattern_report(sample, body.get("direction", "across"))
        elif name == "soft-proof":
            report, image = soft_proof(sample, SoftOptions.parse(body.get("settings", {}), self.plan.sheet.paper))
        elif name == "density":
            report, image = response_patches(sample, body.get("patches"))
            return Result({**report, **self.read_job()}, image, report["patches"])
        elif name == "grain":
            report, image = analyze_neutral_noise(sample)
        else:
            raise Missing("This inspection is not available.")
        return Result({**report, **self.read_job()}, image)

    def measured_optics(self, files, settings, proof_id=None):
        if proof_id:
            proof = self.library.proof(self.card_id, proof_id)
            view_count = len([item for item in proof["snapshot"]["identity"]["views"] if item["enabled"]])
        else:
            view_count = sum(1 for item in self.views if item["enabled"])
        if view_count < 2:
            raise Unprocessable("The selected card or proof must contain at least two active views.")
        report, diagram = measured_angle_set(files, settings, view_count)
        report = {**report, **self.read_job(), "retained_proof": proof_id}
        csv_data = [
            {"capture": row["input"]["label"], "angle": row["angle"], "expected_view": row["expected_view"], **row["result"]}
            for row in report["sources"]
        ]
        return Result(report, diagram, csv_data)

    def retain_measurement(self, tool, title, values):
        report = self.tool(tool, values)
        return retain_result(self.library, self.card_id, title, tool, report)

    def match_exposure(self, body):
        obj = object_value(body, "inspection")
        if "fit_motion" not in obj:
            obj = {**obj, "fit_motion": self.plan.framing.fit_motion}
        report, images = exposure_match(self.library, self.views, obj)
        return Result({**report, **self.read_job()}, images, report["views"])

    def apply_exposure_match(self, body):
        obj = object_value(body, "exposure decision")
        validate_keys(obj, {"marks", "discord_ev", "fit_motion", "version", "proposal_sha"},
                      "exposure decision")
        self.check_receipt_loose(integer(obj.get("version"), "draft version", 0, 99999))
        marks = parse_marks(obj)
        threshold = number(obj.get("discord_ev", 0.18), "inconsistency threshold in stops", 0.02, 1.5)
        fingerprint = proposal_fingerprint(self.views, marks, threshold)
        if fingerprint != obj.get("proposal_sha"):
            raise Conflict("The originals or the marked references changed. Re-run the exposure preview before applying it.")
        report, _ = exposure_match(self.library, self.views, obj)
        confident = [row for row in report["views"] if row["confident"]]
        if not confident:
            raise Unprocessable("No view has a consistent wall reference, so there is nothing confident to apply.")
        updated = self.library.apply_exposure_match(
            self.card_id, confident, self.version, fingerprint=fingerprint,
        )
        return {
            "kind": "angle-exposure-applied",
            "proposal_sha": fingerprint,
            "applied": [
                {"view_id": row["view_id"], "title": row["title"],
                 "exposure_ev": next(item["exposure_ev"] for item in updated if item["id"] == row["view_id"]),
                 "previous_ev": row["current_ev"]}
                for row in confident
            ],
            "kept_manual": [
                {"view_id": row["view_id"], "title": row["title"], "exposure_ev": row["current_ev"]}
                for row in report["views"] if not row["confident"]
            ],
        }

    def check_receipt_loose(self, expected_version):
        if expected_version != self.version:
            raise Conflict("The card changed. Re-run the exposure preview against the current draft.")


def retain_result(library, card_id, title, tool, result, proof_id=None):
    assets = {
        name: library.store_asset(content, "image/png", name + ".png")
        for name, content in result.images.items()
    }
    if result.csv_rows:
        assets["csv"] = library.store_asset(report_csv(result.csv_rows), "text/csv", "lenscard-measurements.csv")
    return library.retain_measurement(card_id, title, tool, result.report, assets, proof_id)


def summarize_issued(library, card_id):
    active, proof = library.approved(card_id)
    seen = []
    for name, reference in proof["assets"].items():
        content = library.asset_bytes(reference)
        seen.append({"artifact": name, "bytes": len(content), "sha256": sha256(content).hexdigest()})
    pdf = library.asset_bytes(proof["assets"]["pdf"])
    plan = proof["snapshot"]["identity"]["plan"]
    expected = [plan["sheet"]["width_mm"], plan["sheet"]["height_mm"]]
    actual = physical_dimensions(pdf)
    verified = (proof["state"] == "approved" and active["proof_id"] == proof["id"]
                and all(abs(a - b) < .001 for a, b in zip(expected, actual, strict=True)))
    return {
        "verified": verified,
        "approval": active,
        "proof_id": proof["id"],
        "base_version": proof["base_version"],
        "fingerprint": proof["fingerprint"],
        "media_box_expected_mm": expected,
        "media_box_actual_mm": actual,
        "assets": seen,
    }
