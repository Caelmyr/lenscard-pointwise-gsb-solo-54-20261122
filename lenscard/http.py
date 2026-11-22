"""Routes for the local LensCard workbench."""
from contextlib import contextmanager
from io import BytesIO
import json
import os
from pathlib import Path
from threading import BoundedSemaphore

from flask import Flask, jsonify, request, send_file, send_from_directory
from werkzeug.exceptions import RequestEntityTooLarge

from . import __version__
from .demo import SAMPLES, create_sample, demo_original, menu, sample_measurements
from .diagnostics import get_original_info
from .imaging import frame_input
from .device_audit import audit_raster
from .lineage import lineage, source_coordinate_receipt
from .preflight import draft_preflight
from .rectify import rectify_board
from .schedule import observation_review, observation_sheet, sweep_from_settings
from .version_compare import compare_versions
from .models import (
    Capacity, Missing, Unprocessable, WorkbenchError, flag, integer,
    number, object_value, require_id, short_text, validate_keys,
)
from .printcheck import device_contact_sheet, measure_ruler_pitch, read_device_region, verify_physical_size
from .storage import Library
from .transfer import (
    MAX_ARCHIVE_BYTES, accept_view_upload, pack_measurement, pack_print,
    pack_view_set, package_read, report_csv,
)
from .workflow import CardJob, Result, data_image, retain_result, summarize_issued


VIEW_UPLOAD_LIMIT = 22 * 1024**2
BODY_LIMIT = 140 * 1024**2
STATIC = Path(__file__).resolve().parents[1] / "static"
TOOLS = {
    "exposure": "Exposure and clipping",
    "line": "Native line profile",
    "spot": "Original pixel probe",
    "sharpness": "Edge energy",
    "texture": "Texture frequency",
    "framing": "Shared framing",
    "view-angle": "Observer angle preview",
    "registered-comparison": "Registered view comparison",
    "soft-proof": "Paper simulation",
    "density": "Density step response",
    "grain": "Approximate grain",
    "lens-fit": "Global lens fit",
}


def _json():
    return object_value(request.get_json(silent=True), "request")


def _media(payload, mime, filename, *, download=False):
    return send_file(BytesIO(payload), mimetype=mime, download_name=filename, as_attachment=download,
                     max_age=0)


def _saved_media(library, record, name):
    ref = record["assets"].get(name)
    if ref is None:
        raise Missing("No such asset exists in this record.")
    return _media(library.asset_bytes(ref), ref["mime"], ref["filename"],
                  download=request.args.get("download", "0") == "1")


def _multipart(name, byte_limit=VIEW_UPLOAD_LIMIT):
    file = request.files.get(name)
    if file is None or not file.filename:
        raise Unprocessable("Choose a file to continue.", field=name)
    payload = file.stream.read(byte_limit + 1)
    if not payload or len(payload) > byte_limit:
        raise Capacity("The selected file is empty or too large.", field=name)
    return file.filename, payload


def _multipart_object(name):
    raw = request.form.get(name, "")
    try:
        return object_value(json.loads(raw), name)
    except (TypeError, ValueError) as error:
        raise Unprocessable("The uploaded settings could not be decoded.", field=name) from error


def _card_version(data):
    return integer(data.get("version"), "draft version", 0, 99999)


def _named_limit(value):
    return short_text(value, "record title", maximum=100)


class WorkSlots:
    def __init__(self, limit):
        self.gate = BoundedSemaphore(limit)

    @contextmanager
    def entry(self):
        if not self.gate.acquire(timeout=8):
            raise ConflictBusy("Another print inspection is busy. Wait before starting the next one.")
        try:
            yield
        finally:
            self.gate.release()


class ConflictBusy(WorkbenchError):
    status = 429


def register_library(app, library):
    @app.get("/api/health")
    def health():
        return {"status": "ready", "version": __version__, "app": "LensCard"}

    @app.get("/api/library")
    def cards():
        return {"cards": library.list_cards(), "tools": TOOLS}

    @app.get("/api/samples")
    def samples():
        return {"samples": list(menu(library))}

    @app.get("/api/samples/<key>/cover")
    def sample_cover(key):
        if key not in SAMPLES:
            raise Missing("This sample is unavailable.")
        sample = SAMPLES[key]
        return _media(demo_original(key, sample.views[len(sample.views) // 2]), "image/jpeg", "synthetic-cover.jpg")

    @app.post("/api/samples/<key>")
    def load_sample(key):
        if key not in SAMPLES:
            raise Missing("This sample is unavailable.")
        return jsonify(card=create_sample(library, SAMPLES[key]))

    @app.post("/api/cards")
    def add_card():
        body = _json()
        validate_keys(body, {"title", "note", "config"})
        card = library.create_card(body.get("title"), body.get("note") or "", plan=body.get("config"))
        return jsonify(card=card), 201

    @app.get("/api/cards/<card_id>")
    def read_card(card_id):
        return jsonify(card=library.get_card(card_id),
                       proofs=library.list_proofs(card_id),
                       measurements=library.measurements(card_id),
                       events=library.event_log(card_id))

    @app.put("/api/cards/<card_id>")
    def update_card(card_id):
        data = _json()
        validate_keys(data, {"title", "note", "config", "version"})
        card = library.update_card(
            card_id, title=data.get("title"), note=data.get("note", ""),
            config=data.get("config"), version=_card_version(data),
        )
        return jsonify(card=card)

    @app.get("/api/cards/<card_id>/events")
    def event_log(card_id):
        count = integer(request.args.get("limit", 30), "events shown", 1, 100)
        return {"events": library.event_log(card_id, count)}

    @app.post("/api/cards/<card_id>/views")
    def import_view(card_id):
        filename, payload = _multipart("view")
        version = integer(request.form.get("version"), "draft version", 0, 99999)
        angle = number(request.form.get("angle", 0), "observation angle", -85, 85)
        title = short_text(request.form.get("title") or Path(filename).stem, "view title")
        source = frame_input(payload, filename, angle, library)
        item = library.add_view(card_id, source, {"title": title, "position": angle}, version=version)
        return jsonify(view=item), 201

    @app.put("/api/cards/<card_id>/views/<view_id>")
    def edit_view(card_id, view_id):
        body = _json()
        validate_keys(body, {"version", "settings"})
        item = library.edit_view(card_id, view_id, body.get("settings"), _card_version(body))
        return {"view": item}

    @app.get("/api/cards/<card_id>/views/<view_id>/original")
    def original(card_id, view_id):
        item = library.view(card_id, view_id)
        original_file = library.source_path(item["sha"])
        if not original_file.exists():
            raise Missing("This original is missing from local storage.")
        return send_file(original_file, download_name=item["name"], as_attachment=True)

    @app.get("/api/cards/<card_id>/views/<view_id>/preview")
    def thumbnail(card_id, view_id):
        item = library.view(card_id, view_id)
        file = library.preview_path(item["sha"])
        if not file.is_file():
            raise Missing("The requested view preview is not available.")
        return send_file(file, mimetype="image/webp", conditional=True, max_age=60)

    @app.get("/api/cards/<card_id>/views/<view_id>/information")
    def source_information(card_id, view_id):
        return {"result": get_original_info(library, card_id, view_id)}


def register_creation(app, library, work):
    @app.post("/api/cards/<card_id>/inspect/<tool>")
    def run_tool(card_id, tool):
        body = _json()
        with work.entry():
            job = CardJob(library, card_id)
            result = job.tool(tool, body)
        return result.public()

    @app.post("/api/cards/<card_id>/caption-check")
    def caption_check(card_id):
        with work.entry():
            result = CardJob(library, card_id).caption_preview()
        return result.public()

    @app.post("/api/cards/<card_id>/retain/<tool>")
    def retain_tool(card_id, tool):
        body = _json()
        name = _named_limit(body.get("title"))
        values = object_value(body.get("settings"), "inspection")
        with work.entry():
            job = CardJob(library, card_id)
            if _card_version(body) != job.version:
                from .models import Conflict
                raise Conflict("The inspected card draft has changed.")
            result = job.retain_measurement(tool, name, values)
        return jsonify(measurement=result), 201

    @app.post("/api/cards/<card_id>/lens-fit/apply")
    def apply_lens_fit(card_id):
        body = _json()
        validate_keys(body, {"ticks", "version", "fit_sha"})
        with work.entry():
            return CardJob(library, card_id).apply_grid(body.get("ticks"), _card_version(body), body.get("fit_sha"))

    @app.post("/api/cards/<card_id>/preview")
    def preview_print(card_id):
        body = _json()
        validate_keys(body, {"angle"})
        with work.entry():
            job = CardJob(library, card_id)
            angle = number(body.get("angle", 0), "display angle", -85, 85)
            return job.preview(angle).public()

    @app.post("/api/cards/<card_id>/proofs")
    def retain_print(card_id):
        body = _json()
        validate_keys(body, {"title", "version", "candidate_sha"})
        with work.entry():
            job = CardJob(library, card_id)
            proof = job.archive(_named_limit(body.get("title")), _card_version(body), body.get("candidate_sha"))
        return jsonify(proof=proof), 201

    @app.get("/api/cards/<card_id>/proofs/<proof_id>")
    def proof_detail(card_id, proof_id):
        return {"proof": library.proof(card_id, proof_id)}

    @app.put("/api/cards/<card_id>/proofs/<proof_id>/approve")
    def approve_proof(card_id, proof_id):
        body = _json()
        validate_keys(body, {"fingerprint", "note"})
        return {"approval": library.release_proof(card_id, proof_id, body.get("fingerprint"), body.get("note", ""))}

    @app.get("/api/cards/<card_id>/proofs/<proof_id>/asset/<media>")
    def proof_file(card_id, proof_id, media):
        if media in ("pdf", "plate"):
            raise Unprocessable("Use the approved download entry for actual-size artwork.")
        return _saved_media(library, library.proof(card_id, proof_id), media)

    @app.get("/api/cards/<card_id>/approved/<media>")
    def approved_file(card_id, media):
        _, proof = library.approved(card_id)
        return _saved_media(library, proof, media)

    @app.get("/api/cards/<card_id>/approved-integrity")
    def approved_integrity(card_id):
        return summarize_issued(library, card_id)


def register_targets(app, library, work):
    @app.post("/api/cards/<card_id>/pitch-target")
    def lens_target(card_id):
        with work.entry():
            return _target_reply(*CardJob(library, card_id).target(_json()))

    @app.get("/api/cards/<card_id>/angle-target")
    def angle_target(card_id):
        with work.entry():
            return _target_reply(*CardJob(library, card_id).angle_target())

    @app.get("/api/cards/<card_id>/synthetic-reading/<int:number>")
    def synthetic_photo(card_id, number):
        data = sample_measurements(library, card_id)
        if not 1 <= number <= data["count"]:
            raise Missing("This illustration has no requested angle photograph.")
        name, payload = data["files"][number - 1]
        return _media(payload, "image/png", name, download=True)

    @app.get("/api/cards/<card_id>/synthetic-reading")
    def synthetic_specs(card_id):
        data = sample_measurements(library, card_id)
        return {key: data[key] for key in ("count", "ticks", "settings")}

    @app.post("/api/cards/<card_id>/measured-optics")
    def measured_optics(card_id):
        spec = _multipart_object("settings")
        files = []
        for file in request.files.getlist("capture"):
            payload = file.stream.read(VIEW_UPLOAD_LIMIT + 1)
            if len(payload) > VIEW_UPLOAD_LIMIT:
                raise Capacity("Use angle test photographs under 22 MiB each.")
            files.append((file.filename, payload))
        with work.entry():
            job = CardJob(library, card_id)
            result = job.measured_optics(files, spec.get("captures", []), spec.get("proof_id"))
            reply = result.public()
            if spec.get("retain") is True:
                record = retain_result(library, card_id, _named_limit(spec.get("title")),
                                       "measured-optics", result, spec.get("proof_id"))
                reply["measurement"] = record
            return reply

    @app.get("/api/cards/<card_id>/measurements/<key>")
    def measurement_detail(card_id, key):
        return {"measurement": library.measurement(card_id, key)}

    @app.get("/api/cards/<card_id>/measurements/<key>/asset/<media>")
    def measurement_file(card_id, key, media):
        return _saved_media(library, library.measurement(card_id, key), media)

    @app.post("/api/cards/<card_id>/proofs/<proof_id>/raster-region")
    def raster_region(card_id, proof_id):
        with work.entry():
            summary, images = read_device_region(library, card_id, proof_id, _json())
        return Result(summary, images, summary["readout"]["views"]).public()

    @app.post("/api/cards/<card_id>/proofs/<proof_id>/size-readback")
    def size_readback(card_id, proof_id):
        return {"report": verify_physical_size(library, card_id, proof_id, _json())}

    @app.get("/api/cards/<card_id>/proofs/<proof_id>/size-proof")
    def size_proof(card_id, proof_id):
        with work.entry():
            report, images, pdf = device_contact_sheet(library, card_id, proof_id)
        return {**Result(report, images).public(), "pdf": _b64_media(pdf, "application/pdf")}

    @app.post("/api/cards/<card_id>/proofs/<proof_id>/pitch-readback")
    def scan_pitch(card_id, proof_id):
        filename, payload = _multipart("scan")
        values = _multipart_object("settings")
        with work.entry():
            report, images = measure_ruler_pitch(library, card_id, proof_id, filename, payload, values)
        return Result(report, images).public()


def register_workbench_audits(app, library, work):
    @app.post("/api/cards/<card_id>/rectify-photo")
    def straighten(card_id):
        filename, payload = _multipart("scan")
        options = _multipart_object("settings")
        with work.entry():
            job = CardJob(library, card_id)
            report, images = rectify_board(payload, filename, job.plan.sheet, options)
        return Result(report, images).public()

    @app.post("/api/cards/<card_id>/angle-sweep")
    def plan_sweep(card_id):
        body = _json()
        with work.entry():
            card, views = library.material_snapshot(card_id)
            report, images = sweep_from_settings(card, views, body)
            printed, table = observation_sheet(report)
        response = Result(report, images, report["rows"]).public()
        response["pdf"] = _b64_media(printed, "application/pdf")
        response["csv"] = _b64_media(table, "text/csv")
        return response

    @app.post("/api/cards/<card_id>/angle-sweep/review")
    def review_sweep(card_id):
        body = _json()
        validate_keys(body, {"settings", "timings"})
        with work.entry():
            card, views = library.material_snapshot(card_id)
            checklist, _ = sweep_from_settings(card, views, body.get("settings", {}))
            summary, images = observation_review(checklist, body.get("timings"))
        result = Result(summary, images, summary["observations"]).public()
        result["csv"] = _b64_media(report_csv(summary["observations"]), "text/csv")
        return result

    @app.post("/api/cards/<card_id>/compare-proofs")
    def compare_recorded(card_id):
        values = _json()
        with work.entry():
            report, images = compare_versions(library, card_id, values)
        return Result(report, images, report["zones"]).public()

    @app.get("/api/cards/<card_id>/proofs/<proof_id>/lineage")
    def proof_lineage(card_id, proof_id):
        with work.entry():
            report = lineage(library, card_id, proof_id)
        if request.args.get("format") == "csv":
            rows = source_coordinate_receipt(report)
            return _media(report_csv(rows), "text/csv", "lenscard-sources.csv", download=True)
        return {"report": report}

    @app.post("/api/cards/<card_id>/proofs/<proof_id>/band-audit")
    def band_audit(card_id, proof_id):
        body = _json()
        with work.entry():
            report, images = audit_raster(library, card_id, proof_id, body)
        return Result(report, images, report["row_prism_counts"]).public()

    @app.get("/api/cards/<card_id>/preflight")
    def angle_source_check(card_id):
        with work.entry():
            card, records = library.material_snapshot(card_id)
            report, images = draft_preflight(library, card, records)
        return Result(report, images, report["records"]).public()


def _b64_media(payload, mime):
    from base64 import b64encode
    return "data:" + mime + ";base64," + b64encode(payload).decode("ascii")


def _target_reply(report, media):
    return {"report": report,
            "images": {name: data_image(contents[0]) for name, contents in media.items() if contents[1] == "image/png"},
            "pdf": _b64_media(media["pdf"][0], "application/pdf")}


def register_transfer(app, library, work):
    @app.get("/api/cards/<card_id>/export-views")
    def export_views(card_id):
        with work.entry():
            payload, filename = pack_view_set(library, *library.material_snapshot(card_id))
        return _media(payload, "application/zip", filename, download=True)

    @app.get("/api/cards/<card_id>/export-approved")
    def export_print(card_id):
        with work.entry():
            active, proof = library.approved(card_id)
            card = library.get_card(card_id)
            result, name = pack_print(library, card, active, proof, request.args.get("originals") == "1")
        return _media(result, "application/zip", name, download=True)

    @app.get("/api/cards/<card_id>/measurements/<key>/export")
    def export_measurement(card_id, key):
        item = library.measurement(card_id, key)
        return _media(pack_measurement(library, item), "application/zip", "lenscard-inspection.zip", download=True)

    @app.post("/api/packages/check")
    def verify_package():
        _, payload = _multipart("archive", MAX_ARCHIVE_BYTES)
        with work.entry():
            report, _ = package_read(payload)
        return {"report": report}

    @app.post("/api/packages/import")
    def load_view_set():
        _, payload = _multipart("archive", MAX_ARCHIVE_BYTES)
        with work.entry():
            result = accept_view_upload(library, payload, request.form.get("digest"))
        return jsonify(card=result), 201


def create_app(data_dir=None):
    base = Path(data_dir or os.environ.get("LENSCARD_DATA", "runtime-data"))
    library = Library(base)
    app = Flask(__name__, static_folder=None)
    app.config["MAX_CONTENT_LENGTH"] = BODY_LIMIT
    work = WorkSlots(1)
    register_library(app, library)
    register_creation(app, library, work)
    register_targets(app, library, work)
    register_transfer(app, library, work)
    register_workbench_audits(app, library, work)

    @app.get("/")
    def page():
        return send_from_directory(STATIC, "index.html", max_age=0)

    @app.get("/static/<path:name>")
    def files(name):
        return send_from_directory(STATIC, name, max_age=0)

    @app.errorhandler(WorkbenchError)
    def operator_error(error):
        return jsonify(error.public()), error.status

    @app.errorhandler(RequestEntityTooLarge)
    def capacity_error(error):
        return jsonify(error="The upload exceeds the local request budget."), 413

    return app
