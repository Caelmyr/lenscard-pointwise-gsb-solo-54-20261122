"""HTTP routes for reverse-salvaging a surviving interlaced print."""
import json
from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile

from flask import jsonify, request, send_file

from .models import Capacity, Missing, Unprocessable, integer, number, object_value, short_text, validate_keys
from .salvage import (
    PROVENANCE, SalvageSettings, backmap_geometry, build_assets, geometry_blob,
    load_print, reconstruct, salvage_report,
)
from .transfer import json_bytes, safe_path, slug

UPLOAD_LIMIT = 24 * 1024**2


def _media(payload, mime, filename, download=False):
    return send_file(BytesIO(payload), mimetype=mime, download_name=filename,
                     as_attachment=download, max_age=0)


def register_salvage(app, store, work):
    @app.post("/api/salvage")
    def create_salvage():
        file = request.files.get("print")
        if file is None or not file.filename:
            raise Unprocessable("Choose the interlaced print image.", field="print")
        payload = file.stream.read(UPLOAD_LIMIT + 1)
        if not payload or len(payload) > UPLOAD_LIMIT:
            raise Capacity("Use a print image under 24 MiB.", field="print")
        try:
            settings = object_value(json.loads(request.form.get("settings") or "{}"), "settings")
        except (TypeError, ValueError) as error:
            raise Unprocessable("The salvage settings could not be decoded.", field="settings") from error
        title = short_text(request.form.get("title") or "Salvaged print", "salvage title", maximum=100)
        with work.entry():
            parsed = SalvageSettings.parse(settings)
            print_data = load_print(payload, file.filename, parsed)
            result = reconstruct(print_data, parsed)
            images, records = build_assets(print_data, result)
            report = salvage_report(print_data, parsed, records, geometry_blob(result),
                                    result["crop_pixels"], result["damage_boxes"])
            assets = {name: store.store_asset(data) for name, data in images.items()}
            identifier = store.save_job(title, report, assets)
        return jsonify(job={"id": identifier, "title": title, "report": report,
                            "images": list(images)}), 201

    @app.get("/api/salvage")
    def list_salvage():
        return {"jobs": store.list_jobs(), "provenance": PROVENANCE}

    @app.get("/api/salvage/<job_id>")
    def get_salvage(job_id):
        return {"job": store.job(job_id)}

    @app.get("/api/salvage/<job_id>/asset/<name>")
    def salvage_asset(job_id, name):
        job = store.job(job_id)
        if not safe_path(name) or name not in job["assets"]:
            raise Missing("This salvage image does not exist.")
        return _media(store.asset_bytes(job["assets"][name]), "image/png", name + ".png")

    @app.post("/api/salvage/<job_id>/locate")
    def salvage_locate(job_id):
        body = object_value(request.get_json(silent=True), "locate request")
        validate_keys(body, {"view", "x", "y"}, "locate request")
        view_index = integer(body.get("view"), "view number", 1, 8) - 1
        x = number(body.get("x"), "horizontal point", 0, 1)
        y = number(body.get("y"), "vertical point", 0, 1)
        job = store.job(job_id)
        geometry = job["report"].get("geometry")
        if not geometry or view_index >= len(geometry):
            raise Missing("That view is not part of this salvage.")
        return {"locate": backmap_geometry(geometry[view_index], x, y)}

    @app.get("/api/salvage/<job_id>/download")
    def salvage_download(job_id):
        job = store.job(job_id)
        buffer = BytesIO()
        with ZipFile(buffer, "w") as archive:
            manifest = {
                "schema": "lenscard-salvage-1",
                "title": job["title"],
                "provenance": PROVENANCE,
                "not_originals": True,
                "report": job["report"],
            }
            archive.writestr("manifest.json", json_bytes(manifest), compress_type=ZIP_DEFLATED)
            for name, reference in sorted(job["assets"].items()):
                archive.writestr(safe_path(name + ".png"), store.asset_bytes(reference),
                                 compress_type=ZIP_DEFLATED)
        return _media(buffer.getvalue(), "application/zip",
                      slug(job["title"], "salvage") + "-salvage.zip", download=True)
