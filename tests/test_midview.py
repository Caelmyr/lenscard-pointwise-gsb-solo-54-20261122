#!/usr/bin/env python3
"""End-to-end checks for two-view middle synthesis through the HTTP API."""
import base64
import io
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lenscard.http import create_app  # noqa: E402

DATA = Path("/tmp/lenscard-midview-test")
FIXTURE = Path("/tmp/lenscard-fixture")


def decode_data_uri(uri):
    raw = base64.b64decode(uri.split(",", 1)[1])
    return raw, np.asarray(Image.open(io.BytesIO(raw)).convert("RGBA"))


def main():
    shutil.rmtree(DATA, ignore_errors=True)
    marks = json.loads((FIXTURE / "marks.json").read_text())
    app = create_app(str(DATA))
    client = app.test_client()

    created = client.post("/api/cards", json={"title": "Sill pair", "note": "test"})
    assert created.status_code == 201, created.get_json()
    card_id = created.get_json()["card"]["id"]

    def upload(path, angle):
        current = client.get(f"/api/cards/{card_id}").get_json()["card"]["version"]
        with open(path, "rb") as stream:
            response = client.post(
                f"/api/cards/{card_id}/views",
                data={"view": (stream, path.name), "version": str(current), "angle": str(angle),
                      "title": path.stem},
                content_type="multipart/form-data",
            )
        assert response.status_code == 201, response.get_json()
        return response.get_json()["view"]["id"]

    left_id = upload(FIXTURE / "left.png", -12)
    version = client.get(f"/api/cards/{card_id}").get_json()["card"]["version"]
    right_id = upload(FIXTURE / "right.png", 12)
    assert version == 1
    body = {
        "left_view": left_id, "right_view": right_id,
        "left_polygon": marks["polygons"][0], "right_polygon": marks["polygons"][1],
        "anchors": marks["anchors"], "mid": 0.5,
    }
    preview = client.post(f"/api/cards/{card_id}/midview/preview", json={"synthesis": body})
    assert preview.status_code == 200, preview.get_json()
    payload = preview.get_json()
    report = payload["report"]
    print("geometry:", json.dumps({k: report[k] for k in
          ("background_shift_px", "foreground_shift_px", "foreground_shift_method")}, indent=2))
    print("attribution:", json.dumps({k: report[k] for k in payload["report"] if k.endswith("pixels") or k == "transparent_fraction"}, indent=2))
    assert abs(report["background_shift_px"][0] - (-marks["bg_shift"])) < 1.5, report
    # The marked masks are the motion authority; hand-drawn outlines and shape
    # change mean either estimator can land a few pixels off the true motion.
    assert abs(report["foreground_shift_px"][0] - marks["fg_shift"]) <= 6.0, report
    assert abs(report["foreground_shift_px"][1]) <= 4.0, report

    middle_raw, middle = decode_data_uri(payload["images"]["middle"])
    marker_raw, marker_view = decode_data_uri(payload["images"]["markers"])
    with Image.open(io.BytesIO(middle_raw)) as middle_image:
        assert middle_image.mode == "RGB"  # checkerboard proof flattens transparency to the board
    with Image.open(io.BytesIO(marker_raw)) as marker_image:
        assert marker_image.mode == "RGBA"  # the retained frame keeps a real alpha channel
    marker_alpha = marker_view[..., 3]
    transparent = marker_alpha == 0
    print("transparent pixels in preview:", int(transparent.sum()))
    assert transparent.sum() > 0, "a region occluded in both views must stay transparent"

    # The checkerboard proof must visibly render the transparent region with
    # the two board tones rather than invented sill texture.
    middle_rgb = middle[..., :3]
    board_a = np.array([214, 214, 214], dtype=np.uint8)
    board_b = np.array([168, 168, 168], dtype=np.uint8)
    board_pixels = np.all((middle_rgb == board_a) | (middle_rgb == board_b), axis=2)
    assert board_pixels.sum() >= transparent.sum(), "transparent area must show the checkerboard"

    # Single ornament silhouette: exactly one connected opaque foreground blob,
    # no duplicated contour band around the pasted ornament.
    from scipy import ndimage
    foreground = ndimage.binary_fill_holes(marker_alpha > 200)
    labels, count = ndimage.label(foreground)
    sizes = ndimage.sum_labels(foreground, labels, range(1, count + 1))
    big = [size for size in sizes if size > 1000]
    assert len(big) == 1, f"expected one ornament blob, got {len(big)}"

    # A gap visible to one view must be filled from that view (opaque, real pixels),
    # never invented: transparent area must be confined to truly unseen regions.
    seen_by_right_only = report["background_from_right_pixels"]
    assert seen_by_right_only > 0, "the visible narrow slit should be filled from one capture"

    # Fingerprint mismatch must be rejected on commit.
    current_version = client.get(f"/api/cards/{card_id}").get_json()["card"]["version"]
    bad = client.post(f"/api/cards/{card_id}/midview/commit",
                      json={"synthesis": {**body, "fingerprint": "0" * 64}, "version": current_version})
    assert bad.status_code == 409, bad.get_json()

    commit = client.post(f"/api/cards/{card_id}/midview/commit",
                         json={"synthesis": {**body, "fingerprint": payload["report"]["synthesis_fingerprint"]},
                               "version": current_version, "title": "Middle view"})
    assert commit.status_code == 201, commit.get_json()
    added = commit.get_json()["view"]
    assert added["position"] == 0.0
    assert added["information"]["synthesized"]["kind"] == "mid-view"

    card = client.get(f"/api/cards/{card_id}").get_json()["card"]
    titles = [view["title"] for view in card["views"]]
    assert titles == ["left", "Middle view", "right"], titles
    assert all(view["enabled"] for view in card["views"])
    assert card["views"][1]["sha"] != card["views"][0]["sha"]

    # The RGBA original must round-trip with its alpha and feed the print pipeline.
    preview_check = client.post(f"/api/cards/{card_id}/preview", json={"angle": 0})
    assert preview_check.status_code == 200, preview_check.get_json()
    framing = client.post(f"/api/cards/{card_id}/inspect/framing", json={})
    assert framing.status_code == 200, framing.get_json()

    # Validation: malformed polygon is rejected.
    bad_body = {**body, "left_polygon": body["left_polygon"][:2]}
    rejected = client.post(f"/api/cards/{card_id}/midview/preview", json={"synthesis": bad_body})
    assert rejected.status_code in (400, 422), rejected.get_json()

    Image.fromarray(middle).save(FIXTURE / "middle_preview.png")
    Image.fromarray(marker_view).save(FIXTURE / "middle_markers.png")
    print("ALL MIDVIEW TESTS PASSED")


if __name__ == "__main__":
    main()
