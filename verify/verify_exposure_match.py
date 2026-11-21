"""End-to-end check for wall-reference angle exposure matching.

Builds four synthetic phone-style captures of one scene containing a matte
grey wall.  Each view gets a known exposure multiplier, and the wall marks are
deliberately partial: the left view only covers the left wall strip, the right
view only the right strip, so the chain must pass through the middle views.
Run with: python -B verify_exposure_match.py
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lenscard.exposure_match import exposure_match
from lenscard.imaging import frame_input
from lenscard.storage import Library


W, H = 480, 640
WALL_SRGB = np.asarray([150.0, 148.0, 152.0]) / 255.0  # near-neutral matte grey


def srgb_to_linear(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055)


def make_view(seed, gain, shift, *, tint=False):
    rng = np.random.default_rng(seed)
    base_srgb = np.zeros((H, W, 3), dtype=np.float32)
    base_srgb[:] = np.asarray([42.0, 66.0, 84.0]) / 255.0  # dark scene
    # Wall band across the upper part, with a couple of darker patches for texture.
    base_srgb[70:300, 20:460] = WALL_SRGB
    base_srgb[150:185, 90:150] = WALL_SRGB * 0.72
    base_srgb[210:245, 300:370] = WALL_SRGB * 0.8
    # Slight per-column brightness gradient on the wall (lighting falloff).
    grad = 1.0 + 0.05 * np.linspace(-1, 1, W)[None, :, None]
    base_srgb[70:300] *= grad
    linear = srgb_to_linear(base_srgb)
    # Camera exposure drift: pure linear gain (the thing exposure_ev corrects).
    linear = np.clip(linear * gain, 0, 1)
    if tint:
        linear[..., 0] *= 1.18  # a deliberately tinted, inconsistent reference view
    noise = rng.normal(0, 0.004, linear.shape).astype(np.float32)
    linear = np.clip(linear + noise, 0, 1)
    shown = (linear_to_srgb(linear) * 255 + 0.5).astype(np.uint8)
    image = Image.fromarray(shown, "RGB")
    # Simulate horizontal framing shift between views (parallax), +/- up to 26 px.
    rolled = np.roll(np.asarray(image), shift, axis=1)
    if shift > 0:
        rolled[:, :shift] = np.asarray(image)[:, :1]
    elif shift < 0:
        rolled[:, shift:] = np.asarray(image)[:, -1:]
    out = Image.fromarray(rolled, "RGB")
    buf = tempfile.SpooledTemporaryFile()
    out.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()


def main():
    data_dir = tempfile.mkdtemp(prefix="lenscard-verify-")
    library = Library(data_dir)
    card = library.create_card("Wall exposure verification", plan={
        "framing": {"mode": "center", "cover": "fill", "fit_motion": True},
    })
    # Four angles, gains in linear light (EV = log2(gain) relative differences).
    angles = [-24.0, -8.0, 9.0, 25.0]
    gains = [1.0, 1.25, 0.82, 1.6]          # +0.32, -0.29, +0.68 EV drifts
    shifts = [22, 8, -10, -24]
    view_ids = []
    entries = []
    for n, (angle, gain, shift) in enumerate(zip(angles, gains, shifts, strict=True)):
        raw = make_view(100 + n, gain, shift, tint=(n == 3))
        frame = frame_input(raw, f"view-{n}.png", angle, library)
        entries.append((frame, {"title": f"Angle {angle:g}", "position": angle, "enabled": True}))
    added = library.add_views(card["id"], entries, version=0)
    view_ids = [row["id"] for row in added]
    card, records = library.material_snapshot(card["id"])

    # Partial wall strips: left view only left strip, right view only right strip,
    # middle views span both sides so the chain can connect.
    marks = [
        {"view": view_ids[0], "region": [0.05, 0.14, 0.46, 0.42]},
        {"view": view_ids[1], "region": [0.18, 0.14, 0.62, 0.42]},
        {"view": view_ids[2], "region": [0.34, 0.14, 0.80, 0.42]},
        {"view": view_ids[3], "region": [0.55, 0.14, 0.94, 0.42]},
    ]
    body = {"marks": marks, "fit_motion": True, "discord_ev": 0.18}
    report, images = exposure_match(library, records, body)

    print("== summary ==")
    print(report["summary"])
    print("\n== views ==")
    for row in report["views"]:
        print(f"{row['index']} {row['title']:>10} current={row['current_ev']:+.3f} "
              f"proposed={row['proposed_ev']:+.3f} delta={row['delta_ev']:+.3f} "
              f"confident={row['confident']} rel={row['relative_exposure_ev']}")
    print("\n== references ==")
    for edge in report["references"]:
        print(f"{edge['left'][:8]}..->{edge['right'][:8]}.. delta_ev={edge['delta_ev']:+.3f} "
              f"overlap={edge['overlap_pixels']} consistent={edge['consistent']}")
    print("\n== inconsistent ==", report["inconsistent"])
    print("\n== images ==", sorted(images))

    # Expectations: views 0-2 form a consistent chain; view 3 (tinted) should be
    # flagged inconsistent and therefore stay uncertain at its current EV (0).
    status = {row["view_id"]: row for row in report["views"]}
    chain = [status[view_ids[i]] for i in range(3)]
    assert all(row["confident"] for row in chain), "first three views must be confident"
    assert not status[view_ids[3]]["confident"], "tinted view must be uncertain"
    assert status[view_ids[3]]["proposed_ev"] == 0.0, "uncertain view keeps its manual EV"
    # Relative EV recovered within the chain must match log2 gain ratios.
    anchor = chain[0]["relative_exposure_ev"]
    for i in range(3):
        expected = np.log2(gains[i] / gains[0])
        got = chain[i]["relative_exposure_ev"] - anchor
        assert abs(got - expected) < 0.12, f"view {i} relative EV {got} vs expected {expected}"
    # Every confident proposal must equal current EV minus its recovered offset.
    for row in chain:
        assert abs(row["proposed_ev"] - (row["current_ev"] - row["relative_exposure_ev"])) < 1e-9
    assert report["summary"]["inconsistent_references"] >= 1
    for key in ("marked-wall", "ev-proposal", "wall-luminance"):
        assert key in images and len(images[key]) > 1000

    # Now exercise the write-back path through the library.
    confident = [row for row in report["views"] if row["confident"]]
    updated = library.apply_exposure_match(card["id"], confident, card["version"],
                                           fingerprint=report["proposal_sha"])
    after = {row["id"]: row["exposure_ev"] for row in updated}
    for row in confident:
        assert abs(after[row["view_id"]] - row["proposed_ev"]) < 1e-3
    assert after[view_ids[3]] == 0.0, "uncertain view must remain untouched in storage"

    # A second apply with a stale version must fail.
    from lenscard.models import Conflict
    try:
        library.apply_exposure_match(card["id"], confident, card["version"],
                                     fingerprint=report["proposal_sha"])
    except Conflict:
        pass
    else:
        raise AssertionError("stale version apply should conflict")

    print("\nALL CHECKS PASSED")
    print("data dir kept for inspection:", data_dir)


if __name__ == "__main__":
    main()
