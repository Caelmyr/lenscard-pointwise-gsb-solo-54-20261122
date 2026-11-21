"""End-to-end check for multi-photograph angle isolation measurement.

Run from the repository root with:  python3 -I scripts/check_multi_photo.py
It builds synthetic captures in memory, feeds them through measured_angle_set
and asserts every selection branch (agree, single source, glare rejection,
blank, reference rejection, legacy single-photo format).
"""
import os
import sys
from pathlib import Path

# Isolated mode (-I) drops PYTHONPATH, so restore the repository and the
# provisioned third-party packages explicitly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
for extra in os.environ.get("POINTWISE_SITE_PACKAGES", "/opt/pointwise-env/site-packages").split(":"):
    if extra and extra not in sys.path:
        sys.path.append(extra)

from lenscard.models import Unprocessable
from lenscard.opticaltest import measured_angle_set, simulated_photograph


class PlanSheet:
    width_mm, height_mm, dpi, mount = 47.0, 62.0, 400, "back"

    @property
    def size(self):
        return round(self.width_mm * self.dpi / 25.4), round(self.height_mm * self.dpi / 25.4)


class PlanLens:
    lpi, phase, shear, order = 46.8, -0.142, .011, "forward"


class Plan:
    sheet = PlanSheet()
    lens = PlanLens()

    def check_sampling(self, count):
        pass


def records(count):
    return [{"enabled": True, "position": -18 + 18 * i} for i in range(count)]


def statuses(group):
    return {patch["index"]: patch["status"] for patch in group["patches"]}


def main():
    count = 3
    frame = [.13, .17, .87, .85]

    # Angle 1: photo A glares on patches 2 and 3; photo B is clean there.
    # Photo B masks patch 1, so patch 1 comes from a single source (A).
    a, _ = simulated_photograph(records(count), Plan(), 1, glare_rows=(2, 3), glare_strength=.6)
    b, _ = simulated_photograph(records(count), Plan(), 1, glare_strength=0.0)
    files = [("a.png", a), ("b.png", b)]
    settings = [{
        "angle": -18.0, "expected": 1, "frame": frame,
        "photos": [
            {"name": "A lower glare", "frame": frame, "exclude": []},
            {"name": "B clean, masks patch 1", "frame": frame, "exclude": [1]},
        ],
    }]
    report, diagrams = measured_angle_set(files, settings, count)
    group = report["angle_groups"][0]
    got = statuses(group)
    print("angle 1 statuses:", got)
    assert got[1] == "single-source", got
    assert group["patches"][0]["source_photo"] == "a1-photo-1"
    assert got[2] == "glare-rejected-other", got
    assert group["patches"][1]["source_photo"] == "a1-photo-2"
    assert "glare" in group["patches"][1]["note"].lower()
    assert got[3] == "glare-rejected-other", got
    assert group["result"]["complete"] is True
    assert group["result"]["passed"] is True, group["result"]
    assert {"a1-photo-1", "a1-photo-2", "contrast"} <= set(diagrams)

    # Angle 2: both captures glare on the same numbered patch -> stays blank.
    c, _ = simulated_photograph(records(count), Plan(), 2, glare_rows=(2,), glare_strength=.6)
    d, _ = simulated_photograph(records(count), Plan(), 2, glare_rows=(2,), glare_strength=.55)
    files2 = [("c.png", c), ("d.png", d)]
    settings2 = [{
        "angle": 0.0, "expected": 2, "frame": frame,
        "photos": [
            {"name": "C glare middle", "frame": frame, "exclude": []},
            {"name": "D glare middle", "frame": frame, "exclude": []},
        ],
    }]
    report2, _ = measured_angle_set(files2, settings2, count)
    g2 = report2["angle_groups"][0]
    print("angle 2 statuses:", statuses(g2))
    assert statuses(g2)[2] in {"ambiguous", "blank"}, statuses(g2)
    assert g2["patches"][1]["ratio"] is None
    assert g2["result"]["complete"] is False
    assert g2["result"]["passed"] is False
    assert 0.0 in report2["incomplete_angles"]
    assert report2["failed_angles"][0]["angle"] == 0.0

    # Angle 3: a clean repeated capture must agree and select the tightest source.
    e, _ = simulated_photograph(records(count), Plan(), 3, angle_yield=.004)
    f, _ = simulated_photograph(records(count), Plan(), 3, angle_yield=.004)
    report3, _ = measured_angle_set(
        [("e.png", e), ("f.png", f)],
        [{"angle": 18.0, "expected": 3, "frame": frame, "photos": [
            {"name": "E", "frame": frame, "exclude": []},
            {"name": "F", "frame": frame, "exclude": []},
        ]}],
        count,
    )
    g3 = report3["angle_groups"][0]
    print("angle 3 statuses:", statuses(g3))
    assert all(status == "agreed" for status in statuses(g3).values()), statuses(g3)
    assert g3["result"]["passed"] is True

    # Reference rows may never be masked: the capture is rejected but the
    # angle group stays measurable from its other photograph.
    report_mask, _ = measured_angle_set(
        [("a.png", a), ("b.png", b)],
        [{"angle": -18.0, "expected": 1, "frame": frame, "photos": [
            {"name": "bad mask", "frame": frame, "exclude": [0]},
            {"name": "good", "frame": frame, "exclude": []},
        ]}],
        count,
    )
    masked_group = report_mask["angle_groups"][0]
    rejected = [photo for photo in masked_group["photos"] if photo["status"] == "rejected"]
    assert len(rejected) == 1 and "dark reference" in rejected[0]["rejection"], rejected
    assert masked_group["result"]["rejected_photos"] == ["a1-photo-1"]
    print("reference mask rejection:", rejected[0]["rejection"])

    # An undecodable file rejects only that photograph, not the whole angle.
    report_bad, _ = measured_angle_set(
        [("broken.png", b"not an image"), ("b.png", b)],
        [{"angle": -18.0, "expected": 1, "frame": frame, "photos": [
            {"name": "broken file", "frame": frame, "exclude": []},
            {"name": "good", "frame": frame, "exclude": []},
        ]}],
        count,
    )
    bad_group = report_bad["angle_groups"][0]
    bad_rejected = [photo for photo in bad_group["photos"] if photo["status"] == "rejected"]
    assert len(bad_rejected) == 1, bad_rejected
    print("decode rejection:", bad_rejected[0]["rejection"])

    # Duplicate angles are rejected; repeated shots belong inside one group.
    try:
        measured_angle_set(
            [("e.png", e), ("f.png", f)],
            [{"angle": 18.0, "expected": 3, "frame": frame, "photos": [
                {"name": "E", "frame": frame, "exclude": []}]},
             {"angle": 18.0, "expected": 3, "frame": frame, "photos": [
                {"name": "F", "frame": frame, "exclude": []}]}],
            count,
        )
    except Unprocessable as error:
        print("duplicate angle rejected:", error)
    else:
        raise AssertionError("duplicate angles must fail")

    # File count must match the listed photographs.
    try:
        measured_angle_set(
            [("e.png", e)],
            [{"angle": 18.0, "expected": 3, "frame": frame, "photos": [
                {"name": "E", "frame": frame, "exclude": []},
                {"name": "F", "frame": frame, "exclude": []},
            ]}],
            count,
        )
    except Unprocessable as error:
        print("file count mismatch rejected:", error)
    else:
        raise AssertionError("missing image files must fail")

    # Legacy single-photo-per-angle settings keep working.
    legacy, _ = measured_angle_set(
        [("e.png", e)],
        [{"angle": 18.0, "expected": 3, "frame": frame, "name": "legacy"}],
        count,
    )
    assert legacy["photo_count"] == 1
    assert legacy["angle_groups"][0]["photo_count"] == 1
    print("legacy format statuses:", statuses(legacy["angle_groups"][0]))

    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    sys.exit(main())
