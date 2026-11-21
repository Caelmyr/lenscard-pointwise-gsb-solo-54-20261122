"""HTTP-level check for the angle exposure match endpoints, using the demo card."""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lenscard.demo import SAMPLES, create_sample
from lenscard.http import create_app
from lenscard.storage import Library


def main():
    data_dir = tempfile.mkdtemp(prefix="lenscard-http-")
    library = Library(data_dir)
    card = create_sample(library, SAMPLES["harbor"])
    app = create_app(data_dir)
    client = app.test_client()
    cid = card["id"]
    views = [v for v in card["views"] if v["enabled"]]
    assert len(views) == 3

    marks = [
        {"view": views[0]["id"], "region": [0.08, 0.10, 0.40, 0.30]},
        {"view": views[1]["id"], "region": [0.20, 0.10, 0.62, 0.30]},
        {"view": views[2]["id"], "region": [0.55, 0.10, 0.92, 0.30]},
    ]
    resp = client.post(f"/api/cards/{cid}/exposure-match", json={
        "marks": marks, "fit_motion": True, "discord_ev": 0.18,
    })
    assert resp.status_code == 200, resp.get_json()
    payload = resp.get_json()
    report = payload["report"]
    assert report["kind"] == "angle-exposure-match"
    assert set(payload["images"]) >= {"marked-wall", "ev-proposal", "wall-luminance"}
    print("preview summary:", report["summary"])
    print("views:", [(r["title"], r["confident"], r["proposed_ev"]) for r in report["views"]])

    # Validation: empty marks rejected.
    bad = client.post(f"/api/cards/{cid}/exposure-match", json={"marks": []})
    assert bad.status_code in (400, 422), bad.status_code

    # Apply with the returned fingerprint and current version.
    apply = client.post(f"/api/cards/{cid}/exposure-match/apply", json={
        "marks": marks, "discord_ev": 0.18, "fit_motion": True,
        "version": report["draft_version"], "proposal_sha": report["proposal_sha"],
    })
    assert apply.status_code == 200, apply.get_json()
    result = apply.get_json()["result"]
    print("applied:", [(a["title"], a["previous_ev"], a["exposure_ev"]) for a in result["applied"]])
    print("kept manual:", result["kept_manual"])

    # Stale version must now conflict (version incremented by the apply).
    stale = client.post(f"/api/cards/{cid}/exposure-match/apply", json={
        "marks": marks, "discord_ev": 0.18, "fit_motion": True,
        "version": report["draft_version"], "proposal_sha": report["proposal_sha"],
    })
    assert stale.status_code == 409, stale.status_code

    # Tampered fingerprint must conflict.
    current = client.get(f"/api/cards/{cid}").get_json()["card"]
    tampered = client.post(f"/api/cards/{cid}/exposure-match/apply", json={
        "marks": marks, "discord_ev": 0.18, "fit_motion": True,
        "version": current["version"], "proposal_sha": "0" * 64,
    })
    assert tampered.status_code == 409, tampered.status_code

    # Existing per-view edit endpoint still works and stores the same EV field.
    target = current["views"][0]
    edit = client.put(f"/api/cards/{cid}/views/{target['id']}", json={
        "version": current["version"],
        "settings": {"title": target["title"], "position": target["position"],
                     "enabled": target["enabled"], "exposure_ev": 0.42},
    })
    assert edit.status_code == 200, edit.get_json()
    assert abs(edit.get_json()["view"]["exposure_ev"] - 0.42) < 1e-9

    # Existing inspection tool still responds.
    exposure = client.post(f"/api/cards/{cid}/inspect/exposure", json={
        "view": target["id"], "region": [0.1, 0.1, 0.5, 0.4],
    })
    assert exposure.status_code == 200, exposure.get_json()
    assert exposure.get_json()["report"]["kind"] == "exposure"

    print("\nHTTP CHECKS PASSED; data dir:", data_dir)


if __name__ == "__main__":
    main()
