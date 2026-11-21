#!/usr/bin/env python3
"""Browser-level check of the middle-view synthesis dialog via Playwright."""
import json
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE = "http://127.0.0.1:19988"
FIXTURE = Path("/tmp/lenscard-fixture")


def main():
    marks = json.loads((FIXTURE / "marks.json").read_text())
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1500, "height": 980})
        errors = []
        page.on("pageerror", lambda exc: errors.append(str(exc)))
        page.on("console", lambda msg: errors.append(msg.text) if msg.type == "error" else None)

        page.goto(BASE)
        page.wait_for_selector(".library-list")
        page.click("#new-button")
        page.wait_for_selector("dialog input[name=title]")
        page.fill("input[name=title]", "Sill browser pair")
        page.click("dialog button[type=submit]")
        page.wait_for_selector(".side h1")

        def upload(path, angle):
            page.click("button[data-action=add-view]")
            page.set_input_files("input[type=file]", str(path))
            page.fill("input[name=angle]", str(angle))
            page.click("dialog button[type=submit]")
            page.wait_for_selector("dialog", state="hidden")

        upload(FIXTURE / "left.png", -12)
        upload(FIXTURE / "right.png", 12)

        page.click("button[data-action=midview-open]")
        page.wait_for_selector(".midview-form")
        canvases = page.locator(".mid-canvas")
        assert canvases.count() == 2

        def mark_outline(side_index, polygon):
            box = canvases.nth(side_index).bounding_box()
            for x, y in polygon:
                page.mouse.click(box["x"] + x * box["width"], box["y"] + y * box["height"])

        # Ornament outlines (looped polygon so closing the shape is easy).
        mark_outline(0, marks["polygons"][0] + [marks["polygons"][0][0]])
        mark_outline(1, marks["polygons"][1] + [marks["polygons"][1][0]])

        def mark_anchor(side_index, point):
            page.click(f'.mid-pane:nth-child({side_index + 1}) button[data-value^="anchor:"]')
            box = canvases.nth(side_index).bounding_box()
            x, y = point
            page.mouse.click(box["x"] + x * box["width"], box["y"] + y * box["height"])

        mark_anchor(0, marks["anchors"][0])
        mark_anchor(1, marks["anchors"][1])

        page.fill("input[name=mid]", "0.5")
        page.dispatch_event("input[name=mid]", "change")
        page.screenshot(path="/tmp/lenscard-fixture/e2e-dialog-marked.png", full_page=True)

        with page.expect_response(lambda r: "/midview/preview" in r.url) as info:
            page.click("button[data-action=mid-preview]")
        response = info.value
        assert response.ok, response.status
        payload = response.json()
        report = payload["report"]
        print("UI preview report:", json.dumps({
            k: report[k] for k in ("background_shift_px", "foreground_shift_px",
                                  "transparent_pixels", "background_from_right_pixels")
        }, indent=2))
        page.wait_for_selector(".mid-result", state="attached")
        page.locator(".mid-result").scroll_into_view_if_needed()
        page.screenshot(path="/tmp/lenscard-fixture/e2e-dialog-preview.png", full_page=True)
        assert report["transparent_pixels"] > 0

        # Undo and clear controls must work without breaking the dialog.
        page.click('.mid-pane:first-child button[data-action=mid-undo]')
        page.wait_for_selector(".midview-form")
        page.click('.mid-pane:first-child button[data-action=mid-clear]')
        # Clear removes both the outline and the corner mark; re-do both.
        page.click('.mid-pane:first-child button[data-value^="polygon:"]')
        mark_outline(0, marks["polygons"][0] + [marks["polygons"][0][0]])
        mark_anchor(0, marks["anchors"][0])
        page.click("button[data-action=mid-preview]")
        page.wait_for_selector(".mid-result", state="attached")

        with page.expect_response(lambda r: "/midview/commit" in r.url) as commit_info:
            page.click("button[data-action=mid-commit]")
        commit_response = commit_info.value
        assert commit_response.ok, commit_response.status
        body = commit_response.json()
        print("committed view:", body["view"]["title"], body["view"]["position"])
        page.wait_for_selector("dialog", state="hidden")
        page.wait_for_selector(".side h1")
        titles = page.locator(".viewlist b").all_inner_texts()
        print("angle list:", titles)
        assert len(titles) == 3
        page.screenshot(path="/tmp/lenscard-fixture/e2e-card-3views.png", full_page=True)

        # The print pipeline must accept the three-view card including RGBA.
        with page.expect_response(lambda r: r.url.endswith("/preview")) as prev_info:
            page.click("button[data-action=preview]")
        assert prev_info.value.ok, prev_info.value.status

        if errors:
            print("BROWSER ERRORS:", errors, file=sys.stderr)
            raise SystemExit(1)
        print("PLAYWRIGHT E2E PASSED")
        browser.close()


if __name__ == "__main__":
    main()
