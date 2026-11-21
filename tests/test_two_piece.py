"""Checks for the two-piece lens lay-up and the unchanged one-piece render."""

import io
import json
import tempfile
import unittest

import numpy as np
from PIL import Image

from lenscard.demo import SAMPLES, create_sample
from lenscard.http import create_app
from lenscard.models import Plan, WorkbenchError
from lenscard.optics import angular_response, build_grid
from lenscard.storage import Library
from lenscard.workflow import CardJob

PAIR = {
    "enabled": True,
    "seam_mm": 22.0,
    "overlap_mm": 1.5,
    "top": "left",
    "edge_guard_pitches": 1.0,
    "left": {"lpi": 47.6, "phase": -0.142, "shear": 0.011},
    "right": {"lpi": 45.4, "phase": 0.3, "shear": -0.005},
}


def card_with(plan_patch=None):
    data = tempfile.mkdtemp()
    library = Library(data)
    card = create_sample(library, SAMPLES["harbor"])
    if plan_patch is not None:
        config = dict(card["config"])
        plan_patch(config)
        card = library.update_card(
            card["id"], version=card["version"], title=card["title"], note=card["note"], config=config,
        )
    return data, library, card


class GridTests(unittest.TestCase):
    def setUp(self):
        self.plan = Plan.parse({"sheet": {"width_mm": 47, "height_mm": 62, "dpi": 400}, "two_piece": PAIR})
        self.grid = build_grid(self.plan)

    def test_each_piece_uses_its_own_phase(self):
        columns = np.asarray([78.0, 629.0])
        rows = np.asarray([40.0, 40.0])
        phases = self.grid.phase_at(columns, rows)
        for column, row, phase, piece in zip(columns, rows, phases, self.grid.pieces(), strict=True):
            direct = ((self.grid.x_mm(column) - piece.shear * self.grid.y_mm(row))
                      / piece.mm_per_lens + piece.phase)
            self.assertAlmostEqual(float(phase), float(direct), places=6)

    def test_gap_is_blank_for_every_view(self):
        gap = self.grid.gap_columns()
        self.assertTrue(gap.any())
        count = 3
        for slot in range(count):
            weights = self.grid.coverage("solid", 0, 40, slot, count)
            self.assertEqual(float(weights[:, gap].sum()), 0.0)
            self.assertGreater(float(weights[:, ~gap].sum()), 0.0)

    def test_view_order_is_shared_across_the_seam(self):
        count = 3
        first = self.grid.coverage("solid", 0, 60, 0, count)
        left = first[:, self.grid.face_mm() < self.grid.split_mm].mean()
        right = first[:, self.grid.face_mm() >= self.grid.split_mm].mean()
        self.assertGreater(left, 0)
        self.assertGreater(right, 0)

    def test_response_mixes_both_pieces(self):
        weights, window = angular_response(self.grid, 0.0, 3, self.plan.sampling)
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=6)
        self.assertEqual(len(window), 2)

    def test_overlap_side_moves_the_boundary(self):
        left = build_grid(self.plan).split_layout
        flipped = Plan.parse({"sheet": {"width_mm": 47, "height_mm": 62, "dpi": 400},
                              "two_piece": {**PAIR, "top": "right"}})
        right = build_grid(flipped).split_layout
        self.assertGreater(left["boundary_mm"], PAIR["seam_mm"])
        self.assertLess(right["boundary_mm"], PAIR["seam_mm"])
        self.assertEqual(left["overlap_face_mm"], [PAIR["seam_mm"], PAIR["seam_mm"] + PAIR["overlap_mm"]])

    def test_zero_guard_single_seam_matches_one_piece(self):
        plain = Plan.parse({
            "sheet": {"width_mm": 47, "height_mm": 62, "dpi": 400},
            "lens": {"lpi": 46.8, "phase": -0.142, "shear": 0.011},
        })
        split = Plan.parse({"sheet": {"width_mm": 47, "height_mm": 62, "dpi": 400}, "two_piece": {
            **PAIR, "overlap_mm": 0.0, "edge_guard_pitches": 0.0,
            "left": {"lpi": 46.8, "phase": -0.142, "shear": 0.011},
            "right": {"lpi": 46.8, "phase": -0.142, "shear": 0.011},
        }})
        first, second = build_grid(plain), build_grid(split)
        np.testing.assert_array_equal(first.phase_matrix(0, 20), second.phase_matrix(0, 20))
        np.testing.assert_array_equal(
            first.coverage("solid", 0, 20, 1, 3), second.coverage("solid", 0, 20, 1, 3),
        )

    def test_piece_pitch_above_dot_budget_is_rejected(self):
        with self.assertRaises(WorkbenchError):
            Plan.parse({"sheet": {"dpi": 400}, "two_piece": {**PAIR, "right": {"lpi": 140}}}
                       ).check_sampling(3)


class ValidationTests(unittest.TestCase):
    def test_bad_setups_are_rejected(self):
        for pair in (
            {**PAIR, "seam_mm": 47.0},
            {**PAIR, "overlap_mm": 30},
            {**PAIR, "seam_mm": 45.0, "top": "left", "overlap_mm": 4},
            {**PAIR, "seam_mm": 46.0, "edge_guard_pitches": 6},
        ):
            with self.assertRaises(WorkbenchError):
                Plan.parse({"sheet": {"width_mm": 47, "height_mm": 62}, "two_piece": pair})

    def test_disabled_pair_round_trips(self):
        settings = Plan.parse({}).public()
        self.assertEqual(Plan.parse(settings).public(), settings)


class RenderTests(unittest.TestCase):
    def test_preview_marks_seam_and_gap(self):
        data, library, card = card_with(lambda config: config.update(two_piece=PAIR))
        result = CardJob(library, card["id"]).preview(0)
        report = result.report
        self.assertTrue(report["bands"]["two_piece"]["enabled"])
        self.assertEqual(report["bands"]["two_piece"]["gap_supported_fraction"], 0.0)
        for name in ("print-preview", "seam-overview", "seam-detail"):
            self.assertIn(name, result.images)
        self.assertGreater(len(result.images["seam-detail"]), 100)

    def test_one_piece_preview_has_no_seam_images(self):
        data, library, card = card_with()
        result = CardJob(library, card["id"]).preview(0)
        self.assertFalse(result.report["bands"]["grid"]["two_piece"]["enabled"])
        self.assertEqual(result.report["seam_detail"], {})
        self.assertNotIn("seam-detail", result.images)

    def test_draft_two_piece_preview_does_not_save(self):
        data, library, card = card_with()
        client = create_app(data).test_client()
        before = client.get(f"/api/cards/{card['id']}").get_json()["card"]["config"]["two_piece"]
        reply = client.post(f"/api/cards/{card['id']}/two-piece-preview", json={"two_piece": PAIR, "angle": 0})
        self.assertEqual(reply.status_code, 200)
        body = reply.get_json()
        self.assertTrue(body["report"]["proposed"])
        self.assertEqual(body["report"]["bands"]["two_piece"]["top"], "left")
        after = client.get(f"/api/cards/{card['id']}").get_json()["card"]["config"]["two_piece"]
        self.assertEqual(before, after)

    def test_approved_pdf_keeps_actual_size_and_two_pieces(self):
        data, library, card = card_with(lambda config: config.update(two_piece=PAIR))
        client = create_app(data).test_client()
        preview = client.post(f"/api/cards/{card['id']}/preview", json={"angle": 0}).get_json()
        proof = client.post(f"/api/cards/{card['id']}/proofs", json={
            "title": "two piece", "version": preview["report"]["draft_version"],
            "candidate_sha": preview["report"]["candidate_sha"],
        }).get_json()["proof"]
        client.put(f"/api/cards/{card['id']}/proofs/{proof['id']}/approve",
                   json={"fingerprint": proof["fingerprint"], "note": ""})
        pdf = client.get(f"/api/cards/{card['id']}/approved/pdf?download=1").data
        plan = card["config"]["sheet"]
        self.assertTrue(pdf.startswith(b"%PDF"))
        integrity = client.get(f"/api/cards/{card['id']}/approved-integrity").get_json()
        self.assertTrue(integrity["verified"])
        self.assertEqual(integrity["media_box_actual_mm"], [plan["width_mm"], plan["height_mm"]])


if __name__ == "__main__":
    unittest.main()
