import copy
import unittest
from pathlib import Path

from scripts import smoke_evaluation as smoke


class EvaluationSmokeTests(unittest.TestCase):
    def setUp(self):
        self.rows = smoke.make_rows()
        self.manifest = smoke.run(self.rows)

    def test_expected_manifest_and_known_macro_f1(self):
        expected = Path(__file__).resolve().parents[1] / "docs" / "synthetic-smoke-result.json"
        self.assertEqual(smoke.canonical_json(self.manifest), expected.read_text())
        self.assertAlmostEqual(self.manifest["macro_f1"]["stub_a"], 1 / 3)
        self.assertEqual(self.manifest["macro_f1"]["stub_b"], 1)
        self.assertEqual(self.manifest["memberships"]["gap_excluded"], ["toy-14", "toy-17"])

    def test_input_order_does_not_change_manifest(self):
        self.assertEqual(self.manifest, smoke.run(list(reversed(self.rows))))

    def test_duplicate_event_rejected(self):
        with self.assertRaisesRegex(ValueError, "IDs must be unique"):
            smoke.run(self.rows + [self.rows[0]])

    def test_membership_controls(self):
        clean = self.manifest["memberships"]
        for move, pattern in [
            (lambda parts: parts["test"].append("toy-00"), "multiple split positions"),
            (lambda parts: parts["test"].pop(), "does not account"),
            (lambda parts: parts["val"].append(parts["gap_excluded"].pop(0)), "temporal gap"),
        ]:
            with self.subTest(pattern=pattern):
                broken = copy.deepcopy(clean)
                move(broken)
                with self.assertRaisesRegex(ValueError, pattern):
                    smoke.audit_memberships(broken, self.rows, 60)
