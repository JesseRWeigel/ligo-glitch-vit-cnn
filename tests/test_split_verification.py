import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "07_verify_split.py"


class SplitVerificationTests(unittest.TestCase):
    def run_verifier(self, defect=None):
        with tempfile.TemporaryDirectory() as directory:
            metadata = Path(directory) / "data" / "metadata"
            metadata.mkdir(parents=True)
            original = [
                [f"toy-{i}", 1_280_000_000 + i * 100, f"class-{i % 23}"]
                for i in range(102)
            ]

            def write_csv(name, rows):
                with (metadata / name).open("w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["gravityspy_id", "event_time", "ml_label"])
                    writer.writerows(rows)

            train = [row.copy() for row in original[:62]]
            if defect == "foreign":
                train[1][0] = "not-in-source"
            elif defect == "duplicate":
                train[1][0] = train[0][0]
            elif defect == "missing":
                train[1][0] = ""
            elif defect == "source_duplicate":
                original[101][0] = original[100][0]
            elif defect == "swapped_labels":
                train[0][2], train[1][2] = train[1][2], train[0][2]
            elif defect == "missing_label":
                train[0][2] = ""
            elif defect == "swapped_source_times":
                train[0][1], train[1][1] = train[1][1], train[0][1]
            elif defect == "shifted_source_time":
                train[1][1] += 0.01
            elif defect == "small_serialization_difference":
                train[1][1] += 0.0000002
            original[100][1] = original[61][1] + 30
            original[101][1] = original[76][1] + 30
            if defect == "wrong_exclusion_same_count":
                original[100][1] = original[10][1] + 30
            if defect == "omission_beyond_buffer":
                original[100][1] = original[61][1] + 90
            elif defect == "omission_at_buffer_edge":
                original[100][1] = original[61][1] + 60
            write_csv("gravity_spy_o3_filtered.csv", list(reversed(original)))
            write_csv("train_manifest.csv", train)
            write_csv("val_manifest.csv", original[62:77])
            write_csv("test_manifest.csv", original[77:100])
            stats = metadata / "split_statistics.json"
            stats.write_text(json.dumps({"samples_excluded_in_gaps": 3 if defect == "bad_count" else 2}))
            result = subprocess.run(
                [sys.executable, str(SCRIPT)], cwd=directory, capture_output=True, text=True
            )
            return result, json.loads(stats.read_text())

    def test_clean_split_reports_source_membership(self):
        result, stats = self.run_verifier()
        self.assertEqual(result.returncode, 0, result.stderr)
        detail = stats["verification_details"]["source_membership"]
        self.assertEqual(detail["source_only_ids"], 2)
        self.assertEqual(detail["assigned_label_mismatch_rows"], 0)
        self.assertEqual(detail["status"], "PASS")

    def test_swapped_labels_fail_even_with_unchanged_ids_and_class_counts(self):
        result, stats = self.run_verifier("swapped_labels")
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        detail = stats["verification_details"]["source_membership"]
        self.assertEqual(detail["assigned_label_mismatch_rows"], 2)
        self.assertFalse(detail["assigned_ids_absent_from_source"])
        self.assertEqual(stats["verification_results"]["class_coverage"], "PASS")

    def test_swapped_event_times_fail_despite_unchanged_split_gaps(self):
        result, stats = self.run_verifier("swapped_source_times")
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertEqual(stats["verification_results"]["no_time_range_overlap"], "PASS")
        self.assertEqual(stats["verification_details"]["source_membership"]["assigned_time_mismatch_rows"], 2)

    def test_absolute_tolerance_does_not_grow_with_gps_time(self):
        result, stats = self.run_verifier("shifted_source_time")
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertEqual(stats["verification_details"]["source_membership"]["assigned_time_mismatch_rows"], 1)

    def test_sub_microsecond_serialization_difference_passes(self):
        result, stats = self.run_verifier("small_serialization_difference")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(stats["verification_details"]["source_membership"]["assigned_time_mismatch_rows"], 0)

    def test_membership_errors_fail_even_if_all_other_checks_pass(self):
        for defect, metric in [
            ("foreign", "assigned_ids_absent_from_source"),
            ("duplicate", "duplicate_assignment_rows"),
            ("missing", "missing_assignment_id_rows"),
            ("source_duplicate", "duplicate_source_rows"),
            ("bad_count", "gap_count_mismatch"),
            ("missing_label", "assigned_label_mismatch_rows"),
        ]:
            with self.subTest(defect=defect):
                result, stats = self.run_verifier(defect)
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                statuses = stats["verification_results"]
                self.assertEqual(statuses["source_membership"], "FAIL")
                self.assertTrue(
                    all(status == "PASS" for key, status in statuses.items() if key != "source_membership")
                )
                self.assertTrue(stats["verification_details"]["source_membership"][metric])

    def test_omitted_event_inside_training_fails_even_with_matching_counts(self):
        result, stats = self.run_verifier("wrong_exclusion_same_count")
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        detail = stats["verification_details"]["source_membership"]
        self.assertEqual(detail["source_only_ids"], 2)
        self.assertFalse(detail["gap_count_mismatch"])
        self.assertEqual(detail["source_only_rows_outside_gaps"], 1)

    def test_source_event_after_the_buffer_must_start_the_next_split(self):
        result, stats = self.run_verifier("omission_beyond_buffer")
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        checks = stats["verification_results"]
        self.assertEqual(checks["no_time_range_overlap"], "PASS")
        detail = stats["verification_details"]["source_membership"]
        self.assertFalse(detail["gap_count_mismatch"])
        self.assertEqual(detail["source_only_rows_outside_gaps"], 1)

    def test_event_exactly_at_buffer_edge_can_be_excluded(self):
        result, stats = self.run_verifier("omission_at_buffer_edge")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        detail = stats["verification_details"]["source_membership"]
        self.assertEqual(detail["source_only_rows_outside_gaps"], 0)
