import csv
import runpy
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "02_parse_metadata.py"
CLASSES = runpy.run_path(str(SCRIPT))["VALID_CLASSES"]
COLUMNS = [
    "gravityspy_id", "event_time", "ifo", "ml_label", "ml_confidence", "snr",
    "peak_frequency", "central_freq", "bandwidth", "duration", "q_value",
]
FILES = [("H1", "O3a", 1238166100), ("H1", "O3b", 1256655700),
         ("L1", "O3a", 1238166100), ("L1", "O3b", 1256655700)]


class MetadataParseTests(unittest.TestCase):
    def run_parser(self, *, missing=(), invalid=False, optimized=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "data/raw/zenodo_5649212"
            raw.mkdir(parents=True)
            for detector, run, start in FILES:
                name = f"{detector}_{run}.csv"
                if name in missing:
                    continue
                with (raw / name).open("w", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(COLUMNS)
                    for index in range(2000):
                        snr = "" if invalid and index == 0 else 10
                        writer.writerow([
                            f"{detector}-{run}-{index}", start + index, detector,
                            CLASSES[index % len(CLASSES)], 0.99, snr, 100, 100, 20, 1, 8,
                        ])
            script = root / "scripts" / SCRIPT.name
            script.parent.mkdir()
            shutil.copyfile(SCRIPT, script)
            output = root / "data/metadata/gravity_spy_o3_filtered.csv"
            output.parent.mkdir()
            output.write_text("previous verified metadata\n")
            command = [sys.executable] + (["-O"] if optimized else []) + [str(script)]
            result = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=20)
            return result, output.read_text()

    def test_missing_partition_fails_and_preserves_previous_output(self):
        result, output = self.run_parser(missing=("L1_O3b.csv",))
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("L1_O3b.csv", result.stderr)
        self.assertEqual(output, "previous verified metadata\n")

    def test_invalid_filtered_metadata_preserves_previous_output(self):
        for optimized in (False, True):
            with self.subTest(optimized=optimized):
                result, output = self.run_parser(invalid=True, optimized=optimized)
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertIn("NaN found in snr", result.stderr)
                self.assertEqual(output, "previous verified metadata\n")

    def test_complete_input_writes_all_partitions(self):
        result, output = self.run_parser()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(output.splitlines()), 8001)
        for detector, run, _ in FILES:
            self.assertIn(f"{detector}-{run}-1999,", output)

    def test_failed_csv_write_preserves_previous_output(self):
        class BrokenWriter:
            def to_csv(self, handle, **kwargs):
                handle.write("partial csv")
                raise OSError("simulated disk write failure")

        save = runpy.run_path(str(SCRIPT))["save_filtered_metadata"]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "filtered.csv"
            output.write_text("previous verified metadata\n")
            with self.assertRaisesRegex(OSError, "simulated disk write failure"):
                save(BrokenWriter(), output)
            self.assertEqual(output.read_text(), "previous verified metadata\n")
            self.assertEqual(list(Path(directory).iterdir()), [output])
