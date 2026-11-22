"""Exercise the portable O4 prediction archive contract without model inference."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from src.evaluation.prediction_export import save_o4_predictions


class PairedPredictionExportTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "o4_predictions.npz"
        self.arrays = dict(
            gravityspy_ids=["0002", "3", "evt-A"],
            expected_labels=[1, 0, 1],
            cnn_labels=[1, 0, 1],
            vit_labels=[1, 0, 1],
            cnn_preds=[0, 0, 1],
            vit_preds=[1, 0, 0],
            class_names=["A", "B"],
        )

    def test_ordered_ids_and_existing_fields_are_pickle_free(self):
        save_o4_predictions(self.path, **self.arrays)
        with np.load(self.path, allow_pickle=False) as archive:
            self.assertEqual(archive["gravityspy_ids"].dtype.kind, "U")
            self.assertEqual(archive["gravityspy_ids"].tolist(), ["0002", "3", "evt-A"])
            self.assertEqual(archive["y_true"].tolist(), [1, 0, 1])
            for key in ("cnn_preds", "vit_preds", "class_names"):
                self.assertEqual(archive[key].tolist(), self.arrays[key])

    def test_bad_rows_are_rejected_before_overwriting_the_output(self):
        self.path.write_bytes(b"previous artifact")
        for field, value in (
            ("cnn_labels", [0, 1, 1]),
            ("vit_labels", [1, 1, 0]),
            ("cnn_preds", [0, 1]),
            ("vit_preds", [[1], [0], [0]]),
            ("vit_preds", [1, 2, 0]),
            ("gravityspy_ids", ["0002", "3", "3"]),
            ("gravityspy_ids", ["0002", "3", None]),
            ("gravityspy_ids", ["0002", "3", " "]),
            ("gravityspy_ids", ["0002", "3"]),
        ):
            with self.subTest(field=field, value=value):
                with self.assertRaises(ValueError):
                    save_o4_predictions(self.path, **{**self.arrays, field: value})
                self.assertEqual(self.path.read_bytes(), b"previous artifact")
