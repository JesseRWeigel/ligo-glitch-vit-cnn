"""Input validation for paired model comparisons, without data or training."""

import unittest

import numpy as np

from src.evaluation.paired_bootstrap import (
    paired_bootstrap_metric,
    paired_bootstrap_rare_f1,
)


class PairedBootstrapContractTests(unittest.TestCase):
    def test_generic_rejects_extra_predictions_before_scoring(self):
        with self.assertRaisesRegex(ValueError, "same number"):
            paired_bootstrap_metric(
                [0, 1], [0, 1, 0], [0, 1],
                lambda *_: self.fail("metric was called before validation"),
                n_resamples=2,
            )

    def test_rare_rejects_multi_output_labels(self):
        with self.assertRaisesRegex(ValueError, "one-dimensional"):
            paired_bootstrap_rare_f1(
                np.array([[0], [1]]), np.array([0, 1]), np.array([0, 1]), [1]
            )

    def test_both_reject_empty_inputs(self):
        for fn, args in (
            (paired_bootstrap_metric, (lambda *_: 0,)),
            (paired_bootstrap_rare_f1, ([1],)),
        ):
            with self.subTest(fn=fn.__name__), self.assertRaisesRegex(ValueError, "non-empty"):
                fn([], [], [], *args)

    def test_list_inputs_preserve_identical_model_pairing(self):
        result = paired_bootstrap_metric(
            [0, 1, 1], [0, 0, 1], [0, 0, 1],
            lambda y, p: float(np.mean(y == p)), n_resamples=25,
        )
        self.assertEqual(result["point_estimate_difference"], 0)
        self.assertEqual((result["ci_lower"], result["ci_upper"]), (0, 0))
