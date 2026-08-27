"""Regression tests for training-tail accuracy and capacity boundaries."""

import os
import unittest

import numpy as np
import pandas as pd

from DelayedCopyTask.utils import summarize_acc


# Importing plot_expA.py is needed for the current boundary implementation. Force a
# non-interactive backend so the tests remain safe in WSL and headless runners.
os.environ.setdefault("MPLBACKEND", "Agg")

from plot_expA import (  # noqa: E402
    compute_mstar_forced,
    compute_mstar_per_seed,
    summarize_mstar_per_seed,
)


def _boundary_frame(rows):
    """Build the minimal ExpA-style table required by boundary functions."""

    records = []
    for seed, max_mem, accuracy in rows:
        records.append(
            {
                "model": "naive",
                "write_mode": "source-only",
                "delay_len": 10,
                "seed": seed,
                "max_mem": max_mem,
                "acc_mean_tail": accuracy,
                "mask_mode": "forced",
                "window_size": 8,
            }
        )
    return pd.DataFrame.from_records(records)


class TrainingTailMetricTests(unittest.TestCase):
    def test_acc_mean_tail_uses_only_requested_suffix(self):
        summary = summarize_acc([0.0, 0.25, 0.5, 0.75, 1.0], tail=2)

        self.assertAlmostEqual(summary["acc_mean_tail"], 0.875)
        self.assertAlmostEqual(summary["acc_std_tail"], 0.125)
        self.assertAlmostEqual(summary["acc_last"], 1.0)

    def test_tail_longer_than_history_uses_complete_history(self):
        summary = summarize_acc([0.25, 0.5, 0.75], tail=500)

        self.assertAlmostEqual(summary["acc_mean_tail"], 0.5)
        self.assertAlmostEqual(summary["acc_std_tail"], np.std([0.25, 0.5, 0.75]))


class CapacityBoundaryTests(unittest.TestCase):
    def test_aggregate_boundary_thresholds_seed_mean_accuracy(self):
        # At m=10, seed 0 passes and seed 1 does not: their mean is 0.945 and
        # remains below tau=0.95.  Both pass at m=20, so aggregate m* is 20.
        df = _boundary_frame(
            [
                (0, 10, 0.99),
                (0, 20, 0.99),
                (1, 10, 0.90),
                (1, 20, 0.99),
            ]
        )

        aggregate = compute_mstar_forced(df, window=8, tau=0.95, save_csv=False)
        per_seed = compute_mstar_per_seed(df, window=8, tau=0.95)

        self.assertEqual(len(aggregate), 1)
        self.assertEqual(int(aggregate.iloc[0]["m_star"]), 20)
        self.assertFalse(bool(aggregate.iloc[0]["m_star_censored"]))
        self.assertEqual(per_seed["m_star"].tolist(), [10, 20])

    def test_censored_seed_is_nan_and_excluded_from_finite_summary(self):
        df = _boundary_frame(
            [
                (0, 10, 0.99),
                (0, 20, 0.99),
                (1, 10, 0.10),
                (1, 20, 0.20),
            ]
        )

        per_seed = compute_mstar_per_seed(df, window=8, tau=0.95)
        seed_one = per_seed.loc[per_seed["seed"] == 1].iloc[0]
        summary = summarize_mstar_per_seed(per_seed).iloc[0]

        self.assertTrue(bool(seed_one["m_star_censored"]))
        self.assertTrue(np.isnan(seed_one["m_star"]))
        self.assertEqual(int(seed_one["max_mem_scanned"]), 20)

        self.assertEqual(int(summary["n_seeds"]), 2)
        self.assertEqual(int(summary["uncensored_n"]), 1)
        self.assertEqual(int(summary["censored_n"]), 1)
        self.assertEqual(float(summary["m_star_mean"]), 10.0)
        self.assertTrue(np.isnan(summary["m_star_std"]))
        self.assertEqual(summary["stats_scope"], "uncensored_seeds_only")

    def test_aggregate_censoring_does_not_impute_scan_limit_as_boundary(self):
        df = _boundary_frame(
            [
                (0, 10, 0.50),
                (0, 20, 0.60),
                (1, 10, 0.40),
                (1, 20, 0.70),
            ]
        )

        aggregate = compute_mstar_forced(df, window=8, tau=0.95, save_csv=False)
        row = aggregate.iloc[0]

        self.assertTrue(bool(row["m_star_censored"]))
        self.assertTrue(np.isnan(row["m_star"]))
        self.assertEqual(int(row["max_mem_scanned"]), 20)


if __name__ == "__main__":
    unittest.main()
