"""Regression tests for local-causal and forced-memory attention masks."""

import unittest

import torch

from DelayedCopyTask.utils import force_no_source_mask, window_attention_mask


class WindowAttentionMaskTests(unittest.TestCase):
    def test_each_row_allows_exactly_its_local_window(self):
        seq_len = 7
        window_size = 3

        mask = window_attention_mask(seq_len, window_size)

        self.assertEqual(tuple(mask.shape), (seq_len, seq_len))
        for row in range(seq_len):
            left = max(0, row - window_size + 1)
            for column in range(seq_len):
                if left <= column <= row:
                    self.assertEqual(mask[row, column].item(), 0.0)
                else:
                    self.assertTrue(torch.isneginf(mask[row, column]).item())

    def test_future_positions_are_always_masked(self):
        seq_len = 6
        mask = window_attention_mask(seq_len, window_size=seq_len)

        for row in range(seq_len):
            self.assertEqual(mask[row, row].item(), 0.0)
            if row + 1 < seq_len:
                self.assertTrue(torch.isneginf(mask[row, row + 1 :]).all().item())


class ForcedNoSourceMaskTests(unittest.TestCase):
    def test_query_rows_cannot_attend_to_source_positions(self):
        seq_len = 10
        source_len = 4
        query_start = seq_len - source_len

        mask = force_no_source_mask(
            seq_len=seq_len,
            window_size=seq_len,
            source_len=source_len,
        )

        blocked_query_to_source = mask[query_start:, :source_len]
        self.assertTrue(torch.isneginf(blocked_query_to_source).all().item())

    def test_only_query_to_source_region_is_added_to_local_mask(self):
        seq_len = 10
        source_len = 4
        window_size = 8
        query_start = seq_len - source_len

        local_mask = window_attention_mask(seq_len, window_size)
        expected = local_mask.clone()
        expected[query_start:, :source_len] = float("-inf")

        forced_mask = force_no_source_mask(seq_len, window_size, source_len)

        self.assertTrue(torch.equal(forced_mask, expected))
        self.assertTrue(torch.equal(forced_mask[:query_start], local_mask[:query_start]))

        # Non-source positions inside the first query token's local window remain
        # available, including the separator/delay suffix and the token itself.
        first_query = query_start
        allowed_start = max(source_len, first_query - window_size + 1)
        self.assertTrue(
            torch.equal(
                forced_mask[first_query, allowed_start : first_query + 1],
                torch.zeros(first_query - allowed_start + 1),
            )
        )


if __name__ == "__main__":
    unittest.main()
