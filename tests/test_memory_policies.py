import unittest

import torch

from DelayedCopyTask.config import (
    SEQ_LEN,
    WRITE_INTERMEDIATE,
    WRITE_PREFIX_ALL,
    WRITE_SOURCE_ONLY,
    WRITE_SOURCE_PINNED,
)
from DelayedCopyTask.memory import WindowKVBuffer
from DelayedCopyTask.memory_ops import write_to_memory


class MemoryWritePolicyTests(unittest.TestCase):
    @staticmethod
    def _hidden_states(noise_len):
        """Create position-coded states for [source][SEP][noise][query]."""
        total_length = 2 * SEQ_LEN + 1 + noise_len
        return torch.arange(total_length, dtype=torch.float32).view(1, total_length, 1)

    def assert_memory_equals(self, memory, expected, expected_labels):
        self.assertTrue(torch.equal(memory.keys, expected))
        self.assertTrue(torch.equal(memory.vals, expected))
        self.assertEqual(memory.labels, expected_labels)
        self.assertEqual(len(memory), expected.size(1))

    def test_source_only_writes_only_source_entries(self):
        hidden = self._hidden_states(noise_len=6)
        memory = WindowKVBuffer(max_mem=SEQ_LEN + 5)

        write_to_memory(hidden, memory, WRITE_SOURCE_ONLY)

        expected = hidden[:, :SEQ_LEN, :]
        self.assert_memory_equals(memory, expected, ["source"] * SEQ_LEN)

    def test_prefix_all_writes_the_complete_pre_query_prefix(self):
        noise_len = 6
        hidden = self._hidden_states(noise_len=noise_len)
        prefix_length = SEQ_LEN + 1 + noise_len
        memory = WindowKVBuffer(max_mem=prefix_length)

        write_to_memory(hidden, memory, WRITE_PREFIX_ALL)

        expected = hidden[:, :prefix_length, :]
        expected_labels = ["source"] * SEQ_LEN + ["sep"] + ["noise"] * noise_len
        self.assert_memory_equals(memory, expected, expected_labels)

    def test_prefix_all_capacity_limit_keeps_latest_prefix_entries(self):
        noise_len = 6
        hidden = self._hidden_states(noise_len=noise_len)
        prefix_length = SEQ_LEN + 1 + noise_len
        memory = WindowKVBuffer(max_mem=4)

        write_to_memory(hidden, memory, WRITE_PREFIX_ALL)

        expected = hidden[:, prefix_length - 4:prefix_length, :]
        self.assert_memory_equals(memory, expected, ["noise"] * 4)

    def test_intermediate_writes_expected_earliest_noise_entries(self):
        noise_len = 8
        noise_write_ratio = 0.5
        noise_count = 4
        hidden = self._hidden_states(noise_len=noise_len)
        memory = WindowKVBuffer(max_mem=SEQ_LEN + 1 + noise_count)

        write_to_memory(
            hidden,
            memory,
            WRITE_INTERMEDIATE,
            noise_write_ratio=noise_write_ratio,
        )

        source = hidden[:, :SEQ_LEN, :]
        separator = hidden[:, SEQ_LEN:SEQ_LEN + 1, :]
        earliest_noise = hidden[:, SEQ_LEN + 1:SEQ_LEN + 1 + noise_count, :]
        expected = torch.cat([source, separator, earliest_noise], dim=1)
        expected_labels = ["source"] * SEQ_LEN + ["sep"] + ["noise"] * noise_count
        self.assert_memory_equals(memory, expected, expected_labels)

    def test_source_pinned_preserves_source_and_keeps_recent_noise(self):
        noise_len = 6
        noise_budget = 3
        hidden = self._hidden_states(noise_len=noise_len)
        memory = WindowKVBuffer(max_mem=SEQ_LEN + noise_budget)

        write_to_memory(hidden, memory, WRITE_SOURCE_PINNED)

        source = hidden[:, :SEQ_LEN, :]
        noise = hidden[:, SEQ_LEN + 1:SEQ_LEN + 1 + noise_len, :]
        expected = torch.cat([source, noise[:, -noise_budget:, :]], dim=1)
        expected_labels = ["source"] * SEQ_LEN + ["noise"] * noise_budget
        self.assert_memory_equals(memory, expected, expected_labels)

    def test_fifo_append_discards_oldest_entries_over_capacity(self):
        memory = WindowKVBuffer(max_mem=3)
        first_keys = torch.tensor([[[0.0], [1.0]]])
        first_vals = torch.tensor([[[10.0], [11.0]]])
        second_keys = torch.tensor([[[2.0], [3.0], [4.0]]])
        second_vals = torch.tensor([[[12.0], [13.0], [14.0]]])

        memory.append(first_keys, first_vals, labels=["source", "sep"])
        memory.append(second_keys, second_vals, labels=["noise"] * 3)

        self.assertTrue(torch.equal(memory.keys, second_keys))
        self.assertTrue(torch.equal(memory.vals, second_vals))
        self.assertEqual(memory.labels, ["noise"] * 3)
        self.assertEqual(len(memory), 3)


if __name__ == "__main__":
    unittest.main()
