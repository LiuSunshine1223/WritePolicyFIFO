import unittest

import torch

from DelayedCopyTask.config import (
    BATCH_SIZE,
    DISTRACTOR_FACT_LEN_EXPD,
    EXPD_ASK_TOKEN,
    EXPD_FACT_TOKEN,
    EXPD_IS_TOKEN,
    EXPD_KEY_END,
    EXPD_KEY_START,
    EXPD_QUERY_TOKEN,
    EXPD_VALUE_END,
    EXPD_VALUE_START,
    SEP_TOKEN,
    SEQ_LEN,
)
from DelayedCopyTask.dataset import generate_batch as generate_delayed_copy_batch
from DelayedCopyTask.dataset_expD import generate_batch_expD


class DelayedCopyProtocolTests(unittest.TestCase):
    def test_input_and_target_shapes_match_delayed_copy_protocol(self):
        delay_len = 7
        data_gen = torch.Generator().manual_seed(101)

        inputs, targets = generate_delayed_copy_batch(delay_len, data_gen)

        self.assertEqual(
            tuple(inputs.shape),
            (BATCH_SIZE, 2 * SEQ_LEN + 1 + delay_len),
        )
        self.assertEqual(tuple(targets.shape), (BATCH_SIZE, SEQ_LEN))
        self.assertTrue(torch.equal(inputs[:, :SEQ_LEN], targets))
        self.assertTrue(torch.all(inputs[:, SEQ_LEN] == SEP_TOKEN).item())
        self.assertTrue(torch.all(inputs[:, -SEQ_LEN:] == 0).item())


class SymbolicKeyValueProtocolTests(unittest.TestCase):
    def setUp(self):
        self.delay_len = 2 * DISTRACTOR_FACT_LEN_EXPD
        data_gen = torch.Generator().manual_seed(202)
        self.inputs, self.targets = generate_batch_expD(self.delay_len, data_gen)

        noise_start = SEQ_LEN + 1
        query_start = noise_start + self.delay_len
        self.source = self.inputs[:, :SEQ_LEN]
        self.noise = self.inputs[:, noise_start:query_start]
        self.query = self.inputs[:, query_start:]
        self.target_keys = self.source[:, 1]

    def test_source_has_symbolic_key_value_record_format(self):
        self.assertEqual(tuple(self.source.shape), (BATCH_SIZE, SEQ_LEN))
        self.assertTrue(torch.all(self.source[:, 0] == EXPD_FACT_TOKEN).item())
        self.assertTrue(torch.all(self.source[:, 2] == EXPD_IS_TOKEN).item())
        self.assertTrue(
            torch.all(
                (self.target_keys >= EXPD_KEY_START)
                & (self.target_keys <= EXPD_KEY_END)
            ).item()
        )

        values = self.source[:, 3:]
        self.assertTrue(
            torch.all(
                (values >= EXPD_VALUE_START) & (values <= EXPD_VALUE_END)
            ).item()
        )
        self.assertTrue(torch.all(self.inputs[:, SEQ_LEN] == SEP_TOKEN).item())

    def test_query_contains_target_key_but_no_value_tokens(self):
        self.assertEqual(tuple(self.query.shape), (BATCH_SIZE, SEQ_LEN))
        self.assertTrue(torch.all(self.query[:, 0] == EXPD_QUERY_TOKEN).item())
        self.assertTrue(torch.equal(self.query[:, 1], self.target_keys))
        self.assertTrue(torch.all(self.query[:, 2:] == EXPD_ASK_TOKEN).item())

        query_has_value_token = (
            (self.query >= EXPD_VALUE_START) & (self.query <= EXPD_VALUE_END)
        ).any()
        self.assertFalse(query_has_value_token.item())

    def test_distractor_keys_never_match_target_key(self):
        for fact_start in range(0, self.delay_len, DISTRACTOR_FACT_LEN_EXPD):
            distractor_fact = self.noise[
                :, fact_start : fact_start + DISTRACTOR_FACT_LEN_EXPD
            ]
            self.assertEqual(
                tuple(distractor_fact.shape),
                (BATCH_SIZE, DISTRACTOR_FACT_LEN_EXPD),
            )
            self.assertTrue(
                torch.all(distractor_fact[:, 0] == EXPD_FACT_TOKEN).item()
            )
            self.assertTrue(torch.all(distractor_fact[:, 2] == EXPD_IS_TOKEN).item())
            self.assertTrue(
                torch.all(distractor_fact[:, 1] != self.target_keys).item()
            )

    def test_target_is_the_complete_source_record(self):
        self.assertEqual(tuple(self.targets.shape), (BATCH_SIZE, SEQ_LEN))
        self.assertTrue(torch.equal(self.targets, self.source))


if __name__ == "__main__":
    unittest.main()
