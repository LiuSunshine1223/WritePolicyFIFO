"""Generate symbolic key-value retrieval minibatches for Experiment D."""

import torch

from .config import (
    BATCH_SIZE,
    SEQ_LEN,
    SEP_TOKEN,
    EXPD_FACT_TOKEN,
    EXPD_QUERY_TOKEN,
    EXPD_IS_TOKEN,
    EXPD_ASK_TOKEN,
    EXPD_KEY_START,
    EXPD_KEY_END,
    EXPD_VALUE_START,
    EXPD_VALUE_END,
    DISTRACTOR_FACT_LEN_EXPD,
)

def _randint_inclusive(low, high, shape, generator=None):
    return torch.randint(
        low=low,
        high=high + 1,
        size=shape,
        generator=generator,
        dtype=torch.long,
    )


def _make_value_tokens(batch_size, value_len, generator=None):
    return _randint_inclusive(
        EXPD_VALUE_START,
        EXPD_VALUE_END,
        (batch_size, value_len),
        generator=generator,
    )


def _sample_target_keys(batch_size, generator=None):
    return _randint_inclusive(
        EXPD_KEY_START,
        EXPD_KEY_END,
        (batch_size,),
        generator=generator,
    )


def _sample_distractor_keys(target_keys, generator=None):
    """Sample one distractor key per row, excluding its target key."""
    num_keys = EXPD_KEY_END - EXPD_KEY_START + 1
    if num_keys <= 1:
        raise ValueError("Experiment D requires at least two key tokens.")

    batch_size = target_keys.size(0)

    raw = torch.randint(
        low=0,
        high=num_keys - 1,
        size=(batch_size,),
        generator=generator,
        dtype=torch.long,
    )

    target_idx = target_keys - EXPD_KEY_START

    # Shift indices at or above the target so the target is never sampled.
    raw = raw + (raw >= target_idx).long()
    keys = EXPD_KEY_START + raw

    return keys


def _make_source_fact(keys, generator=None):
    """Return ``[FACT][KEY][IS][values...]`` records of length ``SEQ_LEN``."""
    batch_size = keys.size(0)
    value_len = SEQ_LEN - 3

    fact_col = torch.full((batch_size, 1), EXPD_FACT_TOKEN, dtype=torch.long)
    key_col = keys.view(-1, 1)
    is_col = torch.full((batch_size, 1), EXPD_IS_TOKEN, dtype=torch.long)
    values = _make_value_tokens(batch_size, value_len, generator=generator)

    source = torch.cat(
        [fact_col, key_col, is_col, values],
        dim=1,
    )

    return source


def _make_query(keys):
    """Return length-``SEQ_LEN`` queries containing the key but no value."""
    batch_size = keys.size(0)
    rest_len = SEQ_LEN - 2

    query_col = torch.full((batch_size, 1), EXPD_QUERY_TOKEN, dtype=torch.long)
    key_col = keys.view(-1, 1)
    rest = torch.full((batch_size, rest_len), EXPD_ASK_TOKEN, dtype=torch.long)

    query = torch.cat(
        [query_col, key_col, rest],
        dim=1,
    )

    return query


def _make_one_distractor_fact(target_keys, fact_len, generator=None):
    """Return ``[FACT][non-target key][IS][values...]`` records."""
    batch_size = target_keys.size(0)

    if fact_len < 3:
        raise ValueError("DISTRACTOR_FACT_LEN_EXPD must be at least 3.")

    keys = _sample_distractor_keys(target_keys, generator=generator)

    fact_col = torch.full((batch_size, 1), EXPD_FACT_TOKEN, dtype=torch.long)
    key_col = keys.view(-1, 1)
    is_col = torch.full((batch_size, 1), EXPD_IS_TOKEN, dtype=torch.long)
    values = _make_value_tokens(batch_size, fact_len - 3, generator=generator)

    fact = torch.cat(
        [fact_col, key_col, is_col, values],
        dim=1,
    )

    return fact


def _make_noise_segment(batch_size, delay_len, target_keys, generator=None):
    """Concatenate distractor records and crop to exactly ``delay_len``."""
    if delay_len <= 0:
        return torch.empty((batch_size, 0), dtype=torch.long)

    fact_len = DISTRACTOR_FACT_LEN_EXPD
    num_facts = (delay_len + fact_len - 1) // fact_len

    parts = []
    for _ in range(num_facts):
        fact = _make_one_distractor_fact(
            target_keys=target_keys,
            fact_len=fact_len,
            generator=generator,
        )
        parts.append(fact)

    noise = torch.cat(parts, dim=1)
    noise = noise[:, :delay_len]

    return noise


def generate_batch_expD(delay_len, data_gen=None):
    """Return a symbolic KV batch with a full-record retrieval target.

    Inputs have shape ``(B, 2 * SEQ_LEN + 1 + delay_len)`` and layout
    ``[source][SEP][distractor records][query]``. Targets have shape
    ``(B, SEQ_LEN)`` and equal a clone of the complete source record.
    """
    batch_size = BATCH_SIZE

    target_keys = _sample_target_keys(batch_size, generator=data_gen)

    source = _make_source_fact(target_keys, generator=data_gen)
    sep = torch.full((batch_size, 1), SEP_TOKEN, dtype=torch.long)
    noise = _make_noise_segment(
        batch_size=batch_size,
        delay_len=delay_len,
        target_keys=target_keys,
        generator=data_gen,
    )
    query = _make_query(target_keys)

    inp = torch.cat([source, sep, noise, query], dim=1)

    # Retrieve the complete key-value record rather than only its value.
    tgt = source.clone()

    return inp, tgt


# Backward-compatible batch-generator name.
generate_batch = generate_batch_expD
