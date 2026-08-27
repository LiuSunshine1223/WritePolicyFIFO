"""Generate fresh delayed-copy minibatches on CPU."""

import torch

from .config import BATCH_SIZE, SEP_TOKEN, SEQ_LEN


def generate_batch(delay_len, data_gen):
    """Return inputs shaped ``(B, 2L + 1 + d)`` and targets ``(B, L)``.

    Inputs follow ``[source][SEP][noise][zero query]`` and targets are the
    source sequence. Device placement is left to the training loop.
    """
    source = torch.randint(0, SEP_TOKEN, (BATCH_SIZE, SEQ_LEN), generator=data_gen)
    sep = torch.full((BATCH_SIZE, 1), SEP_TOKEN)
    noise = torch.randint(0, SEP_TOKEN, (BATCH_SIZE, delay_len), generator=data_gen)
    query = torch.zeros((BATCH_SIZE, SEQ_LEN), dtype=torch.long)

    inp = torch.cat([source, sep, noise, query], dim=1)
    tgt = source

    return inp, tgt
