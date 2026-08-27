"""Shared reproducibility, metric, attention-mask, and CSV helpers."""

import numpy as np
import random
import torch
import csv

from collections.abc import Mapping
from pathlib import Path

def set_seed(seed):
    """Seed Python, NumPy, PyTorch CPU, and all available CUDA devices."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def summarize_loss(loss_history, tail):
    """Summarize the best loss and the trailing ``min(tail, n)`` values."""
    tail = min(tail, len(loss_history))
    tail_losses = loss_history[-tail:]
    return {
        "loss_best": float(np.min(loss_history)),
        "loss_mean_tail": float(np.mean(tail_losses)),
        "loss_std_tail": float(np.std(tail_losses)),
    }

def summarize_acc(acc_history, tail):
    """Summarize best, final, and trailing accuracy values."""
    tail = min(tail, len(acc_history))
    tail_acc = acc_history[-tail:]
    return {
        "acc_best": float(np.max(acc_history)),
        "acc_last": float(acc_history[-1]),
        "acc_mean_tail": float(np.mean(tail_acc)),
        "acc_std_tail": float(np.std(tail_acc)),
    }

def window_attention_mask(seq_len, window_size):
    """Return a causal local-attention mask shaped ``(seq_len, seq_len)``.

    Position ``i`` may attend to itself and at most the preceding
    ``window_size - 1`` positions.
    """
    mask = torch.full((seq_len, seq_len), float("-inf"))

    for i in range(seq_len):
        left = max(0, i - window_size + 1)
        right = i + 1
        mask[i, left : right] = 0.0
    return mask

def force_no_source_mask(seq_len, window_size, source_len):
    """Block query rows from attending directly to source columns.

    The first ``source_len`` positions are the source and the final
    ``source_len`` positions are the query. Other locally visible positions
    retain the causal window defined by :func:`window_attention_mask`.
    """
    mask = window_attention_mask(seq_len, window_size)

    query_start = seq_len - source_len
    source_end = source_len
    mask[query_start:, :source_end] = float("-inf")

    return mask

def save_results_csv(results, output_path, overwrite=False):
    """Save result rows to one CSV file without changing their values.

    The field order is the sorted union of keys across all rows, matching the
    previous experiment CSV convention.  Existing files are protected unless
    ``overwrite=True`` is supplied explicitly.
    """
    rows = list(results)
    if not rows:
        raise ValueError("results must contain at least one row")

    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise TypeError(
                f"results row {index} must be a mapping, "
                f"got {type(row).__name__}"
            )

    fields = {key for row in rows for key in row.keys()}
    if not fields:
        raise ValueError("result rows must contain at least one field")
    if any(not isinstance(key, str) or not key for key in fields):
        raise TypeError("CSV field names must be non-empty strings")
    fieldnames = sorted(fields)

    output_path = Path(output_path)
    if output_path.suffix.lower() != ".csv":
        raise ValueError(f"output_path must end in .csv: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite else "x"

    try:
        with output_path.open(
            mode,
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames,
                extrasaction="ignore",
            )
            writer.writeheader()
            writer.writerows(rows)
    except FileExistsError as exc:
        raise FileExistsError(
            f"Refusing to overwrite existing results file: {output_path}. "
            "Pass overwrite=True only when replacement is intentional."
        ) from exc

    return output_path
