import torch
from collections import Counter

from .config import (
    SEQ_LEN,
    WRITE_SOURCE_ONLY,
    WRITE_PREFIX_ALL,
    WRITE_SOURCE_PINNED,
    WRITE_INTERMEDIATE,
)
from .memory import WindowKVBuffer


# Intermediate policy helper: admit a contiguous prefix of noise states.
def _select_prefix_noise(noise_h, k: int):
    """Return the first ``k`` states from ``noise_h`` of shape ``(B, N, D)``."""
    noise_len = noise_h.size(1)

    if k <= 0:
        return noise_h[:, :0, :]

    if k >= noise_len:
        return noise_h

    # Prefix selection interpolates from source+SEP to prefix-all.
    return noise_h[:, :k, :]

def build_memory_entries(h, memory: WindowKVBuffer, write_mode: str, noise_write_ratio=None):
    """Construct the capacity-limited pre-query memory state.

    ``h`` has shape ``(B, T, D)`` with layout
    ``[source][SEP][noise][query]``. The result is ``mem`` with shape
    ``(B, M, D)`` plus ``M`` semantic labels. Source-only and prefix-all use
    FIFO tail retention; source-pinned keeps source states outside the noise
    FIFO; intermediate writes source, SEP, and a prefix of noise states.
    """
    if memory.max_mem <= 0:
        return None, None

    T = h.size(1)
    noise_len = T - 2 * SEQ_LEN - 1

    if noise_len < 0:
        raise ValueError(
            f"Invalid input length T={T}. Expected at least 2*SEQ_LEN+1."
        )

    source_h = h[:, :SEQ_LEN, :]
    sep_h = h[:, SEQ_LEN:SEQ_LEN + 1, :]
    noise_h = h[:, SEQ_LEN + 1:-SEQ_LEN, :]       # Noise only; SEP is excluded.
    prefix_h = h[:, :-SEQ_LEN, :]                 # Source, SEP, and noise.

    # FIFO policies retain the newest ``max_mem`` candidate entries.
    if write_mode == WRITE_SOURCE_ONLY:
        # Insufficient capacity retains only the most recent source states.
        keep = min(memory.max_mem, source_h.size(1))
        mem = source_h[:, -keep:, :]
        labels = ["source"] * keep
        return mem, labels

    if write_mode == WRITE_PREFIX_ALL:
        keep = min(memory.max_mem, prefix_h.size(1))
        mem = prefix_h[:, -keep:, :]

        prefix_labels = (
            ["source"] * SEQ_LEN
            + ["sep"]
            + ["noise"] * noise_len
        )
        labels = prefix_labels[-keep:]
        return mem, labels

    if write_mode == WRITE_SOURCE_PINNED:
        if memory.max_mem < SEQ_LEN:
            raise ValueError(
                f"{WRITE_SOURCE_PINNED} requires max_mem >= SEQ_LEN. "
                f"Got max_mem={memory.max_mem}, SEQ_LEN={SEQ_LEN}."
            )

        noise_budget = memory.max_mem - SEQ_LEN
        actual_noise = min(noise_budget, noise_h.size(1))

        if actual_noise > 0:
            # Keep the newest noise entries without evicting pinned source entries.
            noise_mem = noise_h[:, -actual_noise:, :]
            mem = torch.cat([source_h, noise_mem], dim=1)
            labels = ["source"] * SEQ_LEN + ["noise"] * actual_noise
        else:
            mem = source_h
            labels = ["source"] * SEQ_LEN

        return mem, labels

    if write_mode == WRITE_INTERMEDIATE:
        if noise_write_ratio is None:
            raise ValueError(
                f"{WRITE_INTERMEDIATE} requires noise_write_ratio, "
                f"but got None."
            )

        p = float(noise_write_ratio)
        if p < 0.0 or p > 1.0:
            raise ValueError(
                f"noise_write_ratio must be in [0, 1]. Got {noise_write_ratio}."
            )

        noise_count = int(round(p * noise_len))
        noise_count = max(0, min(noise_count, noise_len))  # Keep the count within the noise span.

        selected_noise_h = _select_prefix_noise(noise_h, noise_count)

        # Source and SEP are fixed; only the requested noise prefix is admitted.
        candidate_h = torch.cat([source_h, sep_h, selected_noise_h], dim=1)

        candidate_labels = (
                ["source"] * SEQ_LEN
                + ["sep"]
                + ["noise"] * noise_count
        )

        keep = min(memory.max_mem, candidate_h.size(1))
        mem = candidate_h[:, -keep:, :]
        labels = candidate_labels[-keep:]
        return mem, labels

    raise ValueError(f"Invalid write_mode: {write_mode}")


def write_to_memory(h, memory: WindowKVBuffer, write_mode: str, noise_write_ratio=None):
    """Apply the write policy, using selected hidden states as both K and V."""
    mem, labels = build_memory_entries(h, memory, write_mode, noise_write_ratio=noise_write_ratio)

    if mem is None:
        memory.reset()
        return

    memory.set_entries(mem, mem, labels=labels)

def summarize_memory_state(memory: WindowKVBuffer):
    """Summarize entry counts and source survival in the current memory."""
    if memory.keys is None or memory.keys.size(1) == 0:
        return {
            "mem_size": 0,
            "source_count": 0,
            "sep_count": 0,
            "noise_count": 0,
            "source_survival_rate": 0.0,
            "labels": [],
        }

    labels = memory.labels if memory.labels is not None else []
    counts = Counter(labels)

    source_count = counts.get("source", 0)
    sep_count = counts.get("sep", 0)
    noise_count = counts.get("noise", 0)

    return {
        "mem_size": memory.keys.size(1),
        "source_count": source_count,
        "sep_count": sep_count,
        "noise_count": noise_count,
        "source_survival_rate": source_count / float(SEQ_LEN),
        "labels": list(labels),
    }


def summarize_attention_by_label(attn, labels):
    """Summarize ``(B, T, M)`` attention mass by the ``M`` memory labels."""
    if attn is None or labels is None or len(labels) == 0:
        return {
            "attn_mass_source": 0.0,
            "attn_mass_sep": 0.0,
            "attn_mass_noise": 0.0,
            "attn_entropy": 0.0,
        }

    if attn.size(-1) != len(labels):
        raise ValueError(
            f"Attention memory dimension and labels length mismatch: "
            f"attn.size(-1)={attn.size(-1)}, len(labels)={len(labels)}"
        )

    device = attn.device
    dtype = attn.dtype

    def label_mask(label_name):
        return torch.tensor(
            [lab == label_name for lab in labels],
            device=device,
            dtype=dtype,
        ).view(1, 1, -1)

    with torch.no_grad():
        source_mask = label_mask("source")
        sep_mask = label_mask("sep")
        noise_mask = label_mask("noise")

        source_mass = (attn * source_mask).sum(dim=-1).mean()
        sep_mass = (attn * sep_mask).sum(dim=-1).mean()
        noise_mass = (attn * noise_mask).sum(dim=-1).mean()

        eps = 1e-12
        entropy = -(attn.clamp_min(eps) * attn.clamp_min(eps).log()).sum(dim=-1).mean()

    return {
        "attn_mass_source": float(source_mass.detach().cpu().item()),
        "attn_mass_sep": float(sep_mass.detach().cpu().item()),
        "attn_mass_noise": float(noise_mass.detach().cpu().item()),
        "attn_entropy": float(entropy.detach().cpu().item()),
    }

def build_memory_diagnostics(memory: WindowKVBuffer, attn=None):
    """Combine memory-state and attention diagnostics."""
    diag = summarize_memory_state(memory)

    if attn is not None:
        attn_diag = summarize_attention_by_label(attn, memory.labels)
        diag.update(attn_diag)
    else:
        diag.update({
            "attn_mass_source": None,
            "attn_mass_sep": None,
            "attn_mass_noise": None,
            "attn_entropy": None,
        })

    return diag

def read_from_memory_naive(query, memory: WindowKVBuffer, return_diag: bool = False):
    """Read K/V memory for ``query`` shaped ``(B, T, D)`` via residual attention."""
    if memory.keys is None or memory.keys.size(1) == 0:
        if return_diag:
            return query, build_memory_diagnostics(memory, attn=None)
        return query

    mem_keys = memory.keys
    mem_vals = memory.vals

    attn = torch.softmax(
        torch.matmul(query, mem_keys.transpose(-1, -2)) / (query.size(-1) ** 0.5),
        dim=-1,
    )
    retrieved = torch.matmul(attn, mem_vals)
    h_back = query + retrieved

    if return_diag:
        diag = build_memory_diagnostics(memory, attn=attn)
        return h_back, diag

    return h_back


def read_from_memory_gated(query, memory: WindowKVBuffer, gate_layer, return_diag: bool = False):
    """Blend ``(B, T, D)`` queries with retrieved values using a learned gate."""
    if memory.keys is None or memory.keys.size(1) == 0:
        if return_diag:
            return query, build_memory_diagnostics(memory, attn=None)
        return query

    mem_keys = memory.keys
    mem_vals = memory.vals

    attn = torch.softmax(
        torch.matmul(query, mem_keys.transpose(-1, -2)) / (query.size(-1) ** 0.5),
        dim=-1,
    )

    retrieved = torch.matmul(attn, mem_vals)

    gate = torch.sigmoid(gate_layer(query))
    h_back = (1 - gate) * query + gate * retrieved

    if return_diag:
        diag = build_memory_diagnostics(memory, attn=attn)

        with torch.no_grad():
            diag["gate_mean"] = float(gate.mean().detach().cpu().item())
            diag["gate_min"] = float(gate.min().detach().cpu().item())
            diag["gate_max"] = float(gate.max().detach().cpu().item())

        return h_back, diag

    return h_back

@torch.no_grad()
def prediction_token_set_stats(logits, x):
    """Measure whether errors fall in source or noise token sets.

    The categories are not mutually exclusive. ``logits`` has shape
    ``(B, T, vocab)`` and ``x`` uses ``[source][SEP][noise][query]`` layout.
    """
    pred = logits.argmax(dim=-1)
    target = x[:, :SEQ_LEN]
    noise = x[:, SEQ_LEN + 1:-SEQ_LEN]

    correct = pred.eq(target)
    error = ~correct

    acc = correct.float().mean()

    pred_in_source_set = (pred.unsqueeze(-1) == target.unsqueeze(1)).any(dim=-1)

    if noise.size(1) > 0:
        pred_in_noise_set = (pred.unsqueeze(-1) == noise.unsqueeze(1)).any(dim=-1)
    else:
        pred_in_noise_set = torch.zeros_like(pred_in_source_set)

    error_count = error.float().sum().clamp_min(1.0)

    return {
        "copy_acc": float(acc.detach().cpu().item()),
        "error_rate": float(error.float().mean().detach().cpu().item()),
        "error_pred_in_source_set_rate": float(
            ((error & pred_in_source_set).float().sum() / error_count).detach().cpu().item()
        ),
        "error_pred_in_noise_set_rate": float(
            ((error & pred_in_noise_set).float().sum() / error_count).detach().cpu().item()
        ),
    }
