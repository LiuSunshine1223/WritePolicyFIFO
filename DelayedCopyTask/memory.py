import torch
from collections import Counter

# Per-example external FIFO buffer for hidden-state keys and values.
# This is not an autoregressive decoder KV cache.
class WindowKVBuffer:
    """Store per-example hidden-state K/V entries in FIFO order.

    ``keys`` and ``vals`` have shape ``(B, M, D)`` and are reset between
    examples or training steps. Optional labels describe the ``M`` entries.
    """
    def __init__(self, max_mem):
        self.max_mem = max_mem
        self.keys = None
        self.vals = None

        # Optional semantic labels used by memory diagnostics.
        self.labels = None

    def reset(self):
        self.keys = None
        self.vals = None
        self.labels = None

    def __len__(self):
        if self.keys is None:
            return 0
        return self.keys.size(1)

    # Used by streaming variants; experiment policies usually call set_entries.
    def _fifo_crop(self):
        if self.keys is None:
            return

        if self.keys.size(1) <= self.max_mem:
            return

        # Drop the oldest entries along the memory axis.
        overflow = self.keys.size(1) - self.max_mem

        self.keys = self.keys[:, -self.max_mem:, :]
        self.vals = self.vals[:, -self.max_mem:, :]

        if self.labels is not None:
            self.labels = self.labels[overflow:]
    def append(self, k, v, labels=None):
        """Append ``(B, T, D)`` K/V blocks and crop oldest entries by FIFO."""
        # Non-positive capacity disables memory.
        if self.max_mem <= 0:
            self.reset()
            return

        if labels is not None:
            labels = list(labels)
            if len(labels) != k.size(1):
                raise ValueError(
                    f"The length of the label must match the time dimension."
                    f"len(labels)={len(labels)}, k.size(1)={k.size(1)}."
                )

        # Initialize the buffer or append along the memory axis.
        if self.keys is None:
            self.keys = k
            self.vals = v
            self.labels = labels
        else:
            # Existing entries follow the incoming tensors' devices.
            self.keys = self.keys.to(k.device)
            self.vals = self.vals.to(v.device)

            self.keys = torch.cat([self.keys, k], dim=1)
            self.vals = torch.cat([self.vals, v], dim=1)

            if self.labels is not None or labels is not None:
                old_labels = self.labels if self.labels is not None else ["unknown"] * (self.keys.size(1) - k.size(1))
                new_labels = labels if labels is not None else ["unknown"] * k.size(1)
                self.labels = old_labels + new_labels

        self._fifo_crop()

    def set_entries(self, k, v, labels=None):
        """Set a precomputed, capacity-limited memory state.

        ``k`` and ``v`` have shape ``(B, M, D)``; ``labels`` has length ``M``.
        Policy construction, including pinned-source versus FIFO entries, is
        completed before this call.
        """
        if self.max_mem <= 0:
            self.reset()
            return

        if k.size(1) > self.max_mem:
            raise ValueError(
                f"set_entries received {k.size(1)} entries, "
                f"but max_mem={self.max_mem}. "
            )

        if labels is not None:
            labels = list(labels)
            if len(labels) != k.size(1):
                raise ValueError(
                    f"The length of the label must match the time dimension."
                    f"len(labels)={len(labels)}, M={k.size(1)}"
                )

        self.keys = k
        self.vals = v
        self.labels = labels

    def label_counts(self):
        """Return counts of memory labels."""
        if self.labels is None:
            return {}

        return dict(Counter(self.labels))

    def snapshot(self):
        """Return lightweight metadata for the current buffer."""
        if self.keys is None:
            return {
                "mem_size": 0,
                "labels": [],
                "label_counts": {},
            }
        return {
            "mem_size": self.keys.size(1),
            "labels": list(self.labels) if self.labels is not None else [],
            "label_counts": self.label_counts(),
        }
