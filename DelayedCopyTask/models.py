# Transformer baselines and external-memory variants.
import torch
import torch.nn as nn

from .config import (
    VOCAB_SIZE,
    MODEL_DIM,
    NUM_LAYERS,
    NHEAD,
    SEQ_LEN,
    DELAY_LEN,
    WRITE_SOURCE_ONLY,
)
from .utils import window_attention_mask, force_no_source_mask
from .memory import WindowKVBuffer
from .memory_ops import (
    write_to_memory,
    read_from_memory_naive,
    read_from_memory_gated,
)

# Local-attention Transformer baseline.
class CopyMaskTransformer(nn.Module):
    def __init__(self, mask_mode: str):
        super().__init__()
        if mask_mode not in {"natural", "forced"}:
            raise ValueError(f"Invalid mask_mode: {mask_mode}")
        self.mask_mode = mask_mode  # "natural" or "forced"
        self.embed = nn.Embedding(VOCAB_SIZE, MODEL_DIM)

        # Learned absolute positions cover the longest configured input.
        padding = 5
        MAX_TOTAL_LEN = 2 * SEQ_LEN + max(DELAY_LEN) + 1 + padding
        self.pos_emb = nn.Parameter(
            torch.randn(MAX_TOTAL_LEN, MODEL_DIM) * 0.02
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=MODEL_DIM,
            nhead=NHEAD,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, NUM_LAYERS)
        self.fc = nn.Linear(MODEL_DIM, VOCAB_SIZE)

    def _build_attn_mask(self, T, window_size, device):
        # Forced mode blocks every query-to-source edge, including local ones.
        if self.mask_mode == "natural":
            return window_attention_mask(T, window_size).to(device)

        if self.mask_mode == "forced":
            return force_no_source_mask(T, window_size, SEQ_LEN).to(device)

        raise ValueError(f"Invalid mask_mode: {self.mask_mode}")

    def _encode(self, x, window_size):
        T = x.size(1)  # x: (B, T)

        h = self.embed(x)  # (B, T, D)
        pos = self.pos_emb[:T].unsqueeze(0)  # (1, T, D)
        h = h + pos

        attn_mask = self._build_attn_mask(T, window_size, x.device)
        # The additive mask uses zero for allowed edges and -inf for blocked edges.
        h = self.encoder(h, mask=attn_mask)

        return h

    def forward(self, x, window_size, return_diag: bool = False):
        h = self._encode(x, window_size)
        query_h = h[:, -SEQ_LEN:, :]
        logits = self.fc(query_h)

        if return_diag:
            diag = {
                "model_type": "baseline",
                "has_memory": False,
                "mem_size": 0,
                "source_count": 0,
                "sep_count": 0,
                "noise_count": 0,
                "source_survival_rate": 0.0,
                "attn_mass_source": None,
                "attn_mass_sep": None,
                "attn_mass_noise": None,
                "attn_entropy": None,
            }
            return logits, diag

        return logits

class CopyMaskTransformerWithNWMemory(CopyMaskTransformer):
    def __init__(self, memory: WindowKVBuffer, mask_mode: str = "forced", write_mode: str = WRITE_SOURCE_ONLY, noise_write_ratio=None):
        super().__init__(mask_mode)
        self.memory = memory
        self.write_mode = write_mode
        self.noise_write_ratio = noise_write_ratio

    def forward(self, x, window_size, return_diag: bool = False):
        h = self._encode(x, window_size)

        # Build the policy-specific K/V state before reading query positions.
        write_to_memory(h, self.memory, self.write_mode, self.noise_write_ratio)

        # query: (B, SEQ_LEN, D)
        query = h[:, -SEQ_LEN:, :]
        h_back = read_from_memory_naive(query, self.memory, return_diag=False)
        logits = self.fc(h_back)

        if return_diag:
            h_back, diag = read_from_memory_naive(
                query,
                self.memory,
                return_diag=True,
            )

            diag["model_type"] = "naive"
            diag["has_memory"] = True
            diag["write_mode"] = self.write_mode
            diag["noise_write_ratio"] = self.noise_write_ratio

            logits = self.fc(h_back)
            return logits, diag

        return logits

class CopyMaskTransformerWithGWMemory(CopyMaskTransformer):
    def __init__(self, memory: WindowKVBuffer, mask_mode: str = "forced", write_mode: str = WRITE_SOURCE_ONLY, noise_write_ratio=None):
        super().__init__(mask_mode)
        self.memory = memory
        self.gate = nn.Linear(MODEL_DIM, 1)
        self.write_mode = write_mode
        self.noise_write_ratio = noise_write_ratio

    def forward(self, x, window_size, return_diag: bool = False):
        h = self._encode(x, window_size)

        write_to_memory(h, self.memory, self.write_mode, self.noise_write_ratio)

        # query: (B, SEQ_LEN, D)
        query = h[:, -SEQ_LEN:, :]
        h_back = read_from_memory_gated(query, self.memory, self.gate, return_diag=False)
        logits = self.fc(h_back)

        if return_diag:
            h_back, diag = read_from_memory_gated(
                query,
                self.memory,
                self.gate,
                return_diag=True,
            )

            diag["model_type"] = "gated"
            diag["has_memory"] = True
            diag["write_mode"] = self.write_mode
            diag["noise_write_ratio"] = self.noise_write_ratio

            logits = self.fc(h_back)
            return logits, diag

        return logits
