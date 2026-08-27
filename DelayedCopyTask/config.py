# Shared experiment constants.
import torch


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model hyperparameters
BASE_VOCAB_SIZE = 20                  # Data tokens use IDs 0-19.
SEP_TOKEN = BASE_VOCAB_SIZE           # The separator uses ID 20.
VOCAB_SIZE = BASE_VOCAB_SIZE + 1      # Data tokens plus the separator.

MODEL_DIM = 64                        # Token embedding dimension.
NUM_LAYERS = 2                        # Transformer encoder layers.
NHEAD = 4                             # Attention heads.



# Training hyperparameters
BATCH_SIZE = 16
NUM_STEPS = 3001                      # Delayed-copy experiments.
LR = 1e-3

EVAL_TAIL = 500                       # Number of trailing steps summarized.



# Task structure
# input = [source][sep][noise][query]
# input_len = 2 * SEQ_LEN + DELAY_LEN + 1
SEQ_LEN = 20
DELAY_LEN = [10, 20, 40, 80]



# Attention windows.
WINDOW_SIZE = [4, 8, 16, 32]
FORCED_WINDOW = 8

NATURAL_SANITY_MEM = SEQ_LEN           # Natural-mask sanity check only.



# Random seeds
SEEDS_DEBUG = [0]
SEEDS_MAIN = [0, 1, 2, 3, 4, 5, 6]



# Success threshold
TAU_MAIN = 0.95
# Thresholds used for sensitivity analysis.
TAU_LIST = [0.90, 0.95, 0.98]



# Memory grids
# Source-only retains the full source when memory capacity reaches SEQ_LEN.
M_SOURCE_ONLY = [0, 5, 10, 15, 20, 25, 30]

# Prefix-all writes [source][SEP][noise], whose length is 21 + delay.
# Full-source retention boundaries:
# d=10 -> 31
# d=20 -> 41
# d=40 -> 61
# d=80 -> 101
M_PREFIX_ALL = [
    0,
    10,
    20,
    30, 31, 32,
    40, 41, 42,
    50,
    60, 61, 62,
    70,
    80,
    100, 101, 102,
    120
]

# Source-pinned keeps the source fixed and admits a bounded number of noise states.
NOISE_WRITE_BUDGET = [0, 5, 10, 20, 40, 60, 80]
# Total capacity is the source length plus the admitted-noise budget.
M_PINNED = [SEQ_LEN + q for q in NOISE_WRITE_BUDGET]



# Write policies
WRITE_SOURCE_ONLY = "source-only"
WRITE_PREFIX_ALL = "prefix-all"
WRITE_SOURCE_PINNED = "source-pinned-noise-fifo"

# Boundary policies.
WRITE_MODES_BOUNDARY = [
    WRITE_SOURCE_ONLY,
    WRITE_PREFIX_ALL,
]
# Experiment A also separates eviction from retrieval interference.
WRITE_MODES_EXPA = [
    WRITE_SOURCE_ONLY,
    WRITE_PREFIX_ALL,
    WRITE_SOURCE_PINNED,
]



# Model variants
MODEL_VARIANTS = ["naive", "gated"]



# Experiment C: source and SEP are always written; a fraction of noise is admitted.
# p=0 writes source plus SEP, while p=1 is equivalent to prefix-all.
WRITE_INTERMEDIATE = "source-sep-noise-budget"

CONTAMINATION_P = [0.0, 0.25, 0.5, 0.75, 1.0]

# Delays used by Experiment C.
DELAY_LEN_EXPC = [20, 40, 80]

WRITE_MODES_EXPC = [
    WRITE_INTERMEDIATE,
]


# Experiment D: symbolic key-value retrieval with the same model and FIFO memory.
# The input layout remains [source][SEP][noise][query], with source and query
# lengths fixed at SEQ_LEN.

DELAY_LEN_EXPD = [20, 40, 80]
NUM_STEPS_EXPD = 9001

# Each distractor record has ten tokens, yielding 2/4/8 records by delay.
DISTRACTOR_FACT_LEN_EXPD = 10

# Symbolic markers remain inside the base vocabulary.
EXPD_FACT_TOKEN = 0       # Fact marker.
EXPD_QUERY_TOKEN = 1      # Query marker.
EXPD_IS_TOKEN = 2         # Relation marker.
EXPD_ASK_TOKEN = 3        # Query filler.
EXPD_END_TOKEN = 4        # Reserved query/padding marker.
# Key tokens: 5-12.
EXPD_KEY_START = 5
EXPD_KEY_END = 12
# Value tokens: 13-19.
EXPD_VALUE_START = 13
EXPD_VALUE_END = BASE_VOCAB_SIZE - 1

# Experiment D seeds and model variants.
SEEDS_EXPD_DEBUG = [0]
SEEDS_EXPD_MAIN = SEEDS_MAIN

MODEL_VARIANTS_EXPD = ["naive", "gated"]
TAU_EXPD = TAU_MAIN
WINDOW_EXPD = FORCED_WINDOW

WRITE_MODES_EXPD = [
    WRITE_SOURCE_ONLY,
    WRITE_PREFIX_ALL,
    WRITE_SOURCE_PINNED,
]

# Experiment D source-only grid; the retention boundary is SEQ_LEN.
M_SOURCE_ONLY_EXPD = [0, 10, 15, 20, 25, 30]


# Prefix-all full-source retention threshold is SEQ_LEN + 1 + delay.
# d=20 -> 41
# d=40 -> 61
# d=80 -> 101
# Compact grids concentrate capacity near each boundary.
M_PREFIX_ALL_EXPD_BY_DELAY = {
    20: [0, 10, 20, 39, 40, 41, 42, 43, 51, 61],
    40: [0, 10, 20, 59, 60, 61, 62, 63, 71, 81],
    80: [0, 10, 20, 99, 100, 101, 102, 103, 111, 121],
}


# Source-pinned keeps the source fixed while varying admitted-noise capacity.
NOISE_WRITE_BUDGET_EXPD = [0, 20, 40, 80]
M_PINNED_EXPD = [SEQ_LEN + q for q in NOISE_WRITE_BUDGET_EXPD]
