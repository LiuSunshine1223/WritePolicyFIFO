# Experiment B runner for memory, attention, and error diagnostics.
# Paper mode uses the formal protocol; smoke output is isolated under tmp/smoke.

import argparse
from collections import defaultdict
from pathlib import Path

import torch

from DelayedCopyTask.config import (
    SEP_TOKEN,
    SEQ_LEN,
    EVAL_TAIL,
    NUM_STEPS,
    FORCED_WINDOW,
    SEEDS_MAIN,

    WRITE_SOURCE_ONLY,
    WRITE_PREFIX_ALL,
    WRITE_SOURCE_PINNED,
)

from DelayedCopyTask.dataset import generate_batch
from DelayedCopyTask.memory import WindowKVBuffer
from DelayedCopyTask.memory_ops import write_to_memory
from DelayedCopyTask.models import (
    CopyMaskTransformerWithNWMemory,
    CopyMaskTransformerWithGWMemory,
)
from DelayedCopyTask.train import train
from DelayedCopyTask.utils import (
    set_seed,
    summarize_loss,
    summarize_acc,
    save_results_csv,
)


# Repository-rooted paths prevent the working directory from redirecting outputs.
PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "Results" / "expB"
PAPER_OUTPUT_PATH = RESULTS_DIR / "results_expB_paper.csv"
SMOKE_OUTPUT_PATH = (
    PROJECT_ROOT / "tmp" / "smoke" / "expB" / "results_expB_smoke.csv"
)
CHECKPOINT_PATH = PROJECT_ROOT / "tmp" / "expB_checkpoint.csv"

EXPB_WINDOW = FORCED_WINDOW

# The formal protocol evaluates both model variants at delays 40 and 80.
EXPB_DELAYS_MAIN = [40, 80]

EXPB_SEEDS_MAIN = SEEDS_MAIN

EXPB_MODEL_VARIANTS_MAIN = ["naive", "gated"]

EXPB_WRITE_MODES = [
    WRITE_SOURCE_ONLY,
    WRITE_PREFIX_ALL,
    WRITE_SOURCE_PINNED,
]

# Source-only capacities bracket the source length L = 20.
EXPB_M_SOURCE_ONLY = [10, 15, 20, 25, 30]

# Prefix-all grids bracket m = L + 1 + d; paper mode uses delays 40 and 80.
EXPB_M_PREFIX_ALL_BY_DELAY = {
    10: [20, 30, 31, 32, 40],
    20: [30, 40, 41, 42, 50],
    40: [50, 60, 61, 62, 70],
    80: [80, 100, 101, 102, 120],
}

# Source-pinned capacity is m = L + q, where q is the noise FIFO budget.
EXPB_NOISE_WRITE_BUDGET = [0, 5, 10, 20, 40, 60, 80]
EXPB_M_PINNED = [SEQ_LEN + q for q in EXPB_NOISE_WRITE_BUDGET]

EXPB_NUM_EVAL_BATCHES_MAIN = 16


def get_run_config(mode):
    if mode == "paper":
        return {
            "mode": "paper",
            "result_run_mode": "main",
            "seeds": EXPB_SEEDS_MAIN,
            "delays": EXPB_DELAYS_MAIN,
            "model_variants": EXPB_MODEL_VARIANTS_MAIN,
            "write_modes": EXPB_WRITE_MODES,
            "source_only_grid": EXPB_M_SOURCE_ONLY,
            "prefix_all_grid_by_delay": {
                delay: EXPB_M_PREFIX_ALL_BY_DELAY[delay]
                for delay in EXPB_DELAYS_MAIN
            },
            "pinned_grid": EXPB_M_PINNED,
            "num_steps": NUM_STEPS,
            "eval_tail": EVAL_TAIL,
            "eval_batches": EXPB_NUM_EVAL_BATCHES_MAIN,
            "checkpoint_enabled": True,
        }

    if mode == "smoke":
        return {
            "mode": "smoke",
            "result_run_mode": "smoke",
            "seeds": [0],
            "delays": [40],
            "model_variants": ["naive", "gated"],
            "write_modes": EXPB_WRITE_MODES,
            "source_only_grid": [20],
            "prefix_all_grid_by_delay": {40: [61]},
            "pinned_grid": [25],
            "num_steps": 2,
            "eval_tail": 2,
            "eval_batches": 1,
            "checkpoint_enabled": False,
        }

    raise ValueError(f"Invalid mode: {mode}")


def get_expB_memory_grid(write_mode, delay_len, config):
    if write_mode == WRITE_SOURCE_ONLY:
        return config["source_only_grid"]

    if write_mode == WRITE_PREFIX_ALL:
        return config["prefix_all_grid_by_delay"][int(delay_len)]

    if write_mode == WRITE_SOURCE_PINNED:
        return config["pinned_grid"]

    raise ValueError(f"Invalid write_mode: {write_mode}")


def count_experiment_runs(config):
    runs_per_seed_and_model = 0
    for delay_len in config["delays"]:
        for write_mode in config["write_modes"]:
            runs_per_seed_and_model += len(
                get_expB_memory_grid(write_mode, delay_len, config)
            )

    return (
        len(config["seeds"])
        * len(config["model_variants"])
        * runs_per_seed_and_model
    )


def validate_run_config(config):
    for name in ("seeds", "delays", "model_variants", "write_modes"):
        if not config[name]:
            raise ValueError(f"{name} must not be empty")

    for name in ("num_steps", "eval_tail", "eval_batches"):
        if int(config[name]) <= 0:
            raise ValueError(f"{name} must be positive")

    if int(config["eval_tail"]) > int(config["num_steps"]):
        raise ValueError("eval_tail must not exceed num_steps")

    total_runs = count_experiment_runs(config)
    if config["mode"] == "paper":
        expected = {
            "result_run_mode": "main",
            "seeds": [0, 1, 2, 3, 4, 5, 6],
            "delays": [40, 80],
            "model_variants": ["naive", "gated"],
            "write_modes": [
                "source-only",
                "prefix-all",
                "source-pinned-noise-fifo",
            ],
            "source_only_grid": [10, 15, 20, 25, 30],
            "prefix_all_grid_by_delay": {
                40: [50, 60, 61, 62, 70],
                80: [80, 100, 101, 102, 120],
            },
            "pinned_grid": [20, 25, 30, 40, 60, 80, 100],
            "num_steps": 3001,
            "eval_tail": 500,
            "eval_batches": 16,
            "checkpoint_enabled": True,
        }

        for name, expected_value in expected.items():
            if config[name] != expected_value:
                raise ValueError(
                    f"Experiment B paper protocol mismatch for {name}: "
                    f"expected {expected_value!r}, got {config[name]!r}"
                )

        if EXPB_WINDOW != 8:
            raise ValueError(
                f"Experiment B paper protocol requires window 8, got {EXPB_WINDOW}"
            )

        if total_runs != 476:
            raise ValueError(
                "Experiment B paper protocol must contain 476 runs, "
                f"got {total_runs}"
            )

    return total_runs


def resolve_output_path(mode, output_path=None):
    if output_path is None:
        return PAPER_OUTPUT_PATH if mode == "paper" else SMOKE_OUTPUT_PATH

    output_path = Path(output_path)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    return output_path


def resolve_device(device_spec):
    normalized = str(device_spec).strip().lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        device = torch.device(normalized)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(f"Invalid device: {device_spec}") from exc

    if device.type not in {"cpu", "cuda"}:
        raise ValueError("device must be auto, cpu, cuda, or cuda:N")

    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is not available")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError(
                f"CUDA device index {device.index} is unavailable; "
                f"found {torch.cuda.device_count()} CUDA device(s)"
            )

    return device


def checkpoint_pending_path(checkpoint_path=CHECKPOINT_PATH):
    checkpoint_path = Path(checkpoint_path)
    return checkpoint_path.with_name(
        f"{checkpoint_path.stem}.pending{checkpoint_path.suffix}"
    )


def ensure_parent_directory(path, description):
    path = Path(path)
    parent = path.parent

    try:
        parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise OSError(
            f"Unable to prepare {description} directory: {parent}"
        ) from exc

    if not parent.is_dir():
        raise NotADirectoryError(
            f"{description} parent is not a directory: {parent}"
        )

    return parent


def prepare_run_files(config, output_path, overwrite):
    output_path = Path(output_path)
    if output_path.suffix.lower() != ".csv":
        raise ValueError(f"output path must end in .csv: {output_path}")

    if output_path.exists() and output_path.is_dir():
        raise IsADirectoryError(f"output path is a directory: {output_path}")

    ensure_parent_directory(output_path, "output")

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing results file: {output_path}. "
            "Pass --overwrite only when replacement is intentional."
        )

    if not config["checkpoint_enabled"]:
        return

    pending_path = checkpoint_pending_path()
    ensure_parent_directory(CHECKPOINT_PATH, "checkpoint")

    for path in (CHECKPOINT_PATH, pending_path):
        if path.exists() and path.is_dir():
            raise IsADirectoryError(f"checkpoint path is a directory: {path}")

    reserved_paths = {CHECKPOINT_PATH.resolve(), pending_path.resolve()}
    if output_path.resolve() in reserved_paths:
        raise ValueError("output path must not be an Experiment B checkpoint path")

    stale_paths = [
        path for path in (CHECKPOINT_PATH, pending_path) if path.exists()
    ]
    if stale_paths and not overwrite:
        paths = ", ".join(str(path) for path in stale_paths)
        raise FileExistsError(
            f"Stale Experiment B checkpoint found: {paths}. "
            "Inspect it before starting a new paper run, or pass --overwrite "
            "to replace it."
        )

    if overwrite:
        for path in stale_paths:
            path.unlink()


def save_checkpoint(rows, checkpoint_path=CHECKPOINT_PATH):
    checkpoint_path = Path(checkpoint_path)
    pending_path = checkpoint_pending_path(checkpoint_path)
    save_results_csv(rows, pending_path, overwrite=True)
    pending_path.replace(checkpoint_path)
    return checkpoint_path


def remove_checkpoint(checkpoint_path=CHECKPOINT_PATH):
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.unlink(missing_ok=True)
    checkpoint_pending_path(checkpoint_path).unlink(missing_ok=True)


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Run Experiment B memory diagnostics."
    )
    parser.add_argument(
        "--mode",
        choices=("paper", "smoke"),
        default="paper",
        help="paper reproduces the formal protocol; smoke runs a tiny check.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="CSV output path; relative paths are resolved from the project root.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, or an indexed CUDA device such as cuda:0.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of an existing output or stale checkpoint.",
    )
    return parser


def build_model(model_variant, write_mode, memory_size):
    memory = WindowKVBuffer(max_mem=memory_size)

    if model_variant == "naive":
        return CopyMaskTransformerWithNWMemory(
            memory=memory,
            mask_mode="forced",
            write_mode=write_mode,
        )

    if model_variant == "gated":
        return CopyMaskTransformerWithGWMemory(
            memory=memory,
            mask_mode="forced",
            write_mode=write_mode,
        )

    raise ValueError(f"Invalid model_variant: {model_variant}")


def compute_memory_attention(query, memory):
    """Return query-to-memory attention and retrieved states, or the query if memory is empty."""
    if memory.keys is None or memory.keys.size(1) == 0:
        return None, query

    mem_keys = memory.keys
    mem_vals = memory.vals

    attn_logits = torch.matmul(
        query,
        mem_keys.transpose(-1, -2),
    ) / (query.size(-1) ** 0.5)

    attn = torch.softmax(attn_logits, dim=-1)
    retrieved = torch.matmul(attn, mem_vals)

    return attn, retrieved


def memory_label_stats(labels):
    if labels is None:
        labels = []

    source_count = sum(label == "source" for label in labels)
    sep_count = sum(label == "sep" for label in labels)
    noise_count = sum(label == "noise" for label in labels)
    total_count = len(labels)

    return {
        "mem_entries_total": total_count,
        "mem_source_entries": source_count,
        "mem_sep_entries": sep_count,
        "mem_noise_entries": noise_count,
        "source_survival_rate": float(source_count / SEQ_LEN),
    }


def attention_mass_by_label(attn, labels):
    if attn is None or labels is None or len(labels) == 0:
        return {
            "attn_mass_source": 0.0,
            "attn_mass_sep": 0.0,
            "attn_mass_noise": 0.0,
            "attn_mass_other": 0.0,
            "attn_entropy": 0.0,
        }

    if attn.size(-1) != len(labels):
        raise ValueError(
            f"attn memory dim != labels length: "
            f"{attn.size(-1)} vs {len(labels)}"
        )

    device = attn.device
    dtype = attn.dtype

    def make_mask(label_name):
        return torch.tensor(
            [label == label_name for label in labels],
            device=device,
            dtype=dtype,
        ).view(1, 1, -1)

    source_mask = make_mask("source")
    sep_mask = make_mask("sep")
    noise_mask = make_mask("noise")

    known_mask = torch.tensor(
        [label in {"source", "sep", "noise"} for label in labels],
        device=device,
        dtype=dtype,
    ).view(1, 1, -1)

    with torch.no_grad():
        source_mass = (attn * source_mask).sum(dim=-1).mean()
        sep_mass = (attn * sep_mask).sum(dim=-1).mean()
        noise_mass = (attn * noise_mask).sum(dim=-1).mean()
        other_mass = (attn * (1.0 - known_mask)).sum(dim=-1).mean()

        eps = 1e-12
        entropy = -(
            attn.clamp_min(eps) * attn.clamp_min(eps).log()
        ).sum(dim=-1).mean()

    return {
        "attn_mass_source": float(source_mass.detach().cpu().item()),
        "attn_mass_sep": float(sep_mass.detach().cpu().item()),
        "attn_mass_noise": float(noise_mass.detach().cpu().item()),
        "attn_mass_other": float(other_mass.detach().cpu().item()),
        "attn_entropy": float(entropy.detach().cpu().item()),
    }


def classify_error_types(pred, tgt, inp, delay_len):
    """Classify errors by source/noise membership; a token may belong to both sets."""
    pred = pred.detach().cpu()
    tgt = tgt.detach().cpu()
    inp = inp.detach().cpu()

    wrong_mask = pred != tgt

    total_tokens = pred.numel()
    wrong_tokens = int(wrong_mask.sum().item())
    correct_tokens = total_tokens - wrong_tokens

    stats = {
        "eval_total_tokens": total_tokens,
        "eval_correct_tokens": correct_tokens,
        "eval_wrong_tokens": wrong_tokens,
        "eval_error_rate": float(wrong_tokens / total_tokens),

        "err_sep": 0,
        "err_source_only": 0,
        "err_noise_only": 0,
        "err_both_source_and_noise": 0,
        "err_other": 0,
    }

    if wrong_tokens == 0:
        stats.update({
            "err_sep_rate": 0.0,
            "err_source_only_rate": 0.0,
            "err_noise_only_rate": 0.0,
            "err_both_source_and_noise_rate": 0.0,
            "err_other_rate": 0.0,
        })
        return stats

    B, L = pred.shape

    for b in range(B):
        source_set = set(inp[b, :SEQ_LEN].tolist())

        noise_start = SEQ_LEN + 1
        noise_end = SEQ_LEN + 1 + delay_len
        noise_set = set(inp[b, noise_start:noise_end].tolist())

        for j in range(L):
            if not wrong_mask[b, j]:
                continue

            p = int(pred[b, j].item())

            if p == SEP_TOKEN:
                stats["err_sep"] += 1
                continue

            in_source = p in source_set
            in_noise = p in noise_set

            if in_source and in_noise:
                stats["err_both_source_and_noise"] += 1
            elif in_source:
                stats["err_source_only"] += 1
            elif in_noise:
                stats["err_noise_only"] += 1
            else:
                stats["err_other"] += 1

    denom = max(wrong_tokens, 1)

    stats.update({
        "err_sep_rate": float(stats["err_sep"] / denom),
        "err_source_only_rate": float(stats["err_source_only"] / denom),
        "err_noise_only_rate": float(stats["err_noise_only"] / denom),
        "err_both_source_and_noise_rate": float(stats["err_both_source_and_noise"] / denom),
        "err_other_rate": float(stats["err_other"] / denom),
    })

    return stats


@torch.no_grad()
def diagnose_model(
    model,
    delay_len,
    window_size,
    data_gen,
    eval_batches,
    device,
):
    """Aggregate post-training retention, attention, and token-error diagnostics."""
    model.eval()
    model.to(device)

    aggregate = defaultdict(float)
    label_stats_first = None

    for _ in range(eval_batches):
        if hasattr(model, "memory"):
            model.memory.reset()

        inp, tgt = generate_batch(delay_len, data_gen)
        inp = inp.to(device)
        tgt = tgt.to(device)

        h = model._encode(inp, window_size)

        write_to_memory(h, model.memory, model.write_mode)

        query = h[:, -SEQ_LEN:, :]
        attn, retrieved = compute_memory_attention(query, model.memory)

        if hasattr(model, "gate"):
            gate = torch.sigmoid(model.gate(query))
            h_back = (1.0 - gate) * query + gate * retrieved

            aggregate["gate_mean"] += float(gate.mean().detach().cpu().item())
            aggregate["gate_min"] += float(gate.min().detach().cpu().item())
            aggregate["gate_max"] += float(gate.max().detach().cpu().item())
        else:
            h_back = query + retrieved

            aggregate["gate_mean"] += -1.0
            aggregate["gate_min"] += -1.0
            aggregate["gate_max"] += -1.0

        logits = model.fc(h_back)
        pred = logits.argmax(dim=-1)
        eval_acc = (pred == tgt).float().mean().item()

        aggregate["eval_acc"] += float(eval_acc)

        labels = model.memory.labels

        if label_stats_first is None:
            label_stats_first = memory_label_stats(labels)

        attn_stats = attention_mass_by_label(attn, labels)
        for k, v in attn_stats.items():
            aggregate[k] += float(v)

        err_stats = classify_error_types(pred, tgt, inp, delay_len)
        for k, v in err_stats.items():
            aggregate[k] += float(v)

    out = {}
    for k, v in aggregate.items():
        out[k] = float(v / eval_batches)

    if label_stats_first is not None:
        out.update(label_stats_first)

    return out


def relative_or_abs(path):
    path = Path(path).resolve()
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def run_expB(config, output_path, device, overwrite=False):
    total_runs = validate_run_config(config)
    output_path = Path(output_path)
    device = torch.device(device)
    prepare_run_files(config, output_path, overwrite)

    rows = []

    seeds = config["seeds"]
    delays = config["delays"]
    model_variants = config["model_variants"]
    write_modes = config["write_modes"]
    num_steps = int(config["num_steps"])
    eval_tail = int(config["eval_tail"])
    eval_batches = int(config["eval_batches"])

    print("=" * 80)
    print(f"Experiment B: memory diagnostics")
    print(f"mode          : {config['mode']}")
    print(f"device        : {device}")
    print(f"window        : {EXPB_WINDOW}")
    print(f"seeds         : {seeds}")
    print(f"delays        : {delays}")
    print(f"models        : {model_variants}")
    print(f"write modes   : {write_modes}")
    print(f"num steps     : {num_steps}")
    print(f"eval tail     : {eval_tail}")
    print(f"eval batches  : {eval_batches}")
    print(f"total runs    : {total_runs}")
    print(f"output        : {relative_or_abs(output_path)}")
    print("=" * 80)

    run_idx = 0

    for seed in seeds:
        for model_variant in model_variants:
            for delay_len in delays:
                for write_mode in write_modes:
                    memory_grid = get_expB_memory_grid(
                        write_mode,
                        delay_len,
                        config,
                    )

                    print("-" * 80)
                    print(
                        f"Group | seed={seed} | model={model_variant} | "
                        f"delay={delay_len} | write={write_mode}"
                    )
                    print(f"memory grid: {memory_grid}")
                    print("-" * 80)

                    for memory_size in memory_grid:
                        run_idx += 1

                        print(
                            f"[{run_idx:03d}/{total_runs:03d}] "
                            f"mem={memory_size}"
                        )

                        set_seed(seed)

                        train_gen = torch.Generator()
                        train_gen.manual_seed(seed)

                        eval_gen = torch.Generator()
                        eval_gen.manual_seed(seed + 100000)

                        model = build_model(
                            model_variant=model_variant,
                            write_mode=write_mode,
                            memory_size=memory_size,
                        )

                        loss_history, acc_history = train(
                            model=model,
                            window_size=EXPB_WINDOW,
                            delay_len=delay_len,
                            data_gen=train_gen,
                            num_steps=num_steps,
                            device=device,
                        )

                        loss_summary = summarize_loss(loss_history, eval_tail)
                        acc_summary = summarize_acc(acc_history, eval_tail)

                        diag = diagnose_model(
                            model=model,
                            delay_len=delay_len,
                            window_size=EXPB_WINDOW,
                            data_gen=eval_gen,
                            eval_batches=eval_batches,
                            device=device,
                        )

                        row = {
                            "phase": "expB_memory_diagnostics",
                            "run_mode": config["result_run_mode"],
                            "seed": seed,
                            "model": model_variant,
                            "mask_mode": "forced",
                            "window_size": EXPB_WINDOW,
                            "delay_len": delay_len,
                            "write_mode": write_mode,
                            "max_mem": memory_size,
                            "num_steps": num_steps,
                            "eval_tail": eval_tail,
                            "eval_batches": eval_batches,
                        }

                        row.update(loss_summary)
                        row.update(acc_summary)
                        row.update(diag)

                        rows.append(row)

                        if config["checkpoint_enabled"]:
                            save_checkpoint(rows)

                        print(
                            f"    train_tail_acc={row['acc_mean_tail']:.4f} | "
                            f"eval_acc={row['eval_acc']:.4f} | "
                            f"src_survival={row['source_survival_rate']:.3f} | "
                            f"attn_src={row['attn_mass_source']:.3f} | "
                            f"attn_noise={row['attn_mass_noise']:.3f} | "
                            f"mem_src={row['mem_source_entries']} | "
                            f"mem_noise={row['mem_noise_entries']}"
                        )

                        del model

                        if device.type == "cuda":
                            torch.cuda.empty_cache()

    print("=" * 80)
    csv_path = save_results_csv(
        rows,
        output_path,
        overwrite=overwrite,
    )

    if config["checkpoint_enabled"]:
        remove_checkpoint()

    print("Experiment B finished.")
    print(f"Final CSV : {relative_or_abs(csv_path)}")
    print("=" * 80)

    return rows


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    config = get_run_config(args.mode)
    output_path = resolve_output_path(args.mode, args.output)
    device = resolve_device(args.device)

    return run_expB(
        config=config,
        output_path=output_path,
        device=device,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
