# Experiment D runner for symbolic key-value retrieval.
# Paper mode uses the formal protocol; smoke output is isolated under tmp/smoke.

import argparse
from collections import Counter
from pathlib import Path

import torch

from DelayedCopyTask.models import (
    CopyMaskTransformerWithNWMemory,
    CopyMaskTransformerWithGWMemory,
)
from DelayedCopyTask.memory import WindowKVBuffer
from DelayedCopyTask.train import train
from DelayedCopyTask.dataset_expD import generate_batch_expD
from DelayedCopyTask.utils import (
    set_seed,
    summarize_loss,
    summarize_acc,
    save_results_csv,
)
from DelayedCopyTask.config import (
    NHEAD,
    NUM_LAYERS,
    MODEL_DIM,
    SEQ_LEN,
    EVAL_TAIL,
    WINDOW_EXPD,
    DELAY_LEN_EXPD,
    NUM_STEPS_EXPD,
    SEEDS_EXPD_MAIN,
    MODEL_VARIANTS_EXPD,
    WRITE_SOURCE_ONLY,
    WRITE_PREFIX_ALL,
    WRITE_SOURCE_PINNED,
    WRITE_MODES_EXPD,
    M_SOURCE_ONLY_EXPD,
    M_PREFIX_ALL_EXPD_BY_DELAY,
    M_PINNED_EXPD,
)


PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "Results" / "expD"
PAPER_OUTPUT_PATH = RESULTS_DIR / "results_expD_paper.csv"
SMOKE_OUTPUT_PATH = (
    PROJECT_ROOT / "tmp" / "smoke" / "expD" / "results_expD_smoke.csv"
)


def get_model_items(model_variants):
    items = []

    for name in model_variants:
        if name == "naive":
            items.append(("naive", CopyMaskTransformerWithNWMemory))
        elif name == "gated":
            items.append(("gated", CopyMaskTransformerWithGWMemory))
        else:
            raise ValueError(f"Invalid model variant for Experiment D: {name}")

    return items


def get_mem_grid(write_mode, delay_len, config):
    capacity_by_policy = config["capacity_by_policy"]
    if capacity_by_policy is not None:
        if write_mode not in capacity_by_policy:
            raise ValueError(f"No configured capacities for {write_mode}")
        return capacity_by_policy[write_mode]

    if write_mode == WRITE_SOURCE_ONLY:
        grid = M_SOURCE_ONLY_EXPD

    elif write_mode == WRITE_PREFIX_ALL:
        if delay_len not in M_PREFIX_ALL_EXPD_BY_DELAY:
            raise ValueError(
                f"No Experiment D prefix-all grid for delay_len={delay_len}."
            )
        grid = M_PREFIX_ALL_EXPD_BY_DELAY[delay_len]

    elif write_mode == WRITE_SOURCE_PINNED:
        grid = M_PINNED_EXPD

    else:
        raise ValueError(f"Invalid write_mode: {write_mode}")

    return grid


def get_run_config(mode):
    if mode == "paper":
        return {
            "mode": "paper",
            "seeds": SEEDS_EXPD_MAIN,
            "delay_list": DELAY_LEN_EXPD,
            "model_items": get_model_items(MODEL_VARIANTS_EXPD),
            "write_modes": WRITE_MODES_EXPD,
            "capacity_by_policy": None,
            "num_steps": NUM_STEPS_EXPD,
            "eval_tail": EVAL_TAIL,
        }

    if mode == "smoke":
        return {
            "mode": "smoke",
            "seeds": [0],
            "delay_list": [20],
            "model_items": get_model_items(["naive", "gated"]),
            "write_modes": WRITE_MODES_EXPD,
            "capacity_by_policy": {
                WRITE_SOURCE_ONLY: [20],
                WRITE_PREFIX_ALL: [41],
                WRITE_SOURCE_PINNED: [40],
            },
            "num_steps": 2,
            "eval_tail": 2,
        }

    raise ValueError(f"Invalid mode: {mode}")


def get_effective_mem_grid(config, write_mode, delay_len):
    grid = get_mem_grid(write_mode, delay_len, config)
    return [
        max_mem
        for max_mem in grid
        if max_mem > 0
        and not (
            write_mode == WRITE_SOURCE_PINNED and max_mem < SEQ_LEN
        )
    ]


def count_experiment_runs(config):
    runs_per_model_and_seed = 0
    for delay_len in config["delay_list"]:
        for write_mode in config["write_modes"]:
            runs_per_model_and_seed += len(
                get_effective_mem_grid(config, write_mode, delay_len)
            )

    return (
        len(config["model_items"])
        * len(config["seeds"])
        * runs_per_model_and_seed
    )


def validate_run_config(config):
    if config.get("mode") not in {"paper", "smoke"}:
        raise ValueError(f"Invalid mode: {config.get('mode')}")

    for name in ("seeds", "delay_list", "model_items", "write_modes"):
        if not config[name]:
            raise ValueError(f"{name} must not be empty")

    for name in ("num_steps", "eval_tail"):
        if int(config[name]) <= 0:
            raise ValueError(f"{name} must be positive")

    if int(config["eval_tail"]) > int(config["num_steps"]):
        raise ValueError("eval_tail must not exceed num_steps")

    total_runs = count_experiment_runs(config)

    if config["mode"] == "paper":
        expected = {
            "seeds": [0, 1, 2, 3, 4, 5, 6],
            "delay_list": [20, 40, 80],
            "model_items": [
                ("naive", CopyMaskTransformerWithNWMemory),
                ("gated", CopyMaskTransformerWithGWMemory),
            ],
            "write_modes": [
                "source-only",
                "prefix-all",
                "source-pinned-noise-fifo",
            ],
            "capacity_by_policy": None,
            "num_steps": 9001,
            "eval_tail": 500,
        }
        for name, expected_value in expected.items():
            if config[name] != expected_value:
                raise ValueError(
                    f"Experiment D paper protocol mismatch for {name}: "
                    f"expected {expected_value!r}, got {config[name]!r}"
                )

        expected_grids = {
            (20, "source-only"): [10, 15, 20, 25, 30],
            (40, "source-only"): [10, 15, 20, 25, 30],
            (80, "source-only"): [10, 15, 20, 25, 30],
            (20, "prefix-all"): [10, 20, 39, 40, 41, 42, 43, 51, 61],
            (40, "prefix-all"): [10, 20, 59, 60, 61, 62, 63, 71, 81],
            (80, "prefix-all"): [10, 20, 99, 100, 101, 102, 103, 111, 121],
            (20, "source-pinned-noise-fifo"): [20, 40, 60, 100],
            (40, "source-pinned-noise-fifo"): [20, 40, 60, 100],
            (80, "source-pinned-noise-fifo"): [20, 40, 60, 100],
        }
        actual_grids = {
            (delay_len, write_mode): get_effective_mem_grid(
                config,
                write_mode,
                delay_len,
            )
            for delay_len in config["delay_list"]
            for write_mode in config["write_modes"]
        }
        if actual_grids != expected_grids:
            raise ValueError("Experiment D paper capacity grids have changed")

        if WINDOW_EXPD != 8:
            raise ValueError(
                f"Experiment D paper protocol requires window 8, got {WINDOW_EXPD}"
            )
        if total_runs != 756:
            raise ValueError(
                "Experiment D paper protocol must contain 756 runs, "
                f"got {total_runs}"
            )

    if config["mode"] == "smoke":
        expected = {
            "seeds": [0],
            "delay_list": [20],
            "model_items": [
                ("naive", CopyMaskTransformerWithNWMemory),
                ("gated", CopyMaskTransformerWithGWMemory),
            ],
            "write_modes": [
                "source-only",
                "prefix-all",
                "source-pinned-noise-fifo",
            ],
            "capacity_by_policy": {
                "source-only": [20],
                "prefix-all": [41],
                "source-pinned-noise-fifo": [40],
            },
            "num_steps": 2,
            "eval_tail": 2,
        }
        for name, expected_value in expected.items():
            if config[name] != expected_value:
                raise ValueError(
                    f"Experiment D smoke protocol mismatch for {name}: "
                    f"expected {expected_value!r}, got {config[name]!r}"
                )
        if total_runs != 6:
            raise ValueError(
                f"Experiment D smoke protocol must contain 6 runs, got {total_runs}"
            )

    return total_runs


def resolve_output_path(mode, output_path=None):
    if output_path is None:
        return PAPER_OUTPUT_PATH if mode == "paper" else SMOKE_OUTPUT_PATH

    output_path = Path(output_path)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    return output_path


def validate_output_contract(mode, output_path):
    output_path = Path(output_path)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path

    resolved_output = output_path.resolve()
    if mode != "paper" and resolved_output == PAPER_OUTPUT_PATH.resolve():
        raise ValueError(
            "Only Experiment D paper mode may write to "
            f"{PAPER_OUTPUT_PATH}"
        )
    if mode == "paper" and resolved_output == SMOKE_OUTPUT_PATH.resolve():
        raise ValueError(
            "Experiment D paper mode may not write to the reserved smoke path "
            f"{SMOKE_OUTPUT_PATH}"
        )

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


def prepare_output_path(output_path, overwrite):
    output_path = Path(output_path)
    if output_path.exists() and output_path.is_dir():
        raise IsADirectoryError(f"output path is a directory: {output_path}")
    if output_path.suffix.lower() != ".csv":
        raise ValueError(f"output path must end in .csv: {output_path}")

    parent = output_path.parent
    try:
        parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise OSError(f"Unable to prepare output directory: {parent}") from exc
    if not parent.is_dir():
        raise NotADirectoryError(f"output parent is not a directory: {parent}")

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing results file: {output_path}. "
            "Pass --overwrite only when replacement is intentional."
        )


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Run Experiment D symbolic key-value retrieval diagnostics."
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
        help="Allow replacement of an existing output CSV.",
    )
    return parser


def add_expd_memory_diagnostics(summary, write_mode, max_mem, delay_len):
    """Add theoretical source-retention fields without another forward pass."""
    if write_mode == WRITE_SOURCE_ONLY:
        retained_source = min(max_mem, SEQ_LEN)
        noise_write_budget = None
        expected_boundary = SEQ_LEN

    elif write_mode == WRITE_PREFIX_ALL:
        retained_source = max(0, min(SEQ_LEN, max_mem - (1 + delay_len)))
        noise_write_budget = None
        expected_boundary = SEQ_LEN + 1 + delay_len

    elif write_mode == WRITE_SOURCE_PINNED:
        retained_source = SEQ_LEN
        noise_write_budget = max_mem - SEQ_LEN
        expected_boundary = SEQ_LEN

    else:
        retained_source = None
        noise_write_budget = None
        expected_boundary = None

    summary["noise_write_budget"] = noise_write_budget
    summary["theoretical_retained_source"] = retained_source
    summary["expected_boundary"] = expected_boundary

    if retained_source is None:
        summary["theoretical_source_retention_ratio"] = None
    else:
        summary["theoretical_source_retention_ratio"] = retained_source / SEQ_LEN

    return summary


def run_expD_memory(results, config, device):
    print(
        "\n[Experiment D] symbolic key-value retrieval | forced | "
        f"window={WINDOW_EXPD}"
    )

    num_steps = int(config["num_steps"])
    eval_tail = int(config["eval_tail"])

    for model_name, model_class in config["model_items"]:
        for seed in config["seeds"]:
            for delay_len in config["delay_list"]:
                for write_mode in config["write_modes"]:
                    mem_grid = get_mem_grid(
                        write_mode=write_mode,
                        delay_len=delay_len,
                        config=config,
                    )

                    for max_mem in mem_grid:
                        if max_mem <= 0:
                            continue

                        if write_mode == WRITE_SOURCE_PINNED and max_mem < SEQ_LEN:
                            continue

                        set_seed(seed)
                        data_gen = torch.Generator().manual_seed(seed)

                        memory = WindowKVBuffer(max_mem)
                        model = model_class(
                            memory=memory,
                            mask_mode="forced",
                            write_mode=write_mode,
                        )

                        loss_history, acc_history = train(
                            model=model,
                            window_size=WINDOW_EXPD,
                            delay_len=delay_len,
                            data_gen=data_gen,
                            batch_fn=generate_batch_expD,
                            num_steps=num_steps,
                            device=device,
                        )

                        summary = summarize_loss(loss_history, eval_tail)
                        summary.update(summarize_acc(acc_history, eval_tail))

                        summary.update({
                            "phase": "expD_symbolic_kv",
                            "task": "symbolic_key_value_retrieval",
                            "model": model_name,
                            "seed": seed,
                            "window_size": WINDOW_EXPD,
                            "delay_len": delay_len,
                            "max_mem": max_mem,
                            "mask_mode": "forced",
                            "write_mode": write_mode,
                            "eval_tail": eval_tail,
                        })

                        summary = add_expd_memory_diagnostics(
                            summary=summary,
                            write_mode=write_mode,
                            max_mem=max_mem,
                            delay_len=delay_len,
                        )

                        results.append(summary)

                        noise_budget = summary.get("noise_write_budget", None)
                        noise_txt = "-" if noise_budget is None else str(noise_budget)

                        print(
                            f"{model_name:8} | seed={seed} | delay={delay_len:<3} | "
                            f"write={write_mode:25} | mem={max_mem:<3} | "
                            f"noise_budget={noise_txt:<3} | "
                            f"retained_source={summary['theoretical_retained_source']:<3} | "
                            f"acc_mean={summary['acc_mean_tail']:.3f} | "
                            f"acc_std={summary['acc_std_tail']:.3f}"
                        )

                        del model
                        if device.type == "cuda":
                            torch.cuda.empty_cache()


def run_experiments(config, output_path, device, overwrite=False):
    total_runs = validate_run_config(config)
    output_path = validate_output_contract(config["mode"], output_path)
    device = torch.device(device)
    prepare_output_path(output_path, overwrite)

    results = []

    print(
        f"mode={config['mode']} | device={device} | total_runs={total_runs}\n"
        f"model parameters: nhead={NHEAD} | num_layers={NUM_LAYERS} | "
        f"model_dim={MODEL_DIM} | seq_len={SEQ_LEN} | "
        f"num_steps={config['num_steps']} | eval_tail={config['eval_tail']}\n"
        f"seeds={config['seeds']} | delays={config['delay_list']} | "
        f"models={[x[0] for x in config['model_items']]}"
    )

    run_expD_memory(
        results=results,
        config=config,
        device=device,
    )

    model_counts = Counter([r.get("model") for r in results])
    write_counts = Counter([r.get("write_mode") for r in results])

    print("\n[Summary] Experiment D results composition before saving:")
    print("model:", dict(model_counts))
    print("write_mode:", dict(write_counts))

    saved_path = save_results_csv(
        results,
        output_path,
        overwrite=overwrite,
    )
    print(f"\nSaved results: {saved_path}")

    return results


def main(argv=None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    config = get_run_config(args.mode)
    output_path = resolve_output_path(args.mode, args.output)
    device = resolve_device(args.device)

    return run_experiments(
        config=config,
        output_path=output_path,
        device=device,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
