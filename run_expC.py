# Experiment C runner for controlled intermediate writes.
# Paper mode uses the formal protocol; smoke output is isolated under tmp/smoke.
# Source and SEP are always written, with round(p * delay_len) admitted noise states.
# Expected full-source FIFO boundary: L + 1 + round(p * d).

import argparse
from pathlib import Path

import torch

from DelayedCopyTask.models import (
    CopyMaskTransformerWithNWMemory,
    CopyMaskTransformerWithGWMemory,
)
from DelayedCopyTask.utils import (
    set_seed,
    summarize_loss,
    summarize_acc,
    save_results_csv,
)
from DelayedCopyTask.train import train
from DelayedCopyTask.memory import WindowKVBuffer
from DelayedCopyTask.config import (
    FORCED_WINDOW,
    SEQ_LEN,
    EVAL_TAIL,
    NUM_STEPS,
    SEEDS_MAIN,
    CONTAMINATION_P,
    DELAY_LEN_EXPC,
    WRITE_INTERMEDIATE,
)


PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "Results" / "expC"
PAPER_OUTPUT_PATH = RESULTS_DIR / "results_expC_paper.csv"
SMOKE_OUTPUT_PATH = (
    PROJECT_ROOT / "tmp" / "smoke" / "expC" / "results_expC_smoke.csv"
)


def get_run_config(mode):
    if mode == "paper":
        return {
            "mode": "paper",
            "seeds": SEEDS_MAIN,
            "delay_list": DELAY_LEN_EXPC,
            "p_list": CONTAMINATION_P,
            "model_items": [
                ("naive", CopyMaskTransformerWithNWMemory),
                ("gated", CopyMaskTransformerWithGWMemory),
            ],
            "num_steps": NUM_STEPS,
            "eval_tail": EVAL_TAIL,
            "boundary_only": False,
        }

    if mode == "smoke":
        return {
            "mode": "smoke",
            "seeds": [0],
            "delay_list": [20],
            "p_list": [0.0, 0.5, 1.0],
            "model_items": [
                ("naive", CopyMaskTransformerWithNWMemory),
                ("gated", CopyMaskTransformerWithGWMemory),
            ],
            "num_steps": 2,
            "eval_tail": 2,
            "boundary_only": True,
        }

    raise ValueError(f"Invalid mode: {mode}")


def get_noise_write_count(delay_len, p):
    count = int(round(float(p) * delay_len))
    return max(0, min(count, delay_len))


def get_expected_boundary(delay_len, p):
    noise_count = get_noise_write_count(delay_len, p)
    return SEQ_LEN + 1 + noise_count


def get_mem_grid(delay_len, p):
    """Return a compact capacity grid around L + 1 + round(p * d)."""
    expected = get_expected_boundary(delay_len, p)

    candidates = [
        10,
        20,
        expected - 2,
        expected - 1,
        expected,
        expected + 1,
        expected + 2,
        expected + 10,
        expected + 20,
    ]

    candidates = [m for m in candidates if m > 0]
    return sorted(set(candidates))


def get_configured_mem_grid(config, delay_len, p):
    if config["boundary_only"]:
        return [get_expected_boundary(delay_len, p)]
    return get_mem_grid(delay_len, p)


def count_experiment_runs(config):
    runs_per_model_and_seed = 0
    for delay_len in config["delay_list"]:
        for p in config["p_list"]:
            runs_per_model_and_seed += len(
                get_configured_mem_grid(config, delay_len, p)
            )

    return (
        len(config["model_items"])
        * len(config["seeds"])
        * runs_per_model_and_seed
    )


def validate_run_config(config):
    if config.get("mode") not in {"paper", "smoke"}:
        raise ValueError(f"Invalid mode: {config.get('mode')}")

    for name in ("seeds", "delay_list", "p_list", "model_items"):
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
            "p_list": [0.0, 0.25, 0.5, 0.75, 1.0],
            "model_items": [
                ("naive", CopyMaskTransformerWithNWMemory),
                ("gated", CopyMaskTransformerWithGWMemory),
            ],
            "num_steps": 3001,
            "eval_tail": 500,
            "boundary_only": False,
        }

        for name, expected_value in expected.items():
            if config[name] != expected_value:
                raise ValueError(
                    f"Experiment C paper protocol mismatch for {name}: "
                    f"expected {expected_value!r}, got {config[name]!r}"
                )

        expected_grids = {
            (20, 0.0): [10, 19, 20, 21, 22, 23, 31, 41],
            (20, 0.25): [10, 20, 24, 25, 26, 27, 28, 36, 46],
            (20, 0.5): [10, 20, 29, 30, 31, 32, 33, 41, 51],
            (20, 0.75): [10, 20, 34, 35, 36, 37, 38, 46, 56],
            (20, 1.0): [10, 20, 39, 40, 41, 42, 43, 51, 61],
            (40, 0.0): [10, 19, 20, 21, 22, 23, 31, 41],
            (40, 0.25): [10, 20, 29, 30, 31, 32, 33, 41, 51],
            (40, 0.5): [10, 20, 39, 40, 41, 42, 43, 51, 61],
            (40, 0.75): [10, 20, 49, 50, 51, 52, 53, 61, 71],
            (40, 1.0): [10, 20, 59, 60, 61, 62, 63, 71, 81],
            (80, 0.0): [10, 19, 20, 21, 22, 23, 31, 41],
            (80, 0.25): [10, 20, 39, 40, 41, 42, 43, 51, 61],
            (80, 0.5): [10, 20, 59, 60, 61, 62, 63, 71, 81],
            (80, 0.75): [10, 20, 79, 80, 81, 82, 83, 91, 101],
            (80, 1.0): [10, 20, 99, 100, 101, 102, 103, 111, 121],
        }
        actual_grids = {
            (delay_len, p): get_mem_grid(delay_len, p)
            for delay_len in config["delay_list"]
            for p in config["p_list"]
        }
        if actual_grids != expected_grids:
            raise ValueError("Experiment C paper capacity grids have changed")

        if FORCED_WINDOW != 8:
            raise ValueError(
                f"Experiment C paper protocol requires window 8, got {FORCED_WINDOW}"
            )
        if WRITE_INTERMEDIATE != "source-sep-noise-budget":
            raise ValueError(
                "Experiment C paper write policy must be "
                "source-sep-noise-budget"
            )
        if total_runs != 1848:
            raise ValueError(
                "Experiment C paper protocol must contain 1848 runs, "
                f"got {total_runs}"
            )

    if config["mode"] == "smoke":
        expected = {
            "seeds": [0],
            "delay_list": [20],
            "p_list": [0.0, 0.5, 1.0],
            "model_items": [
                ("naive", CopyMaskTransformerWithNWMemory),
                ("gated", CopyMaskTransformerWithGWMemory),
            ],
            "num_steps": 2,
            "eval_tail": 2,
            "boundary_only": True,
        }
        for name, expected_value in expected.items():
            if config[name] != expected_value:
                raise ValueError(
                    f"Experiment C smoke protocol mismatch for {name}: "
                    f"expected {expected_value!r}, got {config[name]!r}"
                )
        if total_runs != 6:
            raise ValueError(
                f"Experiment C smoke protocol must contain 6 runs, got {total_runs}"
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
            "Only Experiment C paper mode may write to "
            f"{PAPER_OUTPUT_PATH}"
        )
    if mode == "paper" and resolved_output == SMOKE_OUTPUT_PATH.resolve():
        raise ValueError(
            "Experiment C paper mode may not write to the reserved smoke path "
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
        description="Run Experiment C continuous-contamination diagnostics."
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


def add_expc_diagnostics(summary, max_mem, delay_len, p):
    noise_count = get_noise_write_count(delay_len, p)
    expected_non_source_writes = 1 + noise_count
    expected_boundary = SEQ_LEN + expected_non_source_writes

    retained_source = max(
        0,
        min(
            SEQ_LEN,
            max_mem - expected_non_source_writes,
        ),
    )

    summary["noise_write_ratio"] = float(p)
    summary["noise_write_count"] = int(noise_count)
    summary["expected_non_source_writes"] = int(expected_non_source_writes)
    summary["expected_boundary"] = int(expected_boundary)
    summary["theoretical_retained_source"] = int(retained_source)
    summary["theoretical_source_retention_ratio"] = retained_source / float(SEQ_LEN)

    return summary


def run_expC(results, config, device):
    print("\n[Experiment C] intermediate write policy | forced | window=FORCED_WINDOW")
    print(f"write_mode={WRITE_INTERMEDIATE}")

    num_steps = int(config["num_steps"])
    eval_tail = int(config["eval_tail"])

    for model_name, model_class in config["model_items"]:
        for seed in config["seeds"]:
            for delay_len in config["delay_list"]:
                for p in config["p_list"]:
                    mem_grid = get_configured_mem_grid(config, delay_len, p)
                    expected_boundary = get_expected_boundary(delay_len, p)
                    noise_count = get_noise_write_count(delay_len, p)

                    for max_mem in mem_grid:
                        set_seed(seed)
                        data_gen = torch.Generator().manual_seed(seed)

                        memory = WindowKVBuffer(max_mem)
                        model = model_class(
                            memory=memory,
                            mask_mode="forced",
                            write_mode=WRITE_INTERMEDIATE,
                            noise_write_ratio=p,
                        )

                        loss_history, acc_history = train(
                            model=model,
                            window_size=FORCED_WINDOW,
                            delay_len=delay_len,
                            data_gen=data_gen,
                            num_steps=num_steps,
                            device=device,
                        )

                        summary = summarize_loss(loss_history, eval_tail)
                        summary.update(summarize_acc(acc_history, eval_tail))
                        summary.update({
                            "phase": "expC_intermediate",
                            "model": model_name,
                            "seed": seed,
                            "window_size": FORCED_WINDOW,
                            "delay_len": delay_len,
                            "max_mem": max_mem,
                            "mask_mode": "forced",
                            "write_mode": WRITE_INTERMEDIATE,
                            "eval_tail": eval_tail,
                        })

                        summary = add_expc_diagnostics(
                            summary,
                            max_mem=max_mem,
                            delay_len=delay_len,
                            p=p,
                        )

                        results.append(summary)

                        print(
                            f"{model_name:8} | seed={seed} | delay={delay_len:<3} | "
                            f"p={p:<4} | noise_count={noise_count:<3} | "
                            f"mem={max_mem:<3} | expected_boundary={expected_boundary:<3} | "
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
        f"mode={config['mode']} | device={device} | "
        f"num_steps={config['num_steps']} | eval_tail={config['eval_tail']} | "
        f"total_runs={total_runs}"
    )

    run_expC(
        results=results,
        config=config,
        device=device,
    )

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
