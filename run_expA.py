# Experiment A runner for the formal paper protocol.
# Smoke output is isolated under tmp/smoke.
import argparse
import torch
from collections import Counter
from pathlib import Path

from DelayedCopyTask.models import (
    CopyMaskTransformer,
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
    NHEAD,
    NUM_LAYERS,
    MODEL_DIM,
    WINDOW_SIZE,
    FORCED_WINDOW,
    NATURAL_SANITY_MEM,
    SEQ_LEN,
    DELAY_LEN,
    EVAL_TAIL,
    SEEDS_MAIN,
    NUM_STEPS,
    M_SOURCE_ONLY,
    M_PREFIX_ALL,
    M_PINNED,
    WRITE_SOURCE_ONLY,
    WRITE_PREFIX_ALL,
    WRITE_SOURCE_PINNED,
    WRITE_MODES_EXPA,
)

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "Results" / "expA"
DEFAULT_PAPER_OUTPUT = RESULTS_DIR / "results_expA_paper.csv"
DEFAULT_SMOKE_OUTPUT = (
    PROJECT_ROOT / "tmp" / "smoke" / "expA" / "results_expA_smoke.csv"
)


def resolve_project_path(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def resolve_device(requested):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        device = torch.device(requested)
    except (RuntimeError, TypeError) as exc:
        raise ValueError(f"Invalid device: {requested}") from exc

    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA was requested, but CUDA is not available.")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError(
                f"CUDA device index {device.index} is unavailable; "
                f"found {torch.cuda.device_count()} CUDA device(s)."
            )
    return device


def get_run_config(mode):
    model_items = [
        ("naive", CopyMaskTransformerWithNWMemory),
        ("gated", CopyMaskTransformerWithGWMemory),
    ]

    if mode == "paper":
        return {
            "seeds": SEEDS_MAIN,
            "window_list": WINDOW_SIZE,
            "delay_list": DELAY_LEN,
            "model_items": model_items,
            "write_modes": WRITE_MODES_EXPA,
            "run_natural_sanity": True,
            "num_steps": NUM_STEPS,
            "eval_tail": EVAL_TAIL,
            "mem_grids": {
                WRITE_SOURCE_ONLY: M_SOURCE_ONLY,
                WRITE_PREFIX_ALL: M_PREFIX_ALL,
                WRITE_SOURCE_PINNED: M_PINNED,
            },
        }

    if mode == "smoke":
        return {
            "seeds": [0],
            "window_list": [FORCED_WINDOW],
            "delay_list": [10],
            "model_items": model_items,
            "write_modes": WRITE_MODES_EXPA,
            "run_natural_sanity": False,
            "num_steps": 2,
            "eval_tail": 2,
            "mem_grids": {
                WRITE_SOURCE_ONLY: [20],
                WRITE_PREFIX_ALL: [31],
                WRITE_SOURCE_PINNED: [25],
            },
        }

    raise ValueError(f"Invalid mode: {mode}")


def get_mem_grid(write_mode, mem_grids):
    try:
        return mem_grids[write_mode]
    except KeyError as exc:
        raise ValueError(f"Invalid write_mode: {write_mode}") from exc


def empty_cuda_cache(device):
    if device.type == "cuda":
        torch.cuda.empty_cache()

def add_memory_diagnostics(summary, write_mode, max_mem, delay_len):
    """Add theoretical memory-retention fields without another forward pass."""
    if write_mode == WRITE_SOURCE_ONLY:
        retained_source = min(max_mem, SEQ_LEN)
        summary["noise_write_budget"] = None

    elif write_mode == WRITE_PREFIX_ALL:
        # Prefix-all writes [source, SEP, noise]; FIFO retains the newest entries.
        # Source retention is therefore max_mem - (1 + delay_len), capped at SEQ_LEN.
        retained_source = max(0, min(SEQ_LEN, max_mem - (1 + delay_len)))
        summary["noise_write_budget"] = None

    elif write_mode == WRITE_SOURCE_PINNED:
        retained_source = SEQ_LEN
        summary["noise_write_budget"] = max_mem - SEQ_LEN

    else:
        retained_source = None
        summary["noise_write_budget"] = None

    summary["theoretical_retained_source"] = retained_source
    if retained_source is None:
        summary["theoretical_source_retention_ratio"] = None
    else:
        summary["theoretical_source_retention_ratio"] = retained_source / SEQ_LEN

    return summary

def run_natural_sanity(
    results,
    seeds,
    window_list,
    delay_list,
    num_steps,
    eval_tail,
    device,
):
    print("\n[Phase 1] baseline | natural | scan window x delay x seed")
    for seed in seeds:
        for window_size in window_list:
            for delay_len in delay_list:
                set_seed(seed)
                data_gen = torch.Generator().manual_seed(seed)

                model = CopyMaskTransformer(mask_mode="natural")
                loss_history, acc_history = train(
                    model,
                    window_size,
                    delay_len,
                    data_gen,
                    num_steps=num_steps,
                    device=device,
                )

                tail = eval_tail
                summary = summarize_loss(loss_history, tail)
                summary.update(summarize_acc(acc_history, tail))
                summary.update({
                    "phase": "natural_baseline",
                    "model": "baseline",
                    "seed": seed,
                    "window_size": window_size,
                    "delay_len": delay_len,
                    "max_mem": 0,
                    "mask_mode": "natural",
                    "write_mode": "none",
                    "noise_write_budget": None,
                    "theoretical_retained_source": None,
                    "theoretical_source_retention_ratio": None,
                    "eval_tail": int(tail),
                })
                results.append(summary)

                print(
                    f"baseline | seed={seed} | window={window_size:<2} | delay={delay_len:<2} | "
                    f"mask=natural | write=none | mem={0:<3} | "
                    f"acc_mean={summary['acc_mean_tail']:.3f} | acc_std={summary['acc_std_tail']:.3f}"
                )

                del model
                empty_cuda_cache(device)

    print("\n[Phase 2] naive/gated | natural | write=source-only | mem=NATURAL_SANITY_MEM")
    for model_name, model_class in [
        ("naive", CopyMaskTransformerWithNWMemory),
        ("gated", CopyMaskTransformerWithGWMemory),
    ]:
        for seed in seeds:
            for window_size in window_list:
                for delay_len in delay_list:
                    set_seed(seed)
                    data_gen = torch.Generator().manual_seed(seed)

                    memory = WindowKVBuffer(NATURAL_SANITY_MEM)
                    model = model_class(
                        memory=memory,
                        mask_mode="natural",
                        write_mode=WRITE_SOURCE_ONLY,
                    )

                    loss_history, acc_history = train(
                        model,
                        window_size,
                        delay_len,
                        data_gen,
                        num_steps=num_steps,
                        device=device,
                    )

                    tail = eval_tail
                    summary = summarize_loss(loss_history, tail)
                    summary.update(summarize_acc(acc_history, tail))
                    summary.update({
                        "phase": "natural_memory",
                        "model": model_name,
                        "seed": seed,
                        "window_size": window_size,
                        "delay_len": delay_len,
                        "max_mem": NATURAL_SANITY_MEM,
                        "mask_mode": "natural",
                        "write_mode": WRITE_SOURCE_ONLY,
                        "noise_write_budget": None,
                        "theoretical_retained_source": SEQ_LEN,
                        "theoretical_source_retention_ratio": 1.0,
                        "eval_tail": int(tail),
                    })
                    results.append(summary)

                    print(
                        f"{model_name:8} | seed={seed} | window={window_size:<2} | delay={delay_len:<2} | "
                        f"mask=natural | write=source-only | mem={NATURAL_SANITY_MEM:<3} | "
                        f"acc_mean={summary['acc_mean_tail']:.3f} | acc_std={summary['acc_std_tail']:.3f}"
                    )

                    del model
                    empty_cuda_cache(device)


def run_forced_memory(
    results,
    seeds,
    delay_list,
    model_items,
    write_modes,
    mem_grids,
    num_steps,
    eval_tail,
    device,
):
    print("\n[Phase 3 / Experiment A] memory models | forced | window=FORCED_WINDOW")

    for model_name, model_class in model_items:
        for seed in seeds:
            for delay_len in delay_list:
                for write_mode in write_modes:
                    mem_grid = get_mem_grid(write_mode, mem_grids)

                    for max_mem in mem_grid:
                        # External-memory models require positive capacity.
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
                            model,
                            FORCED_WINDOW,
                            delay_len,
                            data_gen,
                            num_steps=num_steps,
                            device=device,
                        )

                        tail = eval_tail
                        summary = summarize_loss(loss_history, tail)
                        summary.update(summarize_acc(acc_history, tail))
                        summary.update({
                            "phase": "forced_memory",
                            "model": model_name,
                            "seed": seed,
                            "window_size": FORCED_WINDOW,
                            "delay_len": delay_len,
                            "max_mem": max_mem,
                            "mask_mode": "forced",
                            "write_mode": write_mode,
                            "eval_tail": int(tail),
                        })
                        summary = add_memory_diagnostics(
                            summary,
                            write_mode=write_mode,
                            max_mem=max_mem,
                            delay_len=delay_len,
                        )
                        results.append(summary)

                        noise_budget = summary.get("noise_write_budget", None)
                        noise_txt = "-" if noise_budget is None else str(noise_budget)

                        print(
                            f"{model_name:8} | seed={seed} | window={FORCED_WINDOW:<2} | "
                            f"delay={delay_len:<2} | mask=forced | write={write_mode:25} | "
                            f"mem={max_mem:<3} | noise_budget={noise_txt:<3} | "
                            f"retained_source={summary['theoretical_retained_source']:<3} | "
                            f"acc_mean={summary['acc_mean_tail']:.3f} | acc_std={summary['acc_std_tail']:.3f}"
                        )

                        del model
                        empty_cuda_cache(device)

def run_experiments(mode, cfg, output_path, device, overwrite=False):
    results = []

    print(
        f"mode={mode} | device={device} | output={output_path}\n"
        f"model parameters: nhead={NHEAD} | num_layers={NUM_LAYERS} | "
        f"model_dim={MODEL_DIM} | seq_len={SEQ_LEN} | "
        f"num_steps={cfg['num_steps']} | eval_tail={cfg['eval_tail']}\n"
        f"seeds={cfg['seeds']} | delays={cfg['delay_list']}"
    )

    if cfg["run_natural_sanity"]:
        run_natural_sanity(
            results=results,
            seeds=cfg["seeds"],
            window_list=cfg["window_list"],
            delay_list=cfg["delay_list"],
            num_steps=cfg["num_steps"],
            eval_tail=cfg["eval_tail"],
            device=device,
        )

    run_forced_memory(
        results=results,
        seeds=cfg["seeds"],
        delay_list=cfg["delay_list"],
        model_items=cfg["model_items"],
        write_modes=cfg["write_modes"],
        mem_grids=cfg["mem_grids"],
        num_steps=cfg["num_steps"],
        eval_tail=cfg["eval_tail"],
        device=device,
    )

    mask_counts = Counter([r.get("mask_mode") for r in results])
    model_counts = Counter([r.get("model") for r in results])
    write_counts = Counter([r.get("write_mode") for r in results])

    print("\n[DEBUG] Results composition before saving:")
    print("mask_mode:", dict(mask_counts))
    print("model:", dict(model_counts))
    print("write_mode:", dict(write_counts))

    saved_path = save_results_csv(
        results,
        output_path,
        overwrite=overwrite,
    )
    print(f"\nSaved results: {saved_path}")
    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Experiment A delayed-copy diagnostics."
    )
    parser.add_argument(
        "--mode",
        choices=("paper", "smoke"),
        default="paper",
        help="paper reproduces the full protocol; smoke runs six tiny checks.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path relative to the project root, or an absolute path.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, or a specific CUDA device such as cuda:0.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of the explicitly selected output CSV.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = get_run_config(args.mode)
    device = resolve_device(args.device)

    if args.output is None:
        output_path = (
            DEFAULT_PAPER_OUTPUT
            if args.mode == "paper"
            else DEFAULT_SMOKE_OUTPUT
        )
    else:
        output_path = resolve_project_path(args.output)

    if output_path.suffix.lower() != ".csv":
        raise ValueError(f"--output must end in .csv: {output_path}")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing output before training: {output_path}. "
            "Choose another --output path or pass --overwrite explicitly."
        )

    run_experiments(
        mode=args.mode,
        cfg=cfg,
        output_path=output_path,
        device=device,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
