# Experiment B diagnostic figures and derived tables.
# Input: Results/expB/results_expB_paper.csv
# Outputs: 12 PNG figures and two derived CSV tables.

from pathlib import Path
import argparse

import pandas as pd
import matplotlib.pyplot as plt

from DelayedCopyTask.config import (
    SEQ_LEN,
    WRITE_SOURCE_ONLY,
    WRITE_PREFIX_ALL,
    WRITE_SOURCE_PINNED,
)


# Plot settings.

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "Results" / "expB"
FIGURES_DIR = PROJECT_ROOT / "Figures" / "expB"
DEFAULT_CSV = RESULTS_DIR / "results_expB_paper.csv"
DEFAULT_FIGURE_DIR = FIGURES_DIR
DEFAULT_DERIVED_DIR = RESULTS_DIR

TAU = 0.95

# Font sizes account for LaTeX figure scaling.
FIG_TITLE_SIZE = 18
AX_TITLE_SIZE = 16
LABEL_SIZE = 15
TICK_SIZE = 13
LEGEND_SIZE = 13
LEGEND_TITLE_SIZE = 13

LINE_WIDTH = 2.2
MARKER_SIZE = 5.5
GRID_ALPHA = 0.25

AX_TITLE_PAD = 3
FIG_TITLE_Y = 0.965
LAYOUT_TOP = 0.90

WRITE_MODE_ORDER = [
    WRITE_SOURCE_ONLY,
    WRITE_PREFIX_ALL,
    WRITE_SOURCE_PINNED,
]

EXPECTED_MODELS = ("naive", "gated")
EXPECTED_SEEDS = tuple(range(7))
EXPECTED_DELAYS = (40, 80)
EXPECTED_ROW_COUNT = 476
EXPECTED_COLUMN_COUNT = 47
EXPECTED_PHASE = "expB_memory_diagnostics"
EXPECTED_RUN_MODE = "main"
EXPECTED_MASK_MODE = "forced"
EXPECTED_NUM_STEPS = 3001
EXPECTED_EVAL_TAIL = 500
EXPECTED_EVAL_BATCHES = 16
EXPECTED_WINDOW_SIZE = 8

EXPECTED_MEMORY_GRIDS = {
    WRITE_SOURCE_ONLY: {
        40: (10, 15, 20, 25, 30),
        80: (10, 15, 20, 25, 30),
    },
    WRITE_PREFIX_ALL: {
        40: (50, 60, 61, 62, 70),
        80: (80, 100, 101, 102, 120),
    },
    WRITE_SOURCE_PINNED: {
        40: (20, 25, 30, 40, 60, 80, 100),
        80: (20, 25, 30, 40, 60, 80, 100),
    },
}

WRITE_MODE_TITLES = {
    WRITE_SOURCE_ONLY: "source-only",
    WRITE_PREFIX_ALL: "prefix-all",
    WRITE_SOURCE_PINNED: "source-pinned",
}

GROUP_COLS = [
    "model",
    "write_mode",
    "delay_len",
    "max_mem",
]

DIAGNOSTIC_METRIC_COLS = [
    "eval_acc",
    "acc_mean_tail",
    "source_survival_rate",
    "attn_mass_source",
    "attn_mass_noise",
    "attn_mass_sep",
    "attn_mass_other",
    "attn_entropy",
    "mem_source_entries",
    "mem_noise_entries",
    "mem_sep_entries",
    "eval_error_rate",
    "err_sep_rate",
    "err_source_only_rate",
    "err_noise_only_rate",
    "err_both_source_and_noise_rate",
    "err_other_rate",
]

EXPECTED_COLUMNS = (
    "acc_best",
    "acc_last",
    "acc_mean_tail",
    "acc_std_tail",
    "attn_entropy",
    "attn_mass_noise",
    "attn_mass_other",
    "attn_mass_sep",
    "attn_mass_source",
    "delay_len",
    "err_both_source_and_noise",
    "err_both_source_and_noise_rate",
    "err_noise_only",
    "err_noise_only_rate",
    "err_other",
    "err_other_rate",
    "err_sep",
    "err_sep_rate",
    "err_source_only",
    "err_source_only_rate",
    "eval_acc",
    "eval_batches",
    "eval_correct_tokens",
    "eval_error_rate",
    "eval_tail",
    "eval_total_tokens",
    "eval_wrong_tokens",
    "gate_max",
    "gate_mean",
    "gate_min",
    "loss_best",
    "loss_mean_tail",
    "loss_std_tail",
    "mask_mode",
    "max_mem",
    "mem_entries_total",
    "mem_noise_entries",
    "mem_sep_entries",
    "mem_source_entries",
    "model",
    "num_steps",
    "phase",
    "run_mode",
    "seed",
    "source_survival_rate",
    "window_size",
    "write_mode",
)


# Data loading and validation.

def resolve_project_path(path) -> Path:
    """Resolve relative command-line paths against the project root."""
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def load_results(csv_path=DEFAULT_CSV) -> pd.DataFrame:
    """Load the explicitly selected Experiment-B result CSV."""
    csv_path = resolve_project_path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Experiment-B result CSV not found: {csv_path}")

    print(f"Loading: {csv_path.resolve()}")

    df = pd.read_csv(csv_path)

    for column in ["model", "write_mode", "phase", "run_mode", "mask_mode"]:
        if column in df.columns:
            df[column] = df[column].astype(str).str.strip()

    return df


def validate_paper_data(df: pd.DataFrame) -> None:
    """Validate the complete frozen Experiment-B paper protocol."""
    observed_shape = tuple(df.shape)
    expected_shape = (EXPECTED_ROW_COUNT, EXPECTED_COLUMN_COUNT)
    if observed_shape != expected_shape:
        raise ValueError(
            "Experiment-B paper CSV has the wrong shape: "
            f"expected {expected_shape}, got {observed_shape}."
        )

    observed_columns = tuple(df.columns)
    if observed_columns != EXPECTED_COLUMNS:
        missing_cols = sorted(set(EXPECTED_COLUMNS) - set(observed_columns))
        unexpected_cols = sorted(set(observed_columns) - set(EXPECTED_COLUMNS))
        raise ValueError(
            "Experiment-B paper CSV columns do not match the frozen schema. "
            f"Missing: {missing_cols}; unexpected: {unexpected_cols}; "
            "column order must also match the paper CSV."
        )

    integer_columns = [
        "seed",
        "delay_len",
        "max_mem",
        "num_steps",
        "eval_tail",
        "eval_batches",
        "window_size",
    ]
    for column in integer_columns:
        values = pd.to_numeric(df[column], errors="coerce")
        invalid = values.isna() | (values % 1 != 0)
        if invalid.any():
            raise ValueError(
                f"Experiment-B paper CSV contains invalid integer values in {column}."
            )
        df[column] = values.astype(int)

    expected_models = set(EXPECTED_MODELS)
    observed_models = set(df["model"])
    missing_models = sorted(expected_models - observed_models)
    unexpected_models = sorted(observed_models - expected_models)
    if missing_models or unexpected_models:
        raise ValueError(
            "Experiment-B paper CSV has an invalid model set. "
            f"Missing: {missing_models}; unexpected: {unexpected_models}."
        )

    expected_policies = set(WRITE_MODE_ORDER)
    observed_policies = set(df["write_mode"])
    missing_policies = sorted(expected_policies - observed_policies)
    unexpected_policies = sorted(observed_policies - expected_policies)
    if missing_policies or unexpected_policies:
        raise ValueError(
            "Experiment-B paper CSV has an invalid write-policy set. "
            f"Missing: {missing_policies}; unexpected: {unexpected_policies}."
        )

    exact_value_contract = {
        "seed": set(EXPECTED_SEEDS),
        "delay_len": set(EXPECTED_DELAYS),
        "phase": {EXPECTED_PHASE},
        "run_mode": {EXPECTED_RUN_MODE},
        "mask_mode": {EXPECTED_MASK_MODE},
        "num_steps": {EXPECTED_NUM_STEPS},
        "eval_tail": {EXPECTED_EVAL_TAIL},
        "eval_batches": {EXPECTED_EVAL_BATCHES},
        "window_size": {EXPECTED_WINDOW_SIZE},
    }
    for column, expected_values in exact_value_contract.items():
        observed_values = set(df[column])
        if observed_values != expected_values:
            raise ValueError(
                f"Experiment-B paper CSV has invalid {column} values: "
                f"expected {sorted(expected_values)}, "
                f"got {sorted(observed_values)}."
            )

    key_columns = ["seed", "model", "delay_len", "write_mode", "max_mem"]
    duplicate_mask = df.duplicated(subset=key_columns, keep=False)
    if duplicate_mask.any():
        duplicate_count = int(duplicate_mask.sum())
        raise ValueError(
            "Experiment-B paper CSV contains duplicate protocol keys "
            f"({duplicate_count} rows involved): {key_columns}."
        )

    grid_errors = []
    for seed in EXPECTED_SEEDS:
        for model in EXPECTED_MODELS:
            for delay_len in EXPECTED_DELAYS:
                for write_mode in WRITE_MODE_ORDER:
                    rows = df[
                        (df["seed"] == seed)
                        & (df["model"] == model)
                        & (df["delay_len"] == delay_len)
                        & (df["write_mode"] == write_mode)
                    ]
                    observed_grid = tuple(sorted(rows["max_mem"].tolist()))
                    expected_grid = EXPECTED_MEMORY_GRIDS[write_mode][delay_len]
                    if observed_grid != expected_grid:
                        grid_errors.append(
                            f"seed={seed}, model={model}, delay={delay_len}, "
                            f"write_mode={write_mode}: expected {expected_grid}, "
                            f"got {observed_grid}"
                        )

    if grid_errors:
        details = "\n".join(f"- {error}" for error in grid_errors)
        raise ValueError(
            "Experiment-B paper CSV has incomplete capacity grids:\n"
            f"{details}"
        )


def aggregate_results(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate results across seeds.
    """
    metric_cols = [c for c in DIAGNOSTIC_METRIC_COLS if c in df.columns]

    agg = (
        df.groupby(GROUP_COLS)[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )

    # Flatten aggregation column labels for CSV output.
    flat_cols = []
    for col in agg.columns:
        if isinstance(col, tuple):
            if col[1] == "":
                flat_cols.append(col[0])
            else:
                flat_cols.append(f"{col[0]}_{col[1]}")
        else:
            flat_cols.append(col)

    agg.columns = flat_cols

    return agg


def build_multiseed_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build an appendix-ready long table of per-configuration seed statistics.

    Each row contains one diagnostic metric for one model/write/delay/capacity
    configuration. ``n_seeds`` counts the distinct seeds with a non-missing
    value for that metric.
    """
    metric_cols = [c for c in DIAGNOSTIC_METRIC_COLS if c in df.columns]
    stat_frames = []

    for metric in metric_cols:
        metric_df = df[GROUP_COLS + ["seed", metric]].dropna(subset=[metric])
        if metric_df.empty:
            continue

        metric_stats = (
            metric_df.groupby(GROUP_COLS, as_index=False)
            .agg(
                mean=(metric, "mean"),
                std=(metric, "std"),
                min=(metric, "min"),
                max=(metric, "max"),
                n_seeds=("seed", "nunique"),
            )
        )
        metric_stats.insert(len(GROUP_COLS), "metric", metric)
        stat_frames.append(metric_stats)

    columns = GROUP_COLS + ["metric", "mean", "std", "min", "max", "n_seeds"]
    if not stat_frames:
        return pd.DataFrame(columns=columns)

    return (
        pd.concat(stat_frames, ignore_index=True)
        .sort_values(GROUP_COLS + ["metric"])
        .reset_index(drop=True)
    )


# Plot helpers.

def _get_metric_cols(metric_name):
    mean_col = f"{metric_name}_mean"
    std_col = f"{metric_name}_std"
    return mean_col, std_col


def _sorted_unique(values):
    return sorted(pd.Series(values).dropna().unique().tolist())


FIGURE_NAME_TEMPLATES = (
    "expB_source_survival_{model}.png",
    "expB_eval_accuracy_{model}.png",
    "expB_prefix_all_attention_mass_{model}.png",
    "expB_source_pinned_attention_mass_{model}.png",
    "expB_survival_vs_accuracy_{model}.png",
    "expB_error_type_rates_{model}.png",
)

DERIVED_FILENAMES = (
    "expB_aggregated_summary.csv",
    "expB_multiseed_stats.csv",
)


def expected_output_paths(figure_dir: Path, derived_dir: Path):
    """Return the twelve figure and two derived-table targets."""
    paths = [
        figure_dir / template.format(model=model)
        for template in FIGURE_NAME_TEMPLATES
        for model in EXPECTED_MODELS
    ]
    paths.extend(derived_dir / filename for filename in DERIVED_FILENAMES)
    return paths


def preflight_outputs(paths, input_csv: Path, overwrite: bool) -> None:
    """Reject duplicate targets and existing files before writing anything."""
    normalized = [Path(path).resolve() for path in paths]
    if len(normalized) != 14:
        raise ValueError(
            f"Expected fourteen Experiment-B outputs, got {len(normalized)}."
        )
    if len(normalized) != len(set(normalized)):
        raise ValueError("Experiment-B output paths contain duplicate targets.")

    resolved_input = Path(input_csv).resolve()
    if resolved_input in normalized:
        raise ValueError(
            "The Experiment-B input CSV cannot also be an output target: "
            f"{resolved_input}. This is forbidden even with --overwrite."
        )

    conflicts = [path for path in normalized if path.exists()]
    if conflicts and not overwrite:
        listed = "\n".join(f"- {path}" for path in conflicts)
        raise FileExistsError(
            "Refusing to overwrite existing Experiment-B outputs:\n"
            f"{listed}\nUse --overwrite to replace these exact files."
        )


def verify_outputs_created(paths) -> None:
    """Confirm that all fourteen promised outputs were created as files."""
    normalized = [Path(path).resolve() for path in paths]
    missing = [path for path in normalized if not path.is_file()]
    if missing:
        listed = "\n".join(f"- {path}" for path in missing)
        raise RuntimeError(
            "Experiment-B plotting finished without all fourteen outputs:\n"
            f"{listed}"
        )
    print("Verified: all fourteen Experiment-B outputs exist.")


def prepare_output(path: Path, overwrite: bool) -> Path:
    """Create the parent directory and enforce single-file overwrite policy."""
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing output: {path}. "
            "Use --overwrite to replace it."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_png(fig, out_path: Path, overwrite: bool) -> None:
    out_path = prepare_output(out_path, overwrite)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.03)


def _format_ax(ax):
    ax.tick_params(axis="both", labelsize=TICK_SIZE, pad=2)
    ax.grid(True, alpha=GRID_ALPHA)


def _set_axis_labels(ax, xlabel=None, ylabel=None):
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=LABEL_SIZE)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=LABEL_SIZE)


def _set_ax_title(ax, title):
    ax.set_title(title, fontsize=AX_TITLE_SIZE, pad=AX_TITLE_PAD)


def _set_fig_title(fig, title):
    fig.suptitle(title, fontsize=FIG_TITLE_SIZE, y=FIG_TITLE_Y)


def _tight_layout_with_title(fig):
    """
    Use a unified layout for figures with a whole-figure title.
    This makes title-to-plot spacing more consistent.
    """
    fig.tight_layout(
        rect=[0, 0, 1, LAYOUT_TOP],
        pad=0.4,
        h_pad=0.5,
        w_pad=0.6,
    )


def _tight_layout_with_title_and_bottom_legend(fig):
    """Reserve space for both the figure title and a shared bottom legend."""
    fig.tight_layout(
        rect=[0, 0.12, 1, LAYOUT_TOP],
        pad=0.4,
        h_pad=0.5,
        w_pad=0.6,
    )


def _legend(ax, **kwargs):
    leg = ax.legend(
        fontsize=LEGEND_SIZE,
        title_fontsize=LEGEND_TITLE_SIZE,
        **kwargs,
    )
    return leg


def _plot_line_with_error(ax, x, y, yerr=None, label=None, marker="o"):
    if yerr is not None:
        yerr = yerr.fillna(0.0)

    ax.errorbar(
        x,
        y,
        yerr=yerr,
        marker=marker,
        markersize=MARKER_SIZE,
        capsize=3,
        linewidth=LINE_WIDTH,
        label=label,
    )


# Source survival versus memory capacity.

def plot_source_survival(
    agg: pd.DataFrame,
    figure_dir: Path,
    overwrite: bool,
):
    """
    Plot source survival rate as a function of memory capacity.
    One figure per model.
    Three panels: source-only, prefix-all, source-pinned.
    """
    metric = "source_survival_rate"
    mean_col, std_col = _get_metric_cols(metric)

    for model in EXPECTED_MODELS:
        df_m = agg[agg["model"] == model]

        fig, axes = plt.subplots(
            1,
            3,
            figsize=(15, 4),
            sharey=True,
        )

        for ax, write_mode in zip(axes, WRITE_MODE_ORDER):
            df_w = df_m[df_m["write_mode"] == write_mode]

            for delay in _sorted_unique(df_w["delay_len"]):
                df_d = df_w[df_w["delay_len"] == delay].sort_values("max_mem")

                if df_d.empty:
                    continue

                _plot_line_with_error(
                    ax=ax,
                    x=df_d["max_mem"],
                    y=df_d[mean_col],
                    yerr=df_d[std_col] if std_col in df_d else None,
                    label=f"d={delay}",
                )

            ax.axhline(1.0, linestyle="--", linewidth=1.2)
            ax.axhline(0.0, linestyle="--", linewidth=1.2)
            _set_ax_title(ax, WRITE_MODE_TITLES.get(write_mode, write_mode))
            _set_axis_labels(ax, xlabel="Memory capacity m")
            ax.set_ylim(-0.05, 1.05)
            _format_ax(ax)

        _set_axis_labels(axes[0], ylabel="Source survival rate")
        _legend(axes[-1], title="Delay", loc="best")

        _set_fig_title(fig, f"Source survival rate ({model})")
        _tight_layout_with_title(fig)

        out_path = figure_dir / f"expB_source_survival_{model}.png"
        save_png(fig, out_path, overwrite)
        plt.close(fig)

        print(f"Saved: {out_path.resolve()}")


# Evaluation accuracy versus memory capacity.

def plot_eval_accuracy(
    agg: pd.DataFrame,
    figure_dir: Path,
    overwrite: bool,
):
    """
    Plot eval accuracy as a function of memory capacity.
    One figure per model.
    Three panels: source-only, prefix-all, source-pinned.
    """
    metric = "eval_acc"
    mean_col, std_col = _get_metric_cols(metric)

    for model in EXPECTED_MODELS:
        df_m = agg[agg["model"] == model]

        fig, axes = plt.subplots(
            1,
            3,
            figsize=(15, 4),
            sharey=True,
        )

        for ax, write_mode in zip(axes, WRITE_MODE_ORDER):
            df_w = df_m[df_m["write_mode"] == write_mode]

            for delay in _sorted_unique(df_w["delay_len"]):
                df_d = df_w[df_w["delay_len"] == delay].sort_values("max_mem")

                if df_d.empty:
                    continue

                _plot_line_with_error(
                    ax=ax,
                    x=df_d["max_mem"],
                    y=df_d[mean_col],
                    yerr=df_d[std_col] if std_col in df_d else None,
                    label=f"d={delay}",
                )

            ax.axhline(TAU, linestyle="--", linewidth=1.2)
            _set_ax_title(ax, WRITE_MODE_TITLES.get(write_mode, write_mode))
            _set_axis_labels(ax, xlabel="Memory capacity m")
            ax.set_ylim(-0.05, 1.05)
            _format_ax(ax)

        _set_axis_labels(axes[0], ylabel="Eval copy accuracy")
        _legend(axes[-1], title="Delay", loc="best")

        _set_fig_title(fig, f"Eval accuracy ({model})")
        _tight_layout_with_title(fig)

        out_path = figure_dir / f"expB_eval_accuracy_{model}.png"
        save_png(fig, out_path, overwrite)
        plt.close(fig)

        print(f"Saved: {out_path.resolve()}")


# Prefix-all attention mass.

def plot_prefix_all_attention_mass(
    agg: pd.DataFrame,
    figure_dir: Path,
    overwrite: bool,
):
    """
    For prefix-all, plot attention mass to source / sep / noise.
    One figure per model.
    Panels correspond to delay lengths.
    """
    required = [
        "attn_mass_source_mean",
        "attn_mass_source_std",
        "attn_mass_noise_mean",
        "attn_mass_noise_std",
        "attn_mass_sep_mean",
        "attn_mass_sep_std",
    ]
    missing = [c for c in required if c not in agg.columns]
    if missing:
        raise ValueError(
            "Cannot create prefix-all attention figures; missing aggregated "
            f"columns: {missing}"
        )

    for model in EXPECTED_MODELS:
        df_m = agg[
            (agg["model"] == model)
            & (agg["write_mode"] == WRITE_PREFIX_ALL)
        ]

        delays = _sorted_unique(df_m["delay_len"])

        if not delays:
            raise ValueError(f"No prefix-all rows found for model={model}.")

        n_cols = 2
        n_rows = (len(delays) + 1) // 2

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(10, 3.8 * n_rows),
            sharey=True,
        )

        if n_rows == 1:
            axes = axes.reshape(1, -1)

        axes_flat = axes.flatten()

        for ax, delay in zip(axes_flat, delays):
            df_d = df_m[df_m["delay_len"] == delay].sort_values("max_mem")

            _plot_line_with_error(
                ax=ax,
                x=df_d["max_mem"],
                y=df_d["attn_mass_source_mean"],
                yerr=df_d["attn_mass_source_std"],
                marker="o",
                label="source",
            )
            _plot_line_with_error(
                ax=ax,
                x=df_d["max_mem"],
                y=df_d["attn_mass_noise_mean"],
                yerr=df_d["attn_mass_noise_std"],
                marker="s",
                label="noise",
            )
            _plot_line_with_error(
                ax=ax,
                x=df_d["max_mem"],
                y=df_d["attn_mass_sep_mean"],
                yerr=df_d["attn_mass_sep_std"],
                marker="^",
                label="sep",
            )

            expected_boundary = SEQ_LEN + 1 + int(delay)
            ax.axvline(expected_boundary, linestyle="--", linewidth=1.2)

            _set_ax_title(ax, f"prefix-all, d={delay}")
            _set_axis_labels(ax, xlabel="Memory capacity m")
            ax.set_ylim(-0.05, 1.05)
            _format_ax(ax)

        for ax in axes_flat[len(delays):]:
            ax.axis("off")

        _set_axis_labels(axes_flat[0], ylabel="Attention mass")

        legend_handles, legend_labels = axes_flat[0].get_legend_handles_labels()
        fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=len(legend_labels),
            fontsize=LEGEND_SIZE,
            frameon=True,
            handlelength=1.8,
            handletextpad=0.5,
            columnspacing=1.2,
            borderpad=0.3,
        )

        _set_fig_title(fig, f"Prefix-all attention mass ({model})")
        _tight_layout_with_title_and_bottom_legend(fig)

        out_path = figure_dir / f"expB_prefix_all_attention_mass_{model}.png"
        save_png(fig, out_path, overwrite)
        plt.close(fig)

        print(f"Saved: {out_path.resolve()}")


# Source-pinned attention mass.

def plot_source_pinned_attention_mass(
    agg: pd.DataFrame,
    figure_dir: Path,
    overwrite: bool,
):
    """
    For source-pinned noise-FIFO, plot attention mass against noise budget q.
    q = max_mem - SEQ_LEN.
    """
    required = [
        "attn_mass_source_mean",
        "attn_mass_noise_mean",
    ]
    missing = [c for c in required if c not in agg.columns]
    if missing:
        raise ValueError(
            "Cannot create source-pinned attention figures; missing "
            f"aggregated columns: {missing}"
        )

    for model in EXPECTED_MODELS:
        df_m = agg[
            (agg["model"] == model)
            & (agg["write_mode"] == WRITE_SOURCE_PINNED)
        ].copy()

        if df_m.empty:
            raise ValueError(f"No source-pinned rows found for model={model}.")

        df_m["noise_budget"] = df_m["max_mem"] - SEQ_LEN

        fig, axes = plt.subplots(
            1,
            2,
            figsize=(11, 4),
            sharey=True,
        )

        for delay in _sorted_unique(df_m["delay_len"]):
            df_d = df_m[df_m["delay_len"] == delay].sort_values("noise_budget")

            axes[0].plot(
                df_d["noise_budget"],
                df_d["attn_mass_source_mean"],
                marker="o",
                markersize=MARKER_SIZE,
                linewidth=LINE_WIDTH,
                label=f"d={delay}",
            )

            axes[1].plot(
                df_d["noise_budget"],
                df_d["attn_mass_noise_mean"],
                marker="s",
                markersize=MARKER_SIZE,
                linewidth=LINE_WIDTH,
                label=f"d={delay}",
            )

        _set_ax_title(axes[0], "Attention to source")
        _set_ax_title(axes[1], "Attention to noise")

        for ax in axes:
            _set_axis_labels(ax, xlabel="Admitted noise budget q")
            ax.set_ylim(-0.05, 1.05)
            _format_ax(ax)

        _set_axis_labels(axes[0], ylabel="Attention mass")
        _legend(axes[1], title="Delay", loc="best")

        _set_fig_title(fig, f"Source-pinned attention mass ({model})")
        _tight_layout_with_title(fig)

        out_path = figure_dir / f"expB_source_pinned_attention_mass_{model}.png"
        save_png(fig, out_path, overwrite)
        plt.close(fig)

        print(f"Saved: {out_path.resolve()}")


# Source survival versus evaluation accuracy.

def plot_survival_vs_accuracy(
    agg: pd.DataFrame,
    figure_dir: Path,
    overwrite: bool,
):
    """Plot evaluation accuracy against source survival across configurations."""
    required = [
        "source_survival_rate_mean",
        "source_survival_rate_std",
        "eval_acc_mean",
        "eval_acc_std",
    ]
    missing = [c for c in required if c not in agg.columns]
    if missing:
        raise ValueError(
            "Cannot create survival-versus-accuracy figures; missing "
            f"aggregated columns: {missing}"
        )

    for model in EXPECTED_MODELS:
        df_m = agg[agg["model"] == model]

        fig, ax = plt.subplots(figsize=(6, 5))

        markers = {
            WRITE_SOURCE_ONLY: "o",
            WRITE_PREFIX_ALL: "s",
            WRITE_SOURCE_PINNED: "^",
        }

        for write_mode in WRITE_MODE_ORDER:
            df_w = df_m[df_m["write_mode"] == write_mode]

            if df_w.empty:
                continue

            ax.errorbar(
                df_w["source_survival_rate_mean"],
                df_w["eval_acc_mean"],
                xerr=df_w["source_survival_rate_std"].fillna(0.0),
                yerr=df_w["eval_acc_std"].fillna(0.0),
                fmt=markers.get(write_mode, "o"),
                linestyle="none",
                markersize=MARKER_SIZE + 2.0,
                capsize=3,
                elinewidth=1.2,
                label=WRITE_MODE_TITLES.get(write_mode, write_mode),
                alpha=0.8,
            )

        ax.axhline(TAU, linestyle="--", linewidth=1.2)
        ax.axvline(1.0, linestyle="--", linewidth=1.2)

        _set_axis_labels(
            ax,
            xlabel="Source survival rate",
            ylabel="Eval copy accuracy",
        )
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        _format_ax(ax)
        _legend(ax, loc="best")

        _set_fig_title(fig, f"Source survival vs accuracy ({model})")
        _tight_layout_with_title(fig)

        out_path = figure_dir / f"expB_survival_vs_accuracy_{model}.png"
        save_png(fig, out_path, overwrite)
        plt.close(fig)

        print(f"Saved: {out_path.resolve()}")


# Error-type rates.

def plot_error_type_rates(
    agg: pd.DataFrame,
    figure_dir: Path,
    overwrite: bool,
):
    """
    Optional plot for error type analysis.
    """
    error_cols = [
        "err_sep_rate_mean",
        "err_source_only_rate_mean",
        "err_noise_only_rate_mean",
        "err_both_source_and_noise_rate_mean",
        "err_other_rate_mean",
    ]

    if not all(c in agg.columns for c in error_cols):
        missing = [column for column in error_cols if column not in agg.columns]
        raise ValueError(
            "Cannot create error-type figures; missing aggregated columns: "
            f"{missing}"
        )

    for model in EXPECTED_MODELS:
        df_m = agg[
            (agg["model"] == model)
            & (agg["write_mode"] == WRITE_PREFIX_ALL)
        ]

        if df_m.empty:
            raise ValueError(f"No prefix-all error rows found for model={model}.")

        delays = _sorted_unique(df_m["delay_len"])

        fig, axes = plt.subplots(
            1,
            len(delays),
            figsize=(4.2 * len(delays), 4),
            sharey=True,
        )

        if len(delays) == 1:
            axes = [axes]

        for ax, delay in zip(axes, delays):
            df_d = df_m[df_m["delay_len"] == delay].sort_values("max_mem")

            ax.plot(
                df_d["max_mem"],
                df_d["err_sep_rate_mean"],
                marker="o",
                markersize=MARKER_SIZE,
                linewidth=LINE_WIDTH,
                label="sep",
            )
            ax.plot(
                df_d["max_mem"],
                df_d["err_source_only_rate_mean"],
                marker="s",
                markersize=MARKER_SIZE,
                linewidth=LINE_WIDTH,
                label="source-only",
            )
            ax.plot(
                df_d["max_mem"],
                df_d["err_noise_only_rate_mean"],
                marker="^",
                markersize=MARKER_SIZE,
                linewidth=LINE_WIDTH,
                label="noise-only",
            )
            ax.plot(
                df_d["max_mem"],
                df_d["err_both_source_and_noise_rate_mean"],
                marker="x",
                markersize=MARKER_SIZE + 1,
                linewidth=LINE_WIDTH,
                label="both",
            )
            ax.plot(
                df_d["max_mem"],
                df_d["err_other_rate_mean"],
                marker="d",
                markersize=MARKER_SIZE,
                linewidth=LINE_WIDTH,
                label="other",
            )

            _set_ax_title(ax, f"prefix-all, d={delay}")
            _set_axis_labels(ax, xlabel="Memory capacity m")
            ax.set_ylim(-0.05, 1.05)
            _format_ax(ax)

        _set_axis_labels(axes[0], ylabel="Rate among errors")
        _legend(axes[-1], loc="best")

        _set_fig_title(fig, f"Error type rates ({model})")
        _tight_layout_with_title(fig)

        out_path = figure_dir / f"expB_error_type_rates_{model}.png"
        save_png(fig, out_path, overwrite)
        plt.close(fig)

        print(f"Saved: {out_path.resolve()}")


# Derived summary table.

def save_aggregated_table(
    agg: pd.DataFrame,
    derived_dir: Path,
    overwrite: bool,
):
    out_path = prepare_output(
        derived_dir / "expB_aggregated_summary.csv",
        overwrite,
    )
    agg.to_csv(out_path, index=False)
    print(f"Saved aggregated table: {out_path.resolve()}")


def save_multiseed_stats(
    stats: pd.DataFrame,
    derived_dir: Path,
    overwrite: bool,
):
    out_path = prepare_output(
        derived_dir / "expB_multiseed_stats.csv",
        overwrite,
    )
    stats.to_csv(out_path, index=False)
    print(f"Saved multi-seed stats: {out_path.resolve()}")



def main(
    csv_path=DEFAULT_CSV,
    figure_dir=DEFAULT_FIGURE_DIR,
    derived_dir=DEFAULT_DERIVED_DIR,
    overwrite=False,
):
    csv_path = resolve_project_path(csv_path)
    figure_dir = resolve_project_path(figure_dir)
    derived_dir = resolve_project_path(derived_dir)

    df = load_results(csv_path)
    validate_paper_data(df)

    output_paths = expected_output_paths(figure_dir, derived_dir)
    preflight_outputs(output_paths, csv_path, overwrite)

    agg = aggregate_results(df)
    multiseed_stats = build_multiseed_stats(df)

    save_aggregated_table(agg, derived_dir, overwrite)
    save_multiseed_stats(multiseed_stats, derived_dir, overwrite)

    plot_source_survival(agg, figure_dir, overwrite)
    plot_eval_accuracy(agg, figure_dir, overwrite)
    plot_prefix_all_attention_mass(agg, figure_dir, overwrite)
    plot_source_pinned_attention_mass(agg, figure_dir, overwrite)
    plot_survival_vs_accuracy(agg, figure_dir, overwrite)
    plot_error_type_rates(agg, figure_dir, overwrite)

    verify_outputs_created(output_paths)

    print("\nPlotting finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help="Input CSV (default: Results/expB/results_expB_paper.csv).",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=DEFAULT_FIGURE_DIR,
        help="PNG output directory (default: Figures/expB).",
    )
    parser.add_argument(
        "--derived-dir",
        type=Path,
        default=DEFAULT_DERIVED_DIR,
        help="Derived CSV output directory (default: Results/expB).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of the fourteen expected output files.",
    )
    args = parser.parse_args()

    main(
        csv_path=args.csv,
        figure_dir=args.figure_dir,
        derived_dir=args.derived_dir,
        overwrite=args.overwrite,
    )
