# Plot intermediate write-policy and continuous-contamination results.

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.transforms import ScaledTranslation

from DelayedCopyTask.config import (
    SEQ_LEN,
    TAU_MAIN,
)


PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "Results" / "expC"
FIGURES_DIR = PROJECT_ROOT / "Figures" / "expC"
DEFAULT_CSV = RESULTS_DIR / "results_expC_paper.csv"
DEFAULT_FIGURE_DIR = FIGURES_DIR
DEFAULT_DERIVED_DIR = RESULTS_DIR

EXPECTED_MODELS = ("naive", "gated")
EXPECTED_SEEDS = tuple(range(7))
EXPECTED_DELAYS = (20, 40, 80)
EXPECTED_NOISE_RATIOS = (0.0, 0.25, 0.5, 0.75, 1.0)
EXPECTED_ROW_COUNT = 1848
EXPECTED_COLUMN_COUNT = 22
EXPECTED_SOURCE_LEN = 20
EXPECTED_TAU = 0.95
EXPECTED_PHASE = "expC_intermediate"
EXPECTED_MASK_MODE = "forced"
EXPECTED_WINDOW_SIZE = 8
EXPECTED_EVAL_TAIL = 500
EXPECTED_WRITE_MODE = "source-sep-noise-budget"

EXPECTED_COLUMNS = (
    "acc_best",
    "acc_last",
    "acc_mean_tail",
    "acc_std_tail",
    "delay_len",
    "eval_tail",
    "expected_boundary",
    "expected_non_source_writes",
    "loss_best",
    "loss_mean_tail",
    "loss_std_tail",
    "mask_mode",
    "max_mem",
    "model",
    "noise_write_count",
    "noise_write_ratio",
    "phase",
    "seed",
    "theoretical_retained_source",
    "theoretical_source_retention_ratio",
    "window_size",
    "write_mode",
)


# Plot settings for paper figures.

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
ANNOTATION_SIZE = 10

plt.rcParams.update({
    "font.size": LABEL_SIZE,
    "axes.titlesize": AX_TITLE_SIZE,
    "axes.labelsize": LABEL_SIZE,
    "xtick.labelsize": TICK_SIZE,
    "ytick.labelsize": TICK_SIZE,
    "legend.fontsize": LEGEND_SIZE,
    "figure.titlesize": FIG_TITLE_SIZE,
    "lines.linewidth": LINE_WIDTH,
    "axes.titlepad": AX_TITLE_PAD,
})


def resolve_project_path(path) -> Path:
    """Resolve relative command-line paths against the project root."""
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def prepare_output(path: Path, overwrite: bool) -> Path:
    """Create a parent directory while enforcing the overwrite contract."""
    path = Path(path)
    if path.exists() and path.is_dir():
        raise IsADirectoryError(
            f"Experiment-C output target is an existing directory: {path}"
        )
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing output: {path}. "
            "Use --overwrite to replace it."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_png(fig, out_path: Path, overwrite: bool, dpi=300):
    """Save a tightly cropped PNG while enforcing the overwrite policy."""
    out_path = prepare_output(out_path, overwrite)
    fig.savefig(
        out_path,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.03,
    )


def expected_noise_write_count(delay_len, noise_write_ratio):
    count = int(round(float(noise_write_ratio) * int(delay_len)))
    return max(0, min(count, int(delay_len)))


def expected_memory_grid(delay_len, noise_write_ratio):
    expected = (
        EXPECTED_SOURCE_LEN
        + 1
        + expected_noise_write_count(delay_len, noise_write_ratio)
    )
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
    return tuple(sorted({capacity for capacity in candidates if capacity > 0}))


def load_results(csv_path=DEFAULT_CSV):
    csv_path = resolve_project_path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Experiment-C result CSV not found: {csv_path}")

    print(f"Loading: {csv_path}")

    df = pd.read_csv(csv_path)

    for column in ["model", "phase", "mask_mode", "write_mode"]:
        if column in df.columns:
            df[column] = df[column].astype(str).str.strip()

    return df, csv_path


def validate_paper_data(df):
    """Validate the complete frozen Experiment-C paper protocol."""
    if SEQ_LEN != EXPECTED_SOURCE_LEN or float(TAU_MAIN) != EXPECTED_TAU:
        raise ValueError(
            "Experiment-C config does not match the paper protocol: "
            f"expected SEQ_LEN={EXPECTED_SOURCE_LEN}, TAU_MAIN={EXPECTED_TAU}; "
            f"got SEQ_LEN={SEQ_LEN}, TAU_MAIN={TAU_MAIN}."
        )

    observed_shape = tuple(df.shape)
    expected_shape = (EXPECTED_ROW_COUNT, EXPECTED_COLUMN_COUNT)
    if observed_shape != expected_shape:
        raise ValueError(
            "Experiment-C paper CSV has the wrong shape: "
            f"expected {expected_shape}, got {observed_shape}."
        )

    observed_columns = tuple(df.columns)
    if observed_columns != EXPECTED_COLUMNS:
        missing = sorted(set(EXPECTED_COLUMNS) - set(observed_columns))
        unexpected = sorted(set(observed_columns) - set(EXPECTED_COLUMNS))
        raise ValueError(
            "Experiment-C paper CSV columns do not match the frozen schema. "
            f"Missing: {missing}; unexpected: {unexpected}; "
            "column order must also match the paper CSV."
        )

    integer_columns = [
        "delay_len",
        "eval_tail",
        "expected_boundary",
        "expected_non_source_writes",
        "max_mem",
        "noise_write_count",
        "seed",
        "theoretical_retained_source",
        "window_size",
    ]
    for column in integer_columns:
        values = pd.to_numeric(df[column], errors="coerce")
        invalid = values.isna() | (values % 1 != 0)
        if invalid.any():
            raise ValueError(
                f"Experiment-C paper CSV contains invalid integer values in {column}."
            )
        df[column] = values.astype(int)

    float_columns = [
        "acc_best",
        "acc_last",
        "acc_mean_tail",
        "acc_std_tail",
        "loss_best",
        "loss_mean_tail",
        "loss_std_tail",
        "noise_write_ratio",
        "theoretical_source_retention_ratio",
    ]
    for column in float_columns:
        values = pd.to_numeric(df[column], errors="coerce")
        if values.isna().any():
            raise ValueError(
                f"Experiment-C paper CSV contains invalid numeric values in {column}."
            )
        df[column] = values.astype(float)

    exact_value_contract = {
        "seed": set(EXPECTED_SEEDS),
        "model": set(EXPECTED_MODELS),
        "delay_len": set(EXPECTED_DELAYS),
        "noise_write_ratio": set(EXPECTED_NOISE_RATIOS),
        "phase": {EXPECTED_PHASE},
        "mask_mode": {EXPECTED_MASK_MODE},
        "window_size": {EXPECTED_WINDOW_SIZE},
        "eval_tail": {EXPECTED_EVAL_TAIL},
        "write_mode": {EXPECTED_WRITE_MODE},
    }
    for column, expected_values in exact_value_contract.items():
        observed_values = set(df[column])
        if observed_values != expected_values:
            raise ValueError(
                f"Experiment-C paper CSV has invalid {column} values: "
                f"expected {sorted(expected_values)}, "
                f"got {sorted(observed_values)}."
            )

    key_columns = [
        "seed",
        "model",
        "delay_len",
        "noise_write_ratio",
        "max_mem",
    ]
    duplicate_mask = df.duplicated(subset=key_columns, keep=False)
    if duplicate_mask.any():
        duplicate_count = int(duplicate_mask.sum())
        raise ValueError(
            "Experiment-C paper CSV contains duplicate protocol keys "
            f"({duplicate_count} rows involved): {key_columns}."
        )

    grid_errors = []
    for seed in EXPECTED_SEEDS:
        for model in EXPECTED_MODELS:
            for delay_len in EXPECTED_DELAYS:
                for noise_write_ratio in EXPECTED_NOISE_RATIOS:
                    rows = df[
                        (df["seed"] == seed)
                        & (df["model"] == model)
                        & (df["delay_len"] == delay_len)
                        & (df["noise_write_ratio"] == noise_write_ratio)
                    ]
                    observed_grid = tuple(sorted(rows["max_mem"].tolist()))
                    expected_grid = expected_memory_grid(
                        delay_len,
                        noise_write_ratio,
                    )
                    if observed_grid != expected_grid:
                        grid_errors.append(
                            f"seed={seed}, model={model}, delay={delay_len}, "
                            f"p={noise_write_ratio}: expected {expected_grid}, "
                            f"got {observed_grid}"
                        )

    if grid_errors:
        details = "\n".join(f"- {error}" for error in grid_errors)
        raise ValueError(
            "Experiment-C paper CSV has incomplete capacity grids:\n"
            f"{details}"
        )

    diagnostic_errors = []
    for row in df.itertuples(index=False):
        noise_count = expected_noise_write_count(
            row.delay_len,
            row.noise_write_ratio,
        )
        non_source_writes = 1 + noise_count
        boundary = EXPECTED_SOURCE_LEN + non_source_writes
        retained_source = max(
            0,
            min(EXPECTED_SOURCE_LEN, row.max_mem - non_source_writes),
        )
        retention_ratio = retained_source / float(EXPECTED_SOURCE_LEN)

        mismatches = []
        if row.noise_write_count != noise_count:
            mismatches.append("noise_write_count")
        if row.expected_non_source_writes != non_source_writes:
            mismatches.append("expected_non_source_writes")
        if row.expected_boundary != boundary:
            mismatches.append("expected_boundary")
        if row.theoretical_retained_source != retained_source:
            mismatches.append("theoretical_retained_source")
        if abs(row.theoretical_source_retention_ratio - retention_ratio) > 1e-12:
            mismatches.append("theoretical_source_retention_ratio")

        if mismatches:
            diagnostic_errors.append(
                f"seed={row.seed}, model={row.model}, delay={row.delay_len}, "
                f"p={row.noise_write_ratio}, max_mem={row.max_mem}: "
                f"{mismatches}"
            )

    if diagnostic_errors:
        details = "\n".join(f"- {error}" for error in diagnostic_errors)
        raise ValueError(
            "Experiment-C paper CSV has inconsistent diagnostic fields:\n"
            f"{details}"
        )


def expected_output_paths(figure_dir, derived_dir):
    paths = [
        figure_dir / f"expC_intermediate_boundary_{model}.png"
        for model in EXPECTED_MODELS
    ]
    paths.extend([
        derived_dir / "expC_mean_accuracy_by_capacity.csv",
        derived_dir / "expC_boundary_summary.csv",
        derived_dir / "expC_per_seed_boundary_summary.csv",
    ])
    return paths


def preflight_outputs(paths, input_csv, overwrite):
    """Validate all five targets before any output is written."""
    normalized = [Path(path).resolve() for path in paths]
    if len(normalized) != 5:
        raise ValueError(
            f"Expected five Experiment-C outputs, got {len(normalized)}."
        )
    if len(normalized) != len(set(normalized)):
        raise ValueError("Experiment-C output paths contain duplicate targets.")

    resolved_input = Path(input_csv).resolve()
    if resolved_input in normalized:
        raise ValueError(
            "The Experiment-C input CSV cannot also be an output target: "
            f"{resolved_input}. This is forbidden even with --overwrite."
        )

    directory_targets = [
        path for path in normalized if path.exists() and path.is_dir()
    ]
    if directory_targets:
        listed = "\n".join(f"- {path}" for path in directory_targets)
        raise IsADirectoryError(
            "Experiment-C output targets cannot be existing directories, "
            "even with --overwrite:\n"
            f"{listed}"
        )

    conflicts = [path for path in normalized if path.exists()]
    if conflicts and not overwrite:
        listed = "\n".join(f"- {path}" for path in conflicts)
        raise FileExistsError(
            "Refusing to overwrite existing Experiment-C outputs:\n"
            f"{listed}\nUse --overwrite to replace these exact files."
        )


def validate_derived_row_counts(mean_df, boundary, per_seed_summary):
    expected_counts = {
        "expC_mean_accuracy_by_capacity.csv": 264,
        "expC_boundary_summary.csv": 30,
        "expC_per_seed_boundary_summary.csv": 30,
    }
    observed_counts = {
        "expC_mean_accuracy_by_capacity.csv": len(mean_df),
        "expC_boundary_summary.csv": len(boundary),
        "expC_per_seed_boundary_summary.csv": len(per_seed_summary),
    }
    mismatches = {
        name: (expected_counts[name], observed)
        for name, observed in observed_counts.items()
        if observed != expected_counts[name]
    }
    if mismatches:
        raise ValueError(
            "Experiment-C derived row counts do not match the paper protocol: "
            f"{mismatches}"
        )


def verify_outputs_created(paths):
    """Confirm that all five promised outputs were created as files."""
    normalized = [Path(path).resolve() for path in paths]
    missing = [path for path in normalized if not path.is_file()]
    if missing:
        listed = "\n".join(f"- {path}" for path in missing)
        raise RuntimeError(
            "Experiment-C plotting finished without all five outputs:\n"
            f"{listed}"
        )
    print("Verified: all five Experiment-C outputs exist.")


def compute_boundary(df):
    """Threshold seed-averaged ``acc_mean_tail`` to obtain aggregate boundaries."""
    group_cols = [
        "model",
        "delay_len",
        "noise_write_ratio",
        "max_mem",
        "expected_non_source_writes",
        "expected_boundary",
    ]

    mean_df = (
        df.groupby(group_cols, as_index=False)
        .agg(
            acc_mean=("acc_mean_tail", "mean"),
            acc_std=("acc_mean_tail", "std"),
            num_seeds=("seed", "nunique"),
        )
    )

    rows = []
    boundary_group_cols = [
        "model",
        "delay_len",
        "noise_write_ratio",
        "expected_non_source_writes",
        "expected_boundary",
    ]

    for keys, sub in mean_df.groupby(boundary_group_cols):
        model, delay_len, p, expected_non_source_writes, expected_boundary = keys

        sub = sub.sort_values("max_mem")
        reached = sub[sub["acc_mean"] >= TAU_MAIN]

        max_scanned = int(sub["max_mem"].max())

        if len(reached) == 0:
            censored = True
            m_star = None
            m_star_plot = max_scanned
            acc_at_m_star = None
        else:
            censored = False
            first = reached.iloc[0]
            m_star = int(first["max_mem"])
            m_star_plot = m_star
            acc_at_m_star = float(first["acc_mean"])

        rows.append({
            "model": model,
            "delay_len": int(delay_len),
            "noise_write_ratio": float(p),
            "expected_non_source_writes": int(expected_non_source_writes),
            "expected_boundary": int(expected_boundary),
            "m_star": m_star,
            "m_star_plot": int(m_star_plot),
            "censored": bool(censored),
            "max_scanned": max_scanned,
            "acc_at_m_star": acc_at_m_star,
        })

    boundary = pd.DataFrame(rows)
    boundary = boundary.sort_values(
        ["model", "delay_len", "noise_write_ratio"]
    ).reset_index(drop=True)

    return mean_df, boundary


def compute_per_seed_boundary_summary(df):
    """Compute seed-wise boundaries without treating censoring as a value.

    A seed is right-censored when none of its scanned capacities reaches the
    accuracy threshold.  Its boundary is therefore stored as NaN and excluded
    from the finite-boundary moments below; the censoring count and scan limit
    are retained explicitly.
    """
    group_cols = [
        "model",
        "delay_len",
        "noise_write_ratio",
        "expected_non_source_writes",
        "expected_boundary",
        "seed",
    ]

    rows = []
    for keys, sub in df.groupby(group_cols):
        model, delay_len, p, non_source_writes, expected_boundary, seed = keys
        sub = sub.sort_values("max_mem")
        reached = sub[sub["acc_mean_tail"] >= TAU_MAIN]
        max_scanned = int(sub["max_mem"].max())

        if reached.empty:
            m_star = float("nan")
            censored = True
        else:
            m_star = float(reached.iloc[0]["max_mem"])
            censored = False

        rows.append({
            "model": model,
            "delay_len": int(delay_len),
            "noise_write_ratio": float(p),
            "expected_non_source_writes": int(non_source_writes),
            "expected_boundary": int(expected_boundary),
            "seed": int(seed),
            "m_star": m_star,
            "censored": bool(censored),
            "max_scanned": max_scanned,
        })

    per_seed = pd.DataFrame(rows)
    summary_group_cols = [
        "model",
        "delay_len",
        "noise_write_ratio",
        "expected_non_source_writes",
        "expected_boundary",
    ]
    summary = (
        per_seed.groupby(summary_group_cols, as_index=False)
        .agg(
            mean=("m_star", "mean"),
            std=("m_star", "std"),
            min=("m_star", "min"),
            max=("m_star", "max"),
            uncensored_n=("m_star", "count"),
            censored_n=("censored", "sum"),
            total_seeds=("seed", "nunique"),
            max_scanned=("max_scanned", "max"),
        )
        .sort_values(["model", "delay_len", "noise_write_ratio"])
        .reset_index(drop=True)
    )
    summary["uncensored_n"] = summary["uncensored_n"].astype(int)
    summary["censored_n"] = summary["censored_n"].astype(int)
    summary["total_seeds"] = summary["total_seeds"].astype(int)
    return summary


def save_boundary_tables(
    mean_df,
    boundary,
    per_seed_summary,
    derived_dir,
    overwrite,
):
    mean_path = prepare_output(
        derived_dir / "expC_mean_accuracy_by_capacity.csv",
        overwrite,
    )
    boundary_path = prepare_output(
        derived_dir / "expC_boundary_summary.csv",
        overwrite,
    )
    per_seed_summary_path = prepare_output(
        derived_dir / "expC_per_seed_boundary_summary.csv",
        overwrite,
    )

    mean_df.to_csv(mean_path, index=False)
    boundary.to_csv(boundary_path, index=False)
    per_seed_summary.to_csv(per_seed_summary_path, index=False)

    print(
        f"Saved:\n- {mean_path}\n- {boundary_path}"
        f"\n- {per_seed_summary_path}"
    )


def plot_boundary(boundary, per_seed_summary, figure_dir, overwrite):
    for model in sorted(boundary["model"].unique()):
        sub_model = boundary[boundary["model"] == model].copy()
        delay_colors = {}
        # Display-coordinate lanes separate auxiliary indicators by delay
        # without changing their data coordinates.
        delay_lane_offsets_pt = {
            delay: (index - (len(sub_model["delay_len"].unique()) - 1) / 2.0) * 3.0
            for index, delay in enumerate(sorted(sub_model["delay_len"].unique()))
        }

        fig, ax = plt.subplots(figsize=(7.8, 4.8))

        for delay_len in sorted(sub_model["delay_len"].unique()):
            sub = sub_model[sub_model["delay_len"] == delay_len].sort_values(
                "expected_non_source_writes"
            )

            # A scan-limit cross marks aggregate censoring and is not included
            # in the finite measured-boundary line.
            finite_y = sub["m_star_plot"].where(~sub["censored"].astype(bool))
            line, = ax.plot(
                sub["expected_non_source_writes"],
                finite_y,
                marker="o",
                linewidth=LINE_WIDTH,
                markersize=MARKER_SIZE,
                label=f"d={delay_len}",
            )
            delay_colors[int(delay_len)] = line.get_color()

            for _, row in sub[sub["censored"].astype(bool)].iterrows():
                x = float(row["expected_non_source_writes"])
                marker_transform = (
                    ax.transData
                    + ScaledTranslation(
                        delay_lane_offsets_pt.get(int(delay_len), 0.0) / 72.0,
                        0.0,
                        fig.dpi_scale_trans,
                    )
                )
                y = float(row["max_scanned"])
                ax.scatter(
                    [x],
                    [y],
                    transform=marker_transform,
                    marker="x",
                    color=line.get_color(),
                    s=46,
                    linewidths=1.4,
                    zorder=8,
                )

        xs = sorted(sub_model["expected_non_source_writes"].unique())
        expected_y = [SEQ_LEN + x for x in xs]

        ax.plot(
            xs,
            expected_y,
            linestyle="--",
            linewidth=LINE_WIDTH,
            color="tab:red",
            label=r"expected $L + 1 + \mathrm{round}(pd)$",
        )

        ax.set_xlabel(
            r"Expected written non-source length $1 + \mathrm{round}(pd)$",
            fontsize=LABEL_SIZE,
        )
        ax.set_ylabel(
            r"Measured boundary $m^*(d,p)$",
            fontsize=LABEL_SIZE,
        )

        model_label = str(model).replace("_", "-")
        ax.set_title(
            f"Intermediate Write Policy Boundaries ({model_label})",
            fontsize=FIG_TITLE_SIZE,
            pad=AX_TITLE_PAD,
        )

        ax.tick_params(axis="both", labelsize=TICK_SIZE, pad=2)
        ax.grid(True, alpha=GRID_ALPHA)
        ax.legend(
            fontsize=LEGEND_SIZE,
            title_fontsize=LEGEND_TITLE_SIZE,
            frameon=True,
        )

        # Finite seed-wise ranges require a finite aggregate boundary and at
        # least two observed seed-wise boundaries; censoring is marked separately.
        seed_model = per_seed_summary[
            per_seed_summary["model"] == model
        ].copy()
        aggregate_lookup = {
            (int(row["delay_len"]), float(row["noise_write_ratio"])): row
            for _, row in sub_model.iterrows()
        }

        finite_ranges = seed_model[
            (seed_model["uncensored_n"] >= 2)
            & seed_model["min"].notna()
            & seed_model["max"].notna()
        ]
        for _, row in finite_ranges.iterrows():
            delay_len = int(row["delay_len"])
            aggregate_row = aggregate_lookup.get(
                (delay_len, float(row["noise_write_ratio"]))
            )
            if aggregate_row is None or bool(aggregate_row["censored"]):
                continue

            x = float(row["expected_non_source_writes"])
            y_min = float(row["min"])
            y_max = float(row["max"])
            y_mid = 0.5 * (y_min + y_max)
            color = delay_colors.get(delay_len, "black")
            lane_transform = (
                ax.transData
                + ScaledTranslation(
                    delay_lane_offsets_pt.get(delay_len, 0.0) / 72.0,
                    0.0,
                    fig.dpi_scale_trans,
                )
            )
            ax.errorbar(
                [x],
                [y_mid],
                yerr=[[y_mid - y_min], [y_max - y_mid]],
                fmt="none",
                transform=lane_transform,
                ecolor=color,
                elinewidth=0.8,
                capsize=2.5,
                capthick=0.8,
                alpha=0.45,
                zorder=6,
            )

        censored = seed_model[seed_model["censored_n"] > 0]
        for _, row in censored.iterrows():
            delay_len = int(row["delay_len"])
            aggregate_row = aggregate_lookup.get(
                (delay_len, float(row["noise_write_ratio"]))
            )
            if aggregate_row is None:
                continue

            x = float(row["expected_non_source_writes"])
            if bool(aggregate_row["censored"]):
                y = float(aggregate_row["max_scanned"])
            else:
                y = float(aggregate_row["m_star"])

            triangle_transform = (
                ax.transData
                + ScaledTranslation(
                    delay_lane_offsets_pt.get(delay_len, 0.0) / 72.0,
                    5.0 / 72.0,
                    fig.dpi_scale_trans,
                )
            )
            color = delay_colors.get(delay_len, "black")
            ax.scatter(
                [x],
                [y],
                transform=triangle_transform,
                marker="^",
                facecolors="none",
                edgecolors=color,
                s=18,
                linewidths=0.75,
                alpha=0.70,
                zorder=8,
            )

        # Preserve a small margin for the auxiliary display-coordinate lanes.
        ax.margins(x=0.06)
        fig.tight_layout()

        out_path = figure_dir / f"expC_intermediate_boundary_{model}.png"
        save_png(fig, out_path, overwrite=overwrite, dpi=300)
        plt.close(fig)

        print(f"Saved: {out_path}")


def main(
    csv_path=DEFAULT_CSV,
    figure_dir=DEFAULT_FIGURE_DIR,
    derived_dir=DEFAULT_DERIVED_DIR,
    overwrite=False,
):
    csv_path = resolve_project_path(csv_path)
    figure_dir = resolve_project_path(figure_dir)
    derived_dir = resolve_project_path(derived_dir)

    df, csv_path = load_results(csv_path)
    validate_paper_data(df)

    output_paths = expected_output_paths(figure_dir, derived_dir)
    preflight_outputs(output_paths, csv_path, overwrite)

    mean_df, boundary = compute_boundary(df)
    per_seed_summary = compute_per_seed_boundary_summary(df)
    validate_derived_row_counts(mean_df, boundary, per_seed_summary)

    save_boundary_tables(
        mean_df,
        boundary,
        per_seed_summary,
        derived_dir,
        overwrite,
    )
    plot_boundary(boundary, per_seed_summary, figure_dir, overwrite)

    verify_outputs_created(output_paths)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help="Input CSV (default: Results/expC/results_expC_paper.csv).",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=DEFAULT_FIGURE_DIR,
        help="PNG output directory (default: Figures/expC).",
    )
    parser.add_argument(
        "--derived-dir",
        type=Path,
        default=DEFAULT_DERIVED_DIR,
        help="Derived CSV output directory (default: Results/expC).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of the five expected output files.",
    )
    args = parser.parse_args()

    main(
        csv_path=args.csv,
        figure_dir=args.figure_dir,
        derived_dir=args.derived_dir,
        overwrite=args.overwrite,
    )
