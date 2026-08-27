import argparse
from pathlib import Path

import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.transforms import ScaledTranslation


# Shared experiment constants.
from DelayedCopyTask.config import (
    SEQ_LEN,
    TAU_MAIN,
    WRITE_SOURCE_ONLY,
    WRITE_PREFIX_ALL,
    WRITE_SOURCE_PINNED,
)


PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "Results" / "expD"
FIGURES_DIR = PROJECT_ROOT / "Figures" / "expD"
DEFAULT_CSV = RESULTS_DIR / "results_expD_paper.csv"
DEFAULT_FIGURE_DIR = FIGURES_DIR
DEFAULT_DERIVED_DIR = RESULTS_DIR

BOUNDARY_FIG_TEMPLATE = "expD_symbolic_boundary_{model}.png"

EXPECTED_MODELS = ("naive", "gated")
EXPECTED_SEEDS = tuple(range(7))
EXPECTED_DELAYS = (20, 40, 80)
EXPECTED_ROW_COUNT = 756
EXPECTED_COLUMN_COUNT = 21
EXPECTED_SOURCE_LEN = 20
EXPECTED_TAU = 0.95
EXPECTED_PHASE = "expD_symbolic_kv"
EXPECTED_TASK = "symbolic_key_value_retrieval"
EXPECTED_MASK_MODE = "forced"
EXPECTED_WINDOW_SIZE = 8
EXPECTED_EVAL_TAIL = 500

EXPECTED_COLUMNS = (
    "acc_best",
    "acc_last",
    "acc_mean_tail",
    "acc_std_tail",
    "delay_len",
    "eval_tail",
    "expected_boundary",
    "loss_best",
    "loss_mean_tail",
    "loss_std_tail",
    "mask_mode",
    "max_mem",
    "model",
    "noise_write_budget",
    "phase",
    "seed",
    "task",
    "theoretical_retained_source",
    "theoretical_source_retention_ratio",
    "window_size",
    "write_mode",
)


# Compact plot settings keep the longer legend clear of the prefix-all curve.
FIG_TITLE_SIZE = 18
AX_TITLE_SIZE = 16
LABEL_SIZE = 15
TICK_SIZE = 12
LEGEND_SIZE = 12
LEGEND_TITLE_SIZE = 11

LINE_WIDTH = 2.0
MARKER_SIZE = 5.0
GRID_ALPHA = 0.25

AX_TITLE_PAD = 2
ANNOTATION_SIZE = 8

FIGSIZE_BOUNDARY = (8.2, 4.8)

DPI = 300

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


# Display labels and styles.
WRITE_MODE_ORDER = [
    WRITE_SOURCE_ONLY,
    WRITE_PREFIX_ALL,
    WRITE_SOURCE_PINNED,
]

EXPECTED_MEMORY_GRIDS = {
    WRITE_SOURCE_ONLY: {
        20: (10, 15, 20, 25, 30),
        40: (10, 15, 20, 25, 30),
        80: (10, 15, 20, 25, 30),
    },
    WRITE_PREFIX_ALL: {
        20: (10, 20, 39, 40, 41, 42, 43, 51, 61),
        40: (10, 20, 59, 60, 61, 62, 63, 71, 81),
        80: (10, 20, 99, 100, 101, 102, 103, 111, 121),
    },
    WRITE_SOURCE_PINNED: {
        20: (20, 40, 60, 100),
        40: (20, 40, 60, 100),
        80: (20, 40, 60, 100),
    },
}

AGGREGATE_SUMMARY_COLUMNS = (
    "model",
    "write_mode",
    "delay_len",
    "tau",
    "m_star",
    "censored",
    "max_scanned",
    "expected_boundary",
)

WRITE_MODE_LABEL = {
    WRITE_SOURCE_ONLY: "source-only",
    WRITE_PREFIX_ALL: "prefix-all",
    WRITE_SOURCE_PINNED: "source-pinned",
}

WRITE_MODE_MARKER = {
    WRITE_SOURCE_ONLY: "o",
    WRITE_PREFIX_ALL: "s",
    WRITE_SOURCE_PINNED: "^",
}

WRITE_MODE_ZORDER = {
    WRITE_SOURCE_ONLY: 3,
    WRITE_PREFIX_ALL: 3,
    WRITE_SOURCE_PINNED: 4,
}


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
            f"Experiment-D output target is an existing directory: {path}"
        )
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing output: {path}. "
            "Use --overwrite to replace it."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_png(fig, out_path: Path, overwrite: bool, dpi=DPI):
    """Save a tightly cropped PNG while enforcing the overwrite policy."""
    out_path = prepare_output(out_path, overwrite)
    fig.savefig(
        out_path,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.03,
    )


def load_results(csv_path=DEFAULT_CSV):
    """Load the explicitly selected Experiment-D paper CSV."""
    csv_path = resolve_project_path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Experiment-D result CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    for column in ["model", "write_mode", "phase", "task", "mask_mode"]:
        if column in df.columns:
            df[column] = df[column].astype(str).str.strip()

    print(f"[ExpD plot] Loaded: {csv_path}")
    print("[ExpD plot] write_mode counts:")
    if "write_mode" in df.columns:
        print(df["write_mode"].value_counts())
    print()

    return df, csv_path


def validate_paper_data(df: pd.DataFrame) -> None:
    """Validate the complete frozen Experiment-D paper protocol."""
    if SEQ_LEN != EXPECTED_SOURCE_LEN or float(TAU_MAIN) != EXPECTED_TAU:
        raise ValueError(
            "Experiment-D config does not match the paper protocol: "
            f"expected SEQ_LEN={EXPECTED_SOURCE_LEN}, TAU_MAIN={EXPECTED_TAU}; "
            f"got SEQ_LEN={SEQ_LEN}, TAU_MAIN={TAU_MAIN}."
        )

    observed_shape = tuple(df.shape)
    expected_shape = (EXPECTED_ROW_COUNT, EXPECTED_COLUMN_COUNT)
    if observed_shape != expected_shape:
        raise ValueError(
            "Experiment-D paper CSV has the wrong shape: "
            f"expected {expected_shape}, got {observed_shape}."
        )

    observed_columns = tuple(df.columns)
    if observed_columns != EXPECTED_COLUMNS:
        missing = sorted(set(EXPECTED_COLUMNS) - set(observed_columns))
        unexpected = sorted(set(observed_columns) - set(EXPECTED_COLUMNS))
        raise ValueError(
            "Experiment-D paper CSV columns do not match the frozen schema. "
            f"Missing: {missing}; unexpected: {unexpected}; "
            "column order must also match the paper CSV."
        )

    integer_columns = [
        "delay_len",
        "eval_tail",
        "expected_boundary",
        "max_mem",
        "seed",
        "theoretical_retained_source",
        "window_size",
    ]
    for column in integer_columns:
        values = pd.to_numeric(df[column], errors="coerce")
        invalid = values.isna() | (values % 1 != 0)
        if invalid.any():
            raise ValueError(
                f"Experiment-D paper CSV contains invalid integer values in {column}."
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
        "theoretical_source_retention_ratio",
    ]
    for column in float_columns:
        values = pd.to_numeric(df[column], errors="coerce")
        if values.isna().any():
            raise ValueError(
                f"Experiment-D paper CSV contains invalid numeric values in {column}."
            )
        df[column] = values.astype(float)

    exact_value_contract = {
        "seed": set(EXPECTED_SEEDS),
        "model": set(EXPECTED_MODELS),
        "delay_len": set(EXPECTED_DELAYS),
        "write_mode": set(WRITE_MODE_ORDER),
        "phase": {EXPECTED_PHASE},
        "task": {EXPECTED_TASK},
        "mask_mode": {EXPECTED_MASK_MODE},
        "window_size": {EXPECTED_WINDOW_SIZE},
        "eval_tail": {EXPECTED_EVAL_TAIL},
    }
    for column, expected_values in exact_value_contract.items():
        observed_values = set(df[column])
        if observed_values != expected_values:
            raise ValueError(
                f"Experiment-D paper CSV has invalid {column} values: "
                f"expected {sorted(expected_values)}, "
                f"got {sorted(observed_values)}."
            )

    key_columns = ["seed", "model", "delay_len", "write_mode", "max_mem"]
    duplicate_mask = df.duplicated(subset=key_columns, keep=False)
    if duplicate_mask.any():
        duplicate_count = int(duplicate_mask.sum())
        raise ValueError(
            "Experiment-D paper CSV contains duplicate protocol keys "
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
            "Experiment-D paper CSV has incomplete capacity grids:\n"
            f"{details}"
        )

    diagnostic_errors = []
    for row in df.itertuples(index=False):
        if row.write_mode == WRITE_SOURCE_ONLY:
            expected_boundary = EXPECTED_SOURCE_LEN
            retained_source = min(row.max_mem, EXPECTED_SOURCE_LEN)
            noise_budget_valid = pd.isna(row.noise_write_budget)
        elif row.write_mode == WRITE_PREFIX_ALL:
            expected_boundary = EXPECTED_SOURCE_LEN + 1 + row.delay_len
            retained_source = max(
                0,
                min(
                    EXPECTED_SOURCE_LEN,
                    row.max_mem - (1 + row.delay_len),
                ),
            )
            noise_budget_valid = pd.isna(row.noise_write_budget)
        else:
            expected_boundary = EXPECTED_SOURCE_LEN
            retained_source = EXPECTED_SOURCE_LEN
            expected_noise_budget = row.max_mem - EXPECTED_SOURCE_LEN
            noise_budget = pd.to_numeric(
                pd.Series([row.noise_write_budget]),
                errors="coerce",
            ).iloc[0]
            noise_budget_valid = (
                not pd.isna(noise_budget)
                and float(noise_budget).is_integer()
                and int(noise_budget) == expected_noise_budget
            )

        retention_ratio = retained_source / float(EXPECTED_SOURCE_LEN)
        mismatches = []
        if row.expected_boundary != expected_boundary:
            mismatches.append("expected_boundary")
        if row.theoretical_retained_source != retained_source:
            mismatches.append("theoretical_retained_source")
        if abs(row.theoretical_source_retention_ratio - retention_ratio) > 1e-12:
            mismatches.append("theoretical_source_retention_ratio")
        if not noise_budget_valid:
            mismatches.append("noise_write_budget")

        if mismatches:
            diagnostic_errors.append(
                f"seed={row.seed}, model={row.model}, delay={row.delay_len}, "
                f"write_mode={row.write_mode}, max_mem={row.max_mem}: "
                f"{mismatches}"
            )

    if diagnostic_errors:
        details = "\n".join(f"- {error}" for error in diagnostic_errors)
        raise ValueError(
            "Experiment-D paper CSV has inconsistent diagnostic fields:\n"
            f"{details}"
        )


def expected_output_paths(figure_dir, derived_dir):
    paths = [
        figure_dir / BOUNDARY_FIG_TEMPLATE.format(model=model)
        for model in EXPECTED_MODELS
    ]
    paths.extend([
        derived_dir / "expD_boundary_summary.csv",
        derived_dir / "expD_per_seed_boundary_summary.csv",
    ])
    return paths


def preflight_outputs(paths, input_csv, overwrite):
    """Validate all four targets before any output is written."""
    normalized = [Path(path).resolve() for path in paths]
    if len(normalized) != 4:
        raise ValueError(
            f"Expected four Experiment-D outputs, got {len(normalized)}."
        )
    if len(normalized) != len(set(normalized)):
        raise ValueError("Experiment-D output paths contain duplicate targets.")

    resolved_input = Path(input_csv).resolve()
    if resolved_input in normalized:
        raise ValueError(
            "The Experiment-D input CSV cannot also be an output target: "
            f"{resolved_input}. This is forbidden even with --overwrite."
        )

    directory_targets = [
        path for path in normalized if path.exists() and path.is_dir()
    ]
    if directory_targets:
        listed = "\n".join(f"- {path}" for path in directory_targets)
        raise IsADirectoryError(
            "Experiment-D output targets cannot be existing directories, "
            "even with --overwrite:\n"
            f"{listed}"
        )

    conflicts = [path for path in normalized if path.exists()]
    if conflicts and not overwrite:
        listed = "\n".join(f"- {path}" for path in conflicts)
        raise FileExistsError(
            "Refusing to overwrite existing Experiment-D outputs:\n"
            f"{listed}\nUse --overwrite to replace these exact files."
        )


def verify_outputs_created(paths):
    """Confirm that all four promised outputs were created as files."""
    normalized = [Path(path).resolve() for path in paths]
    missing = [path for path in normalized if not path.is_file()]
    if missing:
        listed = "\n".join(f"- {path}" for path in missing)
        raise RuntimeError(
            "Experiment-D plotting finished without all four outputs:\n"
            f"{listed}"
        )
    print("Verified: all four Experiment-D outputs exist.")


def compute_boundary(
    df_model: pd.DataFrame,
    write_mode: str,
    tau: float,
) -> pd.DataFrame:
    """Threshold seed-averaged accuracy to obtain aggregate boundaries.

    A configuration is right-censored when no scanned capacity reaches ``tau``.
    """
    sub = df_model[df_model["write_mode"] == write_mode].copy()

    if sub.empty:
        return pd.DataFrame(
            columns=["delay_len", "boundary", "censored", "max_scanned"]
        )

    grouped = (
        sub.groupby(["delay_len", "max_mem"], as_index=False)
        .agg(acc_mean=("acc_mean_tail", "mean"))
        .sort_values(["delay_len", "max_mem"])
    )

    rows = []

    for d in sorted(grouped["delay_len"].unique()):
        g = grouped[grouped["delay_len"] == d].sort_values("max_mem")

        passed = g[g["acc_mean"] >= tau]
        max_scanned = int(g["max_mem"].max())

        if len(passed) > 0:
            boundary = int(passed.iloc[0]["max_mem"])
            censored = False
        else:
            boundary = max_scanned
            censored = True

        rows.append(
            {
                "delay_len": int(d),
                "boundary": boundary,
                "censored": censored,
                "max_scanned": max_scanned,
            }
        )

    return pd.DataFrame(rows)


def build_aggregate_boundary_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Build the complete three-policy aggregate boundary table."""
    rows = []
    for model in EXPECTED_MODELS:
        df_model = df[df["model"] == model].copy()
        for write_mode in WRITE_MODE_ORDER:
            boundary = compute_boundary(df_model, write_mode, TAU_MAIN)
            for _, row in boundary.iterrows():
                delay_len = int(row["delay_len"])
                censored = bool(row["censored"])
                if write_mode == WRITE_PREFIX_ALL:
                    expected_boundary = EXPECTED_SOURCE_LEN + 1 + delay_len
                else:
                    expected_boundary = EXPECTED_SOURCE_LEN

                rows.append({
                    "model": model,
                    "write_mode": write_mode,
                    "delay_len": delay_len,
                    "tau": float(TAU_MAIN),
                    "m_star": (
                        float("nan")
                        if censored
                        else float(row["boundary"])
                    ),
                    "censored": censored,
                    "max_scanned": int(row["max_scanned"]),
                    "expected_boundary": int(expected_boundary),
                })

    return (
        pd.DataFrame(rows, columns=AGGREGATE_SUMMARY_COLUMNS)
        .sort_values(["model", "write_mode", "delay_len"])
        .reset_index(drop=True)
    )


def compute_per_seed_boundary_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize finite seed-wise boundaries and explicit censoring counts."""
    summary_columns = [
        "model",
        "write_mode",
        "delay_len",
        "mean",
        "std",
        "min",
        "max",
        "uncensored_n",
        "censored_n",
        "total_seeds",
        "max_scanned",
    ]
    rows = []
    filtered = df[df["write_mode"].isin(WRITE_MODE_ORDER)].copy()

    if filtered.empty:
        return pd.DataFrame(columns=summary_columns)

    for keys, sub in filtered.groupby(
        ["model", "write_mode", "delay_len", "seed"]
    ):
        model, write_mode, delay_len, seed = keys
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
            "model": str(model),
            "write_mode": str(write_mode),
            "delay_len": int(delay_len),
            "seed": int(seed),
            "m_star": m_star,
            "censored": bool(censored),
            "max_scanned": max_scanned,
        })

    per_seed = pd.DataFrame(rows)
    summary = (
        per_seed.groupby(
            ["model", "write_mode", "delay_len"],
            as_index=False,
        )
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
        .sort_values(["model", "write_mode", "delay_len"])
        .reset_index(drop=True)
    )
    summary["uncensored_n"] = summary["uncensored_n"].astype(int)
    summary["censored_n"] = summary["censored_n"].astype(int)
    summary["total_seeds"] = summary["total_seeds"].astype(int)
    return summary


def validate_derived_summaries(
    aggregate_summary: pd.DataFrame,
    per_seed_summary: pd.DataFrame,
) -> None:
    expected_per_seed_columns = (
        "model",
        "write_mode",
        "delay_len",
        "mean",
        "std",
        "min",
        "max",
        "uncensored_n",
        "censored_n",
        "total_seeds",
        "max_scanned",
    )
    if tuple(aggregate_summary.columns) != AGGREGATE_SUMMARY_COLUMNS:
        raise ValueError(
            "Experiment-D aggregate summary has an unexpected schema: "
            f"{list(aggregate_summary.columns)}"
        )
    if tuple(per_seed_summary.columns) != expected_per_seed_columns:
        raise ValueError(
            "Experiment-D per-seed summary has an unexpected schema: "
            f"{list(per_seed_summary.columns)}"
        )
    if len(aggregate_summary) != 18 or len(per_seed_summary) != 18:
        raise ValueError(
            "Experiment-D summaries must each contain eighteen rows: "
            f"aggregate={len(aggregate_summary)}, "
            f"per_seed={len(per_seed_summary)}."
        )

    aggregate_key = ["model", "write_mode", "delay_len"]
    if aggregate_summary.duplicated(subset=aggregate_key).any():
        raise ValueError(
            f"Experiment-D aggregate summary has duplicate keys: {aggregate_key}."
        )
    censored = aggregate_summary["censored"].astype(bool)
    if aggregate_summary.loc[censored, "m_star"].notna().any():
        raise ValueError(
            "Experiment-D aggregate-censored rows must have m_star=NaN."
        )
    if aggregate_summary.loc[~censored, "m_star"].isna().any():
        raise ValueError(
            "Experiment-D uncensored aggregate rows must have finite m_star."
        )

    expected_policy_counts = {write_mode: 6 for write_mode in WRITE_MODE_ORDER}
    aggregate_counts = aggregate_summary["write_mode"].value_counts().to_dict()
    per_seed_counts = per_seed_summary["write_mode"].value_counts().to_dict()
    if aggregate_counts != expected_policy_counts:
        raise ValueError(
            "Experiment-D aggregate summary policy counts are incomplete: "
            f"expected {expected_policy_counts}, got {aggregate_counts}."
        )
    if per_seed_counts != expected_policy_counts:
        raise ValueError(
            "Experiment-D per-seed summary policy counts are incomplete: "
            f"expected {expected_policy_counts}, got {per_seed_counts}."
        )


def save_aggregate_boundary_summary(
    summary: pd.DataFrame,
    derived_dir: Path,
    overwrite: bool,
) -> Path:
    out_path = prepare_output(
        derived_dir / "expD_boundary_summary.csv",
        overwrite,
    )
    summary.to_csv(out_path, index=False)
    return out_path


def save_per_seed_boundary_summary(
    summary: pd.DataFrame,
    derived_dir: Path,
    overwrite: bool,
) -> Path:
    out_path = prepare_output(
        derived_dir / "expD_per_seed_boundary_summary.csv",
        overwrite,
    )
    summary.to_csv(out_path, index=False)
    return out_path


def plot_boundary_for_model(
    df: pd.DataFrame,
    model: str,
    per_seed_summary: pd.DataFrame,
    figure_dir: Path,
    overwrite: bool,
):
    """Plot three policy boundaries against L and L + 1 + d references."""
    df_model = df[df["model"] == model].copy()

    boundary_by_mode = {
        write_mode: compute_boundary(df_model, write_mode, TAU_MAIN)
        for write_mode in WRITE_MODE_ORDER
    }

    if all(boundary.empty for boundary in boundary_by_mode.values()):
        raise ValueError(f"No Experiment-D boundary data for model={model}.")

    delays = sorted(
        set().union(
            *[
                set(boundary["delay_len"].tolist())
                for boundary in boundary_by_mode.values()
                if not boundary.empty
            ]
        )
    )

    expected_source = [SEQ_LEN for _ in delays]
    expected_prefix = [SEQ_LEN + 1 + d for d in delays]

    fig, ax = plt.subplots(figsize=FIGSIZE_BOUNDARY)

    measured_lines = {}

    # Aggregate measured boundaries.
    for write_mode in WRITE_MODE_ORDER:
        boundary = boundary_by_mode[write_mode]
        if boundary.empty:
            continue

        line, = ax.plot(
            boundary["delay_len"],
            boundary["boundary"].where(~boundary["censored"].astype(bool)),
            marker=WRITE_MODE_MARKER[write_mode],
            linewidth=LINE_WIDTH,
            markersize=MARKER_SIZE,
            label=WRITE_MODE_LABEL[write_mode],
            zorder=WRITE_MODE_ZORDER[write_mode],
        )
        measured_lines[write_mode] = line

    # Source-only and source-pinned share the theoretical boundary L.
    expected_source_line, = ax.plot(
        delays,
        expected_source,
        linestyle=":",
        linewidth=LINE_WIDTH,
        color="gray",
        label="expected source/pinned",
        zorder=5,
    )

    expected_prefix_line, = ax.plot(
        delays,
        expected_prefix,
        linestyle="--",
        linewidth=LINE_WIDTH,
        color="tab:red",
        label="expected prefix-all",
        zorder=2,
    )

    # Redraw measured markers so they remain visible over coincident references.
    for write_mode in WRITE_MODE_ORDER:
        boundary = boundary_by_mode[write_mode]
        if boundary.empty or write_mode not in measured_lines:
            continue

        finite_boundary = boundary[~boundary["censored"].astype(bool)]

        ax.scatter(
            finite_boundary["delay_len"],
            finite_boundary["boundary"],
            marker=WRITE_MODE_MARKER[write_mode],
            s=35,
            color=measured_lines[write_mode].get_color(),
            zorder=6,
            label="_nolegend_",
        )

    # Auxiliary seed indicators share a small display-coordinate lane per
    # policy while aggregate finite boundaries retain their exact data x values.
    policy_lane_offsets_pt = {
        WRITE_SOURCE_ONLY: -3.0,
        WRITE_PREFIX_ALL: 0.0,
        WRITE_SOURCE_PINNED: 3.0,
    }

    # Scan-limit crosses mark aggregate censoring and do not join finite curves.
    for write_mode in WRITE_MODE_ORDER:
        boundary = boundary_by_mode[write_mode]
        if boundary.empty:
            continue

        color = measured_lines[write_mode].get_color()
        for _, row in boundary[boundary["censored"].astype(bool)].iterrows():
            x = float(row["delay_len"])
            marker_transform = (
                ax.transData
                + ScaledTranslation(
                    policy_lane_offsets_pt.get(write_mode, 0.0) / 72.0,
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
                color=color,
                s=46,
                linewidths=1.4,
                zorder=8,
                label="_nolegend_",
            )

    ax.set_title(
        f"Symbolic Key-Value Retrieval Boundaries ({model})",
        fontsize=FIG_TITLE_SIZE,
        pad=AX_TITLE_PAD,
    )

    ax.set_xlabel("Distractor Length", fontsize=LABEL_SIZE)
    ax.set_ylabel(r"Minimal memory $m^*(d)$", fontsize=LABEL_SIZE)

    ax.set_xticks(delays)
    ax.tick_params(axis="both", labelsize=TICK_SIZE, pad=2)

    y_values = []
    for boundary in boundary_by_mode.values():
        if not boundary.empty:
            y_values.extend(boundary["boundary"].tolist())
    y_values.extend(expected_source)
    y_values.extend(expected_prefix)

    seed_model = per_seed_summary[
        per_seed_summary["model"] == model
    ].copy()
    aggregate_lookup = {}
    for write_mode, boundary in boundary_by_mode.items():
        for _, aggregate_row in boundary.iterrows():
            aggregate_lookup[
                (write_mode, float(aggregate_row["delay_len"]))
            ] = aggregate_row

    # Finite seed-wise ranges require a finite aggregate boundary and at least
    # two observed seed-wise boundaries; seed scan limits do not set the range.
    finite_seed_ranges = []
    for _, seed_row in seed_model.iterrows():
        write_mode = str(seed_row["write_mode"])
        aggregate_row = aggregate_lookup.get(
            (write_mode, float(seed_row["delay_len"]))
        )
        if aggregate_row is None or bool(aggregate_row["censored"]):
            continue
        if int(seed_row["uncensored_n"]) < 2:
            continue
        if pd.isna(seed_row["min"]) or pd.isna(seed_row["max"]):
            continue

        finite_seed_ranges.append(seed_row)
        y_values.extend([float(seed_row["min"]), float(seed_row["max"])])

    y_min = max(0, min(y_values) - 8)
    y_max = max(y_values) + 10
    ax.set_ylim(y_min, y_max)

    ax.grid(True, alpha=GRID_ALPHA)

    # Keep the compact legend inside unused plot space.
    handles = []
    labels = []

    for write_mode in WRITE_MODE_ORDER:
        if write_mode in measured_lines:
            handles.append(measured_lines[write_mode])
            labels.append(WRITE_MODE_LABEL[write_mode])

    handles.extend([expected_source_line, expected_prefix_line])
    labels.extend([
        "expected source/pinned",
        "expected prefix-all",
    ])

    ax.legend(
        handles,
        labels,
        fontsize=LEGEND_SIZE,
        loc="upper left",
        ncol=2,
        frameon=True,
        framealpha=0.85,
        handlelength=2.0,
        borderpad=0.4,
        labelspacing=0.3,
    )

    out_path = figure_dir / BOUNDARY_FIG_TEMPLATE.format(model=model)

    # Seed-wise bars show only observed finite ranges. The solid marker remains
    # the aggregate boundary computed from seed-averaged accuracy.
    for row in finite_seed_ranges:
        write_mode = str(row["write_mode"])
        x = float(row["delay_len"])
        range_min = float(row["min"])
        range_max = float(row["max"])
        color = measured_lines[write_mode].get_color()
        range_transform = (
            ax.transData
            + ScaledTranslation(
                policy_lane_offsets_pt.get(write_mode, 0.0) / 72.0,
                0.0,
                fig.dpi_scale_trans,
            )
        )

        if range_min == range_max:
            ax.plot(
                [x],
                [range_min],
                linestyle="none",
                marker="_",
                markersize=6,
                markeredgewidth=0.8,
                color=color,
                alpha=0.45,
                transform=range_transform,
                zorder=7,
                label="_nolegend_",
            )
        else:
            range_mid = 0.5 * (range_min + range_max)
            range_half = 0.5 * (range_max - range_min)
            ax.errorbar(
                [x],
                [range_mid],
                yerr=[range_half],
                fmt="none",
                transform=range_transform,
                ecolor=color,
                elinewidth=0.8,
                capsize=2.5,
                capthick=0.8,
                alpha=0.45,
                zorder=7,
                label="_nolegend_",
            )

    # A hollow triangle marks seed-wise censoring without treating a scan limit
    # as a finite boundary.
    censored = seed_model[seed_model["censored_n"] > 0]
    for _, row in censored.iterrows():
        write_mode = str(row["write_mode"])
        x = float(row["delay_len"])
        aggregate_row = aggregate_lookup.get((write_mode, x))
        if aggregate_row is None:
            continue
        if bool(aggregate_row["censored"]):
            y = float(aggregate_row["max_scanned"])
        else:
            y = float(aggregate_row["boundary"])
        color = measured_lines[write_mode].get_color()
        marker_transform = (
            ax.transData
            + ScaledTranslation(
                policy_lane_offsets_pt.get(write_mode, 0.0) / 72.0,
                5.0 / 72.0,
                fig.dpi_scale_trans,
            )
        )
        ax.scatter(
            [x],
            [y],
            transform=marker_transform,
            marker="^",
            facecolors="none",
            edgecolors=color,
            s=18,
            linewidths=0.75,
            alpha=0.70,
            zorder=8,
            label="_nolegend_",
        )

    # Preserve a small margin for display-coordinate policy lanes.
    ax.margins(x=0.06)

    fig.tight_layout()
    save_png(fig, out_path, overwrite=overwrite, dpi=DPI)
    plt.close(fig)

    return out_path


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

    aggregate_summary = build_aggregate_boundary_summary(df)
    per_seed_summary = compute_per_seed_boundary_summary(df)
    validate_derived_summaries(aggregate_summary, per_seed_summary)

    aggregate_path = save_aggregate_boundary_summary(
        aggregate_summary,
        derived_dir,
        overwrite,
    )
    per_seed_path = save_per_seed_boundary_summary(
        per_seed_summary,
        derived_dir,
        overwrite,
    )

    models = sorted(df["model"].unique())

    print(f"[ExpD plot] Models: {models}")
    print(f"[ExpD plot] tau = {TAU_MAIN}")
    print(f"[Saved] {aggregate_path}")
    print(f"[Saved] {per_seed_path}")
    print()

    for model in models:
        boundary_path = plot_boundary_for_model(
            df,
            model,
            per_seed_summary,
            figure_dir,
            overwrite,
        )
        print(f"[Saved] {boundary_path}")

    verify_outputs_created(output_paths)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help="Input CSV (default: Results/expD/results_expD_paper.csv).",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=DEFAULT_FIGURE_DIR,
        help="PNG output directory (default: Figures/expD).",
    )
    parser.add_argument(
        "--derived-dir",
        type=Path,
        default=DEFAULT_DERIVED_DIR,
        help="Derived CSV output directory (default: Results/expD).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of the four expected output files.",
    )
    args = parser.parse_args()

    main(
        csv_path=args.csv,
        figure_dir=args.figure_dir,
        derived_dir=args.derived_dir,
        overwrite=args.overwrite,
    )
