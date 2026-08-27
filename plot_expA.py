# Experiment A plotting and derived-statistics script.
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms


# Shared experiment constants.
from DelayedCopyTask.config import (
    FORCED_WINDOW,
    DELAY_LEN,
    NATURAL_SANITY_MEM,
    SEQ_LEN,
    TAU_MAIN,
    TAU_LIST,
    WRITE_SOURCE_ONLY,
    WRITE_PREFIX_ALL,
    WRITE_SOURCE_PINNED,
    WRITE_MODES_EXPA,
    MODEL_VARIANTS,
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
FIG_TITLE_Y = 0.965
LAYOUT_TOP = 0.90

ANNOTATION_SIZE = 10
HEATMAP_TEXT_SIZE = 10

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


PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "Results" / "expA"
DEFAULT_CSV = RESULTS_DIR / "results_expA_paper.csv"
DEFAULT_FIGURE_DIR = PROJECT_ROOT / "Figures" / "expA"
DEFAULT_DERIVED_DIR = RESULTS_DIR

FIG_PREFIX = "expA_"


def resolve_project_path(path: Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def prepare_output_path(path: Path, overwrite=False) -> Path:
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing output: {path}. "
            "Pass --overwrite only when replacement is intentional."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def fig_path(figure_dir: Path, filename: str) -> Path:
    return Path(figure_dir) / f"{FIG_PREFIX}{filename}"


def save_png(fig, out: Path, dpi=350, overwrite=False):
    """Save a tightly cropped PNG while enforcing the overwrite policy."""
    out = prepare_output_path(out, overwrite=overwrite)
    fig.savefig(
        out,
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.03,
    )


def save_dataframe_csv(df, out: Path, overwrite=False):
    out = prepare_output_path(out, overwrite=overwrite)
    df.to_csv(out, index=False)
    return out


METRIC = "acc_mean_tail"
DEFAULT_TAU = TAU_MAIN


# Write policies included in the main boundary figure.
BOUNDARY_WRITE_MODES_WITH_PINNED = [
    WRITE_SOURCE_ONLY,
    WRITE_PREFIX_ALL,
    WRITE_SOURCE_PINNED,
]


def display_write_mode(write_mode: str) -> str:
    """Return a compact plot label without changing stored policy names."""
    if write_mode == WRITE_SOURCE_ONLY:
        return "source-only"
    if write_mode == WRITE_PREFIX_ALL:
        return "prefix-all"
    if write_mode == WRITE_SOURCE_PINNED:
        return "source-pinned"
    return str(write_mode)


def load_df(csv_path=DEFAULT_CSV):
    csv_path = resolve_project_path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Experiment A result CSV not found: {csv_path}")

    print(f"[INFO] Loading results from: {csv_path}")

    df = pd.read_csv(csv_path)

    for c in ["seed", "window_size", "delay_len", "max_mem", "eval_tail"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

    for c in ["model", "mask_mode", "write_mode", "phase"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    if "noise_write_budget" in df.columns:
        df["noise_write_budget"] = pd.to_numeric(
            df["noise_write_budget"],
            errors="coerce",
        )

    if METRIC not in df.columns:
        raise ValueError(f"Metric column `{METRIC}` not found in result csv.")

    return df, csv_path


def mean_over_seeds(df, group_cols, metric=METRIC):
    return (
        df.groupby(group_cols)[metric]
        .mean()
        .reset_index(name="mean")
    )


def mean_std_over_seeds(df, group_cols, metric=METRIC):
    return (
        df.groupby(group_cols)[metric]
        .agg(mean="mean", std="std", n="count")
        .reset_index()
    )


def expected_boundary(write_mode: str, delay_len: int):
    """Return L for source-only/pinned and L + 1 + d for prefix-all."""
    if write_mode == WRITE_SOURCE_ONLY:
        return SEQ_LEN

    if write_mode == WRITE_PREFIX_ALL:
        return SEQ_LEN + 1 + int(delay_len)

    if write_mode == WRITE_SOURCE_PINNED:
        return SEQ_LEN

    return np.nan


def plot_forced_heatmap(
    df0,
    write_mode,
    window=FORCED_WINDOW,
    dpi=350,
    figure_dir=DEFAULT_FIGURE_DIR,
    overwrite=False,
):
    df = df0[
        (df0["mask_mode"] == "forced") &
        (df0["window_size"] == int(window)) &
        (df0["write_mode"] == str(write_mode)) &
        (df0["max_mem"] > 0)
    ].copy()

    if len(df) == 0:
        print(f"[SKIP] No forced data for write_mode={write_mode}, window={window}")
        return

    models = sorted(df["model"].unique().tolist())
    agg = mean_over_seeds(df, ["model", "max_mem", "delay_len"])

    mems = sorted(agg["max_mem"].dropna().astype(int).unique().tolist())
    delays = sorted(agg["delay_len"].dropna().astype(int).unique().tolist())

    fig, axes = plt.subplots(
        1,
        len(models),
        figsize=(6.4 * len(models), 4.8),
        sharey=True,
        constrained_layout=True,
    )

    if len(models) == 1:
        axes = [axes]

    im = None

    for ax, model in zip(axes, models):
        sub = agg[agg["model"] == model]

        mat = np.full((len(mems), len(delays)), np.nan, dtype=float)
        for i, mem in enumerate(mems):
            for j, d in enumerate(delays):
                row = sub[
                    (sub["max_mem"] == mem) &
                    (sub["delay_len"] == d)
                ]
                if len(row) == 1:
                    mat[i, j] = float(row["mean"].iloc[0])

        im = ax.imshow(
            mat,
            aspect="auto",
            origin="lower",
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
        )

        ax.set_title(model, fontsize=AX_TITLE_SIZE, pad=AX_TITLE_PAD)
        ax.set_xlabel("Delay Length", fontsize=LABEL_SIZE)
        ax.set_xticks(range(len(delays)))
        ax.set_xticklabels(delays, fontsize=TICK_SIZE)

        ax.set_yticks(range(len(mems)))
        ax.set_yticklabels(mems, fontsize=TICK_SIZE)
        ax.set_ylabel("Memory Size", fontsize=LABEL_SIZE)

        for i in range(len(mems)):
            for j in range(len(delays)):
                if not np.isnan(mat[i, j]):
                    val = mat[i, j]
                    ax.text(
                        j,
                        i,
                        f"{val:.2f}",
                        ha="center",
                        va="center",
                        fontsize=HEATMAP_TEXT_SIZE,
                        fontweight="bold",
                    )

    cbar = fig.colorbar(im, ax=axes, fraction=0.035, pad=0.02)
    cbar.set_label("Accuracy", fontsize=LABEL_SIZE)
    cbar.ax.tick_params(labelsize=TICK_SIZE)

    fig.suptitle(
        f"Forced | {display_write_mode(write_mode)} | window={window}",
        fontsize=FIG_TITLE_SIZE,
        y=FIG_TITLE_Y,
    )

    out = fig_path(
        figure_dir,
        f"forced_{write_mode}_window{window}_heatmap.png",
    )
    save_png(fig, out, dpi=dpi, overwrite=overwrite)
    plt.close(fig)
    print("Saved", out)


def compute_mstar_forced(
    df0,
    window=FORCED_WINDOW,
    tau=DEFAULT_TAU,
    metric=METRIC,
    save_csv=True,
    derived_dir=DEFAULT_DERIVED_DIR,
    overwrite=False,
):
    """Threshold seed-averaged accuracy to obtain all three policy boundaries."""
    df = df0[
        (df0["mask_mode"] == "forced") &
        (df0["window_size"] == int(window)) &
        (df0["write_mode"].isin(BOUNDARY_WRITE_MODES_WITH_PINNED)) &
        (df0["max_mem"] > 0)
    ].copy()

    if len(df) == 0:
        print(f"[SKIP] No forced boundary data for window={window}")
        return pd.DataFrame()

    agg = (
        df.groupby(["model", "write_mode", "delay_len", "max_mem"])[metric]
        .mean()
        .reset_index(name="seed_mean")
    )

    rows = []
    for (model, write_mode, delay_len), sub in agg.groupby(
        ["model", "write_mode", "delay_len"]
    ):
        sub = sub.sort_values("max_mem").reset_index(drop=True)

        max_mem_scanned = int(sub["max_mem"].max())

        best_row = sub.loc[sub["seed_mean"].idxmax()]
        best_acc = float(best_row["seed_mean"])
        best_mem = int(best_row["max_mem"])

        ok = sub[sub["seed_mean"] >= float(tau)]

        if len(ok) == 0:
            m_star = np.nan
            acc_at = best_acc
            censored = True
        else:
            m_star = int(ok.iloc[0]["max_mem"])
            acc_at = float(ok.iloc[0]["seed_mean"])
            censored = False

        rows.append({
            "model": str(model),
            "write_mode": str(write_mode),
            "delay_len": int(delay_len),
            "tau": float(tau),
            "m_star": m_star,
            "m_star_censored": bool(censored),
            "acc_at_m_star": acc_at,
            "max_mem_scanned": max_mem_scanned,
            "best_acc_within_scan": best_acc,
            "best_mem_within_scan": best_mem,
            "expected_boundary": expected_boundary(str(write_mode), int(delay_len)),
        })

    out_df = pd.DataFrame(rows).sort_values(["model", "write_mode", "delay_len"])

    if save_csv:
        out_path = (
            Path(derived_dir)
            / f"mstar_forced_window{window}_tau{tau}.csv"
        )
        save_dataframe_csv(out_df, out_path, overwrite=overwrite)
        print("Saved", out_path)

    return out_df


def plot_mstar_forced(
    df0,
    window=FORCED_WINDOW,
    tau=DEFAULT_TAU,
    dpi=350,
    figure_dir=DEFAULT_FIGURE_DIR,
    derived_dir=DEFAULT_DERIVED_DIR,
    overwrite=False,
):
    """Plot aggregate policy boundaries and deterministic FIFO references.

    Thin bars show ranges over observed finite seed-wise boundaries when the
    aggregate boundary is finite; triangles mark seed-wise censoring. Scan
    limits are never treated as finite boundaries.
    """
    mstar = compute_mstar_forced(
        df0,
        window=window,
        tau=tau,
        save_csv=True,
        derived_dir=derived_dir,
        overwrite=overwrite,
    )
    seed_mstar = compute_mstar_per_seed(
        df0,
        window=window,
        tau=tau,
        metric=METRIC,
    )
    seed_summary = summarize_mstar_per_seed(seed_mstar)

    if len(mstar) == 0:
        return

    models = [m for m in MODEL_VARIANTS if m in set(mstar["model"])]
    if not models:
        models = sorted(mstar["model"].unique().tolist())

    policy_order = [
        WRITE_SOURCE_ONLY,
        WRITE_PREFIX_ALL,
        WRITE_SOURCE_PINNED,
    ]

    policy_style = {
        WRITE_SOURCE_ONLY: dict(color="tab:blue", marker="o", linestyle="-"),
        WRITE_PREFIX_ALL: dict(color="tab:orange", marker="s", linestyle="-"),
        WRITE_SOURCE_PINNED: dict(color="tab:green", marker="^", linestyle="-"),
    }

    # Display-coordinate offsets separate coincident auxiliary indicators
    # without changing their semantic delay values.
    policy_lane_dx_points = {
        WRITE_SOURCE_ONLY: -3.0,
        WRITE_PREFIX_ALL: 0.0,
        WRITE_SOURCE_PINNED: 3.0,
    }
    seed_censor_dy_points = 5.0

    for model in models:
        sub_model = mstar[mstar["model"] == model].copy()

        if len(sub_model) == 0:
            continue

        delays = sorted(sub_model["delay_len"].dropna().astype(int).unique().tolist())

        fig, ax = plt.subplots(figsize=(8.0, 4.8))
        displayed_seed_range_y = []
        for write_mode in policy_order:
            sub = sub_model[sub_model["write_mode"] == write_mode].sort_values("delay_len")

            if len(sub) == 0:
                continue

            y_plot = []
            for _, row in sub.iterrows():
                if bool(row["m_star_censored"]):
                    # Break the line rather than treating a scan limit as a finite boundary.
                    y_plot.append(np.nan)
                else:
                    y_plot.append(float(row["m_star"]))

            style = policy_style.get(write_mode, {})

            policy_lane_transform = (
                ax.transData
                + mtransforms.ScaledTranslation(
                    policy_lane_dx_points.get(write_mode, 0.0) / 72.0,
                    0.0,
                    fig.dpi_scale_trans,
                )
            )
            seed_censor_annotation_transform = (
                ax.transData
                + mtransforms.ScaledTranslation(
                    policy_lane_dx_points.get(write_mode, 0.0) / 72.0,
                    seed_censor_dy_points / 72.0,
                    fig.dpi_scale_trans,
                )
            )

            ax.plot(
                sub["delay_len"],
                y_plot,
                label=display_write_mode(write_mode),
                linewidth=LINE_WIDTH,
                markersize=MARKER_SIZE,
                **style,
            )

            aggregate_censored = sub[sub["m_star_censored"] == True]
            if len(aggregate_censored) > 0:
                ax.plot(
                    aggregate_censored["delay_len"],
                    aggregate_censored["max_mem_scanned"],
                    linestyle="none",
                    marker="x",
                    markersize=MARKER_SIZE + 1.5,
                    markeredgewidth=1.6,
                    color=style.get("color", "black"),
                    transform=policy_lane_transform,
                    zorder=7,
                    label="_nolegend_",
                )

            seed_sub = seed_summary[
                (seed_summary["model"] == model) &
                (seed_summary["write_mode"] == write_mode)
            ].sort_values("delay_len")

            # The auxiliary bar is the observed range of finite seed-wise
            # boundaries, not an alternative estimate of the aggregate boundary.
            finite_aggregate_mask = (
                (sub["m_star_censored"] == False)
                & sub["m_star"].notna()
                & np.isfinite(sub["m_star"].astype(float))
            )
            finite_aggregate_delays = set(
                sub.loc[
                    finite_aggregate_mask,
                    "delay_len",
                ].astype(int)
            )
            with_range = seed_sub[
                seed_sub["delay_len"].astype(int).isin(finite_aggregate_delays)
                & (seed_sub["uncensored_n"] >= 2)
                & seed_sub["m_star_min"].notna()
                & seed_sub["m_star_max"].notna()
                & np.isfinite(seed_sub["m_star_min"].astype(float))
                & np.isfinite(seed_sub["m_star_max"].astype(float))
            ].copy()

            if len(with_range) > 0:
                displayed_seed_range_y.extend(
                    with_range["m_star_min"].astype(float).tolist()
                )
                displayed_seed_range_y.extend(
                    with_range["m_star_max"].astype(float).tolist()
                )
                range_midpoint = (
                    with_range["m_star_min"] + with_range["m_star_max"]
                ) / 2.0
                range_yerr = np.vstack([
                    range_midpoint - with_range["m_star_min"],
                    with_range["m_star_max"] - range_midpoint,
                ])
                ax.errorbar(
                    with_range["delay_len"],
                    range_midpoint,
                    yerr=range_yerr,
                    fmt="none",
                    ecolor=style.get("color", "black"),
                    elinewidth=0.8,
                    capsize=2.5,
                    capthick=0.8,
                    alpha=0.45,
                    zorder=4,
                    transform=policy_lane_transform,
                    label="_nolegend_",
                )

            seed_censored = seed_sub[seed_sub["censored_n"] > 0]
            if len(seed_censored) > 0:
                censor_annotations = seed_censored[
                    ["delay_len", "censored_n"]
                ].merge(
                    sub[
                        [
                            "delay_len",
                            "m_star",
                            "m_star_censored",
                            "max_mem_scanned",
                        ]
                    ],
                    on="delay_len",
                    how="inner",
                )
                censor_annotations["anchor_y"] = np.where(
                    censor_annotations["m_star_censored"],
                    censor_annotations["max_mem_scanned"],
                    censor_annotations["m_star"],
                )
                censor_annotations = censor_annotations[
                    censor_annotations["anchor_y"].notna()
                ]

                ax.plot(
                    censor_annotations["delay_len"],
                    censor_annotations["anchor_y"],
                    linestyle="none",
                    marker="^",
                    markersize=3.5,
                    markerfacecolor="none",
                    markeredgecolor=style.get("color", "black"),
                    markeredgewidth=0.75,
                    alpha=0.70,
                    transform=seed_censor_annotation_transform,
                    zorder=8,
                    label="_nolegend_",
                )

        if delays:
            expected_source = [SEQ_LEN for _ in delays]
            expected_prefix = [SEQ_LEN + 1 + d for d in delays]

            ax.plot(
                delays,
                expected_source,
                color="gray",
                linestyle=":",
                linewidth=LINE_WIDTH,
                label="expected source-only / source-pinned",
            )

            ax.plot(
                delays,
                expected_prefix,
                color="tab:red",
                linestyle="--",
                linewidth=LINE_WIDTH,
                label="expected prefix-all",
            )

        ax.set_xticks(delays)
        ax.set_xticklabels(delays, fontsize=TICK_SIZE)
        ax.set_xlabel("Delay Length", fontsize=LABEL_SIZE)
        ax.set_ylabel(r"Minimal memory $m^*(d)$", fontsize=LABEL_SIZE)
        ax.set_title(
            rf"Forced Capacity Boundaries ({model}, $\tau={tau}$)",
            fontsize=FIG_TITLE_SIZE,
            pad=AX_TITLE_PAD,
        )
        ax.tick_params(axis="both", labelsize=TICK_SIZE, pad=2)
        ax.grid(True, alpha=GRID_ALPHA)
        ax.margins(x=0.10)

        finite_aggregate_y = sub_model.loc[
            sub_model["m_star_censored"] == False,
            "m_star",
        ].dropna().astype(float).tolist()
        aggregate_censor_y = sub_model.loc[
            sub_model["m_star_censored"] == True,
            "max_mem_scanned",
        ].dropna().astype(float).tolist()
        theory_y = [float(SEQ_LEN)]
        theory_y.extend(float(SEQ_LEN + 1 + d) for d in delays)
        visible_y = (
            finite_aggregate_y
            + aggregate_censor_y
            + theory_y
            + displayed_seed_range_y
        )
        y_max = max(visible_y) if visible_y else float(SEQ_LEN)
        y_top_padding = max(2.0, y_max * 0.12)
        ax.set_ylim(0, y_max + y_top_padding)

        # A compact legend avoids covering the upper-right prefix-all result.
        ax.legend(
            loc="upper left",
            ncol=1,
            fontsize=LEGEND_SIZE - 1,
            frameon=True,
            framealpha=0.88,
            handlelength=2.0,
            borderpad=0.4,
            labelspacing=0.3,
        )
        fig.tight_layout()

        out = fig_path(
            figure_dir,
            f"mstar_forced_{model}_window{window}_tau{tau}.png",
        )
        save_png(fig, out, dpi=dpi, overwrite=overwrite)
        plt.close(fig)
        print("Saved", out)


def plot_mstar_tau_overlay(
    df0,
    window=FORCED_WINDOW,
    tau_list=TAU_LIST,
    dpi=350,
    figure_dir=DEFAULT_FIGURE_DIR,
    overwrite=False,
):
    """Plot all three policy boundaries across accuracy thresholds."""
    from matplotlib.lines import Line2D

    frames = []
    for tau in tau_list:
        mstar = compute_mstar_forced(
            df0,
            window=window,
            tau=float(tau),
            save_csv=False,
        )
        if len(mstar) > 0:
            frames.append(mstar)

    if len(frames) == 0:
        print("[SKIP] No mstar data for tau overlay.")
        return

    all_mstar = pd.concat(frames, ignore_index=True)

    models = [m for m in MODEL_VARIANTS if m in set(all_mstar["model"])]
    if not models:
        models = sorted(all_mstar["model"].unique().tolist())

    n_models = len(models)

    fig, axes = plt.subplots(
        1,
        n_models,
        figsize=(6.2 * n_models, 4.6),
        sharey=True,
    )

    if n_models == 1:
        axes = [axes]

    colors = {
        WRITE_SOURCE_ONLY: "tab:blue",
        WRITE_PREFIX_ALL: "tab:orange",
        WRITE_SOURCE_PINNED: "tab:green",
    }

    markers = {
        WRITE_SOURCE_ONLY: "o",
        WRITE_PREFIX_ALL: "s",
        WRITE_SOURCE_PINNED: "^",
    }

    linestyles = {
        0.90: "-",
        0.95: "--",
        0.98: "-.",
    }

    # Point offsets separate coincident censoring markers without changing
    # their data-space delay coordinates.
    tau_keys = [round(float(tau), 2) for tau in tau_list]
    tau_offset_center = (len(tau_keys) - 1) / 2.0
    tau_censor_dx_points = {
        tau_key: (idx - tau_offset_center) * 6.0
        for idx, tau_key in enumerate(tau_keys)
    }

    y_max_seen = 0.0

    for ax, model in zip(axes, models):
        sub_model = all_mstar[all_mstar["model"] == model].copy()
        delays = sorted(sub_model["delay_len"].dropna().astype(int).unique().tolist())

        for write_mode in BOUNDARY_WRITE_MODES_WITH_PINNED:
            for tau in tau_list:
                tau_key = round(float(tau), 2)
                sub = sub_model[
                    (sub_model["write_mode"] == write_mode) &
                    (np.isclose(sub_model["tau"], float(tau)))
                ].sort_values("delay_len")

                if len(sub) == 0:
                    continue

                y_values = []
                for _, row in sub.iterrows():
                    if bool(row["m_star_censored"]):
                        # The scan limit marks censoring, not an observed boundary.
                        y_values.append(np.nan)
                        y_max_seen = max(
                            y_max_seen,
                            float(row["max_mem_scanned"]),
                        )
                    else:
                        y = float(row["m_star"])
                        y_values.append(y)
                        y_max_seen = max(y_max_seen, y)

                ax.plot(
                    sub["delay_len"],
                    y_values,
                    marker=markers.get(write_mode, "o"),
                    color=colors.get(write_mode, None),
                    linestyle=linestyles.get(tau_key, "-"),
                    linewidth=LINE_WIDTH,
                    markersize=MARKER_SIZE,
                )

                cens = sub[sub["m_star_censored"] == True]
                if len(cens) > 0:
                    censor_transform = (
                        ax.transData
                        + mtransforms.ScaledTranslation(
                            tau_censor_dx_points.get(tau_key, 0.0) / 72.0,
                            0.0,
                            fig.dpi_scale_trans,
                        )
                    )
                    ax.plot(
                        cens["delay_len"],
                        cens["max_mem_scanned"],
                        linestyle="none",
                        marker="x",
                        markersize=MARKER_SIZE + 1.0,
                        markeredgewidth=1.8,
                        color=colors.get(write_mode, "black"),
                        transform=censor_transform,
                        zorder=7,
                        label="_nolegend_",
                    )

        if delays:
            source_expected = [SEQ_LEN for _ in delays]
            prefix_expected = [SEQ_LEN + 1 + d for d in delays]

            y_max_seen = max(
                y_max_seen,
                max(source_expected),
                max(prefix_expected),
            )

            ax.plot(
                delays,
                source_expected,
                color="gray",
                linestyle=":",
                linewidth=LINE_WIDTH,
            )

            ax.plot(
                delays,
                prefix_expected,
                color="black",
                linestyle=":",
                linewidth=LINE_WIDTH,
            )

        ax.set_title(model, fontsize=AX_TITLE_SIZE, pad=AX_TITLE_PAD)
        ax.set_xlabel("Delay Length", fontsize=LABEL_SIZE)
        ax.tick_params(axis="both", labelsize=TICK_SIZE, pad=2)

        if delays:
            ax.set_xticks(delays)

        ax.grid(True, alpha=GRID_ALPHA)
        ax.margins(x=0.08)

    axes[0].set_ylabel(r"Minimal memory $m^*(d)$", fontsize=LABEL_SIZE)

    policy_handles = [
        Line2D(
            [0],
            [0],
            color=colors[WRITE_SOURCE_ONLY],
            marker=markers[WRITE_SOURCE_ONLY],
            linestyle="-",
            linewidth=LINE_WIDTH,
            markersize=MARKER_SIZE,
            label="source-only",
        ),
        Line2D(
            [0],
            [0],
            color=colors[WRITE_PREFIX_ALL],
            marker=markers[WRITE_PREFIX_ALL],
            linestyle="-",
            linewidth=LINE_WIDTH,
            markersize=MARKER_SIZE,
            label="prefix-all",
        ),
        Line2D(
            [0],
            [0],
            color=colors[WRITE_SOURCE_PINNED],
            marker=markers[WRITE_SOURCE_PINNED],
            linestyle="-",
            linewidth=LINE_WIDTH,
            markersize=MARKER_SIZE,
            label="source-pinned",
        ),
    ]

    tau_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            linestyle=linestyles[round(float(tau), 2)],
            linewidth=LINE_WIDTH,
            label=rf"$\tau={float(tau):.2f}$",
        )
        for tau in tau_list
    ]

    ref_handles = [
        Line2D(
            [0],
            [0],
            color="gray",
            linestyle=":",
            linewidth=LINE_WIDTH,
            label="expected source-only / source-pinned",
        ),
        Line2D(
            [0],
            [0],
            color="black",
            linestyle=":",
            linewidth=LINE_WIDTH,
            label="expected prefix-all",
        ),
    ]

    handles = policy_handles + tau_handles + ref_handles

    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.06),
        ncol=4,
        frameon=True,
        fontsize=LEGEND_SIZE - 1,
    )

    fig.suptitle(
        "Capacity-Boundary Stability Across Thresholds",
        y=FIG_TITLE_Y,
        fontsize=FIG_TITLE_SIZE,
    )

    fig.tight_layout(rect=[0, 0.15, 1, 0.92])

    if y_max_seen > 0:
        for ax in axes:
            ax.set_ylim(0, y_max_seen * 1.12)

    out = fig_path(figure_dir, f"mstar_tau_overlay_window{window}.png")
    save_png(fig, out, dpi=dpi, overwrite=overwrite)
    plt.close(fig)
    print("Saved", out)


def save_tau_sensitivity_table(
    df0,
    window=FORCED_WINDOW,
    tau_list=TAU_LIST,
    derived_dir=DEFAULT_DERIVED_DIR,
    overwrite=False,
):
    frames = []
    for tau in tau_list:
        mstar = compute_mstar_forced(
            df0,
            window=window,
            tau=float(tau),
            save_csv=False,
        )
        if len(mstar) > 0:
            frames.append(mstar)

    if len(frames) == 0:
        print("[SKIP] No mstar data for tau sensitivity table.")
        return

    all_mstar = pd.concat(frames, ignore_index=True)

    def display_mstar(row):
        if bool(row["m_star_censored"]):
            return f">{int(row['max_mem_scanned'])}"
        if pd.isna(row["m_star"]):
            return "NA"
        return str(int(row["m_star"]))

    all_mstar["m_star_display"] = all_mstar.apply(display_mstar, axis=1)

    table = all_mstar.pivot_table(
        index=["model", "write_mode", "delay_len", "expected_boundary"],
        columns="tau",
        values="m_star_display",
        aggfunc="first",
    ).reset_index()

    new_cols = []
    for col in table.columns:
        if isinstance(col, (float, np.floating)):
            new_cols.append(f"tau_{float(col):.2f}")
        else:
            new_cols.append(str(col))
    table.columns = new_cols

    table = table.sort_values(["model", "write_mode", "delay_len"])

    out = Path(derived_dir) / f"boundary_tau_sensitivity_window{window}.csv"
    save_dataframe_csv(table, out, overwrite=overwrite)
    print("Saved", out)


def compute_mstar_per_seed(df0, window=FORCED_WINDOW, tau=DEFAULT_TAU, metric=METRIC):
    df = df0[
        (df0["mask_mode"] == "forced") &
        (df0["window_size"] == int(window)) &
        (df0["write_mode"].isin(BOUNDARY_WRITE_MODES_WITH_PINNED)) &
        (df0["max_mem"] > 0)
    ].copy()

    if len(df) == 0:
        print(f"[SKIP] No forced seed-wise boundary data for window={window}")
        return pd.DataFrame()

    rows = []
    for (model, write_mode, delay_len, seed), sub in df.groupby(
        ["model", "write_mode", "delay_len", "seed"]
    ):
        sub = sub.sort_values("max_mem").reset_index(drop=True)

        max_mem_scanned = int(sub["max_mem"].max())
        ok = sub[sub[metric] >= float(tau)]

        if len(ok) == 0:
            m_star = np.nan
            censored = True
        else:
            m_star = int(ok.iloc[0]["max_mem"])
            censored = False

        rows.append({
            "model": str(model),
            "write_mode": str(write_mode),
            "delay_len": int(delay_len),
            "seed": int(seed),
            "tau": float(tau),
            "m_star": m_star,
            "m_star_censored": bool(censored),
            "max_mem_scanned": max_mem_scanned,
            "expected_boundary": expected_boundary(str(write_mode), int(delay_len)),
        })

    return pd.DataFrame(rows).sort_values(
        ["model", "write_mode", "delay_len", "seed"]
    )


def summarize_mstar_per_seed(seed_df):
    """Summarize observed seed-wise boundaries and censoring separately.

    Finite moments exclude censored seeds because their scan limits are lower
    bounds rather than observed boundaries. All three policies are retained.
    """
    columns = [
        "model",
        "write_mode",
        "delay_len",
        "expected_boundary",
        "m_star_mean",
        "m_star_std",
        "m_star_min",
        "m_star_max",
        "n_seeds",
        "uncensored_n",
        "censored_n",
        "max_mem_scanned",
        "stats_scope",
    ]

    if seed_df is None or len(seed_df) == 0:
        return pd.DataFrame(columns=columns)

    summary = (
        seed_df.groupby(["model", "write_mode", "delay_len", "expected_boundary"])
        .agg(
            m_star_mean=("m_star", "mean"),
            m_star_std=("m_star", "std"),
            m_star_min=("m_star", "min"),
            m_star_max=("m_star", "max"),
            n_seeds=("seed", "nunique"),
            uncensored_n=("m_star", "count"),
            censored_n=("m_star_censored", "sum"),
            max_mem_scanned=("max_mem_scanned", "max"),
        )
        .reset_index()
    )

    for count_col in ["n_seeds", "uncensored_n", "censored_n", "max_mem_scanned"]:
        summary[count_col] = summary[count_col].astype(int)

    summary["stats_scope"] = "uncensored_seeds_only"

    for col in ["m_star_mean", "m_star_std", "m_star_min", "m_star_max"]:
        summary[col] = summary[col].round(2)

    return summary[columns].sort_values(
        ["model", "write_mode", "delay_len"]
    )


def save_seed_stability_table(
    df0,
    window=FORCED_WINDOW,
    tau=DEFAULT_TAU,
    derived_dir=DEFAULT_DERIVED_DIR,
    overwrite=False,
):
    seed_df = compute_mstar_per_seed(df0, window=window, tau=tau)

    if len(seed_df) == 0:
        return

    summary = summarize_mstar_per_seed(seed_df)

    out = (
        Path(derived_dir)
        / f"boundary_seed_stability_window{window}_tau{tau}.csv"
    )
    save_dataframe_csv(summary, out, overwrite=overwrite)
    print("Saved", out)


def plot_source_pinned_noise_budget(
    df0,
    window=FORCED_WINDOW,
    dpi=350,
    figure_dir=DEFAULT_FIGURE_DIR,
    overwrite=False,
):
    df = df0[
        (df0["mask_mode"] == "forced") &
        (df0["window_size"] == int(window)) &
        (df0["write_mode"] == WRITE_SOURCE_PINNED) &
        (df0["max_mem"] > 0)
    ].copy()

    if len(df) == 0:
        print("[SKIP] No source-pinned data found.")
        return

    if "noise_write_budget" not in df.columns:
        df["noise_write_budget"] = df["max_mem"] - SEQ_LEN

    agg = mean_std_over_seeds(
        df,
        ["model", "delay_len", "noise_write_budget"],
    )

    models = sorted(agg["model"].unique().tolist())

    for model in models:
        sub_model = agg[agg["model"] == model]

        fig, ax = plt.subplots(figsize=(7.8, 4.8))

        for delay_len, sub in sub_model.groupby("delay_len"):
            sub = sub.sort_values("noise_write_budget")

            ax.errorbar(
                sub["noise_write_budget"],
                sub["mean"],
                yerr=sub["std"],
                marker="o",
                markersize=MARKER_SIZE,
                capsize=3,
                linewidth=LINE_WIDTH,
                label=f"delay={int(delay_len)}",
            )

        ax.set_xlabel("Allowed noise writes into memory", fontsize=LABEL_SIZE)
        ax.set_ylabel("Accuracy", fontsize=LABEL_SIZE)
        ax.set_title(
            f"Source-pinned interference | {model} | window={window}",
            fontsize=FIG_TITLE_SIZE,
            pad=AX_TITLE_PAD,
        )
        ax.set_ylim(0.0, 1.05)
        ax.tick_params(axis="both", labelsize=TICK_SIZE, pad=2)
        ax.grid(True, alpha=GRID_ALPHA)
        ax.legend(fontsize=LEGEND_SIZE)
        fig.tight_layout()

        out = fig_path(
            figure_dir,
            f"source_pinned_noise_budget_{model}_window{window}.png",
        )
        save_png(fig, out, dpi=dpi, overwrite=overwrite)
        plt.close(fig)
        print("Saved", out)


def plot_policy_curves_at_delay(
    df0,
    delay=80,
    window=FORCED_WINDOW,
    model="naive",
    dpi=350,
    figure_dir=DEFAULT_FIGURE_DIR,
    overwrite=False,
):
    df = df0[
        (df0["mask_mode"] == "forced") &
        (df0["window_size"] == int(window)) &
        (df0["delay_len"] == int(delay)) &
        (df0["model"] == str(model)) &
        (df0["max_mem"] > 0)
    ].copy()

    if len(df) == 0:
        print(f"[SKIP] No data for delay={delay}, model={model}")
        return

    agg = mean_std_over_seeds(
        df,
        ["write_mode", "max_mem"],
    )

    fig, ax = plt.subplots(figsize=(7.8, 4.8))

    policy_order = [
        WRITE_SOURCE_ONLY,
        WRITE_PREFIX_ALL,
        WRITE_SOURCE_PINNED,
    ]

    for write_mode in policy_order:
        sub = agg[agg["write_mode"] == write_mode].sort_values("max_mem")

        if len(sub) == 0:
            continue

        ax.errorbar(
            sub["max_mem"],
            sub["mean"],
            yerr=sub["std"],
            marker="o",
            markersize=MARKER_SIZE,
            capsize=3,
            linewidth=LINE_WIDTH,
            label=display_write_mode(write_mode),
        )

    ax.set_xlabel("Memory Size", fontsize=LABEL_SIZE)
    ax.set_ylabel("Accuracy", fontsize=LABEL_SIZE)
    ax.set_title(
        f"Forced policy comparison | model={model} | "
        f"delay={delay} | window={window}",
        fontsize=FIG_TITLE_SIZE,
        pad=AX_TITLE_PAD,
    )
    ax.set_ylim(0.0, 1.05)
    ax.tick_params(axis="both", labelsize=TICK_SIZE, pad=2)
    ax.grid(True, alpha=GRID_ALPHA)
    ax.legend(fontsize=LEGEND_SIZE)
    fig.tight_layout()

    out = fig_path(
        figure_dir,
        f"policy_curves_model{model}_delay{delay}_window{window}.png",
    )
    save_png(fig, out, dpi=dpi, overwrite=overwrite)
    plt.close(fig)
    print("Saved", out)


def plot_natural_sanity(
    df0,
    delay=80,
    mem=NATURAL_SANITY_MEM,
    write_mode=WRITE_SOURCE_ONLY,
    dpi=350,
    figure_dir=DEFAULT_FIGURE_DIR,
    overwrite=False,
):
    baseline = df0[
        (df0["mask_mode"] == "natural") &
        (df0["model"] == "baseline") &
        (df0["delay_len"] == int(delay))
    ].copy()

    memory_models = df0[
        (df0["mask_mode"] == "natural") &
        (df0["model"].isin(["naive", "gated"])) &
        (df0["delay_len"] == int(delay)) &
        (df0["max_mem"] == int(mem)) &
        (df0["write_mode"] == str(write_mode))
    ].copy()

    df = pd.concat([baseline, memory_models], ignore_index=True)

    if len(df) == 0:
        print("[SKIP] No natural-sanity rows found in the input CSV.")
        return

    agg = mean_std_over_seeds(df, ["model", "window_size"]).sort_values("window_size")

    fig, ax = plt.subplots(figsize=(7.4, 4.8))

    for model in ["baseline", "naive", "gated"]:
        sub = agg[agg["model"] == model].sort_values("window_size")
        if len(sub) == 0:
            continue

        ax.errorbar(
            sub["window_size"],
            sub["mean"],
            yerr=sub["std"],
            marker="o",
            markersize=MARKER_SIZE,
            capsize=3,
            linewidth=LINE_WIDTH,
            label=model,
        )

    ax.set_xlabel("Attention Window Size", fontsize=LABEL_SIZE)
    ax.set_ylabel("Accuracy", fontsize=LABEL_SIZE)
    ax.set_title(
        f"Natural sanity | delay={delay} | "
        f"mem={mem} | write={display_write_mode(write_mode)}",
        fontsize=FIG_TITLE_SIZE,
        pad=AX_TITLE_PAD,
    )
    ax.set_ylim(0.0, 1.05)
    ax.tick_params(axis="both", labelsize=TICK_SIZE, pad=2)
    ax.grid(True, alpha=GRID_ALPHA)
    ax.legend(fontsize=LEGEND_SIZE)
    fig.tight_layout()

    out = fig_path(
        figure_dir,
        f"natural_sanity_delay{delay}_mem{mem}_write{write_mode}.png",
    )
    save_png(fig, out, dpi=dpi, overwrite=overwrite)
    plt.close(fig)
    print("Saved", out)


def expected_output_paths(figure_dir, derived_dir, window, tau):
    figure_names = [
        f"{FIG_PREFIX}forced_{write_mode}_window{window}_heatmap.png"
        for write_mode in WRITE_MODES_EXPA
    ]
    figure_names.extend(
        f"{FIG_PREFIX}mstar_forced_{model}_window{window}_tau{tau}.png"
        for model in MODEL_VARIANTS
    )
    figure_names.append(f"{FIG_PREFIX}mstar_tau_overlay_window{window}.png")
    figure_names.extend(
        f"{FIG_PREFIX}source_pinned_noise_budget_{model}_window{window}.png"
        for model in MODEL_VARIANTS
    )
    figure_names.extend(
        f"{FIG_PREFIX}policy_curves_model{model}_delay{delay}_window{window}.png"
        for model in MODEL_VARIANTS
        for delay in DELAY_LEN
    )
    figure_names.append(
        f"{FIG_PREFIX}natural_sanity_delay80_mem{NATURAL_SANITY_MEM}_"
        f"write{WRITE_SOURCE_ONLY}.png"
    )

    derived_names = [
        f"mstar_forced_window{window}_tau{tau}.csv",
        f"boundary_tau_sensitivity_window{window}.csv",
        f"boundary_seed_stability_window{window}_tau{tau}.csv",
    ]
    figure_paths = [Path(figure_dir) / name for name in figure_names]
    derived_paths = [Path(derived_dir) / name for name in derived_names]
    return figure_paths, derived_paths


def validate_paper_plot_data(df0, window):
    forced = df0[
        (df0["mask_mode"] == "forced")
        & (df0["window_size"] == int(window))
    ].copy()
    available = {
        (str(row.model), str(row.write_mode), int(row.delay_len))
        for row in forced[["model", "write_mode", "delay_len"]]
        .drop_duplicates()
        .itertuples(index=False)
    }
    expected = {
        (model, write_mode, int(delay))
        for model in MODEL_VARIANTS
        for write_mode in WRITE_MODES_EXPA
        for delay in DELAY_LEN
    }
    missing = sorted(expected - available)
    if missing:
        raise ValueError(
            "The input CSV is incomplete for the 17-figure ExpA set; "
            f"missing forced configurations: {missing}"
        )

    natural = df0[
        (df0["mask_mode"] == "natural")
        & (df0["delay_len"] == 80)
    ]
    natural_models = set(natural["model"].astype(str))
    required_natural_models = {"baseline", "naive", "gated"}
    if not required_natural_models.issubset(natural_models):
        missing_models = sorted(required_natural_models - natural_models)
        raise ValueError(
            "The input CSV lacks natural-sanity rows for: "
            f"{missing_models}"
        )


def preflight_outputs(paths, overwrite):
    paths = [Path(path) for path in paths]
    if len(paths) != len(set(paths)):
        raise ValueError("Duplicate ExpA output paths were constructed.")
    conflicts = [path for path in paths if path.exists()]
    if conflicts and not overwrite:
        formatted = "\n".join(f"- {path}" for path in conflicts)
        raise FileExistsError(
            "Refusing to overwrite existing ExpA outputs:\n"
            f"{formatted}\nPass --overwrite only when replacement is intentional."
        )


def main():
    parser = argparse.ArgumentParser(
        description="Generate the complete Experiment A figure and table set."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help="Formal ExpA CSV, relative to the project root or absolute.",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=DEFAULT_FIGURE_DIR,
        help="PNG output directory, relative to the project root or absolute.",
    )
    parser.add_argument(
        "--derived-dir",
        type=Path,
        default=DEFAULT_DERIVED_DIR,
        help="Derived CSV output directory, relative to the project root or absolute.",
    )
    parser.add_argument("--window", type=int, default=FORCED_WINDOW)
    parser.add_argument("--tau", type=float, default=DEFAULT_TAU)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of the 20 explicitly generated outputs.",
    )
    args = parser.parse_args()

    csv_path = resolve_project_path(args.csv)
    figure_dir = resolve_project_path(args.figure_dir)
    derived_dir = resolve_project_path(args.derived_dir)
    df0, csv_path = load_df(csv_path)
    validate_paper_plot_data(df0, args.window)

    figure_paths, derived_paths = expected_output_paths(
        figure_dir=figure_dir,
        derived_dir=derived_dir,
        window=args.window,
        tau=args.tau,
    )
    if len(figure_paths) != 17 or len(derived_paths) != 3:
        raise RuntimeError("Unexpected ExpA output inventory.")
    preflight_outputs(figure_paths + derived_paths, overwrite=args.overwrite)

    print(f"[INFO] csv={csv_path}")
    print(f"[INFO] figure_dir={figure_dir}")
    print(f"[INFO] derived_dir={derived_dir}")
    print(f"[INFO] window={args.window}")
    print(f"[INFO] main tau={args.tau}")
    print(f"[INFO] tau list={TAU_LIST}")
    print(f"[INFO] boundary write modes with pinned={BOUNDARY_WRITE_MODES_WITH_PINNED}")
    print(f"[INFO] write modes expA={WRITE_MODES_EXPA}")

    # Heatmaps for all write policies.
    for write_mode in WRITE_MODES_EXPA:
        plot_forced_heatmap(
            df0,
            write_mode=write_mode,
            window=args.window,
            figure_dir=figure_dir,
            overwrite=args.overwrite,
        )

    # Main boundary figure for all three policies.
    plot_mstar_forced(
        df0,
        window=args.window,
        tau=args.tau,
        figure_dir=figure_dir,
        derived_dir=derived_dir,
        overwrite=args.overwrite,
    )

    # Threshold and seed stability.
    plot_mstar_tau_overlay(
        df0,
        window=args.window,
        tau_list=TAU_LIST,
        figure_dir=figure_dir,
        overwrite=args.overwrite,
    )
    save_tau_sensitivity_table(
        df0,
        window=args.window,
        tau_list=TAU_LIST,
        derived_dir=derived_dir,
        overwrite=args.overwrite,
    )
    save_seed_stability_table(
        df0,
        window=args.window,
        tau=args.tau,
        derived_dir=derived_dir,
        overwrite=args.overwrite,
    )

    # Source-pinned control indexed by the number of allowed noise writes.
    plot_source_pinned_noise_budget(
        df0,
        window=args.window,
        figure_dir=figure_dir,
        overwrite=args.overwrite,
    )

    # Policy comparisons by delay and model.
    forced = df0[
        (df0["mask_mode"] == "forced") &
        (df0["window_size"] == int(args.window))
    ].copy()

    if len(forced) > 0:
        for model in sorted(forced["model"].unique().tolist()):
            for delay in sorted(forced["delay_len"].dropna().astype(int).unique().tolist()):
                plot_policy_curves_at_delay(
                    df0,
                    delay=delay,
                    window=args.window,
                    model=model,
                    figure_dir=figure_dir,
                    overwrite=args.overwrite,
                )

    # Optional natural-attention diagnostic for full-mode data.
    plot_natural_sanity(
        df0,
        delay=80,
        mem=NATURAL_SANITY_MEM,
        write_mode=WRITE_SOURCE_ONLY,
        figure_dir=figure_dir,
        overwrite=args.overwrite,
    )

    missing_outputs = [
        path for path in figure_paths + derived_paths if not path.is_file()
    ]
    if missing_outputs:
        formatted = "\n".join(f"- {path}" for path in missing_outputs)
        raise RuntimeError(f"ExpA generation was incomplete:\n{formatted}")

    print(
        f"[DONE] Generated {len(figure_paths)} PNG files and "
        f"{len(derived_paths)} derived CSV files."
    )


if __name__ == "__main__":
    main()
