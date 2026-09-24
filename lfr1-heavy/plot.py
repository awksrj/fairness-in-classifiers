"""Plot LFR admissions and, optionally, learned SAT/Gender prototypes."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


# ============================================================
# CONFIGURATION
# ============================================================

# Comment out either "prototypes_path" entry to hide prototypes in that plot.
PLOTS = [
    {
        "csv_path": Path(r"lfr1-heavy/5-test_representations_4k_2d_min_discrimination.csv"),
        "prototypes_path": Path(r"lfr1-heavy/3-prototypes_4k_2d_min_discrimination.csv"),
        "output_path": Path(r"lfr1-heavy/plots/1-result-mindisc.png"),
        "title": "Admission Outcomes: Minimum Discrimination",
    },
    {
        "csv_path": Path(r"lfr1-heavy/5-test_representations_4k_2d_max_delta.csv"),
        "prototypes_path": Path(r"lfr1-heavy/3-prototypes_4k_2d_max_delta.csv"),
        "output_path": Path(r"lfr1-heavy/plots/2-result-maxdelta.png"),
        "title": "Admission Outcomes: Maximum Delta",
    },
]

GENDER_COL = "Gender"
SAT_COL = "SAT"
ADMISSION_COL = "Predicted_Admission"  # Or "Admission"

SAT_MIN = 350
SAT_MAX = 1700
SAT_TICK_INTERVAL = 100
JITTER_WIDTH = 0.12
RANDOM_SEED = 42
POINT_SIZE = 70
POINT_ALPHA = 0.85
BOUNDARY_LINE_WIDTH = 2.5
BOUNDARY_LINE_STYLE = "--"
BOUNDARY_COLOR = "black"
COLUMN_HALF_WIDTH = 0.25

# Exact horizontal coordinate: Male = 0, Female = 1, prototype gender in [0, 1].
# Move student dots outward only when displaying prototypes, keeping the middle clear.
PROTOTYPE_COLOR = "#ed64aa"
PROTOTYPE_SIZE = 180


def load_prototypes(path):
    prototypes = pd.read_csv(path)
    required = {"SAT", "Gender_Coordinate"}
    missing = required - set(prototypes.columns)
    if missing:
        raise ValueError(f"Prototype file is missing columns: {sorted(missing)}")
    for column in required:
        prototypes[column] = pd.to_numeric(prototypes[column], errors="raise")
    if prototypes[list(required)].isna().any().any():
        raise ValueError("Prototype SAT and Gender_Coordinate cannot be missing")
    if not prototypes["Gender_Coordinate"].between(0, 1).all():
        raise ValueError("Prototype Gender_Coordinate must be between 0 and 1")
    return prototypes


def plot_result(config):
    prototype_path = config.get("prototypes_path")
    prototypes = load_prototypes(prototype_path) if prototype_path is not None else None

    plot_df = pd.read_csv(config["csv_path"]).copy()
    plot_df["x"] = plot_df[GENDER_COL].map({"M": 0, "F": 1})
    if plot_df["x"].isna().any():
        raise ValueError("Gender must contain only M and F")

    rng = np.random.default_rng(RANDOM_SEED)
    if prototypes is None:
        jitter = rng.uniform(-JITTER_WIDTH, JITTER_WIDTH, size=len(plot_df))
    else:
        # Preserve the exact x-axis: student dots appear outside [0, 1], while
        # prototypes retain their actual Gender_Coordinate in the clear center.
        jitter = rng.uniform(0.02, JITTER_WIDTH, size=len(plot_df))
        jitter *= np.where(plot_df[GENDER_COL].eq("M"), -1, 1)
    plot_df["x_jittered"] = plot_df["x"] + jitter

    boundaries = {}
    for gender in ("M", "F"):
        group = plot_df[plot_df[GENDER_COL].eq(gender)]
        accepted = group.loc[group[ADMISSION_COL].eq("Yes"), SAT_COL]
        rejected = group.loc[group[ADMISSION_COL].eq("No"), SAT_COL]
        boundaries[gender] = (
            (accepted.min() + rejected.max()) / 2
            if not accepted.empty and not rejected.empty else None
        )

    male_boundary, female_boundary = boundaries["M"], boundaries["F"]
    print()
    print(f"  Male boundary:   {male_boundary:.1f} SAT" if male_boundary is not None else "  Male boundary:   N/A")
    print(f"  Female boundary: {female_boundary:.1f} SAT" if female_boundary is not None else "  Female boundary: N/A")
    if male_boundary is not None and female_boundary is not None:
        slope = female_boundary - male_boundary
        print(f"\n  Boundary equation: SAT = {slope:.1f} * Gender + {male_boundary:.1f}")
        print(f"\n  Gender gap / slope: {abs(slope):.1f} SAT points")
    print()

    fig, ax = plt.subplots(figsize=(11, 9) if prototypes is not None else (8, 8))
    for admission, color in (("Yes", "blue"), ("No", "red")):
        subset = plot_df[plot_df[ADMISSION_COL].eq(admission)]
        ax.scatter(
            subset["x_jittered"], subset[SAT_COL], color=color,
            s=POINT_SIZE, edgecolors="black", linewidths=0.7,
            alpha=POINT_ALPHA, zorder=2,
        )

    for x, boundary in ((0, male_boundary), (1, female_boundary)):
        if boundary is not None:
            ax.plot(
                [x - COLUMN_HALF_WIDTH, x + COLUMN_HALF_WIDTH],
                [boundary, boundary], color=BOUNDARY_COLOR,
                linestyle=BOUNDARY_LINE_STYLE,
                linewidth=BOUNDARY_LINE_WIDTH, zorder=3,
            )

    if prototypes is not None:
        # Gender_Coordinate stays on the true 0–1 horizontal scale. Labels are
        # offset in screen points only, so SAT/Gender dot positions stay exact.
        for row_number, (_, row) in enumerate(prototypes.iterrows(), start=1):
            x, sat = row["Gender_Coordinate"], row["SAT"]
            ax.scatter(
                x, sat, s=PROTOTYPE_SIZE, color=PROTOTYPE_COLOR,
                edgecolors="black", linewidths=1.0, zorder=5,
            )
            dx = 10 if x < 0.8 else -10
            align = "left" if dx > 0 else "right"
            dy = 12 if row_number % 2 else -14
            ax.annotate(
                f"P{row_number} ({sat:.0f}, {x:.2f})", (x, sat),
                xytext=(dx, dy), textcoords="offset points",
                ha=align, va="center", fontsize=9, color="#8c1457",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.5},
                zorder=6,
            )

    ax.set_xticks([0, 1], ["Male", "Female"])
    ax.set_yticks(range(SAT_MIN, SAT_MAX + 1, SAT_TICK_INTERVAL))
    ax.set_ylim(SAT_MIN, SAT_MAX)
    ax.set_xlim(-0.4, 1.4)
    ax.set_xlabel("Gender (prototype x = gender coordinate)")
    ax.set_ylabel("SAT Score")
    ax.set_title(config["title"])
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", label="Admission: Yes", markerfacecolor="blue", markeredgecolor="black", markersize=9),
        Line2D([0], [0], marker="o", color="w", label="Admission: No", markerfacecolor="red", markeredgecolor="black", markersize=9),
        Line2D([0], [0], color=BOUNDARY_COLOR, linestyle=BOUNDARY_LINE_STYLE, linewidth=BOUNDARY_LINE_WIDTH, label="Decision boundary"),
    ]
    if prototypes is not None:
        legend_elements.append(Line2D([0], [0], marker="o", color="w", label="Prototype (SAT, Gender)", markerfacecolor=PROTOTYPE_COLOR, markeredgecolor="black", markersize=11))
    ax.legend(handles=legend_elements, loc="best")
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    output_path = config["output_path"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved plot to: {output_path}")
    plt.show()
    plt.close(fig)


def main():
    for config in PLOTS:
        print(f"Plotting: {config['title']}")
        plot_result(config)


if __name__ == "__main__":
    main()
