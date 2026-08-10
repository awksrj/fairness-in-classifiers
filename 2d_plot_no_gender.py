import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# ============================================================
# CONFIGURATION
# ============================================================

CSV_PATH = (
    "fairness_through_awareness_3/"
    "training_dataset.csv"
)

# Column names
SAT_COL = "SAT"
HOBBY_COL = "Hobby"
ADMISSION_COL = "Admission"
GENDER_COL = "Gender"

# SAT axis
SAT_MIN = 350
SAT_MAX = 1700
SAT_TICK_INTERVAL = 100

# Horizontal jitter
JITTER_WIDTH = 0.12

# Reproducible jitter
RANDOM_SEED = 42

# Point appearance
POINT_SIZE = 70
POINT_ALPHA = 0.85

# Admission colors
ADMISSION_COLORS = {
    "Yes": "blue",
    "No": "red"
}

# ============================================================
# FEMALE BOX CONFIGURATION
# ============================================================

FEMALE_BOX_COLOR = "pink"

# Square marker size
# Increase/decrease this to change the box size.
#
# 32 is approximately a small ~2 mm visual square.

FEMALE_BOX_SIZE = 180

FEMALE_BOX_LINEWIDTH = 3

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(CSV_PATH)

plot_df = df[
    [
        SAT_COL,
        HOBBY_COL,
        ADMISSION_COL,
        GENDER_COL
    ]
].copy()

plot_df = plot_df.dropna(
    subset=[
        SAT_COL,
        HOBBY_COL,
        ADMISSION_COL,
        GENDER_COL
    ]
)

# ============================================================
# FIND HOBBIES
# ============================================================

hobbies = sorted(
    plot_df[HOBBY_COL].unique()
)

num_hobbies = len(hobbies)

# ============================================================
# MAP HOBBIES TO X-AXIS
# ============================================================

hobby_to_x = {
    hobby: i
    for i, hobby in enumerate(hobbies)
}

plot_df["x"] = plot_df[
    HOBBY_COL
].map(hobby_to_x)

# ============================================================
# ADD HORIZONTAL JITTER
# ============================================================

np.random.seed(RANDOM_SEED)

plot_df["x_jittered"] = (
    plot_df["x"]
    + np.random.uniform(
        -JITTER_WIDTH,
        JITTER_WIDTH,
        size=len(plot_df)
    )
)

# ============================================================
# CREATE PLOT
# ============================================================

fig, ax = plt.subplots(
    figsize=(8, 8)
)

# ============================================================
# PLOT ADMISSION OUTCOMES
# ============================================================

for admission, color in ADMISSION_COLORS.items():

    subset = plot_df[
        plot_df[ADMISSION_COL] == admission
    ]

    ax.scatter(
        subset["x_jittered"],
        subset[SAT_COL],
        color=color,
        s=POINT_SIZE,
        edgecolors="black",
        linewidths=0.7,
        alpha=POINT_ALPHA,
        zorder=2
    )

# ============================================================
# BOX EACH FEMALE DOT
#
# Square pink outline
# No fill
# ============================================================

female_subset = plot_df[
    plot_df[GENDER_COL] == "F"
]

ax.scatter(
    female_subset["x_jittered"],
    female_subset[SAT_COL],

    marker="s",

    # No fill
    facecolors="none",

    # Pink border
    edgecolors=FEMALE_BOX_COLOR,

    # Box size
    s=FEMALE_BOX_SIZE,

    # Border thickness
    linewidths=FEMALE_BOX_LINEWIDTH,

    zorder=3
)

# ============================================================
# X-AXIS
# ============================================================

ax.set_xticks(
    range(num_hobbies)
)

ax.set_xticklabels(
    hobbies
)

if num_hobbies == 1:

    ax.set_xlim(
        -0.5,
        0.5
    )

else:

    ax.set_xlim(
        -0.4,
        num_hobbies - 1 + 0.4
    )

# ============================================================
# Y-AXIS
# ============================================================

ax.set_yticks(
    range(
        SAT_MIN,
        SAT_MAX + 1,
        SAT_TICK_INTERVAL
    )
)

ax.set_ylim(
    SAT_MIN,
    SAT_MAX
)

# ============================================================
# LABELS
# ============================================================

ax.set_xlabel("Hobby")
ax.set_ylabel("SAT Score")

ax.set_title(
    "Admission Outcomes by Hobby"
)

# ============================================================
# LEGEND
# ============================================================

legend_elements = [

    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        label="Admission: Yes",
        markerfacecolor="blue",
        markeredgecolor="black",
        markersize=9
    ),

    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        label="Admission: No",
        markerfacecolor="red",
        markeredgecolor="black",
        markersize=9
    ),

    Line2D(
        [0],
        [0],
        marker="s",
        color="pink",
        markerfacecolor="none",
        markeredgecolor="pink",
        label="Female applicant",
        markersize=8
    )
]

ax.legend(
    handles=legend_elements
)

# ============================================================
# GRID
# ============================================================

ax.grid(
    axis="y",
    linestyle=":",
    alpha=0.4,
    zorder=0
)

plt.tight_layout()

# ============================================================
# SHOW
# ============================================================

plt.show()