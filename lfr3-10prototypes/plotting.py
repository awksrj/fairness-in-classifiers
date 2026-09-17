import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# ============================================================
# CONFIGURATION
# ============================================================

CSV_PATH = "lfr1-4prototypes/1-training_data.csv"

# Column names
GENDER_COL = "Gender"
SAT_COL = "SAT"
ADMISSION_COL = "Admission"

# SAT axis
SAT_MIN = 350
SAT_MAX = 1700
SAT_TICK_INTERVAL = 100

# Jitter amount
JITTER_WIDTH = 0.12

# Reproducible jitter
RANDOM_SEED = 42

# Training point appearance
POINT_SIZE = 70
POINT_ALPHA = 0.85

# Prototype appearance
PROTOTYPE_SIZE = 250
PROTOTYPE_COLOR = "green"
PROTOTYPE_MARKER = "*"

# Prototypes have no gender, so place them
# halfway between Male and Female.
PROTOTYPE_X = 0.5


# ============================================================
# LOAD TRAINING DATA
# ============================================================

df = pd.read_csv(CSV_PATH)

plot_df = df.copy()


# ============================================================
# TERMINAL INPUT: PROTOTYPES
# ============================================================

print("=" * 65)
print("LFR TRAINING DATA + PROTOTYPE PLOT")
print("=" * 65)
print()

K = int(
    input(
        "Number of prototypes K: "
    )
)

print()
print(
    "Enter each prototype's SAT location "
    "and Admission score."
)
print()

prototypes = []

for k in range(K):

    sat = float(
        input(
            f"Prototype {k + 1} SAT location: "
        )
    )

    score = float(
        input(
            f"Prototype {k + 1} Admission score: "
        )
    )

    prototypes.append({
        "Prototype": k + 1,
        "SAT": sat,
        "Admission_Score": score
    })

    print()

prototype_df = pd.DataFrame(
    prototypes
)


# ============================================================
# OUTPUT PROTOTYPES TO TERMINAL
# ============================================================

print()
print("=" * 65)
print("PROTOTYPES")
print("=" * 65)
print()

print(
    prototype_df.to_string(
        index=False
    )
)

print()


# ============================================================
# MAP GENDER TO X-AXIS
# ============================================================

gender_to_x = {
    "M": 0,
    "F": 1
}

plot_df["x"] = (
    plot_df[GENDER_COL]
    .map(gender_to_x)
)


# ============================================================
# ADD HORIZONTAL JITTER
# ============================================================

np.random.seed(
    RANDOM_SEED
)

plot_df["x_jittered"] = (
    plot_df["x"]
    + np.random.uniform(
        -JITTER_WIDTH,
        JITTER_WIDTH,
        size=len(plot_df)
    )
)


# ============================================================
# ADMISSION COLORS
# ============================================================

admission_colors = {
    "Yes": "blue",
    "No": "red"
}


# ============================================================
# CALCULATE TRAINING DATA DECISION BOUNDARIES
#
# Boundary = midpoint between:
#   highest rejected SAT
#   lowest accepted SAT
# ============================================================

boundaries = {}

for gender in ["M", "F"]:

    gender_df = plot_df[
        plot_df[GENDER_COL] == gender
    ]

    accepted = gender_df[
        gender_df[ADMISSION_COL] == "Yes"
    ][SAT_COL]

    rejected = gender_df[
        gender_df[ADMISSION_COL] == "No"
    ][SAT_COL]

    if (
        len(accepted) == 0
        or len(rejected) == 0
    ):
        boundaries[gender] = None
        continue

    lowest_accepted = accepted.min()
    highest_rejected = rejected.max()

    boundary = (
        lowest_accepted
        + highest_rejected
    ) / 2

    boundaries[gender] = boundary


# ============================================================
# OUTPUT DECISION BOUNDARIES
# ============================================================

print("=" * 65)
print("TRAINING DATA DECISION BOUNDARIES")
print("=" * 65)
print()

male_boundary = boundaries["M"]
female_boundary = boundaries["F"]

if male_boundary is not None:

    print(
        f"Male boundary:   "
        f"{male_boundary:.1f} SAT"
    )

else:

    print(
        "Male boundary:   N/A"
    )


if female_boundary is not None:

    print(
        f"Female boundary: "
        f"{female_boundary:.1f} SAT"
    )

else:

    print(
        "Female boundary: N/A"
    )


if (
    male_boundary is not None
    and female_boundary is not None
):

    slope = (
        female_boundary
        - male_boundary
    )

    print()
    print(
        f"Gender gap: "
        f"{abs(slope):.1f} SAT points"
    )

print()


# ============================================================
# CREATE PLOT
# ============================================================

plt.figure(
    figsize=(8, 8)
)


# ============================================================
# PLOT TRAINING DATA
# ============================================================

for admission, color in admission_colors.items():

    subset = plot_df[
        plot_df[ADMISSION_COL]
        == admission
    ]

    plt.scatter(
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
# PLOT PROTOTYPES
# ============================================================

for _, prototype in prototype_df.iterrows():

    prototype_number = int(
        prototype["Prototype"]
    )

    prototype_sat = (
        prototype["SAT"]
    )

    prototype_score = (
        prototype["Admission_Score"]
    )

    # Plot prototype
    plt.scatter(
        PROTOTYPE_X,
        prototype_sat,
        marker=PROTOTYPE_MARKER,
        s=PROTOTYPE_SIZE,
        color=PROTOTYPE_COLOR,
        edgecolors="black",
        linewidths=1.0,
        zorder=5
    )

    # Label prototype
    plt.annotate(
        (
            f"P{prototype_number}\n"
            f"SAT={prototype_sat:.0f}\n"
            f"score={prototype_score:.3f}"
        ),
        xy=(
            PROTOTYPE_X,
            prototype_sat
        ),
        xytext=(
            12,
            0
        ),
        textcoords="offset points",
        va="center",
        fontsize=9
    )


# ============================================================
# AXES
# ============================================================

plt.xticks(
    [0, 0.5, 1],
    [
        "Male",
        "Prototypes",
        "Female"
    ]
)

plt.yticks(
    range(
        SAT_MIN,
        SAT_MAX + 1,
        SAT_TICK_INTERVAL
    )
)

plt.ylim(
    SAT_MIN,
    SAT_MAX
)

plt.xlim(
    -0.4,
    1.4
)

plt.xlabel(
    "Gender / Prototypes"
)

plt.ylabel(
    "SAT Score"
)

plt.title(
    "Training Data and LFR Prototypes"
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
        marker=PROTOTYPE_MARKER,
        color="w",
        label="LFR Prototype",
        markerfacecolor=PROTOTYPE_COLOR,
        markeredgecolor="black",
        markersize=14
    )
]

plt.legend(
    handles=legend_elements
)


# ============================================================
# GRID
# ============================================================

plt.grid(
    axis="y",
    linestyle=":",
    alpha=0.4
)

plt.tight_layout()


# ============================================================
# SHOW
# ============================================================

plt.show()