import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# ============================================================
# CONFIGURATION
# ============================================================

CSV_PATH = "causal/3-repaired_data_discrete.csv"

# Column names
GENDER_COL = "Gender"
SAT_COL = "SAT"
HOBBY_COL = "Hobby"
ADMISSION_COL = "Admission"  # use FTA_Prediction for the FTA results, # or Admission for the training dataset

# Select which hobby to visualize
SELECTED_HOBBY = "Soccer"

# SAT axis
SAT_MIN = 350
SAT_MAX = 1700
SAT_TICK_INTERVAL = 100

# Jitter amount
# 0.10 = small spread
# 0.20 = moderate spread
# 0.30 = large spread
JITTER_WIDTH = 0.12

# Reproducible jitter
RANDOM_SEED = 42

# Point appearance
POINT_SIZE = 70
POINT_ALPHA = 0.85

# Decision boundary appearance
BOUNDARY_LINE_WIDTH = 2.5
BOUNDARY_LINE_STYLE = "--"
BOUNDARY_COLOR = "black"

# Horizontal extent of each gender column
# The lines will NOT cross into the other gender column.
COLUMN_HALF_WIDTH = 0.25

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(CSV_PATH)

# Keep only the selected hobby
plot_df = df[
    df[HOBBY_COL] == SELECTED_HOBBY
].copy()

# ============================================================
# MAP GENDER TO X-AXIS
# ============================================================

gender_to_x = {
    "M": 0,
    "F": 1
}

plot_df["x"] = plot_df[GENDER_COL].map(gender_to_x)

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
# ADMISSION COLORS
# ============================================================

admission_colors = {
    "Yes": "blue",
    "No": "red"
}

# ============================================================
# CALCULATE DECISION BOUNDARIES
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

    # Make sure both classes exist
    if len(accepted) == 0 or len(rejected) == 0:
        boundaries[gender] = None
        continue

    lowest_accepted = accepted.min()
    highest_rejected = rejected.max()

    boundary = (
        lowest_accepted + highest_rejected
    ) / 2

    boundaries[gender] = boundary

# ============================================================
# CALCULATE MUTUAL DECISION BOUNDARY STATISTICS
# ============================================================

male_boundary = boundaries["M"]
female_boundary = boundaries["F"]

if male_boundary is not None and female_boundary is not None:

   # Equation:
   # SAT = slope * Gender + intercept
   #
   # Gender:
   # M = 0
   # F = 1
   #
   # Therefore:
   # intercept = Male boundary
   # slope = Female boundary - Male boundary

   slope = female_boundary - male_boundary
   intercept = male_boundary

else:
   slope = None
   intercept = None
#
# ============================================================
# OUTPUT DECISION BOUNDARY STATS
# ============================================================
#
print()
print(f"{SELECTED_HOBBY}:")
print()

if male_boundary is not None:
   print(
       f"  Male boundary:   "
       f"{male_boundary:.1f} SAT"
   )
else:
   print(
       "  Male boundary:   "
       "N/A"
   )

if female_boundary is not None:
   print(
       f"  Female boundary: "
       f"{female_boundary:.1f} SAT"
   )
else:
   print(
       "  Female boundary: N/A"
   )

if slope is not None:
   print()
   print(
       f"  Boundary equation: "
       f"SAT = {slope:.1f} * Gender + "
       f"{intercept:.1f}"
   )

   print()
   print(
       f"  Gender gap / slope: "
       f"{abs(slope):.1f} SAT points"
   )

print()

# ============================================================
# CREATE PLOT
# ============================================================

plt.figure(figsize=(8, 8))

# ============================================================
# PLOT ADMISSION OUTCOMES
# ============================================================

for admission, color in admission_colors.items():

    subset = plot_df[
        plot_df[ADMISSION_COL] == admission
    ]

    plt.scatter(
        subset["x_jittered"],
        subset[SAT_COL],
        color=color,
        s=POINT_SIZE,
        edgecolors="black",
        linewidths=0.7,
        alpha=POINT_ALPHA
    )

# ============================================================
# DRAW MALE DECISION BOUNDARY
# ============================================================

if male_boundary is not None:

   plt.plot(
       [
           0 - COLUMN_HALF_WIDTH,
           0 + COLUMN_HALF_WIDTH
       ],
       [
           male_boundary,
           male_boundary
       ],
       color=BOUNDARY_COLOR,
       linestyle=BOUNDARY_LINE_STYLE,
       linewidth=BOUNDARY_LINE_WIDTH
   )

# ============================================================
# DRAW FEMALE DECISION BOUNDARY
# ============================================================

if female_boundary is not None:

   plt.plot(
       [
           1 - COLUMN_HALF_WIDTH,
           1 + COLUMN_HALF_WIDTH
       ],
       [
           female_boundary,
           female_boundary
       ],
       color=BOUNDARY_COLOR,
       linestyle=BOUNDARY_LINE_STYLE,
       linewidth=BOUNDARY_LINE_WIDTH
   )

# ============================================================
# AXES
# ============================================================

plt.xticks(
    [0, 1],
    ["Male", "Female"]
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

plt.xlabel("Gender")
plt.ylabel("SAT Score")

plt.title(
    f"Admission Outcomes — {SELECTED_HOBBY}"
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
        color=BOUNDARY_COLOR,
        linestyle=BOUNDARY_LINE_STYLE,
        linewidth=BOUNDARY_LINE_WIDTH,
        label="Decision boundary"
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