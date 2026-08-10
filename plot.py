import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


# ============================================================
# CONFIGURATION — change these for different datasets
# ============================================================

# CSV_FILE = "fairness_through_awareness_redo/training_dataset_outliner.csv"
CSV_FILE = "fairness_through_awareness_3/training_dataset.csv"

# Column names
GENDER_COL = "Gender"
HOBBY_COL = "Hobby"
SAT_COL = "SAT"

# Column containing the labels to visualize
# Examples:
#   "Admission"
#   "FTA_Prediction"
#   "Prediction"
ADMISSION_COL = "Admission"

# Gender values
MALE_VALUE = "M"
FEMALE_VALUE = "F"

# Hobby values
SOCCER_VALUE = "Soccer"
DANCE_VALUE = "Dance"

# Admission values
ACCEPTED_VALUE = "Yes"
REJECTED_VALUE = "No"

# Plot title
PLOT_TITLE = "FTA Result with Decision Boundaries"

# SAT axis range
SAT_MIN = 550
SAT_MAX = 1650

# Jitter
JITTER = 0.05
RANDOM_SEED = 42

# Decision boundary color
BOUNDARY_COLOR = "orange"


# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv(CSV_FILE)


# ============================================================
# CONVERT CATEGORICAL VARIABLES TO NUMERIC
# ============================================================

gender_map = {
    MALE_VALUE: 0,
    FEMALE_VALUE: 1
}

hobby_map = {
    SOCCER_VALUE: 0,
    DANCE_VALUE: 1
}

admission_map = {
    ACCEPTED_VALUE: "tab:blue",
    REJECTED_VALUE: "tab:red"
}

df["GenderNum"] = df[GENDER_COL].map(gender_map)
df["HobbyNum"] = df[HOBBY_COL].map(hobby_map)

colors = df[ADMISSION_COL].map(admission_map)


# ============================================================
# ADD SLIGHT JITTER
# ============================================================

np.random.seed(RANDOM_SEED)

x = (
    df["GenderNum"]
    + np.random.uniform(-JITTER, JITTER, len(df))
)

y = (
    df["HobbyNum"]
    + np.random.uniform(-JITTER, JITTER, len(df))
)

z = df[SAT_COL]


# ============================================================
# CREATE FIGURE
# ============================================================

fig = plt.figure(figsize=(10, 8))

ax = fig.add_subplot(
    111,
    projection="3d"
)


# ============================================================
# PLOT APPLICANTS
# ============================================================

ax.scatter(
    x,
    y,
    z,
    c=colors,
    s=60,
    edgecolors="black"
)


# ============================================================
# FIND DECISION BOUNDARY
#
# Boundary = midpoint between:
#   highest rejected SAT
#   lowest accepted SAT
#
# for each Gender + Hobby group.
# ============================================================

def find_boundary(data):

    accepted = data.loc[
        data[ADMISSION_COL] == ACCEPTED_VALUE,
        SAT_COL
    ]

    rejected = data.loc[
        data[ADMISSION_COL] == REJECTED_VALUE,
        SAT_COL
    ]

    if len(accepted) > 0 and len(rejected) > 0:

        lowest_accepted = accepted.min()
        highest_rejected = rejected.max()

        return (
            lowest_accepted + highest_rejected
        ) / 2

    return None


# ============================================================
# CALCULATE DECISION BOUNDARIES
# ============================================================

hobbies = [
    SOCCER_VALUE,
    DANCE_VALUE
]

genders = [
    MALE_VALUE,
    FEMALE_VALUE
]

boundaries = {}

for hobby in hobbies:

    boundaries[hobby] = {}

    for gender in genders:

        subset = df[
            (df[HOBBY_COL] == hobby)
            & (df[GENDER_COL] == gender)
        ]

        boundary = find_boundary(subset)

        boundaries[hobby][gender] = boundary


# ============================================================
# PRINT DECISION BOUNDARIES
# ============================================================

print("\nDecision boundaries:")

for hobby in hobbies:

    male_boundary = boundaries[hobby][MALE_VALUE]
    female_boundary = boundaries[hobby][FEMALE_VALUE]

    print(f"\n{hobby}:")

    if male_boundary is None:
        print("  Male boundary:   N/A")
    else:
        print(
            f"  Male boundary:   "
            f"{male_boundary:.1f} SAT"
        )

    if female_boundary is None:
        print("  Female boundary: N/A")
    else:
        print(
            f"  Female boundary: "
            f"{female_boundary:.1f} SAT"
        )

    # Only calculate slope if both boundaries exist
    if (
        male_boundary is not None
        and female_boundary is not None
    ):

        # Gender:
        # Male = 0
        # Female = 1
        #
        # SAT = slope * Gender + intercept

        slope = (
            female_boundary
            - male_boundary
        )

        intercept = male_boundary

        print(
            f"  Boundary equation: "
            f"SAT = {slope:.1f} * Gender "
            f"+ {intercept:.1f}"
        )

        print(
            f"  Gender gap / slope: "
            f"{slope:.1f} SAT points"
        )


# ============================================================
# PLOT DECISION BOUNDARY LINES
#
# One line for Soccer:
#   Male → Female
#
# One line for Dance:
#   Male → Female
# ============================================================

for hobby in hobbies:

    hobby_num = hobby_map[hobby]

    male_boundary = boundaries[hobby][MALE_VALUE]
    female_boundary = boundaries[hobby][FEMALE_VALUE]

    # Skip if a boundary cannot be calculated
    if (
        male_boundary is None
        or female_boundary is None
    ):
        continue

    # Gender coordinates
    boundary_x = np.array([0, 1])

    # SAT coordinates
    boundary_z = np.array([
        male_boundary,
        female_boundary
    ])

    # Keep the line at the Hobby's y-coordinate
    boundary_y = np.array([
        hobby_num,
        hobby_num
    ])

    ax.plot(
        boundary_x,
        boundary_y,
        boundary_z,
        color=BOUNDARY_COLOR,
        linewidth=3,
        label=f"{hobby} decision boundary"
    )


# ============================================================
# STANDARD 3D GRAPH VIEW
# ============================================================

ax.view_init(
    elev=20,
    azim=-60
)


# ============================================================
# AXIS LABELS
# ============================================================

ax.set_xlabel("Gender")
ax.set_ylabel("Hobby")
ax.set_zlabel("SAT Score")


# ============================================================
# CATEGORICAL TICK LABELS
# ============================================================

ax.set_xticks([0, 1])

ax.set_xticklabels([
    "Male",
    "Female"
])

ax.set_yticks([0, 1])

ax.set_yticklabels([
    "Soccer",
    "Dance"
])


# ============================================================
# SAT RANGE
# ============================================================

ax.set_zlim(
    SAT_MIN,
    SAT_MAX
)


# ============================================================
# LEGEND
# ============================================================

accepted = plt.Line2D(
    [0],
    [0],
    marker="o",
    color="w",
    markerfacecolor="tab:blue",
    markeredgecolor="black",
    markersize=8,
    label="Accepted"
)

rejected = plt.Line2D(
    [0],
    [0],
    marker="o",
    color="w",
    markerfacecolor="tab:red",
    markeredgecolor="black",
    markersize=8,
    label="Rejected"
)

ax.legend(
    handles=[
        accepted,
        rejected
    ],
    loc="upper left"
)


# ============================================================
# TITLE
# ============================================================

ax.set_title(PLOT_TITLE)


# ============================================================
# DISPLAY
# ============================================================

plt.tight_layout()
plt.show()

