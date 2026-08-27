import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ============================================================
# Configuration
# ============================================================
# CSV_FILE = "causal-2/1-training_data.csv"
# CSV_FILE = "causal-2/3-repaired_data.csv"
CSV_FILE = "causal-2/8-evaluation_result.csv"


# ============================================================
# Load dataset
# ============================================================

df = pd.read_csv(CSV_FILE)

# ============================================================
# Encode categorical variables
# ============================================================

gender_map = {
    "Male": 0,
    "Female": 1
}

department_map = {
    "A": 0,
    "B": 1
}

# Q1 = highest
# Q4 = lowest
qualification_map = {
    "Q1": 4,
    "Q2": 3,
    "Q3": 2,
    "Q4": 1
}

df["Gender_num"] = df["Gender"].map(gender_map)
df["Department_num"] = df["Department"].map(department_map)
df["Qualification_num"] = df["Qualification"].map(
    qualification_map
)

# ============================================================
# Add jitter
# ============================================================

np.random.seed(42)

df["x"] = (
    df["Gender_num"]
    + np.random.uniform(-0.12, 0.12, len(df))
)

df["y"] = (
    df["Department_num"]
    + np.random.uniform(-0.12, 0.12, len(df))
)

df["z"] = (
    df["Qualification_num"]
    + np.random.uniform(-0.08, 0.08, len(df))
)

# ============================================================
# Admission colors
# ============================================================

colors = df["Admission"].map({
    "Yes": "blue",
    "No": "red"
})

# ============================================================
# Function to calculate decision boundary
# ============================================================

def get_boundary(group):

    admitted = group[
        group["Admission"] == "Yes"
    ]

    rejected = group[
        group["Admission"] == "No"
    ]

    # Everyone admitted
    if len(rejected) == 0:
        return 0.5

    # Everyone rejected
    if len(admitted) == 0:
        return 4.5

    lowest_admitted = (
        admitted["Qualification_num"].min()
    )

    highest_rejected = (
        rejected["Qualification_num"].max()
    )

    return (
        lowest_admitted
        + highest_rejected
    ) / 2


# ============================================================
# Calculate boundaries
# ============================================================

boundaries = {}

for department in ["A", "B"]:

    for gender in ["Male", "Female"]:

        group = df[
            (df["Department"] == department)
            &
            (df["Gender"] == gender)
        ]

        boundaries[
            (department, gender)
        ] = get_boundary(group)


# ============================================================
# PRINT DECISION BOUNDARY DATA
# ============================================================

print()
print("=" * 60)
print("DECISION BOUNDARIES")
print("=" * 60)

for department in ["A", "B"]:

    male_boundary = boundaries[
        (department, "Male")
    ]

    female_boundary = boundaries[
        (department, "Female")
    ]

    slope = (
        female_boundary
        - male_boundary
    )

    print()
    print(f"Department {department}")
    print(
        f"  Male boundary:   "
        f"{male_boundary:.2f} Qualification"
    )

    print(
        f"  Female boundary: "
        f"{female_boundary:.2f} Qualification"
    )

    print(
        f"  Boundary equation: "
        f"Qualification = "
        f"{slope:.2f} * Gender + "
        f"{male_boundary:.2f}"
    )

    print(
        f"  Gender gap / slope: "
        f"{slope:.2f} Qualification points"
    )

print()
print("=" * 60)


# ============================================================
# CREATE 3D PLOT
# ============================================================

fig = plt.figure(figsize=(12, 9))

ax = fig.add_subplot(
    111,
    projection="3d"
)

ax.scatter(
    df["x"],
    df["y"],
    df["z"],
    c=colors,
    s=45,
    alpha=0.75,
    edgecolors="black",
    linewidths=0.3
)

# ============================================================
# Draw decision boundaries
# ============================================================

for department, y_pos in department_map.items():

    male_boundary = boundaries[
        (department, "Male")
    ]

    female_boundary = boundaries[
        (department, "Female")
    ]

    # Male -> Female
    x_line = np.array([
        0,
        1
    ])

    # Department stays fixed
    y_line = np.array([
        y_pos,
        y_pos
    ])

    # Boundary changes across gender
    z_line = np.array([
        male_boundary,
        female_boundary
    ])

    ax.plot(
        x_line,
        y_line,
        z_line,
        color="gray",
        linewidth=3,
        linestyle="--",
        alpha=0.7
    )


# ============================================================
# Axis labels
# ============================================================

ax.set_xlabel(
    "Gender",
    labelpad=12
)

ax.set_ylabel(
    "Department",
    labelpad=12
)

ax.set_zlabel(
    "Qualification",
    labelpad=12
)

# ============================================================
# Gender
# ============================================================

ax.set_xticks([
    0,
    1
])

ax.set_xticklabels([
    "Male",
    "Female"
])

# ============================================================
# Department
# ============================================================

ax.set_yticks([
    0,
    1
])

ax.set_yticklabels([
    "A",
    "B"
])

# ============================================================
# Qualification
# ============================================================

ax.set_zticks([
    1,
    2,
    3,
    4
])

ax.set_zticklabels([
    "Q4 (Lowest)",
    "Q3",
    "Q2",
    "Q1 (Highest)"
])

ax.set_zlim(
    0.5,
    4.5
)

# ============================================================
# Legend
# ============================================================

legend_elements = [

    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        label="Admitted",
        markerfacecolor="blue",
        markeredgecolor="black",
        markersize=9
    ),

    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        label="Rejected",
        markerfacecolor="red",
        markeredgecolor="black",
        markersize=9
    ),

    Line2D(
        [0],
        [0],
        color="gray",
        linewidth=3,
        linestyle="--",
        label="Decision boundary"
    )
]

ax.legend(
    handles=legend_elements,
    title="Admission",
    loc="upper left"
)

# ============================================================
# Title
# ============================================================

ax.set_title(
    "College Admission Dataset\n"
    "Gender × Department × Qualification",
    pad=20
)

# ============================================================
# Viewing angle
# ============================================================

ax.view_init(
    elev=25,
    azim=-55
)

plt.tight_layout()

plt.show()