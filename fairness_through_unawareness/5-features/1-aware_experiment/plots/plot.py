import os

import matplotlib.pyplot as plt
import mplcursors
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_SEED = 42

INPUT_CSV_PATH = (
    "fairness_through_unawareness\\1-aware_experiment\\3-aware_predictions.csv"
)

OUTPUT_PNG_PATH = (
    "fairness_through_unawareness\\1-aware_experiment\\plots"
    "\\aware_predictions_sat_gender_major_hobbyfiltered.png"
)

TITLE = "Aware Predictions: SAT vs Gender vs Major (only Dance and Art)"

X_COLUMN = "Gender"
Y_COLUMN = "Major"
Z_COLUMN = "SAT"

LABEL_COLUMN = "Predicted_Admission"

# Set to None to include all hobbies.
# Examples:
# HOBBY_FILTER = ["Dance"]
HOBBY_FILTER = ["Dance", "Art"]
# HOBBY_FILTER = None

X_CATEGORY_ORDER = [
    "Male",
    "Female",
]

Y_CATEGORY_ORDER = [
    "Computer Science",
    "Humanities",
]

Z_TICKS = [
    600,
    800,
    1000,
    1200,
    1400,
    1600,
]

Z_LIMITS = (
    600,
    1600,
)

ACCEPTED_VALUE = 1
ACCEPTED_COLOR = "blue"
REJECTED_COLOR = "red"

POINT_SIZE = 35
POINT_ALPHA = 0.65

X_JITTER = 0.08
Y_JITTER = 0.10

BOUNDARY_COLOR = "black"
BOUNDARY_LINESTYLE = "--"
BOUNDARY_LINEWIDTH = 2.0

TOOLTIP_COLUMNS = [
    "Gender",
    "SAT",
    "GPA",
    "Hobby",
    "Major",
    LABEL_COLUMN,
]


# ============================================================
# HELPERS
# ============================================================

def category_positions(categories):
    return {
        category: position
        for position, category in enumerate(categories)
    }


def decision_boundary_by_group(group_df, x_column, z_column, label_column):
    boundary_df = group_df[
        [x_column, z_column, label_column]
    ].dropna()

    if boundary_df[label_column].nunique() < 2:
        return None

    X = pd.DataFrame({
        "x_position": boundary_df[x_column].map(x_positions),
        z_column: boundary_df[z_column],
    })

    y = boundary_df[label_column].astype(int)

    model = LogisticRegression(
        random_state=RANDOM_SEED
    )

    model.fit(
        X,
        y
    )

    intercept = model.intercept_[0]
    x_coefficient = model.coef_[0][0]
    z_coefficient = model.coef_[0][1]

    if np.isclose(z_coefficient, 0):
        return None

    boundary_points = []

    for category in X_CATEGORY_ORDER:
        x_position = x_positions[category]

        z_boundary = -(
            intercept
            + x_coefficient * x_position
        ) / z_coefficient

        boundary_points.append(
            (
                x_position,
                z_boundary
            )
        )

    return boundary_points


def format_tooltip(row):
    lines = []

    for column in TOOLTIP_COLUMNS:
        if column in row.index:
            lines.append(
                f"{column}: {row[column]}"
            )

    return "\n".join(lines)


def add_click_tooltips(scatter_to_data):
    cursor = mplcursors.cursor(
        list(scatter_to_data.keys()),
        hover=False
    )

    @cursor.connect("add")
    def on_add(selection):
        source_df = scatter_to_data[
            selection.artist
        ]

        row = source_df.iloc[
            int(selection.index)
        ]

        selection.annotation.set_text(
            format_tooltip(row)
        )

        selection.annotation.get_bbox_patch().set(
            alpha=0.9
        )


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(
    INPUT_CSV_PATH
)

required_columns = [
    X_COLUMN,
    Y_COLUMN,
    Z_COLUMN,
    LABEL_COLUMN,
    "Hobby",
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# ============================================================
# PREPARE PLOT DATA
# ============================================================

x_positions = category_positions(
    X_CATEGORY_ORDER
)

y_positions = category_positions(
    Y_CATEGORY_ORDER
)

plot_df = df[
    df[X_COLUMN].isin(X_CATEGORY_ORDER)
    & df[Y_COLUMN].isin(Y_CATEGORY_ORDER)
].copy()

if HOBBY_FILTER is not None:
    plot_df = plot_df[
        plot_df["Hobby"].isin(HOBBY_FILTER)
    ].copy()

plot_df[Z_COLUMN] = pd.to_numeric(
    plot_df[Z_COLUMN],
    errors="coerce"
)

plot_df = plot_df.dropna(
    subset=[
        X_COLUMN,
        Y_COLUMN,
        Z_COLUMN,
        LABEL_COLUMN,
    ]
)

rng = np.random.default_rng(
    RANDOM_SEED
)

plot_df["x_position"] = (
    plot_df[X_COLUMN]
    .map(x_positions)
)

plot_df["y_position"] = (
    plot_df[Y_COLUMN]
    .map(y_positions)
)

plot_df["x_jitter"] = (
    plot_df["x_position"]
    + rng.uniform(
        -X_JITTER,
        X_JITTER,
        len(plot_df)
    )
)

plot_df["y_jitter"] = (
    plot_df["y_position"]
    + rng.uniform(
        -Y_JITTER,
        Y_JITTER,
        len(plot_df)
    )
)

accepted = plot_df[
    plot_df[LABEL_COLUMN] == ACCEPTED_VALUE
].reset_index(
    drop=True
)

rejected = plot_df[
    plot_df[LABEL_COLUMN] != ACCEPTED_VALUE
].reset_index(
    drop=True
)


# ============================================================
# PLOT
# ============================================================

fig = plt.figure(
    figsize=(11, 8)
)

ax = fig.add_subplot(
    111,
    projection="3d"
)

rejected_scatter = ax.scatter(
    rejected["x_jitter"],
    rejected["y_jitter"],
    rejected[Z_COLUMN],
    color=REJECTED_COLOR,
    s=POINT_SIZE,
    alpha=POINT_ALPHA,
    label="Rejected"
)

accepted_scatter = ax.scatter(
    accepted["x_jitter"],
    accepted["y_jitter"],
    accepted[Z_COLUMN],
    color=ACCEPTED_COLOR,
    s=POINT_SIZE,
    alpha=POINT_ALPHA,
    label="Accepted"
)


# ============================================================
# DECISION BOUNDARIES
# ============================================================

for y_category in Y_CATEGORY_ORDER:
    group_df = plot_df[
        plot_df[Y_COLUMN] == y_category
    ]

    boundary_points = decision_boundary_by_group(
        group_df,
        X_COLUMN,
        Z_COLUMN,
        LABEL_COLUMN
    )

    if boundary_points is None:
        continue

    x_values = [
        point[0]
        for point in boundary_points
    ]

    z_values = [
        np.clip(
            point[1],
            Z_LIMITS[0],
            Z_LIMITS[1]
        )
        for point in boundary_points
    ]

    y_value = y_positions[y_category]

    ax.plot(
        x_values,
        [
            y_value,
            y_value,
        ],
        z_values,
        color=BOUNDARY_COLOR,
        linestyle=BOUNDARY_LINESTYLE,
        linewidth=BOUNDARY_LINEWIDTH,
    )


# ============================================================
# AXES AND OUTPUT
# ============================================================

ax.set_xticks(
    list(x_positions.values())
)

ax.set_xticklabels(
    X_CATEGORY_ORDER
)

ax.set_yticks(
    list(y_positions.values())
)

ax.set_yticklabels(
    Y_CATEGORY_ORDER
)

ax.set_zticks(
    Z_TICKS
)

ax.set_zlim(
    Z_LIMITS
)

ax.set_xlabel(
    X_COLUMN
)

ax.set_ylabel(
    Y_COLUMN
)

ax.set_zlabel(
    Z_COLUMN
)

ax.set_title(
    TITLE
)

ax.view_init(
    elev=24,
    azim=-55
)

ax.legend()

add_click_tooltips({
    rejected_scatter: rejected,
    accepted_scatter: accepted,
})

plt.tight_layout()

os.makedirs(
    os.path.dirname(OUTPUT_PNG_PATH),
    exist_ok=True
)

plt.savefig(
    OUTPUT_PNG_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.show()
