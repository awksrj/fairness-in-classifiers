import ast
import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ============================================================
# CONFIGURATION
# ============================================================

TRAINING_CSV = "lfr-prototype-locations/1-training_data.csv"

RESULT_CSV = "lfr-prototype-locations/2-test_cases.csv"

OUTPUT_DIR = "lfr-prototype-locations/plots"


# ============================================================
# COLUMN NAMES
# ============================================================

GENDER_COL = "Gender"
SAT_COL = "SAT"
ADMISSION_COL = "Admission"

TEST_COL = "Test"

PROTOTYPE_COL = (
    "Prototype locations and scores"
)


# ============================================================
# SAT AXIS
# ============================================================

SAT_MIN = 350
SAT_MAX = 1700
SAT_TICK_INTERVAL = 100


# ============================================================
# POINT APPEARANCE
# ============================================================

JITTER_WIDTH = 0.12
RANDOM_SEED = 42

POINT_SIZE = 70
POINT_ALPHA = 0.85


# ============================================================
# PROTOTYPE APPEARANCE
# ============================================================

PROTOTYPE_COLOR = "hotpink"

PROTOTYPE_SIZE = 180

PROTOTYPE_LINE_WIDTH = 2.5

PROTOTYPE_MARKER = "D"

# Horizontal location of prototype markers.
# Prototypes are not gender-specific, so they are placed
# between the Male and Female columns.
PROTOTYPE_X = 0.5


# ============================================================
# LOAD DATA
# ============================================================

training_df = pd.read_csv(
    TRAINING_CSV
)

result_df = pd.read_csv(
    RESULT_CSV
)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# MAP GENDER TO X-AXIS
# ============================================================

gender_to_x = {
    "M": 0,
    "F": 1
}

training_df["x"] = (
    training_df[GENDER_COL]
    .map(gender_to_x)
)


# ============================================================
# ADD HORIZONTAL JITTER
# ============================================================

np.random.seed(
    RANDOM_SEED
)

training_df["x_jittered"] = (
    training_df["x"]
    + np.random.uniform(
        -JITTER_WIDTH,
        JITTER_WIDTH,
        size=len(training_df)
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
# PARSE PROTOTYPES
# ============================================================

def parse_prototypes(value):

    """
    Example input:

    (500,0), (850,0.3), (1250,0.7), (1580,1)

    Returns:

    locations:
        [500, 850, 1250, 1580]

    scores:
        [0, 0.3, 0.7, 1]
    """

    value = str(value).strip()

    parsed = ast.literal_eval(
        "[" + value + "]"
    )

    locations = []
    scores = []

    for prototype in parsed:

        if (
            not isinstance(prototype, tuple)
            or len(prototype) != 2
        ):

            raise ValueError(
                f"Invalid prototype: {prototype}"
            )

        location, score = prototype

        locations.append(
            float(location)
        )

        scores.append(
            float(score)
        )

    return (
        np.array(locations),
        np.array(scores)
    )


# ============================================================
# SAFE FILE NAME
# ============================================================

def safe_filename(name):

    """
    Convert experiment name into a filename safe
    for Windows.

    Example:

    "L1: Spread"
        ->
    "L1_Spread.png"
    """

    name = str(name).strip()

    # Replace invalid Windows filename characters
    name = re.sub(
        r'[<>:"/\\|?*]',
        "_",
        name
    )

    # Replace spaces with underscores
    name = re.sub(
        r"\s+",
        "_",
        name
    )

    # Remove repeated underscores
    name = re.sub(
        r"_+",
        "_",
        name
    )

    return name.strip("_")


# ============================================================
# LOOP THROUGH ALL EXPERIMENTS
# ============================================================

for experiment_index, row in result_df.iterrows():

    # --------------------------------------------------------
    # EXPERIMENT INFORMATION
    # --------------------------------------------------------

    test_name = row[
        TEST_COL
    ]

    prototype_string = row[
        PROTOTYPE_COL
    ]

    prototypes_sat, prototype_scores = (
        parse_prototypes(
            prototype_string
        )
    )


    # ========================================================
    # CREATE PLOT
    # ========================================================

    fig, ax = plt.subplots(
        figsize=(8, 8)
    )


    # ========================================================
    # PLOT ORIGINAL TRAINING DATA
    # ========================================================

    for admission, color in admission_colors.items():

        subset = training_df[
            training_df[
                ADMISSION_COL
            ] == admission
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


    # ========================================================
    # PLOT PROTOTYPES
    # ========================================================
    #
    # Prototype is placed between Male and Female because
    # the prototype itself is not gender-specific.
    #
    # y-axis = SAT location
    # label  = admission score
    #
    # ========================================================

    for k, (
        prototype_sat,
        prototype_score
    ) in enumerate(
        zip(
            prototypes_sat,
            prototype_scores
        ),
        start=1
    ):

        ax.scatter(
            PROTOTYPE_X,
            prototype_sat,

            color=PROTOTYPE_COLOR,

            marker=PROTOTYPE_MARKER,

            s=PROTOTYPE_SIZE,

            edgecolors="black",

            linewidths=1.2,

            zorder=5
        )


        # ----------------------------------------------------
        # LABEL PROTOTYPE
        # ----------------------------------------------------

        ax.annotate(

            f"P{k}: "
            f"SAT={prototype_sat:.0f}, "
            f"score={prototype_score:.2f}",

            xy=(
                PROTOTYPE_X,
                prototype_sat
            ),

            xytext=(
                12,
                0
            ),

            textcoords="offset points",

            verticalalignment="center",

            fontsize=9,

            zorder=6
        )


    # ========================================================
    # AXES
    # ========================================================

    ax.set_xticks(
        [0, 1]
    )

    ax.set_xticklabels(
        [
            "Male",
            "Female"
        ]
    )


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


    ax.set_xlim(
        -0.4,
        1.4
    )


    ax.set_xlabel(
        "Gender"
    )

    ax.set_ylabel(
        "SAT Score"
    )


    # ========================================================
    # TITLE
    # ========================================================

    ax.set_title(
        f"{test_name}\n"
        f"Prototype Locations and Scores"
    )


    # ========================================================
    # LOSS INFORMATION
    # ========================================================
    #
    # If these columns exist in the result CSV,
    # display them directly on the plot.
    #
    # ========================================================

    if all(
        column in result_df.columns
        for column in [
            "Alpha",
            "Lz",
            "Lx",
            "Ly",
            "L"
        ]
    ):

        loss_text = (
            f"α = {row['Alpha']:.4f}\n"
            f"Lz = {row['Lz']:.4f}\n"
            f"Lx = {row['Lx']:.4f}\n"
            f"Ly = {row['Ly']:.4f}\n"
            f"L = {row['L']:.4f}"
        )

        ax.text(
            0.02,
            0.98,

            loss_text,

            transform=ax.transAxes,

            verticalalignment="top",

            horizontalalignment="left",

            fontsize=9,

            bbox={
                "boxstyle": "round",
                "facecolor": "white",
                "alpha": 0.8
            },

            zorder=10
        )


    # ========================================================
    # LEGEND
    # ========================================================

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

            label="Prototype",

            markerfacecolor=PROTOTYPE_COLOR,

            markeredgecolor="black",

            markersize=10
        )
    ]


    ax.legend(
        handles=legend_elements,
        loc="lower right"
    )


    # ========================================================
    # GRID
    # ========================================================

    ax.grid(
        axis="y",
        linestyle=":",
        alpha=0.4
    )


    plt.tight_layout()


    # ========================================================
    # SAVE PLOT
    # ========================================================

    filename = (
        safe_filename(
            test_name
        )
        + ".png"
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        filename
    )


    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )


    # Important when producing many plots
    plt.close(
        fig
    )


    print(
        f"Saved: {output_path}"
    )


# ============================================================
# FINISHED
# ============================================================

print()

print(
    f"Finished creating "
    f"{len(result_df)} plots."
)

print(
    f"Plots saved to: "
    f"{OUTPUT_DIR}"
)