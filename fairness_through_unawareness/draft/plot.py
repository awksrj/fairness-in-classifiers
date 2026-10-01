# ============================================================
# plot.py
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURATION
# ============================================================

BEFORE_CSV_PATH = "fairness_through_unawareness\\draft\\csv\\before_ftu.csv"
AFTER_CSV_PATH = "fairness_through_unawareness\\draft\\csv\\after_ftu.csv"

BEFORE_PNG_PATH = "fairness_through_unawareness\\draft\\png\\before_ftu.png"
AFTER_PNG_PATH = "fairness_through_unawareness\\draft\\png\\after_ftu.png"

THRESHOLD = 0.5

RANDOM_SEED = 42
JITTER = 0.07

FIGURE_WIDTH = 7
FIGURE_HEIGHT = 7
POINT_SIZE = 30
POINT_ALPHA = 0.8

Y_MIN = 0.0
Y_MAX = 1.0
Y_TICK_STEP = 0.1

X_MIN = -0.35
X_MAX = 1.35

BEFORE_TITLE = (
    "Aware Model: Before Fairness Through Unawareness"
)

AFTER_TITLE = (
    "Unaware Model: After Removing Gender"
)

X_LABEL = "Gender"
Y_LABEL = "Predicted Admission Probability"

MALE_X = 0
FEMALE_X = 1


# ============================================================
# PLOTTING FUNCTION
# ============================================================

def plot_distribution(
    csv_path,
    title,
    output_path
):
    np.random.seed(RANDOM_SEED)

    df = pd.read_csv(csv_path)

    x_map = {
        "Male": MALE_X,
        "Female": FEMALE_X
    }

    df["x"] = df["Gender"].map(x_map)

    # Small horizontal jitter so points do not overlap
    df["x_jitter"] = (
        df["x"]
        + np.random.uniform(
            -JITTER,
            JITTER,
            len(df)
        )
    )

    accepted = df[
        df["Probability"] >= THRESHOLD
    ]

    unaccepted = df[
        df["Probability"] < THRESHOLD
    ]

    plt.figure(
        figsize=(
            FIGURE_WIDTH,
            FIGURE_HEIGHT
        )
    )

    # --------------------------------------------------------
    # Unaccepted points
    # --------------------------------------------------------

    plt.scatter(
        unaccepted["x_jitter"],
        unaccepted["Probability"],
        color="red",
        s=POINT_SIZE,
        alpha=POINT_ALPHA,
        label="Unaccepted"
    )

    # --------------------------------------------------------
    # Accepted points
    # --------------------------------------------------------

    plt.scatter(
        accepted["x_jitter"],
        accepted["Probability"],
        color="blue",
        s=POINT_SIZE,
        alpha=POINT_ALPHA,
        label="Accepted"
    )

    # --------------------------------------------------------
    # Decision threshold
    # --------------------------------------------------------

    plt.axhline(
        y=THRESHOLD,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label=f"Decision Threshold = {THRESHOLD}"
    )

    # --------------------------------------------------------
    # Axes
    # --------------------------------------------------------

    plt.xticks(
        [MALE_X, FEMALE_X],
        ["Male", "Female"],
        fontsize=12
    )

    plt.yticks(
        np.arange(
            Y_MIN,
            Y_MAX + Y_TICK_STEP,
            Y_TICK_STEP
        )
    )

    plt.ylim(
        Y_MIN,
        Y_MAX
    )

    plt.xlim(
        X_MIN,
        X_MAX
    )

    plt.xlabel(
        X_LABEL,
        fontsize=12
    )

    plt.ylabel(
        Y_LABEL,
        fontsize=12
    )

    plt.title(
        title,
        fontsize=14
    )

    plt.legend()

    plt.grid(
        axis="y",
        linestyle=":",
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()


# ============================================================
# BEFORE FTU
# ============================================================

plot_distribution(
    csv_path=BEFORE_CSV_PATH,
    title=BEFORE_TITLE,
    output_path=BEFORE_PNG_PATH
)


# ============================================================
# AFTER FTU
# ============================================================

plot_distribution(
    csv_path=AFTER_CSV_PATH,
    title=AFTER_TITLE,
    output_path=AFTER_PNG_PATH
)