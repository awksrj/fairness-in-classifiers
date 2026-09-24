import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# ============================================================
# CONFIGURATION
# ============================================================

CSV_PATH = "lfr2-4prototypes-unsucessful/1-training_data.csv"

# Column names
GENDER_COL = "Gender"
SAT_COL = "SAT"
ADMISSION_COL = "Admission"

# LFR objective weights
AZ = 1.0
AX = 1.0
AY = 1.0

EPSILON = 1e-10

# SAT plot axis
PLOT_SAT_MIN = 350
PLOT_SAT_MAX = 1700
SAT_TICK_INTERVAL = 100

# Jitter amount
JITTER_WIDTH = 0.12

# Reproducible jitter
RANDOM_SEED = 42

# Training point appearance
POINT_SIZE = 70
POINT_ALPHA = 0.85

# Prototype appearance
PROTOTYPE_SIZE = 180
PROTOTYPE_COLOR = "pink"
PROTOTYPE_MARKER = "o"

# Prototypes have no gender, so place them
# halfway between Male and Female
PROTOTYPE_X = 0.5


# ============================================================
# LOAD TRAINING DATA
# ============================================================

df = pd.read_csv(CSV_PATH)

plot_df = df.copy()


# ============================================================
# PREPARE TRAINING DATA FOR LFR
# ============================================================

X_original = (
    df[SAT_COL]
    .astype(float)
    .to_numpy()
)

# IMPORTANT:
# These are the min/max of the actual training data.
# They are used for LFR normalization.
DATA_SAT_MIN = X_original.min()
DATA_SAT_MAX = X_original.max()

X = (
    X_original - DATA_SAT_MIN
) / (
    DATA_SAT_MAX - DATA_SAT_MIN
)

gender = df[GENDER_COL].to_numpy()

male_mask = gender == "M"
female_mask = gender == "F"

Y = (
    df[ADMISSION_COL]
    .map({
        "Yes": 1.0,
        "No": 0.0
    })
    .to_numpy()
)


# ============================================================
# TERMINAL INPUT: PROTOTYPES
# ============================================================

print("=" * 65)
print("LFR TRAINING DATA + PROTOTYPE PLOT")
print("=" * 65)
print()

print(
    f"Training SAT range: "
    f"{DATA_SAT_MIN:.0f} - {DATA_SAT_MAX:.0f}"
)

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


# ------------------------------------------------------------
# Prototype SAT locations and scores
# ------------------------------------------------------------

prototypes_sat = []
prototype_scores = []

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

    prototypes_sat.append(
        sat
    )

    prototype_scores.append(
        score
    )

    print()


prototypes_sat = np.array(
    prototypes_sat,
    dtype=float
)

prototype_scores = np.array(
    prototype_scores,
    dtype=float
)


# ------------------------------------------------------------
# Alpha
# ------------------------------------------------------------

alpha = float(
    input(
        "Alpha: "
    )
)


# ============================================================
# VALIDATE INPUT
# ============================================================

if np.any(
    prototypes_sat < DATA_SAT_MIN
) or np.any(
    prototypes_sat > DATA_SAT_MAX
):

    print()
    print(
        "WARNING: At least one prototype "
        "is outside the training SAT range."
    )


if np.any(
    prototype_scores < 0
) or np.any(
    prototype_scores > 1
):

    raise ValueError(
        "Prototype Admission scores "
        "must be between 0 and 1."
    )


if alpha <= 0:

    raise ValueError(
        "Alpha must be greater than 0."
    )


# ============================================================
# NORMALIZE PROTOTYPE LOCATIONS
# ============================================================

# Use exactly the same normalization as the training data.

prototypes_normalized = (
    prototypes_sat - DATA_SAT_MIN
) / (
    DATA_SAT_MAX - DATA_SAT_MIN
)


# ============================================================
# CREATE PROTOTYPE DATAFRAME
# ============================================================

prototype_df = pd.DataFrame({
    "SAT": prototypes_sat,
    "normalized_sat": prototypes_normalized,
    "Admission_Score": prototype_scores
})


# ============================================================
# PROTOTYPE MEMBERSHIP
# ============================================================

def calculate_membership(
    X,
    prototypes,
    alpha
):
    """
    M_nk = P(Z = k | x_n)

    Membership is based on squared distance
    between each data point and prototype.
    """

    distances = alpha * (
        X[:, np.newaxis]
        - prototypes[np.newaxis, :]
    ) ** 2

    logits = -distances

    # Numerical stability
    logits -= logits.max(
        axis=1,
        keepdims=True
    )

    exp_logits = np.exp(
        logits
    )

    M = exp_logits / (
        exp_logits.sum(
            axis=1,
            keepdims=True
        )
        + EPSILON
    )

    return M


# ============================================================
# FAIRNESS LOSS Lz
# ============================================================

def fairness_loss(M):
    """
    Lz = sum_k |
        mean_male(M_k)
        -
        mean_female(M_k)
    |
    """

    male_distribution = (
        M[male_mask]
        .mean(axis=0)
    )

    female_distribution = (
        M[female_mask]
        .mean(axis=0)
    )

    Lz = np.sum(
        np.abs(
            male_distribution
            - female_distribution
        )
    )

    return (
        Lz,
        male_distribution,
        female_distribution
    )


# ============================================================
# RECONSTRUCTION LOSS Lx
# ============================================================

def reconstruction_loss(
    X,
    M,
    prototypes
):
    """
    Reconstructed SAT:

        x_hat_n = sum_k M_nk * prototype_k

    Reconstruction loss:

        Lx = mean((x - x_hat)^2)
    """

    X_hat = np.sum(
        M
        * prototypes[
            np.newaxis,
            :
        ],
        axis=1
    )

    Lx = np.mean(
        (X - X_hat) ** 2
    )

    return Lx


# ============================================================
# CLASSIFICATION LOSS Ly
# ============================================================

def classification_loss(
    Y,
    M,
    prototype_scores
):
    """
    Predicted Admission score:

        y_hat_n = sum_k M_nk * score_k

    Classification loss:

        Binary cross-entropy
    """

    Y_hat = np.sum(
        M
        * prototype_scores[
            np.newaxis,
            :
        ],
        axis=1
    )

    Y_hat = np.clip(
        Y_hat,
        EPSILON,
        1.0 - EPSILON
    )

    Ly = -np.mean(
        Y * np.log(Y_hat)
        +
        (1.0 - Y)
        * np.log(
            1.0 - Y_hat
        )
    )

    return (
        Ly,
        Y_hat
    )


# ============================================================
# CALCULATE MEMBERSHIP MATRIX
# ============================================================

M = calculate_membership(
    X,
    prototypes_normalized,
    alpha
)


# ============================================================
# CALCULATE LOSSES
# ============================================================

Lz, male_distribution, female_distribution = (
    fairness_loss(M)
)

Lx = reconstruction_loss(
    X,
    M,
    prototypes_normalized
)

Ly, Y_hat = classification_loss(
    Y,
    M,
    prototype_scores
)

L = (
    AZ * Lz
    + AX * Lx
    + AY * Ly
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

print(
    f"Alpha = {alpha:.6f}"
)


# ============================================================
# OUTPUT GROUP PROTOTYPE DISTRIBUTIONS
# ============================================================

print()
print("=" * 65)
print("GROUP PROTOTYPE DISTRIBUTIONS")
print("=" * 65)
print()

for k in range(K):

    difference = (
        male_distribution[k]
        - female_distribution[k]
    )

    print(
        f"Prototype {k + 1}: "
        f"Male = {male_distribution[k]:.6f}, "
        f"Female = {female_distribution[k]:.6f}, "
        f"|Difference| = {abs(difference):.6f}"
    )


# ============================================================
# OUTPUT LOSSES TO TERMINAL
# ============================================================

print()
print("=" * 65)
print("LFR LOSSES")
print("=" * 65)
print()

print(
    f"Fairness loss       Lz = {Lz:.8f}"
)

print(
    f"Reconstruction loss Lx = {Lx:.8f}"
)

print(
    f"Classification loss Ly = {Ly:.8f}"
)

print()

print(
    f"L = "
    f"{AZ}({Lz:.8f}) + "
    f"{AX}({Lx:.8f}) + "
    f"{AY}({Ly:.8f})"
)

print(
    f"Total loss           L = {L:.8f}"
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
# CREATE PLOT
# ============================================================

fig, ax = plt.subplots(
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
# PLOT PROTOTYPES
# ============================================================

for _, prototype in prototype_df.iterrows():

    prototype_sat = (
        prototype["SAT"]
    )

    prototype_score = (
        prototype["Admission_Score"]
    )

    # Pink circle
    ax.scatter(
        PROTOTYPE_X,
        prototype_sat,
        marker=PROTOTYPE_MARKER,
        s=PROTOTYPE_SIZE,
        color=PROTOTYPE_COLOR,
        edgecolors="black",
        linewidths=1.2,
        zorder=5
    )

    # Label:
    # location, score
    #
    # Example:
    # 1150, 0.472
    ax.annotate(
        (
            f"{prototype_sat:.0f}, "
            f"{prototype_score:.3f}"
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
# DISPLAY LOSSES IN TOP LEFT
# ============================================================

loss_text = (
    f"L  = {L:.6f}\n"
    f"Lz = {Lz:.6f}\n"
    f"Lx = {Lx:.6f}\n"
    f"Ly = {Ly:.6f}"
)

ax.text(
    0.02,
    0.98,
    loss_text,
    transform=ax.transAxes,
    fontsize=10,
    verticalalignment="top",
    horizontalalignment="left",
    bbox=dict(
        boxstyle="round",
        facecolor="white",
        edgecolor="black",
        alpha=0.85
    ),
    zorder=10
)


# ============================================================
# AXES
# ============================================================

ax.set_xticks(
    [0, 0.5, 1]
)

ax.set_xticklabels(
    [
        "Male",
        "Prototypes",
        "Female"
    ]
)

ax.set_yticks(
    range(
        PLOT_SAT_MIN,
        PLOT_SAT_MAX + 1,
        SAT_TICK_INTERVAL
    )
)

ax.set_ylim(
    PLOT_SAT_MIN,
    PLOT_SAT_MAX
)

ax.set_xlim(
    -0.4,
    1.4
)

ax.set_xlabel(
    "Gender / Prototypes"
)

ax.set_ylabel(
    "SAT Score"
)

ax.set_title(
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
        marker="o",
        color="w",
        label="LFR Prototype (SAT, score)",
        markerfacecolor=PROTOTYPE_COLOR,
        markeredgecolor="black",
        markersize=11
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
    alpha=0.4
)

plt.tight_layout()


# ============================================================
# SHOW
# ============================================================

plt.show()