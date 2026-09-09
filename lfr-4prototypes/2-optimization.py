import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

# ============================================================
# CONFIGURATION
# ============================================================

CSV_PATH = "lfr/1-training_data.csv"

# Output files
PROTOTYPE_OUTPUT_CSV = "lfr/3-prototypes.csv"
REPRESENTATION_OUTPUT_CSV = "lfr/4-representations.csv"

# Columns
GENDER_COL = "Gender"
SAT_COL = "SAT"
ADMISSION_COL = "Admission"

# Number of prototypes
K = 4

# ------------------------------------------------------------
# LFR objective weights
#
# L = Az * Lz + Ax * Lx + Ay * Ly
# ------------------------------------------------------------

AZ = 1.0
AX = 1.0
AY = 1.0

# Optimization
RANDOM_SEED = 42
MAX_ITER = 500
POP_SIZE = 20

# Numerical stability
EPSILON = 1e-10


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(CSV_PATH)

# SAT
X_original = df[SAT_COL].astype(float).to_numpy()

# Normalize SAT to [0, 1]
SAT_MIN = X_original.min()
SAT_MAX = X_original.max()

X = (
    X_original - SAT_MIN
) / (
    SAT_MAX - SAT_MIN
)

# Gender
gender = df[GENDER_COL].to_numpy()

male_mask = gender == "M"
female_mask = gender == "F"

# Admission
Y = (
    df[ADMISSION_COL]
    .map({"Yes": 1.0, "No": 0.0})
    .to_numpy()
)


# ============================================================
# PROTOTYPE REPRESENTATION
# ============================================================

def calculate_membership(X, prototypes, alpha):
    """
    Calculate:

        M_nk = P(Z=k | x_n)

    X:
        normalized SAT values

    prototypes:
        normalized SAT locations of prototypes

    alpha:
        feature weight
    """

    # Squared distance between every student
    # and every prototype
    distances = alpha * (
        X[:, np.newaxis]
        - prototypes[np.newaxis, :]
    ) ** 2

    # Softmax over negative distances
    logits = -distances

    # Numerical stability
    logits -= logits.max(
        axis=1,
        keepdims=True
    )

    exp_logits = np.exp(logits)

    M = exp_logits / (
        exp_logits.sum(
            axis=1,
            keepdims=True
        )
        + EPSILON
    )

    return M


# ============================================================
# FAIRNESS LOSS
# ============================================================

def fairness_loss(M):
    """
    Lz = sum_k |M_k^male - M_k^female|
    """

    male_distribution = M[male_mask].mean(axis=0)
    female_distribution = M[female_mask].mean(axis=0)

    Lz = np.sum(
        np.abs(
            male_distribution
            - female_distribution
        )
    )

    return Lz


# ============================================================
# RECONSTRUCTION LOSS
# ============================================================

def reconstruction_loss(X, M, prototypes):
    """
    Reconstruct:

        x_hat = sum_k M_nk * v_k
    """

    X_hat = np.sum(
        M * prototypes[np.newaxis, :],
        axis=1
    )

    Lx = np.mean(
        (X - X_hat) ** 2
    )

    return Lx


# ============================================================
# CLASSIFICATION LOSS
# ============================================================

def classification_loss(Y, M, prototype_scores):
    """
    Predict Admission using:

        y_hat_n = sum_k M_nk * w_k
    """

    Y_hat = np.sum(
        M * prototype_scores[np.newaxis, :],
        axis=1
    )

    Y_hat = np.clip(
        Y_hat,
        EPSILON,
        1.0 - EPSILON
    )

    Ly = -np.mean(
        Y * np.log(Y_hat)
        + (1.0 - Y)
        * np.log(1.0 - Y_hat)
    )

    return Ly


# ============================================================
# LFR OBJECTIVE
# ============================================================

def objective(params):

    # First K parameters:
    # prototype locations
    prototypes = params[:K]

    # Next K parameters:
    # prototype Admission scores
    prototype_scores = params[K:2 * K]

    # Last parameter:
    # alpha
    alpha = params[-1]

    # Calculate Z representation
    M = calculate_membership(
        X,
        prototypes,
        alpha
    )

    # Fairness
    Lz = fairness_loss(M)

    # Reconstruction
    Lx = reconstruction_loss(
        X,
        M,
        prototypes
    )

    # Classification
    Ly = classification_loss(
        Y,
        M,
        prototype_scores
    )

    # Combined objective
    loss = (
        AZ * Lz
        + AX * Lx
        + AY * Ly
    )

    return loss


# ============================================================
# OPTIMIZATION BOUNDS
# ============================================================

prototype_bounds = [
    (0.0, 1.0)
    for _ in range(K)
]

score_bounds = [
    (0.0, 1.0)
    for _ in range(K)
]

alpha_bounds = [
    (0.01, 100.0)
]

bounds = (
    prototype_bounds
    + score_bounds
    + alpha_bounds
)


# ============================================================
# RUN OPTIMIZATION
# ============================================================

print("Running LFR optimization...")
print()

result = differential_evolution(
    objective,
    bounds=bounds,
    seed=RANDOM_SEED,
    maxiter=MAX_ITER,
    popsize=POP_SIZE,
    polish=True,
    updating="immediate"
)


# ============================================================
# EXTRACT LEARNED PARAMETERS
# ============================================================

params = result.x

prototypes_normalized = params[:K]

prototype_scores = params[K:2 * K]

alpha = params[-1]


# ============================================================
# CONVERT PROTOTYPE LOCATIONS TO ORIGINAL SAT SCALE
# ============================================================

prototypes_sat = (
    prototypes_normalized
    * (SAT_MAX - SAT_MIN)
    + SAT_MIN
)


# ============================================================
# SORT PROTOTYPES BY SAT LOCATION
# ============================================================

order = np.argsort(prototypes_sat)

prototypes_sat = prototypes_sat[order]

prototype_scores = prototype_scores[order]

prototypes_normalized = (
    prototypes_normalized[order]
)


# ============================================================
# CALCULATE FINAL Z REPRESENTATION
# ============================================================

M = calculate_membership(
    X,
    prototypes_normalized,
    alpha
)


# ============================================================
# CALCULATE LFR ADMISSION SCORE
# ============================================================

Y_hat = np.sum(
    M * prototype_scores[np.newaxis, :],
    axis=1
)


# ============================================================
# CALCULATE FINAL LOSSES
# ============================================================

Lz = fairness_loss(M)

Lx = reconstruction_loss(
    X,
    M,
    prototypes_normalized
)

Ly = classification_loss(
    Y,
    M,
    prototype_scores
)

total_loss = (
    AZ * Lz
    + AX * Lx
    + AY * Ly
)


# ============================================================
# SAVE LEARNED PROTOTYPES
# ============================================================

prototype_df = pd.DataFrame({
    "Prototype": np.arange(1, K + 1),
    "SAT": prototypes_sat,
    "Admission_Score": prototype_scores,
    "Alpha": alpha
})

prototype_df.to_csv(
    PROTOTYPE_OUTPUT_CSV,
    index=False
)


# ============================================================
# OUTPUT LEARNED PROTOTYPES
# ============================================================

print("=" * 60)
print("LEARNED LFR PROTOTYPES")
print("=" * 60)

for k in range(K):

    print(
        f"Prototype {k + 1}: "
        f"SAT = {prototypes_sat[k]:.2f}, "
        f"Admission score = "
        f"{prototype_scores[k]:.4f}"
    )

print()

print(f"Alpha: {alpha:.4f}")

print()

print(
    f"Saved prototypes to: "
    f"{PROTOTYPE_OUTPUT_CSV}"
)

print()


# ============================================================
# CREATE 4-REPRESENTATIONS.CSV
# ============================================================
#
# Each student gets:
#
# ID
# Gender
# SAT
# Admission
# v1
# v2
# v3
# v4
# LFR_Score
# Predicted_Admission
#
# ------------------------------------------------------------
# Z representation:
#
# Z_i = [v1, v2, v3, v4]
#
# where:
#
# v_k = P(Z = prototype k | SAT_i)
#
# ------------------------------------------------------------
# LFR score:
#
# LFR_Score =
#
#     v1 * score1
#   + v2 * score2
#   + v3 * score3
#   + v4 * score4
#
# ============================================================


# ============================================================
# DETERMINE ORIGINAL ADMISSION DISTRIBUTION
# ============================================================

original_yes_count = (
    df[ADMISSION_COL] == "Yes"
).sum()

original_no_count = (
    df[ADMISSION_COL] == "No"
).sum()


# ============================================================
# DETERMINE PREDICTED ADMISSION
# ============================================================
#
# Instead of using an arbitrary threshold such as 0.5,
# rank students by LFR_Score.
#
# The top N students are predicted Yes, where:
#
#     N = number of original Yes labels
#
# This ensures the predicted Yes count matches the
# original Yes count exactly.
#
# ============================================================

sorted_indices = np.argsort(
    -Y_hat
)

predicted_admission = np.array(
    ["No"] * len(df),
    dtype=object
)

# Students with the highest LFR scores
# receive Yes until we reach the original
# number of Yes labels.

yes_indices = sorted_indices[
    :original_yes_count
]

predicted_admission[
    yes_indices
] = "Yes"


# ============================================================
# DETERMINE EFFECTIVE THRESHOLD
# ============================================================

if original_yes_count > 0:

    threshold = np.min(
        Y_hat[yes_indices]
    )

else:

    threshold = np.inf


# ============================================================
# CREATE REPRESENTATION DATAFRAME
# ============================================================

representation_df = pd.DataFrame({
    "ID": df["ID"],
    "Gender": df["Gender"],
    "SAT": df["SAT"],
    "Admission": df["Admission"],

    "v1": M[:, 0],
    "v2": M[:, 1],
    "v3": M[:, 2],
    "v4": M[:, 3],

    "LFR_Score": Y_hat,

    "Predicted_Admission":
        predicted_admission
})


# ============================================================
# SAVE REPRESENTATION CSV
# ============================================================

representation_df.to_csv(
    REPRESENTATION_OUTPUT_CSV,
    index=False
)


# ============================================================
# OUTPUT LFR REPRESENTATION SUMMARY
# ============================================================

print("=" * 60)
print("LFR REPRESENTATION")
print("=" * 60)

print()

print(
    f"Original Admission: "
    f"Yes = {original_yes_count}, "
    f"No = {original_no_count}"
)

predicted_yes_count = (
    predicted_admission == "Yes"
).sum()

predicted_no_count = (
    predicted_admission == "No"
).sum()

print(
    f"Predicted Admission: "
    f"Yes = {predicted_yes_count}, "
    f"No = {predicted_no_count}"
)

print()

print(
    f"Prediction threshold: "
    f"{threshold:.6f}"
)

print()

print(
    f"Saved representations to: "
    f"{REPRESENTATION_OUTPUT_CSV}"
)

print()


# ============================================================
# OUTPUT FIRST 10 REPRESENTATIONS
# ============================================================

print("=" * 60)
print("FIRST 10 LFR REPRESENTATIONS")
print("=" * 60)

print()

print(
    representation_df.head(10).to_string(
        index=False
    )
)

print()


# ============================================================
# CALCULATE FINAL GROUP PROTOTYPE DISTRIBUTIONS
# ============================================================

print("=" * 60)
print("GROUP PROTOTYPE DISTRIBUTIONS")
print("=" * 60)

print()

male_distribution = (
    M[male_mask].mean(axis=0)
)

female_distribution = (
    M[female_mask].mean(axis=0)
)

for k in range(K):

    print(
        f"Prototype {k + 1}: "
        f"Male = {male_distribution[k]:.4f}, "
        f"Female = {female_distribution[k]:.4f}"
    )

print()


# ============================================================
# OUTPUT LOSSES
# ============================================================

print("=" * 60)
print("LFR LOSSES")
print("=" * 60)

print()

print(
    f"Fairness loss Lz:        {Lz:.6f}"
)

print(
    f"Reconstruction loss Lx:  {Lx:.6f}"
)

print(
    f"Classification loss Ly:  {Ly:.6f}"
)

print(
    f"Total objective:         {total_loss:.6f}"
)

print()


# ============================================================
# OPTIMIZATION STATUS
# ============================================================

print("=" * 60)
print("OPTIMIZATION STATUS")
print("=" * 60)

print()

print(
    f"Success: {result.success}"
)

print(
    f"Message: {result.message}"
)

print(
    f"Objective: {result.fun:.6f}"
)

print()


# ============================================================
# OUTPUT PROTOTYPE CSV
# ============================================================

print("=" * 60)
print("PROTOTYPE CSV")
print("=" * 60)

print()

print(
    prototype_df.to_string(
        index=False
    )
)

print()


# ============================================================
# OUTPUT REPRESENTATION CSV
# ============================================================

print("=" * 60)
print("REPRESENTATION CSV")
print("=" * 60)

print()

print(
    representation_df.to_string(
        index=False
    )
)

print()