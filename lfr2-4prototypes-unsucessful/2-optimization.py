import numpy as np
import pandas as pd
from scipy.optimize import minimize

# ============================================================
# CONFIGURATION
# ============================================================

CSV_PATH = "lfr/1-training_data.csv"

PROTOTYPE_OUTPUT_CSV = "lfr/3-prototypes_4k.csv"
REPRESENTATION_OUTPUT_CSV = "lfr/4-representations_4k.csv"

GENDER_COL = "Gender"
SAT_COL = "SAT"
ADMISSION_COL = "Admission"

K = 4

# LFR objective weights
AZ = 1.0
AX = 1.0
AY = 1.0

RANDOM_SEED = 42

EPSILON = 1e-10


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(CSV_PATH)

X_original = df[SAT_COL].astype(float).to_numpy()

SAT_MIN = X_original.min()
SAT_MAX = X_original.max()

X = (
    X_original - SAT_MIN
) / (
    SAT_MAX - SAT_MIN
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
# PROTOTYPE MEMBERSHIP
# ============================================================

def calculate_membership(X, prototypes, alpha):
    """
    M_nk = P(Z = k | x_n)
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

    male_distribution = (
        M[male_mask].mean(axis=0)
    )

    female_distribution = (
        M[female_mask].mean(axis=0)
    )

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

def reconstruction_loss(
    X,
    M,
    prototypes
):
    """
    x_hat_n = sum_k M_nk * v_k
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

def classification_loss(
    Y,
    M,
    prototype_scores
):
    """
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
        +
        (1.0 - Y)
        * np.log(1.0 - Y_hat)
    )

    return Ly


# ============================================================
# LFR OBJECTIVE
# ============================================================

def objective(params):

    # First K values:
    # prototype locations
    prototypes = params[:K]

    # Next K values:
    # prototype Admission scores
    prototype_scores = params[
        K:2 * K
    ]

    # Last value:
    # alpha
    alpha = params[-1]

    M = calculate_membership(
        X,
        prototypes,
        alpha
    )

    Lz = fairness_loss(M)

    Lx = reconstruction_loss(
        X,
        M,
        prototypes
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

    return total_loss


# ============================================================
# INITIAL PARAMETERS
# ============================================================

rng = np.random.default_rng(
    RANDOM_SEED
)

# ------------------------------------------------------------
# Initial prototype locations
#
# Instead of completely random locations, initialize them
# approximately evenly across [0, 1].
# ------------------------------------------------------------

initial_prototypes = np.linspace(
    0.05,
    0.95,
    K
)

# Add a tiny amount of noise so prototypes do not all
# follow a perfectly symmetric initialization.
initial_prototypes += rng.normal(
    0,
    0.01,
    K
)

initial_prototypes = np.clip(
    initial_prototypes,
    0.0,
    1.0
)


# ------------------------------------------------------------
# Initial Admission scores
# ------------------------------------------------------------

initial_scores = rng.uniform(
    0.25,
    0.75,
    K
)


# ------------------------------------------------------------
# Initial alpha
# ------------------------------------------------------------

initial_alpha = np.array([
    10.0
])


# Full parameter vector:
#
# [v1 ... vK,
#  w1 ... wK,
#  alpha]
# ------------------------------------------------------------

initial_params = np.concatenate([
    initial_prototypes,
    initial_scores,
    initial_alpha
])


# ============================================================
# PARAMETER BOUNDS
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
# RUN L-BFGS-B
# ============================================================

print("Running LFR optimization with L-BFGS-B...")
print()

print(
    f"Initial objective: "
    f"{objective(initial_params):.6f}"
)

result = minimize(
    objective,
    x0=initial_params,
    method="L-BFGS-B",
    bounds=bounds,
    options={
        "maxiter": 5000,
        "ftol": 1e-12,
        "gtol": 1e-8,
        "maxls": 50
    }
)


# ============================================================
# EXTRACT LEARNED PARAMETERS
# ============================================================

params = result.x

prototypes_normalized = params[:K]

prototype_scores = params[
    K:2 * K
]

alpha = params[-1]


# ============================================================
# CONVERT PROTOTYPES BACK TO SAT SCALE
# ============================================================

prototypes_sat = (
    prototypes_normalized
    * (SAT_MAX - SAT_MIN)
    + SAT_MIN
)


# ============================================================
# SORT PROTOTYPES BY SAT
# ============================================================

order = np.argsort(
    prototypes_sat
)

prototypes_sat = (
    prototypes_sat[order]
)

prototypes_normalized = (
    prototypes_normalized[order]
)

prototype_scores = (
    prototype_scores[order]
)


# ============================================================
# FINAL MEMBERSHIP MATRIX
# ============================================================

M = calculate_membership(
    X,
    prototypes_normalized,
    alpha
)


# ============================================================
# FINAL LFR SCORE
# ============================================================

Y_hat = np.sum(
    M
    * prototype_scores[
        np.newaxis,
        :
    ],
    axis=1
)


# ============================================================
# FINAL LOSSES
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
# SAVE PROTOTYPES
# ============================================================

prototype_df = pd.DataFrame({
    "Prototype":
        np.arange(
            1,
            K + 1
        ),

    "SAT":
        prototypes_sat,

    "Admission_Score":
        prototype_scores,

    "Alpha":
        alpha
})

prototype_df.to_csv(
    PROTOTYPE_OUTPUT_CSV,
    index=False
)


# ============================================================
# DETERMINE ORIGINAL YES COUNT
# ============================================================

original_yes_count = (
    df[
        ADMISSION_COL
    ] == "Yes"
).sum()


# ============================================================
# RANK-BASED PREDICTIONS
# ============================================================

sorted_indices = np.argsort(
    -Y_hat
)

predicted_admission = np.array(
    ["No"] * len(df),
    dtype=object
)

yes_indices = sorted_indices[
    :original_yes_count
]

predicted_admission[
    yes_indices
] = "Yes"


# ============================================================
# EFFECTIVE THRESHOLD
# ============================================================

if original_yes_count > 0:

    threshold = np.min(
        Y_hat[
            yes_indices
        ]
    )

else:

    threshold = np.inf


# ============================================================
# CREATE REPRESENTATION DATAFRAME
# ============================================================

representation_df = pd.DataFrame({
    "ID":
        df["ID"],

    "Gender":
        df["Gender"],

    "SAT":
        df["SAT"],

    "Admission":
        df["Admission"]
})

for k in range(K):

    representation_df[
        f"v{k + 1}"
    ] = M[:, k]

representation_df[
    "LFR_Score"
] = Y_hat

representation_df[
    "Predicted_Admission"
] = predicted_admission


representation_df.to_csv(
    REPRESENTATION_OUTPUT_CSV,
    index=False
)


# ============================================================
# OUTPUT LEARNED PROTOTYPES
# ============================================================

print("=" * 60)
print("LEARNED LFR PROTOTYPES")
print("=" * 60)

print()

print(
    prototype_df.to_string(
        index=False
    )
)

print()


# ============================================================
# GROUP PROTOTYPE DISTRIBUTIONS
# ============================================================

male_distribution = (
    M[male_mask].mean(
        axis=0
    )
)

female_distribution = (
    M[female_mask].mean(
        axis=0
    )
)

print("=" * 60)
print("GROUP PROTOTYPE DISTRIBUTIONS")
print("=" * 60)

print()

for k in range(K):

    print(
        f"Prototype {k + 1}: "
        f"Male = "
        f"{male_distribution[k]:.4f}, "
        f"Female = "
        f"{female_distribution[k]:.4f}"
    )

print()


# ============================================================
# LOSSES
# ============================================================

print("=" * 60)
print("LFR LOSSES")
print("=" * 60)

print()

print(
    f"Fairness loss Lz:       "
    f"{Lz:.6f}"
)

print(
    f"Reconstruction loss Lx: "
    f"{Lx:.6f}"
)

print(
    f"Classification loss Ly: "
    f"{Ly:.6f}"
)

print(
    f"Total objective:        "
    f"{total_loss:.6f}"
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
    f"Success: "
    f"{result.success}"
)

print(
    f"Message: "
    f"{result.message}"
)

print(
    f"Iterations: "
    f"{result.nit}"
)

print(
    f"Function evaluations: "
    f"{result.nfev}"
)

print(
    f"Final objective: "
    f"{result.fun:.6f}"
)

print()


# ============================================================
# REPRESENTATION OUTPUT
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

print(
    f"Prediction threshold: "
    f"{threshold:.6f}"
)

print(
    f"Saved prototypes to: "
    f"{PROTOTYPE_OUTPUT_CSV}"
)

print(
    f"Saved representations to: "
    f"{REPRESENTATION_OUTPUT_CSV}"
)