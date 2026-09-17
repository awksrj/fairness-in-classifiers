import numpy as np
import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================

CSV_PATH = "lfr-prototype-locations/1-training_data.csv"

GENDER_COL = "Gender"
SAT_COL = "SAT"
ADMISSION_COL = "Admission"

EXPLAIN_MEMBERSHIP_IDS = [
    28,   # F, 1240, No
    12    # M, 900, Yes
]

# LFR objective weights
AZ = 1.0
AX = 1.0
AY = 1.0

EPSILON = 1e-10


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(CSV_PATH)

X_original = (
    df[SAT_COL]
    .astype(float)
    .to_numpy()
)

SAT_MIN = X_original.min()
SAT_MAX = X_original.max()

# Same normalization as optimization code
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
# TERMINAL INPUT
# ============================================================

print("=" * 65)
print("LFR LOSS CALCULATOR")
print("=" * 65)
print()

print(
    f"Training SAT range: "
    f"{SAT_MIN:.0f} - {SAT_MAX:.0f}"
)
print()

K = int(
    input(
        "Number of prototypes K: "
    )
)

print()
print(
    "Enter prototype SAT locations "
    "on the ORIGINAL SAT scale."
)
print()


# ------------------------------------------------------------
# Prototype SAT locations
# ------------------------------------------------------------

prototypes_sat = []

for k in range(K):

    location = float(
        input(
            f"Prototype {k + 1} SAT location: "
        )
    )

    prototypes_sat.append(
        location
    )

prototypes_sat = np.array(
    prototypes_sat,
    dtype=float
)


# ------------------------------------------------------------
# Prototype Admission scores
# ------------------------------------------------------------

print()
print(
    "Enter the Admission score "
    "for each prototype."
)
print()

prototype_scores = []

for k in range(K):

    score = float(
        input(
            f"Prototype {k + 1} "
            f"Admission score: "
        )
    )

    prototype_scores.append(
        score
    )

prototype_scores = np.array(
    prototype_scores,
    dtype=float
)


# ------------------------------------------------------------
# Alpha
# ------------------------------------------------------------

print()

alpha = float(
    input(
        "Alpha: "
    )
)


# ============================================================
# VALIDATE INPUT
# ============================================================

if np.any(
    prototypes_sat < SAT_MIN
) or np.any(
    prototypes_sat > SAT_MAX
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

prototypes_normalized = (
    prototypes_sat - SAT_MIN
) / (
    SAT_MAX - SAT_MIN
)


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
# EXPLAIN MEMBERSHIP FOR SELECTED DATA POINTS
# ============================================================

def explain_membership(
    row_index,
    M,
    X,
    prototypes_normalized,
    prototypes_sat,
    prototype_scores,
    alpha
):
    """
    Print how memberships and the final LFR score are
    calculated for one selected data point.
    """

    row = df.iloc[row_index]

    student_id = row["ID"]
    gender = row[GENDER_COL]
    sat = row[SAT_COL]
    admission = row[ADMISSION_COL]

    x = X[row_index]

    print()
    print("=" * 70)
    print(
        f"MEMBERSHIP EXPLANATION: "
        f"ID {student_id}, {gender}, {sat}, {admission}"
    )
    print("=" * 70)
    print()

    print(
        f"Normalized SAT x = "
        f"({sat} - {SAT_MIN:.0f}) / "
        f"({SAT_MAX:.0f} - {SAT_MIN:.0f}) "
        f"= {x:.6f}"
    )

    print(f"Alpha = {alpha:.6f}")
    print()

    # --------------------------------------------------------
    # Calculate raw membership weights
    #
    # r_k = exp(-alpha * (x - z_k)^2)
    # --------------------------------------------------------

    raw_weights = []

    print("RAW MEMBERSHIP WEIGHTS")
    print("-" * 70)

    for k in range(K):

        squared_distance = (
            x - prototypes_normalized[k]
        ) ** 2

        raw_weight = np.exp(
            -alpha * squared_distance
        )

        raw_weights.append(raw_weight)

        print(
            f"Prototype {k + 1}: "
            f"SAT = {prototypes_sat[k]:.2f}"
        )

        print(
            f"  r{k + 1} = "
            f"exp(-{alpha:.4f} * "
            f"({x:.4f} - "
            f"{prototypes_normalized[k]:.4f})^2)"
        )

        print(
            f"     = {raw_weight:.6f}"
        )

    raw_weights = np.array(
        raw_weights
    )

    denominator = raw_weights.sum()

    print()
    print("MEMBERSHIP NORMALIZATION")
    print("-" * 70)

    denominator_expression = " + ".join(
        f"{weight:.6f}"
        for weight in raw_weights
    )

    print(
        f"Denominator = "
        f"{denominator_expression}"
    )

    print(
        f"            = {denominator:.6f}"
    )

    print()

    # Use M directly so displayed memberships are exactly
    # those used by the loss calculation.
    memberships = M[row_index]

    for k in range(K):

        print(
            f"v{k + 1} = "
            f"{raw_weights[k]:.6f} / "
            f"{denominator:.6f} "
            f"= {memberships[k]:.6f}"
        )

    print()

    # --------------------------------------------------------
    # Display representation
    # --------------------------------------------------------

    membership_expression = ", ".join(
        f"v{k + 1}={memberships[k]:.4f}"
        for k in range(K)
    )

    print(
        f"Representation = "
        f"[{membership_expression}]"
    )

    # --------------------------------------------------------
    # LFR / prediction score
    #
    # y_hat = sum_k v_k * w_k
    # --------------------------------------------------------

    print()
    print("LFR SCORE / PREDICTION CALCULATION")
    print("-" * 70)

    terms = [
        f"{memberships[k]:.4f}"
        f"*{prototype_scores[k]:.4f}"
        for k in range(K)
    ]

    products = (
        memberships
        * prototype_scores
    )

    score = products.sum()

    print(
        "LFR_Score = "
        + " + ".join(terms)
    )

    product_expression = " + ".join(
        f"{product:.6f}"
        for product in products
    )

    print(
        "          = "
        + product_expression
    )

    print(
        f"          = {score:.6f}"
    )

    # --------------------------------------------------------
    # Individual classification loss
    # --------------------------------------------------------

    y = Y[row_index]

    clipped_score = np.clip(
        score,
        EPSILON,
        1.0 - EPSILON
    )

    individual_ly = -(
        y * np.log(clipped_score)
        +
        (1.0 - y)
        * np.log(1.0 - clipped_score)
    )

    print()
    print("INDIVIDUAL CLASSIFICATION LOSS")
    print("-" * 70)

    print(
        f"Actual Y = {int(y)} "
        f"({'Yes' if y == 1 else 'No'})"
    )

    if y == 1:

        print(
            f"Loss = -log({score:.6f})"
        )

    else:

        print(
            f"Loss = -log(1 - {score:.6f})"
        )

    print(
        f"     = {individual_ly:.6f}"
    )

    print()


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
    x_hat_n = sum_k M_nk * prototype_k

    Lx = mean squared reconstruction error
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
    y_hat_n = sum_k M_nk * score_k

    Ly = binary cross-entropy
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

    return Ly


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

Ly = classification_loss(
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
# OUTPUT INPUT PARAMETERS
# ============================================================

print()
print("=" * 65)
print("INPUT PARAMETERS")
print("=" * 65)
print()

for k in range(K):

    print(
        f"Prototype {k + 1}: "
        f"SAT = {prototypes_sat[k]:.2f}, "
        f"Normalized SAT = "
        f"{prototypes_normalized[k]:.6f}, "
        f"Admission Score = "
        f"{prototype_scores[k]:.6f}"
    )

print()
print(
    f"Alpha = {alpha:.6f}"
)


# ============================================================
# OUTPUT GROUP MEMBERSHIP DISTRIBUTIONS
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
# OUTPUT LOSSES
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
# EXPLAIN SELECTED DATA POINTS
# ============================================================

print()
print("=" * 70)
print("SELECTED DATA POINT MEMBERSHIPS")
print("=" * 70)

for student_id in EXPLAIN_MEMBERSHIP_IDS:

    matching_indices = df.index[
        df["ID"] == student_id
    ].tolist()

    if not matching_indices:

        print()
        print(
            f"WARNING: ID {student_id} "
            f"was not found."
        )

        continue

    row_index = matching_indices[0]

    explain_membership(
        row_index=row_index,
        M=M,
        X=X,
        prototypes_normalized=prototypes_normalized,
        prototypes_sat=prototypes_sat,
        prototype_scores=prototype_scores,
        alpha=alpha
    )