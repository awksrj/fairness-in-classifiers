import ast
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

TRAINING_CSV = "lfr-prototype-locations/1-training_data.csv"
TEST_CASE_CSV = "lfr-prototype-locations/2-test_cases.csv"

GENDER_COL = "Gender"
SAT_COL = "SAT"
ADMISSION_COL = "Admission"

# ------------------------------------------------------------
# Fixed alpha for ALL test cases
# ------------------------------------------------------------

ALPHA = 17.067778487885167

# ------------------------------------------------------------
# LFR objective weights
# ------------------------------------------------------------

AZ = 1.0
AX = 1.0
AY = 1.0

EPSILON = 1e-10


# ============================================================
# LOAD ORIGINAL TRAINING DATA
# ============================================================

df = pd.read_csv(TRAINING_CSV)

X_original = (
    df[SAT_COL]
    .astype(float)
    .to_numpy()
)

SAT_MIN = X_original.min()
SAT_MAX = X_original.max()

# Same normalization used in the original LFR code
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
# PARSE PROTOTYPE STRING
# ============================================================

def parse_prototypes(value):
    """
    Convert a CSV string such as:

        (500,0), (850,0.3), (1250,0.7), (1580,1)

    into:

        locations = [500, 850, 1250, 1580]
        scores    = [0, 0.3, 0.7, 1]
    """

    value = str(value).strip()

    # Add outer brackets so Python can interpret the
    # sequence of tuples as a list.
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
                f"Invalid prototype: {prototype}. "
                f"Expected format (SAT, score)."
            )

        location, score = prototype

        locations.append(float(location))
        scores.append(float(score))

    locations = np.array(
        locations,
        dtype=float
    )

    scores = np.array(
        scores,
        dtype=float
    )

    return locations, scores


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

           exp(-alpha * (x_n - v_k)^2)
    M_nk = ---------------------------
           sum_j exp(-alpha*(x_n-v_j)^2)
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
    Lz = sum_k |
        mean(M_male,k)
        -
        mean(M_female,k)
    |
    """

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
    Reconstruct each candidate's normalized SAT:

        x_hat_n = sum_k M_nk * v_k
    """

    X_hat = np.sum(
        M
        * prototypes[np.newaxis, :],
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
    Calculate predicted admission score:

        y_hat_n = sum_k M_nk * w_k

    Then calculate binary cross entropy.
    """

    Y_hat = np.sum(
        M
        * prototype_scores[np.newaxis, :],
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
# LOAD TEST CASES
# ============================================================

test_df = pd.read_csv(
    TEST_CASE_CSV
)


required_columns = [
    "Test",
    "Prototype locations and scores",
    "What this isolates"
]

for column in required_columns:

    if column not in test_df.columns:

        raise ValueError(
            f"Missing required column: "
            f"'{column}'"
        )


# ============================================================
# RUN ALL TEST CASES
# ============================================================

results = []

print()
print("=" * 70)
print("LFR PROTOTYPE TEST CASES")
print("=" * 70)

print(
    f"Training data: {TRAINING_CSV}"
)

print(
    f"Test cases: {TEST_CASE_CSV}"
)

print(
    f"SAT range: "
    f"{SAT_MIN:.0f} - {SAT_MAX:.0f}"
)

print(
    f"Fixed alpha: {ALPHA}"
)

print()


for index, row in test_df.iterrows():

    test_name = row["Test"]

    prototype_string = (
        row[
            "Prototype locations and scores"
        ]
    )

    print("-" * 70)
    print(test_name)

    try:

        # ====================================================
        # PARSE PROTOTYPES
        # ====================================================

        prototypes_sat, prototype_scores = (
            parse_prototypes(
                prototype_string
            )
        )

        K = len(
            prototypes_sat
        )


        # ====================================================
        # VALIDATION
        # ====================================================

        if np.any(
            prototype_scores < 0
        ) or np.any(
            prototype_scores > 1
        ):

            raise ValueError(
                "Prototype scores must "
                "be between 0 and 1."
            )


        if np.any(
            prototypes_sat < SAT_MIN
        ) or np.any(
            prototypes_sat > SAT_MAX
        ):

            raise ValueError(
                f"Prototype SAT locations "
                f"must be between "
                f"{SAT_MIN:.0f} and "
                f"{SAT_MAX:.0f}."
            )


        # ====================================================
        # NORMALIZE PROTOTYPE LOCATIONS
        # ====================================================

        prototypes_normalized = (
            prototypes_sat - SAT_MIN
        ) / (
            SAT_MAX - SAT_MIN
        )


        # ====================================================
        # MEMBERSHIP
        # ====================================================

        M = calculate_membership(
            X,
            prototypes_normalized,
            ALPHA
        )


        # ====================================================
        # LOSSES
        # ====================================================

        Lz = fairness_loss(
            M
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

        total_loss = (
            AZ * Lz
            + AX * Lx
            + AY * Ly
        )


        # ====================================================
        # STORE RESULT
        # ====================================================

        results.append({
            "K": K,
            "Alpha": ALPHA,
            "Lz": Lz,
            "Lx": Lx,
            "Ly": Ly,
            "L": total_loss
        })


        # ====================================================
        # TERMINAL SUMMARY
        # ====================================================

        print(
            f"Prototypes: "
            f"{prototype_string}"
        )

        print(
            f"K = {K}"
        )

        print(
            f"Lz = {Lz:.6f}"
        )

        print(
            f"Lx = {Lx:.6f}"
        )

        print(
            f"Ly = {Ly:.6f}"
        )

        print(
            f"L  = {total_loss:.6f}"
        )

        print()


    except Exception as error:

        print(
            f"ERROR: {error}"
        )

        print()

        results.append({
            "K": np.nan,
            "Alpha": ALPHA,
            "Lz": np.nan,
            "Lx": np.nan,
            "Ly": np.nan,
            "L": np.nan
        })


# ============================================================
# ADD RESULTS TO ORIGINAL TEST CASE DATAFRAME
# ============================================================

result_df = pd.DataFrame(
    results
)

test_df[
    "K"
] = result_df["K"]

test_df[
    "Alpha"
] = result_df["Alpha"]

test_df[
    "Lz"
] = result_df["Lz"]

test_df[
    "Lx"
] = result_df["Lx"]

test_df[
    "Ly"
] = result_df["Ly"]

test_df[
    "L"
] = result_df["L"]


# ============================================================
# WRITE RESULTS BACK TO SAME CSV
# ============================================================

test_df.to_csv(
    TEST_CASE_CSV,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("ALL TEST CASE RESULTS")
print("=" * 70)
print()

print(
    test_df[
        [
            "Test",
            "K",
            "Alpha",
            "Lz",
            "Lx",
            "Ly",
            "L"
        ]
    ].to_string(
        index=False
    )
)

print()

print(
    f"Results written to: "
    f"{TEST_CASE_CSV}"
)