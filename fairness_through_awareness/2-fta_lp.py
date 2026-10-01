import pandas as pd
import numpy as np
from scipy.optimize import linprog

# ============================================================
# CONFIGURATION
# ============================================================

CSV_PATH = (
    # "fairness_through_awareness_3/"
    "training_dataset.csv"
)

# Column names

GENDER_COL = "Gender"
SAT_COL = "SAT"
HOBBY_COL = "Hobby"
ADMISSION_COL = "Admission"

# ============================================================
# FTA DISTANCE CONFIGURATION
# ============================================================

# Smaller value = SAT differences matter more.
#
# Examples:
#
# SAT_SCALE = 1000
# 100 SAT points -> distance 0.10
#
# SAT_SCALE = 600
# 100 SAT points -> distance 0.167
#
# SAT_SCALE = 500
# 100 SAT points -> distance 0.20
#
# A smaller SAT_SCALE allows FTA to assign larger
# probability differences to applicants with large
# SAT differences.

SAT_SCALE = 600.0

# ============================================================
# LOAD TRAINING DATASET
# ============================================================

df = pd.read_csv(CSV_PATH)

# ============================================================
# ENCODE VARIABLES
# ============================================================

df["GenderNum"] = df[GENDER_COL].map({
    "M": 0,
    "F": 1
})

df["HobbyNum"] = df[HOBBY_COL].map({
    "Soccer": 0,
    "Dance": 1
})

df["Y"] = df[ADMISSION_COL].map({
    "No": 0,
    "Yes": 1
})

# ============================================================
# CONVERT COLUMNS TO ARRAYS
# ============================================================

sat = df[SAT_COL].to_numpy(dtype=float)
hobby = df["HobbyNum"].to_numpy(dtype=int)
y = df["Y"].to_numpy(dtype=float)

n = len(df)

# ============================================================
# ORIGINAL ACCEPTANCE STATISTICS
# ============================================================

original_accepted = int(y.sum())
original_acceptance_rate = original_accepted / n

print("Original training dataset:")
print(
    f"Accepted: {original_accepted}/{n} "
    f"({original_acceptance_rate:.1%})"
)

# ============================================================
# FTA DISTANCE
#
# Gender is NOT included.
#
# Similarity is based only on:
#   - SAT
#   - Hobby
#
# d(i,j) =
#     |SAT_i - SAT_j| / SAT_SCALE
#     + hobby_difference
#
# Same hobby:
#     hobby_difference = 0
#
# Different hobby:
#     hobby_difference = 1
#
# ============================================================

def distance(i, j):

    # --------------------------------------------------------
    # SAT component
    # --------------------------------------------------------

    sat_distance = (
        abs(sat[i] - sat[j])
        / SAT_SCALE
    )

    # --------------------------------------------------------
    # Hobby component
    # --------------------------------------------------------

    hobby_distance = (
        0
        if hobby[i] == hobby[j]
        else 1
    )

    return sat_distance + hobby_distance


# ============================================================
# OBJECTIVE FUNCTION
#
# For Y = 1:
#
#     loss = 1 - p_i
#
# For Y = 0:
#
#     loss = p_i
#
# Ignoring constants:
#
#     Y = 1 -> coefficient -1
#     Y = 0 -> coefficient +1
#
# Therefore the LP tries to:
#
#     push positive examples toward 1
#     push negative examples toward 0
#
# while satisfying the FTA fairness constraints.
# ============================================================

c = np.where(
    y == 1,
    -1.0,
    1.0
)

# ============================================================
# FTA FAIRNESS CONSTRAINTS
#
# |p_i - p_j| <= d(i,j)
#
# becomes:
#
#     p_i - p_j <= d(i,j)
#     p_j - p_i <= d(i,j)
#
# ============================================================

A_ub = []
b_ub = []

for i in range(n):

    for j in range(i + 1, n):

        d = distance(i, j)

        # ----------------------------------------------------
        # p_i - p_j <= d
        # ----------------------------------------------------

        constraint_1 = np.zeros(n)

        constraint_1[i] = 1
        constraint_1[j] = -1

        A_ub.append(constraint_1)
        b_ub.append(d)

        # ----------------------------------------------------
        # p_j - p_i <= d
        # ----------------------------------------------------

        constraint_2 = np.zeros(n)

        constraint_2[j] = 1
        constraint_2[i] = -1

        A_ub.append(constraint_2)
        b_ub.append(d)

# ============================================================
# PROBABILITY BOUNDS
#
# 0 <= p_i <= 1
# ============================================================

bounds = [
    (0, 1)
    for _ in range(n)
]

# ============================================================
# SOLVE FTA LP
# ============================================================

result = linprog(
    c=c,
    A_ub=np.array(A_ub),
    b_ub=np.array(b_ub),
    bounds=bounds,
    method="highs"
)

# ============================================================
# CHECK OPTIMIZATION
# ============================================================

if not result.success:

    print("FTA LP failed:")
    print(result.message)

    raise RuntimeError(
        "FTA LP failed"
    )

print("\nFTA LP solved successfully.")

print(
    f"SAT scale used by FTA: "
    f"{SAT_SCALE}"
)

# ============================================================
# STORE FTA PROBABILITIES
#
# IMPORTANT:
# No threshold is applied here.
#
# The continuous FTA probability is passed directly
# to the downstream classifier.
# ============================================================

df["FTA_Probability"] = result.x

# ============================================================
# CHECK FAIRNESS CONSTRAINTS
# ============================================================

max_violation = 0.0
binding_constraints = 0
total_pairs = 0

for i in range(n):

    for j in range(i + 1, n):

        d = distance(i, j)

        probability_difference = abs(
            result.x[i] - result.x[j]
        )

        violation = (
            probability_difference - d
        )

        max_violation = max(
            max_violation,
            violation
        )

        # A constraint is considered binding if the
        # probability difference is approximately equal
        # to the allowed distance.

        if abs(
            probability_difference - d
        ) < 1e-7:

            binding_constraints += 1

        total_pairs += 1

# ============================================================
# PRINT CONSTRAINT DIAGNOSTICS
# ============================================================

print("\nFTA constraint diagnostics:")

print(
    f"Total applicant pairs: "
    f"{total_pairs}"
)

print(
    f"Total inequality constraints: "
    f"{total_pairs * 2}"
)

print(
    f"Binding pairwise relationships: "
    f"{binding_constraints}"
)

print(
    f"Maximum fairness violation: "
    f"{max_violation:.10f}"
)

# ============================================================
# DISPLAY FTA PROBABILITIES
#
# No FTA_Prediction is generated.
# ============================================================

print("\nFTA probabilities:\n")

print(
    df[
        [
            "ID",
            GENDER_COL,
            SAT_COL,
            HOBBY_COL,
            ADMISSION_COL,
            "FTA_Probability"
        ]
    ].to_string(index=False)
)

# ============================================================
# SAVE RESULT
#
# The output contains:
#
#   ID
#   Gender
#   SAT
#   Hobby
#   Admission
#   FTA_Probability
#
# GenderNum, HobbyNum, and Y are internal columns and are
# removed before saving.
# ============================================================

output_path = (
    # "fairness_through_awareness_3/"
    "lp_result.csv"
)

output_df = df[
    [
        "ID",
        GENDER_COL,
        SAT_COL,
        HOBBY_COL,
        ADMISSION_COL,
        "FTA_Probability"
    ]
].copy()

output_df.to_csv(
    output_path,
    index=False
)

print(
    f"\nSaved to: {output_path}"
)