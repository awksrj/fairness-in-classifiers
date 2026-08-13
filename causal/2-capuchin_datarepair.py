import pandas as pd
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_CSV = "causal/1-training_data.csv"
OUTPUT_CSV = "causal/3-repaired_data.csv"

# ------------------------------------------------------------
# CI:
#
#     Gender ⟂ Admission | SAT, Hobby
#
# In the paper's notation:
#
#     X = Gender
#     Y = Admission
#     Z = {SAT, Hobby}
# ------------------------------------------------------------

X_COLS = ["Gender"]
Y_COLS = ["Admission"]
Z_COLS = ["SAT", "Hobby"]

ID_COL = "ID"


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_CSV)

print("Original dataset:")
print(df)
print()

# Check required columns
required_columns = (
    X_COLS +
    Y_COLS +
    Z_COLS
)

for col in required_columns:
    if col not in df.columns:
        raise ValueError(
            f"Missing required column: {col}"
        )


# ============================================================
# PAPER STEP 1:
# CONSTRUCT BAG B
# ============================================================
#
# The paper defines B as the smallest bag representing Pr.
#
# Your training CSV is uniform:
#
#     Pr(t) = 1 / |D|
#
# and every row occurs once.
#
# Therefore:
#
#     B = D
#
# ============================================================

bag = df.drop(columns=[ID_COL], errors="ignore").copy()

print(f"Original bag size: {len(bag)}")


# ============================================================
# PAPER STEP 2:
# ADD FRESH ATTRIBUTE K
# ============================================================
#
# Every occurrence gets a distinct K.
#
# This converts the bag into a set DB.
#
#     DB = {(i,t) | t occurs in B}
#
# ============================================================

db = bag.copy()

db.insert(
    0,
    "K",
    range(1, len(db) + 1)
)

print(f"Database DB size: {len(db)}")


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def make_key(row, columns):
    """
    Return a tuple representing the values of `columns`
    in a database row.
    """
    return tuple(row[col] for col in columns)


def tuple_from_values(values, columns):
    """
    Convert a tuple of values into a dictionary.
    """
    return {
        col: value
        for col, value in zip(columns, values)
    }


# ============================================================
# PAPER STEP 3:
# CONSTRUCT D*
# ============================================================
#
# The paper defines:
#
#     D* = Π_XZ(D) × Π_ZY(D)
#
# For every:
#
#     X1, Z  appearing in D
#
# and
#
#     Z, Y2 appearing in D
#
# create:
#
#     (X1, Y2, Z)
#
# ============================================================

XZ_COLS = X_COLS + Z_COLS
ZY_COLS = Z_COLS + Y_COLS
ALL_COLS = X_COLS + Y_COLS + Z_COLS

# Projection Π_XZ(D)
xz_projection = (
    db[XZ_COLS]
    .drop_duplicates()
)

# Projection Π_ZY(D)
zy_projection = (
    db[ZY_COLS]
    .drop_duplicates()
)

# Join the two projections on Z.
#
# This is equivalent to:
#
#     Π_XZ(D) × Π_ZY(D)
#
# with matching Z values.

d_star = xz_projection.merge(
    zy_projection,
    on=Z_COLS,
    how="inner"
)

# Keep only X,Y,Z
d_star = d_star[ALL_COLS].drop_duplicates()

d_star = d_star.reset_index(drop=True)

print(f"D* size: {len(d_star)}")
print()


# ============================================================
# MAP EVERY POSSIBLE TUPLE TO A BOOLEAN VARIABLE
# ============================================================
#
# The paper associates a Boolean variable X_t to every
# candidate tuple t ∈ D*.
#
# X_t = 1  -> tuple exists in repaired database
# X_t = 0  -> tuple does not exist
#
# ============================================================

def tuple_key_from_row(row):
    return tuple(
        row[col]
        for col in ALL_COLS
    )


candidate_keys = [
    tuple_key_from_row(row)
    for _, row in d_star.iterrows()
]

candidate_index = {
    key: i
    for i, key in enumerate(candidate_keys)
}

num_variables = len(candidate_keys)

print(
    f"Number of Boolean variables: "
    f"{num_variables}"
)


# ============================================================
# ORIGINAL DATABASE SET
# ============================================================

original_keys = {
    tuple(
        row[col]
        for col in ALL_COLS
    )
    for _, row in db.iterrows()
}


# ============================================================
# PAPER STEP 4:
# CONSTRUCT HARD CLAUSES
# ============================================================
#
# For every pair:
#
#     (X1,Y1,Z)
#     (X2,Y2,Z)
#
# the MVD requires:
#
#     (X1,Y2,Z)
#
# to exist.
#
# Therefore the hard clause is:
#
#     ¬X_t1 ∨ ¬X_t2 ∨ X_t3
#
# This is exactly Equation / Algorithm 1 in the paper.
#
# ============================================================

hard_clauses = []

# Group candidates by Z
grouped = {}

for idx, row in d_star.iterrows():

    z_key = make_key(row, Z_COLS)

    grouped.setdefault(
        z_key,
        []
    ).append(idx)


for z_key, indices in grouped.items():

    # Every pair of X/Y combinations sharing Z
    for i in indices:

        row_i = d_star.iloc[i]

        for j in indices:

            row_j = d_star.iloc[j]

            # t1 = (X1, Y1, Z)
            t1 = tuple_key_from_row(row_i)

            # t2 = (X2, Y2, Z)
            t2 = tuple_key_from_row(row_j)

            # t3 = (X1, Y2, Z)
            t3_values = []

            for col in X_COLS:
                t3_values.append(row_i[col])

            for col in Y_COLS:
                t3_values.append(row_j[col])

            for col in Z_COLS:
                t3_values.append(row_i[col])

            t3 = tuple(t3_values)

            idx1 = candidate_index[t1]
            idx2 = candidate_index[t2]
            idx3 = candidate_index[t3]

            hard_clauses.append(
                (idx1, idx2, idx3)
            )


# Remove duplicate clauses
hard_clauses = list(
    set(hard_clauses)
)

print(
    f"Hard MVD clauses: "
    f"{len(hard_clauses)}"
)


# ============================================================
# PAPER STEP 5:
# CONSTRUCT WEIGHTED SOFT CLAUSES
# ============================================================
#
# For t ∈ D:
#
#     soft clause: X_t
#
# For t ∈ D* - D:
#
#     soft clause: ¬X_t
#
# Every clause has cost 1 because your empirical distribution
# is uniform.
#
# Maximizing satisfied soft clauses is equivalent to minimizing
# the symmetric difference:
#
#     |Δ(B,B')|
#
# ============================================================

original_variables = set()

for key in original_keys:

    if key in candidate_index:
        original_variables.add(
            candidate_index[key]
        )

missing_variables = (
    set(range(num_variables))
    - original_variables
)


# ============================================================
# MILP FORMULATION OF WEIGHTED MAXSAT
# ============================================================
#
# x_i ∈ {0,1}
#
# x_i = 1 means candidate tuple i is retained.
#
# Objective:
#
# maximize:
#
#     Σ_{t∈D} x_t
#       +
#     Σ_{t∈D*-D} (1-x_t)
#
# Equivalent to:
#
# maximize:
#
#     Σ_{t∈D} x_t
#       -
#     Σ_{t∈D*-D} x_t
#
# The constant |D*-D| can be ignored.
#
# Since scipy.milp minimizes, we minimize:
#
#     -Σ_{t∈D} x_t
#     +
#      Σ_{t∈D*-D} x_t
#
# ============================================================

objective = np.zeros(num_variables)

for i in original_variables:
    objective[i] = -1

for i in missing_variables:
    objective[i] = 1


# ============================================================
# HARD CLAUSES:
#
#     ¬x1 ∨ ¬x2 ∨ x3
#
# This is equivalent to:
#
#     x1 + x2 - x3 <= 1
#
# ============================================================

if hard_clauses:

    A = lil_matrix(
        (
            len(hard_clauses),
            num_variables
        ),
        dtype=float
    )

    lower_bounds = np.full(
        len(hard_clauses),
        -np.inf
    )

    upper_bounds = np.ones(
        len(hard_clauses)
    )

    for row_idx, (i, j, k) in enumerate(
        hard_clauses
    ):

        A[row_idx, i] = 1
        A[row_idx, j] = 1
        A[row_idx, k] = -1

    constraints = LinearConstraint(
        A.tocsr(),
        lower_bounds,
        upper_bounds
    )

else:

    constraints = None


# ============================================================
# SOLVE
# ============================================================

print("Solving weighted MaxSAT formulation...")

result = milp(
    c=objective,
    integrality=np.ones(num_variables),
    bounds=Bounds(
        np.zeros(num_variables),
        np.ones(num_variables)
    ),
    constraints=constraints,
    options={
        "disp": False
    }
)

if not result.success:

    raise RuntimeError(
        "Repair optimization failed:\n"
        + result.message
    )

print("Repair optimization succeeded.")
print()


# ============================================================
# EXTRACT REPAIRED DATABASE D'_B
# ============================================================

solution = np.rint(
    result.x
).astype(int)

repaired_keys = [
    candidate_keys[i]
    for i in range(num_variables)
    if solution[i] == 1
]

print(
    f"Original DB tuples:  {len(original_keys)}"
)

print(
    f"Repaired DB tuples:  {len(repaired_keys)}"
)

print()


# ============================================================
# PAPER STEP 6:
# PROJECT AWAY K
# ============================================================
#
# The paper constructs:
#
#     D' = Π_V(D'_B)
#
# where V is the original set of attributes.
#
# K is therefore removed.
#
# ============================================================

repaired_rows = []

for key in repaired_keys:

    row = {}

    for col, value in zip(
        ALL_COLS,
        key
    ):
        row[col] = value

    repaired_rows.append(row)


repaired_df = pd.DataFrame(
    repaired_rows,
    columns=ALL_COLS
)


# ============================================================
# RESTORE ID
# ============================================================
#
# ID is not part of the CI/MVD calculation.
# Give the repaired dataset fresh sequential IDs.
#
# ============================================================

repaired_df.insert(
    0,
    ID_COL,
    range(
        1,
        len(repaired_df) + 1
    )
)


# ============================================================
# SAVE
# ============================================================

repaired_df.to_csv(
    OUTPUT_CSV,
    index=False
)

print(
    f"Repaired dataset saved to:\n"
    f"{OUTPUT_CSV}"
)

print()
print("Repaired dataset:")
print(repaired_df)