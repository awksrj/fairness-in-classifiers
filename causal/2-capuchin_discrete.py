import pandas as pd
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_CSV = "causal/1-training_data.csv"

REPAIRED_CSV = "causal/3-repaired_data_discrete.csv"
OPERATIONS_CSV = "causal/3-repair_operations.csv"

ID_COL = "ID"
GENDER_COL = "Gender"
SAT_COL = "SAT"
HOBBY_COL = "Hobby"
ADMISSION_COL = "Admission"

SAT_BIN_SIZE = 100


# ============================================================
# SAT BINNING
# ============================================================

def sat_to_bin(sat):
    """
    Convert SAT into a 100-point bin.

    Examples:

        1580 -> 1500-1600
        1480 -> 1400-1500
        1420 -> 1400-1500
        1380 -> 1300-1400
    """

    lower = (int(sat) // SAT_BIN_SIZE) * SAT_BIN_SIZE
    upper = lower + SAT_BIN_SIZE

    return f"{lower}-{upper}"


def bin_to_sat(sat_bin):
    """
    Convert a SAT bin into a representative SAT value.

    Example:

        1400-1500 -> 1450
    """

    lower, upper = sat_bin.split("-")

    return (
        int(lower) + int(upper)
    ) // 2


# ============================================================
# LOAD TRAINING DATA
# ============================================================

df = pd.read_csv(INPUT_CSV)

required_columns = [
    ID_COL,
    GENDER_COL,
    SAT_COL,
    HOBBY_COL,
    ADMISSION_COL
]

for column in required_columns:

    if column not in df.columns:

        raise ValueError(
            f"Missing required column: {column}"
        )


# ============================================================
# ADD SAT_BIN
# ============================================================

df["SAT_BIN"] = df[SAT_COL].apply(
    sat_to_bin
)


print()
print("=" * 60)
print("SAT DISCRETIZATION")
print("=" * 60)

print(
    df[
        [
            SAT_COL,
            "SAT_BIN"
        ]
    ]
    .drop_duplicates()
    .sort_values(SAT_COL)
    .to_string(index=False)
)


# ============================================================
# FAIRNESS CONSTRAINT
# ============================================================
#
# We want:
#
#     Gender ⟂ Admission | SAT_BIN, Hobby
#
# Therefore:
#
#     X = Gender
#     Y = Admission
#     Z = {SAT_BIN, Hobby}
#
# The paper reduces the CI to the MVD:
#
#     Z ↠ K,X
#
# ============================================================

X_COLS = [
    GENDER_COL
]

Y_COLS = [
    ADMISSION_COL
]

Z_COLS = [
    "SAT_BIN",
    HOBBY_COL
]

# These are the attributes that participate
# in the repair.
REPAIR_COLS = (
    X_COLS
    + Y_COLS
    + Z_COLS
)


# ============================================================
# ORIGINAL DATABASE D
# ============================================================
#
# ID is NOT part of the fairness relation.
#
# It is only used afterward to identify the original
# applicants.
#
# ============================================================

def tuple_key(row):
    """
    Return the tuple used by the MVD repair.
    """

    return tuple(
        row[column]
        for column in REPAIR_COLS
    )


original_keys = [
    tuple_key(row)
    for _, row in df.iterrows()
]

original_key_set = set(
    original_keys
)


# ============================================================
# CONSTRUCT D*
# ============================================================
#
# Paper:
#
#     D* = Π_XZ(D) × Π_ZY(D)
#
# ============================================================

XZ_COLS = (
    X_COLS
    + Z_COLS
)

ZY_COLS = (
    Z_COLS
    + Y_COLS
)


projection_XZ = (
    df[XZ_COLS]
    .drop_duplicates()
)

projection_ZY = (
    df[ZY_COLS]
    .drop_duplicates()
)


D_star = projection_XZ.merge(
    projection_ZY,
    on=Z_COLS,
    how="inner"
)


D_star = (
    D_star[
        REPAIR_COLS
    ]
    .drop_duplicates()
    .reset_index(drop=True)
)


print()
print("=" * 60)
print("REPAIR PROBLEM")
print("=" * 60)

print(
    f"Original distinct tuples: "
    f"{len(original_key_set)}"
)

print(
    f"Candidate tuples D*: "
    f"{len(D_star)}"
)


# ============================================================
# BOOLEAN VARIABLE FOR EACH t ∈ D*
# ============================================================
#
# x_t = 1:
#     tuple exists in repaired database
#
# x_t = 0:
#     tuple does not exist
#
# ============================================================

candidate_keys = [
    tuple_key(row)
    for _, row in D_star.iterrows()
]

candidate_index = {
    key: index
    for index, key in enumerate(
        candidate_keys
    )
}


num_variables = len(
    candidate_keys
)


# ============================================================
# ORIGINAL / NEW CANDIDATES
# ============================================================

original_variables = set()

for key in original_key_set:

    if key in candidate_index:

        original_variables.add(
            candidate_index[key]
        )


new_variables = (
    set(range(num_variables))
    - original_variables
)


# ============================================================
# BUILD MVD HARD CLAUSES
# ============================================================
#
# For two tuples:
#
#     t1 = (X1, Y1, Z)
#     t2 = (X2, Y2, Z)
#
# MVD requires:
#
#     t3 = (X1, Y2, Z)
#
# Therefore:
#
#     ¬x_t1 ∨ ¬x_t2 ∨ x_t3
#
# which becomes:
#
#     x_t1 + x_t2 - x_t3 <= 1
#
# ============================================================

hard_clauses = []


# Group D* by Z
groups = {}

for index, row in D_star.iterrows():

    z_key = tuple(
        row[column]
        for column in Z_COLS
    )

    groups.setdefault(
        z_key,
        []
    ).append(index)


# Generate MVD constraints
for z_key, indices in groups.items():

    for i in indices:

        row_i = D_star.iloc[i]

        for j in indices:

            row_j = D_star.iloc[j]

            # --------------------------------------------
            # t1 = (X1, Y1, Z)
            # --------------------------------------------

            t1 = tuple(
                row_i[column]
                for column in REPAIR_COLS
            )

            # --------------------------------------------
            # t2 = (X2, Y2, Z)
            # --------------------------------------------

            t2 = tuple(
                row_j[column]
                for column in REPAIR_COLS
            )

            # --------------------------------------------
            # t3 = (X1, Y2, Z)
            # --------------------------------------------

            t3_values = []

            # X1
            for column in X_COLS:

                t3_values.append(
                    row_i[column]
                )

            # Y2
            for column in Y_COLS:

                t3_values.append(
                    row_j[column]
                )

            # Z
            for column in Z_COLS:

                t3_values.append(
                    row_i[column]
                )

            t3 = tuple(
                t3_values
            )

            idx1 = candidate_index[t1]
            idx2 = candidate_index[t2]
            idx3 = candidate_index[t3]

            hard_clauses.append(
                (
                    idx1,
                    idx2,
                    idx3
                )
            )


# Remove duplicates
hard_clauses = list(
    set(hard_clauses)
)


print(
    f"Hard MVD clauses: "
    f"{len(hard_clauses)}"
)


# ============================================================
# WEIGHTED MAXSAT OBJECTIVE
# ============================================================
#
# The paper wants to minimize:
#
#     |Δ(D, D')|
#
# Therefore:
#
# Original tuple:
#
#     reward keeping it
#
#     coefficient = -1
#
# Candidate insertion:
#
#     penalize inserting it
#
#     coefficient = +1
#
# scipy.optimize.milp minimizes the objective.
#
# ============================================================

objective = np.zeros(
    num_variables
)


# Keep original tuples
for index in original_variables:

    objective[index] = -1


# Penalize inserted tuples
for index in new_variables:

    objective[index] = 1


# ============================================================
# ENCODE HARD CLAUSES
# ============================================================

if len(hard_clauses) > 0:

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

    for row_index, (
        i,
        j,
        k
    ) in enumerate(hard_clauses):

        A[row_index, i] = 1
        A[row_index, j] = 1
        A[row_index, k] = -1

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

print()
print(
    "Solving weighted MaxSAT / MILP..."
)

result = milp(
    c=objective,

    integrality=np.ones(
        num_variables
    ),

    bounds=Bounds(
        np.zeros(
            num_variables
        ),
        np.ones(
            num_variables
        )
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


print(
    "Repair optimization succeeded."
)


# ============================================================
# EXTRACT REPAIRED TUPLES
# ============================================================

solution = np.rint(
    result.x
).astype(int)


repaired_keys = {
    candidate_keys[index]
    for index in range(num_variables)
    if solution[index] == 1
}


# ============================================================
# DETERMINE OPERATIONS
# ============================================================
#
# Original tuple:
#
#     if retained -> Keep
#     if removed  -> Delete
#
# New tuple:
#
#     Insert
#
# ============================================================

deleted_keys = (
    original_key_set
    - repaired_keys
)

inserted_keys = (
    repaired_keys
    - original_key_set
)


print()
print("=" * 60)
print("REPAIR RESULT")
print("=" * 60)

print(
    f"Original distinct tuples: "
    f"{len(original_key_set)}"
)

print(
    f"Repaired tuples: "
    f"{len(repaired_keys)}"
)

print(
    f"Deleted tuples: "
    f"{len(deleted_keys)}"
)

print(
    f"Inserted tuples: "
    f"{len(inserted_keys)}"
)


# ============================================================
# CREATE REPAIR OPERATIONS
# ============================================================
#
# Every original applicant gets a row here.
#
# This preserves the original ID and exact SAT.
#
# ============================================================

operation_rows = []


for _, row in df.iterrows():

    key = tuple_key(row)

    if key in deleted_keys:

        operation = "Delete"

    else:

        operation = "Keep"

    operation_rows.append(
        {
            ID_COL: row[ID_COL],
            GENDER_COL: row[GENDER_COL],
            SAT_COL: row[SAT_COL],
            HOBBY_COL: row[HOBBY_COL],
            ADMISSION_COL: row[ADMISSION_COL],
            "SAT_BIN": row["SAT_BIN"],
            "Operation": operation
        }
    )


# ============================================================
# ADD INSERTED TUPLES TO OPERATION LOG
# ============================================================
#
# Inserted tuples don't have original IDs.
#
# Give them IDs after the original maximum.
#
# Their SAT is represented by the midpoint of their bin.
#
# ============================================================

next_id = int(
    df[ID_COL].max()
) + 1


for key in sorted(
    inserted_keys,
    key=str
):

    gender = key[0]
    admission = key[1]
    sat_bin = key[2]
    hobby = key[3]

    reconstructed_sat = bin_to_sat(
        sat_bin
    )

    operation_rows.append(
        {
            ID_COL: next_id,
            GENDER_COL: gender,
            SAT_COL: reconstructed_sat,
            HOBBY_COL: hobby,
            ADMISSION_COL: admission,
            "SAT_BIN": sat_bin,
            "Operation": "Insert"
        }
    )

    next_id += 1


# ============================================================
# SAVE REPAIR OPERATIONS
# ============================================================

operations_df = pd.DataFrame(
    operation_rows
)

operations_df = operations_df[
    [
        ID_COL,
        GENDER_COL,
        SAT_COL,
        HOBBY_COL,
        ADMISSION_COL,
        "SAT_BIN",
        "Operation"
    ]
]

operations_df.to_csv(
    OPERATIONS_CSV,
    index=False
)


# ============================================================
# CREATE FINAL REPAIRED DATASET
# ============================================================
#
# Keep:
#     original rows marked Keep
#
# Insert:
#     newly inserted rows
#
# Delete:
#     excluded
#
# ============================================================

repaired_df = operations_df[
    operations_df["Operation"].isin(
        [
            "Keep",
            "Insert"
        ]
    )
].copy()


# The SAT_BIN was only needed for the repair.
# The final training dataset uses the original
# continuous SAT column.

repaired_df = repaired_df[
    [
        ID_COL,
        GENDER_COL,
        SAT_COL,
        HOBBY_COL,
        ADMISSION_COL
    ]
]


# Sort by ID
repaired_df = (
    repaired_df
    .sort_values(ID_COL)
    .reset_index(drop=True)
)


# ============================================================
# SAVE REPAIRED DATASET
# ============================================================

repaired_df.to_csv(
    REPAIRED_CSV,
    index=False
)


# ============================================================
# PRINT SUMMARY
# ============================================================

print()
print("=" * 60)
print("FILES CREATED")
print("=" * 60)

print(
    f"Repaired dataset:"
    f" {REPAIRED_CSV}"
)

print(
    f"Repair operations:"
    f" {OPERATIONS_CSV}"
)


# ============================================================
# PRINT DELETIONS
# ============================================================

print()
print("=" * 60)
print("DELETED ORIGINAL ROWS")
print("=" * 60)

deleted_operations = operations_df[
    operations_df["Operation"] == "Delete"
]

if len(deleted_operations) == 0:

    print("None")

else:

    print(
        deleted_operations.to_string(
            index=False
        )
    )


# ============================================================
# PRINT INSERTIONS
# ============================================================

print()
print("=" * 60)
print("INSERTED TUPLES")
print("=" * 60)

inserted_operations = operations_df[
    operations_df["Operation"] == "Insert"
]

if len(inserted_operations) == 0:

    print("None")

else:

    print(
        inserted_operations.to_string(
            index=False
        )
    )