import pandas as pd
import itertools
import os

# ============================================================
# Configuration
# ============================================================

INPUT_FILE = "causal-2/1-training_data.csv"

REPAIR_DIR = "causal-2/"

OPERATIONS_FILE = os.path.join(
    REPAIR_DIR,
    "3-repair_operations.csv"
)

REPAIRED_FILE = os.path.join(
    REPAIR_DIR,
    "3-repaired_data.csv"
)

os.makedirs(REPAIR_DIR, exist_ok=True)


# ============================================================
# Load training data
# ============================================================

df = pd.read_csv(INPUT_FILE)

required_columns = [
    "ID",
    "Gender",
    "Qualification",
    "Department",
    "Admission"
]

missing = [
    col for col in required_columns
    if col not in df.columns
]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )

df = df[required_columns].copy()


# ============================================================
# Normalize values
# ============================================================

df["Gender"] = df["Gender"].astype(str)
df["Qualification"] = df["Qualification"].astype(str)
df["Department"] = df["Department"].astype(str)
df["Admission"] = df["Admission"].astype(str)


# ============================================================
# Operation records
# ============================================================

operations = []

# Track original rows
for _, row in df.iterrows():

    record = row.to_dict()
    record["Operation"] = "Keep"

    operations.append(record)


# ============================================================
# Helper functions
# ============================================================

def cell_key(gender, admission):
    return (gender, admission)


def get_cell_rows(group, gender, admission):
    """
    Return original row indices belonging to a particular
    Gender × Admission cell.
    """

    return group[
        (group["Gender"] == gender) &
        (group["Admission"] == admission)
    ]


def valid_rectangles(observed_genders, observed_admissions):
    """
    A relation satisfies:

        (D,Q) ->-> G

    when the observed Gender × Admission combinations form
    the Cartesian product of the observed genders and
    observed admission values.

    For example:

        genders = {M,F}
        admissions = {Yes,No}

    requires:

        M,Yes
        M,No
        F,Yes
        F,No
    """

    return {
        (g, a)
        for g in observed_genders
        for a in observed_admissions
    }


# ============================================================
# Find the minimum-edit repair
# ============================================================

# We use set semantics for the MVD condition:
#
#     (D,Q) ->-> G
#
# Each Gender × Admission cell only needs to exist.
#
# There are only four possible cells:
#
#     Male,Yes
#     Male,No
#     Female,Yes
#     Female,No
#
# For every D,Q block, enumerate all possible valid
# rectangles and choose the one with minimum edit cost.
# ============================================================

next_id = int(df["ID"].max()) + 1

grouped = df.groupby(
    ["Department", "Qualification"],
    sort=False
)

for (department, qualification), group in grouped:

    # --------------------------------------------------------
    # Existing cells
    # --------------------------------------------------------

    all_cells = [
        ("Male", "Yes"),
        ("Male", "No"),
        ("Female", "Yes"),
        ("Female", "No")
    ]

    existing_cells = set()

    cell_counts = {}

    for gender, admission in all_cells:

        cell = cell_key(gender, admission)

        count = len(
            get_cell_rows(
                group,
                gender,
                admission
            )
        )

        cell_counts[cell] = count

        if count > 0:
            existing_cells.add(cell)

    # --------------------------------------------------------
    # Enumerate possible valid rectangles
    # --------------------------------------------------------
    #
    # A valid MVD configuration is:
    #
    #   Gender set × Admission set
    #
    # where Gender set can be:
    #
    #   {M}
    #   {F}
    #   {M,F}
    #
    # and Admission set can be:
    #
    #   {Yes}
    #   {No}
    #   {Yes,No}
    #
    # We select the rectangle requiring the fewest edits.
    # --------------------------------------------------------

    gender_sets = [
        {"Male"},
        {"Female"},
        {"Male", "Female"}
    ]

    admission_sets = [
        {"Yes"},
        {"No"},
        {"Yes", "No"}
    ]

    candidates = []

    for genders in gender_sets:

        for admissions in admission_sets:

            target_cells = valid_rectangles(
                genders,
                admissions
            )

            insert_cost = 0
            delete_cost = 0

            # Existing cell not in target:
            # all tuples in that cell must be deleted.
            for cell, count in cell_counts.items():

                if count > 0 and cell not in target_cells:
                    delete_cost += count

            # Target cell that doesn't exist:
            # one tuple must be inserted.
            for cell in target_cells:

                if cell_counts[cell] == 0:
                    insert_cost += 1

            total_cost = insert_cost + delete_cost

            candidates.append({
                "target_cells": target_cells,
                "insert_cost": insert_cost,
                "delete_cost": delete_cost,
                "total_cost": total_cost
            })

    # --------------------------------------------------------
    # Select minimum-edit repair
    # --------------------------------------------------------

    best = min(
        candidates,
        key=lambda x: (
            x["total_cost"],
            x["insert_cost"]
        )
    )

    target_cells = best["target_cells"]

    # --------------------------------------------------------
    # Mark original rows for deletion
    # --------------------------------------------------------

    for idx in group.index:

        row = df.loc[idx]

        cell = (
            row["Gender"],
            row["Admission"]
        )

        if cell not in target_cells:

            # Find corresponding operation record
            for record in operations:

                if record["ID"] == row["ID"]:

                    record["Operation"] = "Delete"
                    break

    # --------------------------------------------------------
    # Insert missing cells
    # --------------------------------------------------------

    for gender, admission in target_cells:

        if cell_counts[(gender, admission)] == 0:

            new_record = {
                "ID": next_id,
                "Gender": gender,
                "Qualification": qualification,
                "Department": department,
                "Admission": admission,
                "Operation": "Insert"
            }

            operations.append(new_record)

            next_id += 1


# ============================================================
# Create operations dataframe
# ============================================================

operations_df = pd.DataFrame(
    operations,
    columns=[
        "ID",
        "Gender",
        "Qualification",
        "Department",
        "Admission",
        "Operation"
    ]
)


# ============================================================
# Create repaired dataset
# ============================================================

repaired_df = operations_df[
    operations_df["Operation"].isin(
        ["Keep", "Insert"]
    )
].copy()

repaired_df = repaired_df[
    required_columns
]


# ============================================================
# Save outputs
# ============================================================

operations_df.to_csv(
    OPERATIONS_FILE,
    index=False
)

repaired_df.to_csv(
    REPAIRED_FILE,
    index=False
)


# ============================================================
# Print summary
# ============================================================

print("\n========================================")
print("CAPUCHIN-STYLE REPAIR")
print("========================================")

print("\nConstraint:")
print("    Admission ⫫ Gender | Department, Qualification")

print("\nMVD:")
print("    (Department, Qualification) ->-> Gender")

print("\nOperations:")
print(
    operations_df["Operation"]
    .value_counts()
    .to_string()
)

print("\nOriginal rows:", len(df))

print(
    "Repaired rows:",
    len(repaired_df)
)

print(
    "Inserted rows:",
    (
        operations_df["Operation"] == "Insert"
    ).sum()
)

print(
    "Deleted rows:",
    (
        operations_df["Operation"] == "Delete"
    ).sum()
)

print("\nOutput files:")
print(OPERATIONS_FILE)
print(REPAIRED_FILE)