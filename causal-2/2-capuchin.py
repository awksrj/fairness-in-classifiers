import pandas as pd
from itertools import product
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = "causal-2/1-training_data.csv"

OPERATIONS_FILE = "causal-2/3-repair_operations.csv"
REPAIRED_FILE = "causal-2/3-repaired-data.csv"

ID_COL = "ID"
GENDER_COL = "Gender"
QUAL_COL = "Qualification"
DEPT_COL = "Department"
ADMISSION_COL = "Admission"

# CI:
#     Admission ⟂ Gender | Department, Qualification
#
# Equivalent saturated MVD:
#     (Department, Qualification) ->-> Gender


# ============================================================
# HELPER: COMPLETE A BLOCK
# ============================================================

def repair_block(block):
    """
    Repair one (Department, Qualification) block.

    For a fixed (D, Q), the MVD requires:

        Gender <-> Admission

    to be independent.

    Because this dataset is represented as a bag, duplicate
    occurrences matter.

    Example:

        Male   Yes = 4
        Female No  = 10

    A pure insertion repair requires:

        Male   No  = 4
        Female Yes = 10

    = 14 insertions.

    But deleting the 4 Male/Yes occurrences costs only 4,
    so deletion is the minimal repair.

    We enumerate possible target count tables and select the
    one with minimum symmetric-difference distance from the
    original block.
    """

    # --------------------------------------------------------
    # Count occurrences
    # --------------------------------------------------------

    genders = sorted(block[GENDER_COL].unique())
    admissions = sorted(block[ADMISSION_COL].unique())

    # For this experiment we expect exactly:
    # Gender = Male, Female
    # Admission = Yes, No

    if len(genders) == 0 or len(admissions) == 0:
        return block.copy(), [], []

    # Count original bag
    counts = {}

    for g in genders:
        for a in admissions:
            counts[(g, a)] = len(
                block[
                    (block[GENDER_COL] == g) &
                    (block[ADMISSION_COL] == a)
                ]
            )

    # --------------------------------------------------------
    # MVD condition
    #
    # For two genders and two admission values, independence
    # means the 2x2 table must have rank 1:
    #
    #   n(M,Y) * n(F,N)
    #       =
    #   n(M,N) * n(F,Y)
    #
    # We find the minimum symmetric-difference repair.
    # --------------------------------------------------------

    # If there is only one gender or one admission value,
    # the MVD is automatically satisfied.
    if len(genders) < 2 or len(admissions) < 2:
        return block.copy(), [], []

    g1, g2 = genders[0], genders[1]
    a1, a2 = admissions[0], admissions[1]

    n11 = counts[(g1, a1)]
    n12 = counts[(g1, a2)]
    n21 = counts[(g2, a1)]
    n22 = counts[(g2, a2)]

    # Already satisfies independence?
    if n11 * n22 == n12 * n21:
        return block.copy(), [], []

    # --------------------------------------------------------
    # Search possible repaired count tables
    #
    # We need:
    #
    #       x11 * x22 = x12 * x21
    #
    # The original number of tuples in this block is small
    # enough for exhaustive search.
    #
    # We search target counts from 0 to a safe upper bound.
    # --------------------------------------------------------

    original_total = len(block)

    # Any minimal repair never needs an arbitrarily large number
    # of tuples. We only need to consider counts up to the
    # original total plus the largest original cell.
    max_count = original_total + max(
        n11, n12, n21, n22
    )

    best_cost = float("inf")
    best_counts = None

    for x11 in range(max_count + 1):
        for x12 in range(max_count + 1):
            for x21 in range(max_count + 1):

                # Determine x22 from the independence condition.
                #
                # x11*x22 = x12*x21
                #
                # If x11 != 0, x22 must be exactly:
                #
                # x12*x21 / x11
                #
                # Otherwise we handle the zero case separately.

                if x11 != 0:

                    numerator = x12 * x21

                    if numerator % x11 != 0:
                        continue

                    x22 = numerator // x11

                    if x22 > max_count:
                        continue

                    target = {
                        (g1, a1): x11,
                        (g1, a2): x12,
                        (g2, a1): x21,
                        (g2, a2): x22,
                    }

                else:

                    # If x11 = 0, then:
                    #
                    # 0 * x22 = x12 * x21
                    #
                    # Therefore:
                    #
                    # x12 = 0 OR x21 = 0

                    # Case x12 = 0
                    if x12 == 0:

                        for x22 in range(max_count + 1):

                            target = {
                                (g1, a1): 0,
                                (g1, a2): 0,
                                (g2, a1): x21,
                                (g2, a2): x22,
                            }

                            cost = sum(
                                abs(
                                    target[key] - counts[key]
                                )
                                for key in target
                            )

                            if cost < best_cost:
                                best_cost = cost
                                best_counts = target

                    # Case x21 = 0
                    if x21 == 0:

                        for x22 in range(max_count + 1):

                            target = {
                                (g1, a1): 0,
                                (g1, a2): x12,
                                (g2, a1): 0,
                                (g2, a2): x22,
                            }

                            cost = sum(
                                abs(
                                    target[key] - counts[key]
                                )
                                for key in target
                            )

                            if cost < best_cost:
                                best_cost = cost
                                best_counts = target

                    continue

                # ------------------------------------------------
                # Symmetric difference distance
                #
                # For each cell:
                #
                # old = original multiplicity
                # new = repaired multiplicity
                #
                # |new-old|
                #
                # is the number of insertions/deletions.
                # ------------------------------------------------

                cost = sum(
                    abs(target[key] - counts[key])
                    for key in target
                )

                if cost < best_cost:
                    best_cost = cost
                    best_counts = target

    # --------------------------------------------------------
    # Construct repaired block
    # --------------------------------------------------------

    repaired_rows = []

    operations = []

    # Keep original rows according to the target multiplicity.
    #
    # If target < original:
    #   keep target rows
    #   delete remaining rows
    #
    # If target >= original:
    #   keep all original rows
    #   insert additional rows
    # --------------------------------------------------------

    next_id = None

    # IDs are handled globally later.
    # Here we use None for inserted IDs.

    for key, original_count in counts.items():

        gender, admission = key
        target_count = best_counts[key]

        matching_rows = block[
            (block[GENDER_COL] == gender) &
            (block[ADMISSION_COL] == admission)
        ].copy()

        # ----------------------------------------------------
        # KEEP existing rows
        # ----------------------------------------------------

        keep_count = min(
            original_count,
            target_count
        )

        keep_rows = matching_rows.iloc[:keep_count]

        for _, row in keep_rows.iterrows():
            repaired_rows.append(row.copy())

            operations.append(
                (
                    row.copy(),
                    "Keep"
                )
            )

        # ----------------------------------------------------
        # DELETE excess original rows
        # ----------------------------------------------------

        delete_count = max(
            0,
            original_count - target_count
        )

        delete_rows = matching_rows.iloc[
            keep_count:
        ]

        for _, row in delete_rows.iterrows():

            operations.append(
                (
                    row.copy(),
                    "Delete"
                )
            )

        # ----------------------------------------------------
        # INSERT missing rows
        # ----------------------------------------------------

        insert_count = max(
            0,
            target_count - original_count
        )

        for _ in range(insert_count):

            new_row = {
                ID_COL: None,
                GENDER_COL: gender,
                QUAL_COL: block.iloc[0][QUAL_COL],
                DEPT_COL: block.iloc[0][DEPT_COL],
                ADMISSION_COL: admission,
            }

            new_row = pd.Series(new_row)

            repaired_rows.append(new_row)

            operations.append(
                (
                    new_row,
                    "Insert"
                )
            )

    repaired_block = pd.DataFrame(repaired_rows)

    return repaired_block, operations, best_counts


# ============================================================
# MAIN REPAIR
# ============================================================

def capuchin_repair():

    # --------------------------------------------------------
    # Read training data
    # --------------------------------------------------------

    df = pd.read_csv(INPUT_FILE)

    required_columns = [
        ID_COL,
        GENDER_COL,
        QUAL_COL,
        DEPT_COL,
        ADMISSION_COL
    ]

    missing = [
        c for c in required_columns
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    print("=" * 60)
    print("CAPUCHIN BAG-BASED MVD REPAIR")
    print("=" * 60)

    print(f"Original rows: {len(df)}")

    # --------------------------------------------------------
    # Track operations for ORIGINAL rows
    # --------------------------------------------------------

    operation_records = []

    # Repaired dataset rows
    repaired_rows = []

    # New IDs start after the largest original ID
    max_id = pd.to_numeric(
        df[ID_COL],
        errors="coerce"
    ).max()

    if pd.isna(max_id):
        max_id = len(df)

    next_id = int(max_id) + 1

    # --------------------------------------------------------
    # Repair each (Department, Qualification) block
    # --------------------------------------------------------

    grouped = df.groupby(
        [DEPT_COL, QUAL_COL],
        sort=False
    )

    for (department, qualification), block in grouped:

        print()
        print(
            f"Block: Department={department}, "
            f"Qualification={qualification}"
        )

        print(
            block[
                [GENDER_COL, ADMISSION_COL]
            ].value_counts()
        )

        repaired_block, operations, best_counts = repair_block(
            block
        )

        # ----------------------------------------------------
        # Assign IDs to inserted rows
        # ----------------------------------------------------

        for row, operation in operations:

            row = row.copy()

            if operation == "Insert":

                row[ID_COL] = next_id
                next_id += 1

            operation_records.append({
                ID_COL: row[ID_COL],
                GENDER_COL: row[GENDER_COL],
                QUAL_COL: row[QUAL_COL],
                DEPT_COL: row[DEPT_COL],
                ADMISSION_COL: row[ADMISSION_COL],
                "Operation": operation
            })

            if operation in ["Keep", "Insert"]:
                repaired_rows.append(row)

        # ----------------------------------------------------
        # Print repair summary
        # ----------------------------------------------------

        counts = block.groupby(
            [GENDER_COL, ADMISSION_COL]
        ).size().to_dict()

        print("Original counts:")
        print(counts)

        print("Repaired counts:")
        print(best_counts)

        deletes = sum(
            1
            for _, op in operations
            if op == "Delete"
        )

        inserts = sum(
            1
            for _, op in operations
            if op == "Insert"
        )

        print(
            f"Repair: {deletes} deletion(s), "
            f"{inserts} insertion(s)"
        )

    # --------------------------------------------------------
    # Create operation dataframe
    # --------------------------------------------------------

    operations_df = pd.DataFrame(
        operation_records,
        columns=[
            ID_COL,
            GENDER_COL,
            QUAL_COL,
            DEPT_COL,
            ADMISSION_COL,
            "Operation"
        ]
    )

    # --------------------------------------------------------
    # Create repaired dataframe
    # --------------------------------------------------------

    repaired_df = pd.DataFrame(
        repaired_rows,
        columns=[
            ID_COL,
            GENDER_COL,
            QUAL_COL,
            DEPT_COL,
            ADMISSION_COL
        ]
    )

    # Make IDs integers
    operations_df[ID_COL] = pd.to_numeric(
        operations_df[ID_COL]
    ).astype(int)

    repaired_df[ID_COL] = pd.to_numeric(
        repaired_df[ID_COL]
    ).astype(int)

    # --------------------------------------------------------
    # Make output directories
    # --------------------------------------------------------

    Path(OPERATIONS_FILE).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    Path(REPAIRED_FILE).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    operations_df.to_csv(
        OPERATIONS_FILE,
        index=False
    )

    repaired_df.to_csv(
        REPAIRED_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Final statistics
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("REPAIR COMPLETE")
    print("=" * 60)

    print(
        f"Original rows: {len(df)}"
    )

    print(
        "Kept:",
        (operations_df["Operation"] == "Keep").sum()
    )

    print(
        "Deleted:",
        (operations_df["Operation"] == "Delete").sum()
    )

    print(
        "Inserted:",
        (operations_df["Operation"] == "Insert").sum()
    )

    print(
        f"Repaired rows: {len(repaired_df)}"
    )

    print()
    print(
        f"Operations file: {OPERATIONS_FILE}"
    )

    print(
        f"Repaired data:   {REPAIRED_FILE}"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    capuchin_repair()