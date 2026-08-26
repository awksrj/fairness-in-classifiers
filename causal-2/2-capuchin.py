import pandas as pd
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

TRAINING_COLUMNS = [
    ID_COL,
    GENDER_COL,
    QUAL_COL,
    DEPT_COL,
    ADMISSION_COL
]


# ============================================================
# REPAIR ONE (DEPARTMENT, QUALIFICATION) BLOCK
# ============================================================

def repair_block(block):
    """
    Repair one (Department, Qualification) block.

    Target CI:

        Admission ⟂ Gender | Department, Qualification

    Equivalent saturated MVD:

        (Department, Qualification) ->-> Gender

    For a fixed (Department, Qualification), the Gender and
    Admission values must satisfy the MVD.

    Because Capuchin works with a BAG, duplicate occurrences
    matter.

    Example:

        Male, Yes    x4
        Female, No   x10

    Possible insertion repair:

        Male, No     x4
        Female, Yes  x10

    Cost = 14.

    But deleting the 4 Male/Yes tuples costs only 4.

    Therefore deletion is the minimal repair.
    """

    # --------------------------------------------------------
    # Unique values in this block
    # --------------------------------------------------------

    genders = sorted(
        block[GENDER_COL].unique()
    )

    admissions = sorted(
        block[ADMISSION_COL].unique()
    )

    # If there is only one gender or one admission value,
    # the MVD is automatically satisfied.
    if len(genders) < 2 or len(admissions) < 2:
        operations = []

        for _, row in block.iterrows():
            operations.append(
                (row.copy(), "Keep")
            )

        return block.copy(), operations

    # --------------------------------------------------------
    # This experiment has two genders and two admission values
    # --------------------------------------------------------

    g1, g2 = genders[0], genders[1]
    a1, a2 = admissions[0], admissions[1]

    # --------------------------------------------------------
    # Original multiplicities
    # --------------------------------------------------------

    counts = {}

    for g in genders:
        for a in admissions:

            counts[(g, a)] = len(
                block[
                    (block[GENDER_COL] == g)
                    &
                    (block[ADMISSION_COL] == a)
                ]
            )

    n11 = counts[(g1, a1)]
    n12 = counts[(g1, a2)]
    n21 = counts[(g2, a1)]
    n22 = counts[(g2, a2)]

    # --------------------------------------------------------
    # MVD / independence condition
    #
    # For a 2x2 table:
    #
    #     n11 * n22 = n12 * n21
    #
    # --------------------------------------------------------

    if n11 * n22 == n12 * n21:

        operations = []

        for _, row in block.iterrows():
            operations.append(
                (row.copy(), "Keep")
            )

        return block.copy(), operations

    # --------------------------------------------------------
    # Search for minimum symmetric-difference repair
    #
    # We search possible target multiplicities:
    #
    #     x11, x12, x21, x22
    #
    # satisfying:
    #
    #     x11*x22 = x12*x21
    #
    # Cost:
    #
    #     Σ |xij - nij|
    #
    # This corresponds to bag symmetric difference.
    # --------------------------------------------------------

    original_total = len(block)

    max_original_cell = max(
        n11,
        n12,
        n21,
        n22
    )

    # A safe finite search range for this small experiment.
    max_count = (
        original_total
        + max_original_cell
    )

    best_cost = float("inf")
    best_counts = None

    # --------------------------------------------------------
    # Exhaustive search
    # --------------------------------------------------------

    for x11 in range(max_count + 1):

        for x12 in range(max_count + 1):

            for x21 in range(max_count + 1):

                # --------------------------------------------
                # Case x11 > 0
                #
                # x11*x22 = x12*x21
                #
                # Therefore:
                #
                # x22 = x12*x21 / x11
                # --------------------------------------------

                if x11 > 0:

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
                        (g2, a2): x22
                    }

                    cost = sum(
                        abs(
                            target[key]
                            - counts[key]
                        )
                        for key in target
                    )

                    if cost < best_cost:

                        best_cost = cost
                        best_counts = target

                # --------------------------------------------
                # Case x11 = 0
                #
                # Then:
                #
                #     0*x22 = x12*x21
                #
                # Therefore:
                #
                #     x12 = 0
                #
                # OR
                #
                #     x21 = 0
                # --------------------------------------------

                else:

                    # ----------------------------------------
                    # Case x12 = 0
                    # ----------------------------------------

                    if x12 == 0:

                        for x22 in range(
                            max_count + 1
                        ):

                            target = {
                                (g1, a1): 0,
                                (g1, a2): 0,
                                (g2, a1): x21,
                                (g2, a2): x22
                            }

                            cost = sum(
                                abs(
                                    target[key]
                                    - counts[key]
                                )
                                for key in target
                            )

                            if cost < best_cost:

                                best_cost = cost
                                best_counts = target

                    # ----------------------------------------
                    # Case x21 = 0
                    # ----------------------------------------

                    if x21 == 0:

                        for x22 in range(
                            max_count + 1
                        ):

                            target = {
                                (g1, a1): 0,
                                (g1, a2): x12,
                                (g2, a1): 0,
                                (g2, a2): x22
                            }

                            cost = sum(
                                abs(
                                    target[key]
                                    - counts[key]
                                )
                                for key in target
                            )

                            if cost < best_cost:

                                best_cost = cost
                                best_counts = target

    # --------------------------------------------------------
    # Safety check
    # --------------------------------------------------------

    if best_counts is None:
        raise RuntimeError(
            "Could not find an MVD repair."
        )

    # --------------------------------------------------------
    # Construct operations
    #
    # For each cell:
    #
    # original_count
    # target_count
    #
    # target < original:
    #     Keep target rows
    #     Delete remaining rows
    #
    # target > original:
    #     Keep all original rows
    #     Insert additional rows
    # --------------------------------------------------------

    operations = []

    for key in counts:

        gender, admission = key

        original_count = counts[key]
        target_count = best_counts[key]

        matching_rows = block[
            (block[GENDER_COL] == gender)
            &
            (block[ADMISSION_COL] == admission)
        ].copy()

        # ----------------------------------------------------
        # KEEP
        # ----------------------------------------------------

        keep_count = min(
            original_count,
            target_count
        )

        keep_rows = matching_rows.iloc[
            :keep_count
        ]

        for _, row in keep_rows.iterrows():

            operations.append(
                (
                    row.copy(),
                    "Keep"
                )
            )

        # ----------------------------------------------------
        # DELETE
        # ----------------------------------------------------

        if target_count < original_count:

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
        # INSERT
        # ----------------------------------------------------

        if target_count > original_count:

            insert_count = (
                target_count
                - original_count
            )

            for _ in range(insert_count):

                new_row = pd.Series({
                    ID_COL: None,
                    GENDER_COL: gender,
                    QUAL_COL: block.iloc[0][QUAL_COL],
                    DEPT_COL: block.iloc[0][DEPT_COL],
                    ADMISSION_COL: admission
                })

                operations.append(
                    (
                        new_row,
                        "Insert"
                    )
                )

    return None, operations


# ============================================================
# MAIN CAPUCHIN REPAIR
# ============================================================

def capuchin_repair():

    # --------------------------------------------------------
    # READ INPUT
    # --------------------------------------------------------

    df = pd.read_csv(
        INPUT_FILE
    )

    # --------------------------------------------------------
    # Validate columns
    # --------------------------------------------------------

    missing_columns = [
        column
        for column in TRAINING_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Training dataset is missing columns: "
            + str(missing_columns)
        )

    # --------------------------------------------------------
    # Make sure ID is numeric
    # --------------------------------------------------------

    df[ID_COL] = pd.to_numeric(
        df[ID_COL]
    )

    # --------------------------------------------------------
    # Starting ID for inserted tuples
    # --------------------------------------------------------

    next_id = (
        int(df[ID_COL].max())
        + 1
    )

    # --------------------------------------------------------
    # Store ALL operations
    # --------------------------------------------------------

    operation_records = []

    # --------------------------------------------------------
    # Store only Keep + Insert
    #
    # This becomes repaired-data.csv
    # --------------------------------------------------------

    repaired_records = []

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    total_keep = 0
    total_delete = 0
    total_insert = 0

    # --------------------------------------------------------
    # Process each (Department, Qualification) block
    # --------------------------------------------------------

    grouped = df.groupby(
        [
            DEPT_COL,
            QUAL_COL
        ],
        sort=False
    )

    print()
    print("=" * 70)
    print("CAPUCHIN BAG-BASED MVD REPAIR")
    print("=" * 70)

    print(
        f"Input dataset: {INPUT_FILE}"
    )

    print(
        f"Original rows: {len(df)}"
    )

    print()

    for (
        department,
        qualification
    ), block in grouped:

        print(
            "-" * 70
        )

        print(
            f"Department = {department}, "
            f"Qualification = {qualification}"
        )

        # ----------------------------------------------------
        # Show original block
        # ----------------------------------------------------

        original_counts = (
            block
            .groupby(
                [
                    GENDER_COL,
                    ADMISSION_COL
                ]
            )
            .size()
            .to_dict()
        )

        print(
            "Original counts:"
        )

        for key, count in original_counts.items():

            print(
                f"    {key}: {count}"
            )

        # ----------------------------------------------------
        # Repair block
        # ----------------------------------------------------

        _, operations = repair_block(
            block
        )

        # ----------------------------------------------------
        # Process operations
        # ----------------------------------------------------

        block_keep = 0
        block_delete = 0
        block_insert = 0

        for row, operation in operations:

            row = row.copy()

            # -----------------------------------------------
            # Assign new ID to inserted tuples
            # -----------------------------------------------

            if operation == "Insert":

                row[ID_COL] = next_id

                next_id += 1

            # -----------------------------------------------
            # Record in repair_operations.csv
            #
            # ALL training columns + Operation
            # -----------------------------------------------

            operation_records.append({
                ID_COL: row[ID_COL],
                GENDER_COL: row[GENDER_COL],
                QUAL_COL: row[QUAL_COL],
                DEPT_COL: row[DEPT_COL],
                ADMISSION_COL: row[ADMISSION_COL],
                "Operation": operation
            })

            # -----------------------------------------------
            # Record Keep + Insert in repaired dataset
            #
            # NO Operation column
            # -----------------------------------------------

            if operation in [
                "Keep",
                "Insert"
            ]:

                repaired_records.append({
                    ID_COL: row[ID_COL],
                    GENDER_COL: row[GENDER_COL],
                    QUAL_COL: row[QUAL_COL],
                    DEPT_COL: row[DEPT_COL],
                    ADMISSION_COL: row[ADMISSION_COL]
                })

            # -----------------------------------------------
            # Statistics
            # -----------------------------------------------

            if operation == "Keep":

                block_keep += 1
                total_keep += 1

            elif operation == "Delete":

                block_delete += 1
                total_delete += 1

            elif operation == "Insert":

                block_insert += 1
                total_insert += 1

        # ----------------------------------------------------
        # Repaired counts for this block
        # ----------------------------------------------------

        repaired_block_records = [
            row
            for row in repaired_records
            if (
                row[DEPT_COL] == department
                and
                row[QUAL_COL] == qualification
            )
        ]

        repaired_block_df = pd.DataFrame(
            repaired_block_records
        )

        if len(repaired_block_df) > 0:

            repaired_counts = (
                repaired_block_df
                .groupby(
                    [
                        GENDER_COL,
                        ADMISSION_COL
                    ]
                )
                .size()
                .to_dict()
            )

        else:

            repaired_counts = {}

        # ----------------------------------------------------
        # Print repair
        # ----------------------------------------------------

        print(
            f"Repair:"
            f" {block_keep} Keep,"
            f" {block_delete} Delete,"
            f" {block_insert} Insert"
        )

        if block_delete > 0 or block_insert > 0:

            print(
                "  >>> VIOLATION REPAIRED"
            )

        print(
            "Repaired counts:"
        )

        for key, count in repaired_counts.items():

            print(
                f"    {key}: {count}"
            )

    # ========================================================
    # CREATE repair_operations.csv
    #
    # ALL TRAINING COLUMNS + Operation
    # ========================================================

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

    # ========================================================
    # CREATE repaired-data.csv
    #
    # ONLY TRAINING COLUMNS
    #
    # Only Keep + Insert rows are included.
    # ========================================================

    repaired_df = pd.DataFrame(
        repaired_records,
        columns=[
            ID_COL,
            GENDER_COL,
            QUAL_COL,
            DEPT_COL,
            ADMISSION_COL
        ]
    )

    # --------------------------------------------------------
    # Ensure IDs are integers
    # --------------------------------------------------------

    operations_df[ID_COL] = (
        pd.to_numeric(
            operations_df[ID_COL]
        ).astype(int)
    )

    repaired_df[ID_COL] = (
        pd.to_numeric(
            repaired_df[ID_COL]
        ).astype(int)
    )

    # ========================================================
    # CREATE OUTPUT DIRECTORY
    # ========================================================

    Path("causal-2").mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # SAVE FILE 1
    #
    # causal-2/3-repair_operations.csv
    # ========================================================

    operations_df.to_csv(
        OPERATIONS_FILE,
        index=False
    )

    # ========================================================
    # SAVE FILE 2
    #
    # causal-2/3-repaired-data.csv
    # ========================================================

    repaired_df.to_csv(
        REPAIRED_FILE,
        index=False
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 70)
    print("REPAIR COMPLETE")
    print("=" * 70)

    print(
        f"Original rows : {len(df)}"
    )

    print(
        f"Keep          : {total_keep}"
    )

    print(
        f"Delete        : {total_delete}"
    )

    print(
        f"Insert        : {total_insert}"
    )

    print(
        f"Repaired rows : {len(repaired_df)}"
    )

    print()

    print(
        "Operations file:"
    )

    print(
        f"  {OPERATIONS_FILE}"
    )

    print()

    print(
        "Repaired dataset:"
    )

    print(
        f"  {REPAIRED_FILE}"
    )

    print()
    print(
        "Output columns:"
    )

    print(
        "  repair_operations.csv:"
    )

    print(
        "  ",
        list(operations_df.columns)
    )

    print()

    print(
        "  repaired-data.csv:"
    )

    print(
        "  ",
        list(repaired_df.columns)
    )

    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    capuchin_repair()