import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================

CSV_PATH = "lfr2-light/1-data.csv"

# Column names
GENDER_COL = "Gender"
SAT_COL = "SAT"
ADMISSION_COL = "Admission"

# SAT range definitions
# Format: (label, lower_bound, upper_bound)
SAT_RANGES = [
    ("1550-1600", 1550, 1600),
    ("1200-1250", 1200, 1250),
    ("850-900",    850,  900),
    ("400-600",    400,  600),
]

# Desired distribution:
# (Gender, SAT range, Admission, expected count)
EXPECTED_DISTRIBUTION = [
    ("M", "1550-1600", "Yes", 6),
    ("M", "1200-1250", "Yes", 0),
    ("M", "850-900",    "Yes", 6),
    ("M", "400-600",    "No",  10),

    ("F", "1550-1600", "Yes", 1),
    ("F", "1200-1250", "No",  5),
    ("F", "850-900",    "No",  0),
    ("F", "400-600",    "No",  16),
]

GENDER_ORDER = ["M", "F"]
ADMISSION_ORDER = ["Yes", "No"]


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(CSV_PATH)


# ============================================================
# DEFINE SAT RANGES
# ============================================================

labels = [r[0] for r in SAT_RANGES]
lower_bounds = [r[1] for r in SAT_RANGES]
upper_bounds = [r[2] for r in SAT_RANGES]

# pd.cut requires increasing bin boundaries.
# Create bins from the lowest to highest SAT range.
sorted_ranges = sorted(SAT_RANGES, key=lambda x: x[1])

bins = [sorted_ranges[0][1] - 1] + [
    r[2] for r in sorted_ranges
]

sorted_labels = [r[0] for r in sorted_ranges]

df["SAT range"] = pd.cut(
    df[SAT_COL],
    bins=bins,
    labels=sorted_labels,
    include_lowest=True
)


# ============================================================
# GROUP AND COUNT
# ============================================================

result = (
    df.groupby(
        [GENDER_COL, "SAT range", ADMISSION_COL],
        observed=False
    )
    .size()
    .reset_index(name="#")
)


# ============================================================
# CREATE ALL COMBINATIONS
# This ensures zero-count groups are displayed.
# ============================================================

range_order = labels

all_combinations = pd.MultiIndex.from_product(
    [
        GENDER_ORDER,
        range_order,
        ADMISSION_ORDER
    ],
    names=[
        GENDER_COL,
        "SAT range",
        ADMISSION_COL
    ]
)

result = (
    result
    .set_index(
        [GENDER_COL, "SAT range", ADMISSION_COL]
    )
    .reindex(
        all_combinations,
        fill_value=0
    )
    .reset_index()
)


# ============================================================
# KEEP ONLY THE COMBINATIONS IN THE EXPECTED DISTRIBUTION
# ============================================================

desired_combinations = [
    (gender, sat_range, admission)
    for gender, sat_range, admission, count
    in EXPECTED_DISTRIBUTION
]

result = result[
    result[
        [GENDER_COL, "SAT range", ADMISSION_COL]
    ]
    .apply(tuple, axis=1)
    .isin(desired_combinations)
]


# ============================================================
# SORT IN THE DESIRED ORDER
# ============================================================

result["Gender"] = pd.Categorical(
    result["Gender"],
    categories=GENDER_ORDER,
    ordered=True
)

result["SAT range"] = pd.Categorical(
    result["SAT range"],
    categories=range_order,
    ordered=True
)

result["Admission"] = pd.Categorical(
    result["Admission"],
    categories=ADMISSION_ORDER,
    ordered=True
)

result = result.sort_values(
    ["Gender", "SAT range", "Admission"]
)


# ============================================================
# CONVERT GENDER NAMES
# ============================================================

result["Gender"] = result["Gender"].map({
    "M": "Male",
    "F": "Female"
})


# ============================================================
# PRINT MARKDOWN TABLE
# ============================================================

print("| **Group** | **SAT range** | **Admission** | **#** |")
print("| --------- | ------------- | ------------- | ----- |")

for _, row in result.iterrows():
    print(
        f"| {row['Gender']:<9} "
        f"| {row['SAT range']:<13} "
        f"| {row['Admission']:<9} "
        f"| {row['#']} |"
    )