# ============================================================
# generate_ftu_data.py
# ============================================================

import pandas as pd
import numpy as np

# ============================================================
# CONFIGURATION
# ============================================================

BEFORE_CSV_PATH = "fairness_through_unawareness\\csv\\before_ftu.csv"
AFTER_CSV_PATH = "fairness_through_unawareness\\csv\\after_ftu.csv"

THRESHOLD = 0.5

# Male distribution
MALE_BELOW_START = 0.40
MALE_BELOW_END = 0.49
MALE_BELOW_COUNT = 8

MALE_ABOVE_START = 0.52
MALE_ABOVE_END = 0.80
MALE_ABOVE_COUNT = 12

# Female distribution before FtU
FEMALE_LOW_START = 0.20
FEMALE_LOW_END = 0.44
FEMALE_LOW_COUNT = 11

FEMALE_BORDERLINE_BEFORE = [
    0.45,
    0.46,
    0.47,
    0.48,
    0.49,
    0.495
]

FEMALE_HIGH = [
    0.52,
    0.56,
    0.60
]

# Female borderline points after FtU
FEMALE_BORDERLINE_AFTER = [
    0.51,
    0.52,
    0.53,
    0.54,
    0.55,
    0.56
]


# ============================================================
# GENERATE MALE DATA
# ============================================================

male_below = np.linspace(
    MALE_BELOW_START,
    MALE_BELOW_END,
    MALE_BELOW_COUNT
)

male_above = np.linspace(
    MALE_ABOVE_START,
    MALE_ABOVE_END,
    MALE_ABOVE_COUNT
)

male_scores = np.concatenate([
    male_below,
    male_above
])


# ============================================================
# GENERATE FEMALE DATA - BEFORE FTU
# ============================================================

female_low = np.linspace(
    FEMALE_LOW_START,
    FEMALE_LOW_END,
    FEMALE_LOW_COUNT
)

female_scores_before = np.concatenate([
    female_low,
    FEMALE_BORDERLINE_BEFORE,
    FEMALE_HIGH
])


# ============================================================
# CREATE BEFORE-FTU DATASET
# ============================================================

before = pd.DataFrame({
    "Gender": (
        ["Male"] * len(male_scores)
        + ["Female"] * len(female_scores_before)
    ),
    "Probability": np.concatenate([
        male_scores,
        female_scores_before
    ])
})

before["Decision"] = np.where(
    before["Probability"] >= THRESHOLD,
    "Accepted",
    "Unaccepted"
)

before.to_csv(
    BEFORE_CSV_PATH,
    index=False
)


# ============================================================
# GENERATE FEMALE DATA - AFTER FTU
# ============================================================

female_scores_after = np.concatenate([
    female_low,
    FEMALE_BORDERLINE_AFTER,
    FEMALE_HIGH
])


# ============================================================
# CREATE AFTER-FTU DATASET
# ============================================================

after = pd.DataFrame({
    "Gender": (
        ["Male"] * len(male_scores)
        + ["Female"] * len(female_scores_after)
    ),
    "Probability": np.concatenate([
        male_scores,
        female_scores_after
    ])
})

after["Decision"] = np.where(
    after["Probability"] >= THRESHOLD,
    "Accepted",
    "Unaccepted"
)

after.to_csv(
    AFTER_CSV_PATH,
    index=False
)


# ============================================================
# OUTPUT
# ============================================================

print("Created:")
print(f"  {BEFORE_CSV_PATH}")
print(f"  {AFTER_CSV_PATH}")