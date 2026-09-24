import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_CSV = "lfr-10prototypes/4-representations_10k.csv"
OUTPUT_CSV = "lfr-10prototypes/4-representations_readable.csv"

GENDER_COL = "Gender"
SAT_COL = "SAT"
SCORE_COL = "LFR_Score"
PREDICTION_COL = "Predicted_Admission"

# Middle regions from this thought experiment
MALE_MIDDLE_MIN = 800
MALE_MIDDLE_MAX = 900

FEMALE_MIDDLE_MIN = 1200
FEMALE_MIDDLE_MAX = 1300


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_CSV)


# ============================================================
# CREATE READABLE VERSION
# ============================================================

readable_df = df[
    [
        GENDER_COL,
        SAT_COL,
        SCORE_COL,
        PREDICTION_COL
    ]
].copy()

# Round LFR score to 3 decimal places
readable_df[SCORE_COL] = readable_df[SCORE_COL].round(3)

# Save cleaner output
readable_df.to_csv(
    OUTPUT_CSV,
    index=False
)

print("=" * 60)
print("READABLE LFR REPRESENTATIONS")
print("=" * 60)

print(readable_df.to_string(index=False))

print(f"\nSaved readable output to: {OUTPUT_CSV}")


# ============================================================
# MIDDLE-REGION ANALYSIS
# ============================================================

male_middle = df[
    (df[GENDER_COL] == "M")
    & (df[SAT_COL].between(
        MALE_MIDDLE_MIN,
        MALE_MIDDLE_MAX
    ))
]

female_middle = df[
    (df[GENDER_COL] == "F")
    & (df[SAT_COL].between(
        FEMALE_MIDDLE_MIN,
        FEMALE_MIDDLE_MAX
    ))
]


male_min = male_middle[SCORE_COL].min()
male_max = male_middle[SCORE_COL].max()

female_min = female_middle[SCORE_COL].min()
female_max = female_middle[SCORE_COL].max()


print("\n" + "=" * 60)
print("MIDDLE-REGION LFR SCORE SUMMARY")
print("=" * 60)

print(
    f"Male middle region "
    f"({MALE_MIDDLE_MIN}-{MALE_MIDDLE_MAX} SAT):"
)
print(
    f"  LFR scores run from "
    f"{male_min:.3f} to {male_max:.3f}"
)

print(
    f"\nFemale middle region "
    f"({FEMALE_MIDDLE_MIN}-{FEMALE_MIDDLE_MAX} SAT):"
)
print(
    f"  LFR scores run from "
    f"{female_min:.3f} to {female_max:.3f}"
)


# ============================================================
# OPTIONAL: PREDICTION COUNTS FOR MIDDLE REGIONS
# ============================================================

print("\nMiddle-region prediction counts:")

print("\nMale:")
print(
    male_middle[PREDICTION_COL]
    .value_counts()
    .to_string()
)

print("\nFemale:")
print(
    female_middle[PREDICTION_COL]
    .value_counts()
    .to_string()
)