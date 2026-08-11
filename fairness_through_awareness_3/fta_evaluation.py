import pandas as pd
import joblib
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

EVALUATION_DATASET_PATH = (
    # "fairness_through_awareness_3/"
    "evaluation_dataset.csv"
)

MODEL_PATH = (
    # "fairness_through_awareness_3/"
    "fta_model.pkl"
)

OUTPUT_PATH = (
    # "fairness_through_awareness_3/"
    "fta_evaluation_result.csv"
)

# ============================================================
# THRESHOLD CONFIGURATION
# ============================================================

# Applicants with FTA probability >= this value
# will be predicted as "Yes".

FTA_THRESHOLD = 0.50

# ============================================================
# MODEL FEATURES
# ============================================================

# Gender is intentionally NOT included.
#
# Gender is retained in the evaluation dataset only so that
# we can evaluate whether the model produces different
# outcomes for different gender groups.

FEATURE_COLUMNS = [
    "SAT",
    "Hobby"
]

# ============================================================
# LOAD EVALUATION DATASET
# ============================================================

df = pd.read_csv(
    EVALUATION_DATASET_PATH
)

print("Loaded evaluation dataset:")
print(
    f"Number of applicants: {len(df)}"
)

# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "ID",
    "Gender",
    "SAT",
    "Hobby"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    raise ValueError(
        "Missing required columns: "
        + ", ".join(missing_columns)
    )

# ============================================================
# LOAD TRAINED FTA MODEL
# ============================================================

model = joblib.load(
    MODEL_PATH
)

print(
    f"Loaded FTA model: {MODEL_PATH}"
)

# ============================================================
# GENERATE FTA PROBABILITIES
# ============================================================

X = df[
    FEATURE_COLUMNS
]

df["FTA_Probability"] = model.predict(X)

# Keep probabilities within [0, 1].

df["FTA_Probability"] = (
    df["FTA_Probability"]
    .clip(0.0, 1.0)
)

# ============================================================
# APPLY THRESHOLD
# ============================================================

df["FTA_Prediction"] = (
    df["FTA_Probability"]
    >= FTA_THRESHOLD
).map({
    True: "Yes",
    False: "No"
})

# ============================================================
# OVERALL STATISTICS
# ============================================================

total_applicants = len(df)

predicted_accepted = (
    df["FTA_Prediction"] == "Yes"
).sum()

acceptance_rate = (
    predicted_accepted
    / total_applicants
)

print("\nFTA evaluation:")
print(
    f"Threshold: {FTA_THRESHOLD:.2f}"
)

print(
    f"Predicted accepted: "
    f"{predicted_accepted}/{total_applicants} "
    f"({acceptance_rate:.1%})"
)

# ============================================================
# GENDER-LEVEL STATISTICS
#
# Gender is NOT used to generate the predictions.
# It is used only for fairness evaluation.
# ============================================================

print("\nFTA results by gender:\n")

for gender in sorted(
    df["Gender"].unique()
):

    group = df[
        df["Gender"] == gender
    ]

    group_size = len(group)

    group_accepted = (
        group["FTA_Prediction"] == "Yes"
    ).sum()

    group_acceptance_rate = (
        group_accepted
        / group_size
    )

    mean_probability = (
        group["FTA_Probability"]
        .mean()
    )

    print(
        f"{gender}: "
        f"Accepted = "
        f"{group_accepted}/{group_size} "
        f"({group_acceptance_rate:.1%}), "
        f"Mean probability = "
        f"{mean_probability:.4f}"
    )

# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\nFTA evaluation results:\n")

print(
    df[
        [
            "ID",
            "Gender",
            "SAT",
            "Hobby",
            "FTA_Probability",
            "FTA_Prediction"
        ]
    ].to_string(index=False)
)

# ============================================================
# SAVE RESULTS
# ============================================================

Path(OUTPUT_PATH).parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUTPUT_PATH,
    index=False
)

print(
    f"\nSaved evaluation results to:"
)

print(
    OUTPUT_PATH
)