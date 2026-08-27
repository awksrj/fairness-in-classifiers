import pandas as pd
import joblib

# ============================================================
# Configuration
# ============================================================

EVALUATION_FILE = "causal-2/6-evaluation_data.csv"
MODEL_FILE = "causal-2/5-capuchin_logistic_model.pkl"
OUTPUT_FILE = "causal-2/8-evaluation_result_logistic.csv"


# ============================================================
# Load evaluation data
# ============================================================

df = pd.read_csv(EVALUATION_FILE)

print("=" * 60)
print("CAPUCHIN LOGISTIC REGRESSION MODEL EVALUATION")
print("=" * 60)

print(f"Evaluation data: {EVALUATION_FILE}")
print(f"Number of tuples: {len(df)}")
print()


# ============================================================
# Load trained model
# ============================================================

model = joblib.load(MODEL_FILE)

print(f"Loaded model: {MODEL_FILE}")
print()


# ============================================================
# Validate columns
# ============================================================

required_columns = [
    "ID",
    "Gender",
    "Qualification",
    "Department"
]

missing = [
    col for col in required_columns
    if col not in df.columns
]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )


# ============================================================
# Prepare features
# ============================================================

# IMPORTANT:
# Gender IS intentionally used as a model feature.
#
# The classifier was trained using:
#   Qualification
#   Department
#   Gender
#
# This allows us to test whether Capuchin's repaired
# training data reduces the classifier's dependence
# on the protected attribute.

FEATURES = [
    "Qualification",
    "Department",
    "Gender"
]

X = df[FEATURES]


# ============================================================
# Predict Admission
# ============================================================

predictions = model.predict(X)

# Convert model output:
# 0 -> No
# 1 -> Yes

predictions = pd.Series(predictions).map({
    0: "No",
    1: "Yes"
})


# ============================================================
# Predict Admission Probability
# ============================================================

# Probability of the positive class (Yes)

probabilities = model.predict_proba(X)[:, 1]

df["Admission_Probability"] = probabilities


# ============================================================
# Add predictions to evaluation dataset
# ============================================================

df["Admission"] = predictions


# ============================================================
# Save evaluation results
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print(f"Results saved to: {OUTPUT_FILE}")
print()


# ============================================================
# Summary
# ============================================================

print("Predicted Admission distribution:")
print(
    df["Admission"].value_counts()
)

print()


print("Predicted Admission by Gender:")
print(
    pd.crosstab(
        df["Gender"],
        df["Admission"]
    )
)

print()


print("Predicted Admission rate by Gender:")
print(
    df.groupby("Gender")["Admission"]
      .apply(lambda x: (x == "Yes").mean())
)

print()


# ============================================================
# Average predicted probability by Gender
# ============================================================

print("Average predicted admission probability by Gender:")
print(
    df.groupby("Gender")["Admission_Probability"]
      .mean()
)

print()


# ============================================================
# Predictions by Department and Gender
# ============================================================

print("Predicted Admission rate by Department and Gender:")
print(
    df.groupby(
        ["Department", "Gender"]
    )["Admission"]
    .apply(lambda x: (x == "Yes").mean())
)

print()


print("=" * 60)
print("DONE")
print("=" * 60)