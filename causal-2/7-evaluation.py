import pandas as pd
import joblib

# ============================================================
# Configuration
# ============================================================

EVALUATION_FILE = "causal-2/6-evaluation_data.csv"
MODEL_FILE = "causal-2/5-capuchin_model.pkl"
OUTPUT_FILE = "causal-2/8-evaluation_result.csv"


# ============================================================
# Load evaluation data
# ============================================================

df = pd.read_csv(EVALUATION_FILE)

print("=" * 60)
print("CAPUCHIN MODEL EVALUATION")
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
# Prepare features
# ============================================================

# IMPORTANT:
# Gender is intentionally NOT used as a model feature.
#
# The classifier was trained using:
#   Qualification
#   Department
#
# Gender remains in the evaluation dataframe so that we can
# analyze the model's predictions across genders afterward.

FEATURES = [
    "Qualification",
    "Department"
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
print(df["Admission"].value_counts())

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

print("=" * 60)
print("DONE")
print("=" * 60)