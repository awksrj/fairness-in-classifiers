import pandas as pd
import pickle


# ============================================================
# CONFIGURATION
# ============================================================

EVALUATION_DATA = "causal/6-evaluation_data.csv"
MODEL_PATH = "causal/5-capuchin_model.pkl"
OUTPUT_PATH = "causal/8-evaluation_result.csv"


# ============================================================
# LOAD EVALUATION DATA
# ============================================================

df = pd.read_csv(EVALUATION_DATA)


# ============================================================
# LOAD TRAINED CAPUCHIN MODEL
# ============================================================

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)


# ============================================================
# PREPARE FEATURES
# ============================================================

FEATURES = [
    "Gender",
    "SAT",
    "Hobby"
]

X = df[FEATURES]


# ============================================================
# MAKE PREDICTIONS
# ============================================================

predictions = model.predict(X)


# Convert:
#     1 -> Yes
#     0 -> No

df["Admission"] = [
    "Yes" if prediction == 1 else "No"
    for prediction in predictions
]


# ============================================================
# SAVE RESULTS
# ============================================================

df.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("=" * 60)
print("CAPUCHIN EVALUATION")
print("=" * 60)

print(f"Evaluation data: {EVALUATION_DATA}")
print(f"Model:           {MODEL_PATH}")
print(f"Output:          {OUTPUT_PATH}")

print()
print("Predicted Admission:")
print(
    df["Admission"].value_counts()
)

print()
print("Evaluation results:")
print(df.to_string(index=False))