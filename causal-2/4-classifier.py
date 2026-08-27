import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier


# ============================================================
# Configuration
# ============================================================

INPUT_FILE = "causal-2/3-repaired_data.csv"
MODEL_FILE = "causal-2/5-capuchin_model.pkl"

TARGET = "Admission"

# Protected attribute
PROTECTED_ATTRIBUTE = "Gender"

# Features used by the classifier
FEATURES = [
    "Qualification",
    "Department"
]


# ============================================================
# Load repaired training data
# ============================================================

df = pd.read_csv(INPUT_FILE)

print("=" * 60)
print("CAPUCHIN MODEL TRAINING")
print("=" * 60)

print(f"Training data: {INPUT_FILE}")
print(f"Number of rows: {len(df)}")
print()


# ============================================================
# Validate columns
# ============================================================

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


# ============================================================
# Show repaired-data distribution
# ============================================================

print("Repaired data distribution:")
print()

print(
    df.groupby(
        ["Department", "Gender", "Qualification", "Admission"]
    ).size()
)

print()


# ============================================================
# Prepare X and y
# ============================================================

X = df[FEATURES]

y = df[TARGET]


# Convert Yes/No to 1/0

y = y.map({
    "No": 0,
    "Yes": 1
})


if y.isna().any():
    raise ValueError(
        "Admission contains values other than Yes/No."
    )


# ============================================================
# Preprocessing
# ============================================================

categorical_features = [
    "Qualification",
    "Department"
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        )
    ]
)


# ============================================================
# Classifier
# ============================================================

classifier = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# ============================================================
# Build pipeline
# ============================================================

model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            classifier
        )
    ]
)


# ============================================================
# Train
# ============================================================

print("Training model...")

model.fit(
    X,
    y
)

print("Training complete.")
print()


# ============================================================
# Save model
# ============================================================

joblib.dump(
    model,
    MODEL_FILE
)

print(
    f"Model saved to: {MODEL_FILE}"
)

print()
print("=" * 60)
print("DONE")
print("=" * 60)