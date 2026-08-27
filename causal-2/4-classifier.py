import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression


# ============================================================
# Configuration
# ============================================================

INPUT_FILE = "causal-2/3-repaired_data.csv"
MODEL_FILE = "causal-2/5-capuchin_logistic_model.pkl"

TARGET = "Admission"

# Protected attribute
PROTECTED_ATTRIBUTE = "Gender"

# Features used by the classifier
#
# IMPORTANT:
# Gender IS intentionally included.
#
# This allows us to test whether Capuchin's repaired
# training data reduces the classifier's dependence on Gender.
FEATURES = [
    "Qualification",
    "Department",
    "Gender"
]


# ============================================================
# Load repaired training data
# ============================================================

df = pd.read_csv(INPUT_FILE)

print("=" * 60)
print("CAPUCHIN LOGISTIC REGRESSION MODEL TRAINING")
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
    "Department",
    "Gender"
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

classifier = LogisticRegression(
    max_iter=1000,
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

print("Training logistic regression model...")

model.fit(
    X,
    y
)

print("Training complete.")
print()


# ============================================================
# Show model coefficients
# ============================================================

print("Model coefficients:")
print()

feature_names = (
    model
    .named_steps["preprocessor"]
    .get_feature_names_out()
)

coefficients = (
    model
    .named_steps["classifier"]
    .coef_[0]
)

for feature, coefficient in zip(
    feature_names,
    coefficients
):
    print(
        f"{feature:40s} {coefficient:+.6f}"
    )

print()

print(
    f"Intercept: "
    f"{model.named_steps['classifier'].intercept_[0]:+.6f}"
)

print()


# ============================================================
# Gender coefficient
# ============================================================

gender_features = [
    (feature, coefficient)
    for feature, coefficient in zip(
        feature_names,
        coefficients
    )
    if "Gender" in feature
]

print("Gender-related coefficients:")
print()

for feature, coefficient in gender_features:
    print(
        f"{feature:40s} {coefficient:+.6f}"
    )

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