import pandas as pd
import pickle

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_CSV = "causal/3-repaired_data_discrete.csv"
MODEL_PATH = "causal/5-capuchin_model.pkl"

FEATURES = [
    "Gender",
    "SAT",
    "Hobby"
]

TARGET = "Admission"


# ============================================================
# LOAD REPAIRED DATA
# ============================================================

df = pd.read_csv(INPUT_CSV)

print("=" * 60)
print("REPAIRED TRAINING DATA")
print("=" * 60)

print(df)


# ============================================================
# FEATURES AND TARGET
# ============================================================

X = df[FEATURES]

y = df[TARGET].map({
    "Yes": 1,
    "No": 0
})


# ============================================================
# PREPROCESSING
# ============================================================

# Gender and Hobby are categorical.
# SAT is numerical.

categorical_features = [
    "Gender",
    "Hobby"
]

numerical_features = [
    "SAT"
]


preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        ),

        (
            "numerical",
            StandardScaler(),
            numerical_features
        )
    ]
)


# ============================================================
# CLASSIFIER
# ============================================================

classifier = LogisticRegression(
    random_state=42,
    max_iter=1000
)


# ============================================================
# COMPLETE PIPELINE
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
# TRAIN
# ============================================================

model.fit(
    X,
    y
)


# ============================================================
# SAVE MODEL
# ============================================================

with open(
    MODEL_PATH,
    "wb"
) as f:

    pickle.dump(
        model,
        f
    )


# ============================================================
# TRAINING INFORMATION
# ============================================================

print()
print("=" * 60)
print("CAPUCHIN CLASSIFIER")
print("=" * 60)

print(
    f"Features: {FEATURES}"
)

print(
    f"Target: {TARGET}"
)

print(
    f"Number of training examples: {len(df)}"
)

print(
    f"Positive examples (Yes): {(y == 1).sum()}"
)

print(
    f"Negative examples (No): {(y == 0).sum()}"
)

print()
print(
    f"Model saved to: {MODEL_PATH}"
)