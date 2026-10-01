import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_PATH = (
    # "fairness_through_awareness_3/"
    "lp_result.csv"
)

MODEL_PATH = (
    # "fairness_through_awareness_3/"
    "fta_model.pkl"
)

# Features used by the downstream model.
#
# IMPORTANT:
# Gender is intentionally NOT included.
#
FEATURE_COLUMNS = [
    "SAT",
    "Hobby"
]

TARGET_COLUMN = "FTA_Probability"


# ============================================================
# LOAD FTA LP RESULTS
# ============================================================

df = pd.read_csv(INPUT_PATH)

print("Loaded FTA LP results:")
print(f"Number of training examples: {len(df)}")

# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = (
    FEATURE_COLUMNS
    + [TARGET_COLUMN]
)

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
# PREPARE FEATURES AND TARGET
# ============================================================

X = df[FEATURE_COLUMNS].copy()
y = df[TARGET_COLUMN].copy()

print("\nFeatures:")
print(FEATURE_COLUMNS)

print(
    f"\nTarget: {TARGET_COLUMN}"
)

# ============================================================
# PREPROCESSING
# ============================================================

# SAT is already numeric.
#
# Hobby is categorical, so encode it using one-hot encoding.
#
# Gender is NOT included.

preprocessor = ColumnTransformer(
    transformers=[
        (
            "hobby",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            ["Hobby"]
        ),
        (
            "sat",
            "passthrough",
            ["SAT"]
        )
    ]
)


# ============================================================
# CLASSIFIER / REGRESSOR
# ============================================================

# We are learning continuous FTA probabilities.
#
# Therefore this is technically a regression model rather
# than a binary classifier.
#
# Random Forest can learn nonlinear relationships between
# SAT, Hobby, and the FTA probabilities.

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)


# ============================================================
# BUILD PIPELINE
# ============================================================

pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            model
        )
    ]
)


# ============================================================
# TRAIN MODEL
# ============================================================

print("\nTraining FTA probability model...")

pipeline.fit(
    X,
    y
)

print("Training completed.")


# ============================================================
# TRAINING-SET DIAGNOSTICS
# ============================================================

# These metrics only measure how well the model reproduces
# the LP outputs on the training dataset.
#
# They are NOT the final evaluation of fairness or
# generalization.

training_predictions = pipeline.predict(X)

mae = mean_absolute_error(
    y,
    training_predictions
)

rmse = mean_squared_error(
    y,
    training_predictions
) ** 0.5

r2 = r2_score(
    y,
    training_predictions
)

print("\nTraining-set diagnostics:")

print(
    f"MAE:  {mae:.6f}"
)

print(
    f"RMSE: {rmse:.6f}"
)

print(
    f"R²:   {r2:.6f}"
)


# ============================================================
# DISPLAY LEARNED PROBABILITIES
# ============================================================

results = df[
    [
        "ID",
        "Gender",
        "SAT",
        "Hobby",
        "Admission",
        TARGET_COLUMN
    ]
].copy()

results["Model_Prediction"] = (
    training_predictions
)

print("\nModel predictions:\n")

print(
    results.to_string(
        index=False
    )
)


# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(
    pipeline,
    MODEL_PATH
)

print(
    f"\nSaved trained FTA model to:"
)

print(
    MODEL_PATH
)