# ============================================================
# aware_lr_pipeline.py
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_SEED = 42

# ------------------------------------------------------------
# Files
# ------------------------------------------------------------

DATA_CSV_PATH = "fairness_through_unawareness\\1-aware_experiment\\2-original_data.csv"
PREDICTIONS_CSV_PATH = "fairness_through_unawareness\\1-aware_experiment\\3-aware_predictions.csv"
COEFFICIENTS_CSV_PATH = "fairness_through_unawareness\\1-aware_experiment\\4-aware_coefficients.csv"
PLOT_PNG_PATH = "fairness_through_unawareness\\1-aware_experiment\\5-aware_predictions.png"

# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------

N_SAMPLES = 1000

TEST_SIZE = 0.25

# ------------------------------------------------------------
# Classification
# ------------------------------------------------------------

THRESHOLD = 0.50

# Region considered "near threshold"
NEAR_THRESHOLD_LOW = 0.40
NEAR_THRESHOLD_HIGH = 0.60

# ------------------------------------------------------------
# Logistic regression
# ------------------------------------------------------------

MAX_ITER = 1000

# ------------------------------------------------------------
# Feature columns
# ------------------------------------------------------------

NUMERIC_FEATURES = [
    "SAT",
    "GPA"
]

CATEGORICAL_FEATURES = [
    "Gender",
    "Hobby",
    "Major"
]

FEATURE_COLUMNS = (
    NUMERIC_FEATURES
    + CATEGORICAL_FEATURES
)

TARGET_COLUMN = "Admission"

PROTECTED_COLUMN = "Gender"


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    os.path.dirname(DATA_CSV_PATH),
    exist_ok=True
)


# ============================================================
# GENERATE SYNTHETIC DATA
# ============================================================

rng = np.random.default_rng(RANDOM_SEED)

rows = []


for i in range(N_SAMPLES):

    # --------------------------------------------------------
    # Gender
    # --------------------------------------------------------

    gender = rng.choice(
        ["Male", "Female"],
        p=[0.5, 0.5]
    )

    # --------------------------------------------------------
    # SAT
    #
    # Keep SAT distributions intentionally similar across
    # genders so Gender / proxy variables can be studied.
    # --------------------------------------------------------

    sat = rng.normal(
        loc=1200,
        scale=180
    )

    sat = np.clip(
        sat,
        400,
        1600
    )

    # --------------------------------------------------------
    # GPA
    #
    # Also intentionally similar across gender.
    # --------------------------------------------------------

    gpa = rng.normal(
        loc=3.20,
        scale=0.45
    )

    gpa = np.clip(
        gpa,
        0.0,
        4.0
    )

    # --------------------------------------------------------
    # Hobby
    #
    # Proxy for Gender.
    #
    # Male students are more likely to have:
    #   Soccer / Gaming
    #
    # Female students are more likely to have:
    #   Dance / Art
    # --------------------------------------------------------

    if gender == "Male":

        hobby = rng.choice(
            [
                "Soccer",
                "Gaming",
                "Dance",
                "Art"
            ],
            p=[
                0.40,
                0.35,
                0.10,
                0.15
            ]
        )

    else:

        hobby = rng.choice(
            [
                "Soccer",
                "Gaming",
                "Dance",
                "Art"
            ],
            p=[
                0.10,
                0.15,
                0.40,
                0.35
            ]
        )

    # --------------------------------------------------------
    # Major
    #
    # Second proxy for Gender.
    # --------------------------------------------------------

    if gender == "Male":

        major = rng.choice(
            [
                "Engineering",
                "Computer Science",
                "Biology",
                "Humanities"
            ],
            p=[
                0.40,
                0.35,
                0.15,
                0.10
            ]
        )

    else:

        major = rng.choice(
            [
                "Engineering",
                "Computer Science",
                "Biology",
                "Humanities"
            ],
            p=[
                0.15,
                0.15,
                0.35,
                0.35
            ]
        )

    # --------------------------------------------------------
    # TRUE ADMISSION SCORE
    #
    # SAT and GPA are important determinants.
    #
    # Gender also directly influences the historical outcome,
    # creating a disparity that an aware LR model can learn.
    #
    # Hobby and Major also contribute so that they can later
    # behave as proxies when Gender is removed.
    # --------------------------------------------------------

    score = -7.0

    # SAT contribution
    score += 0.0035 * sat

    # GPA contribution
    score += 0.90 * gpa

    # Direct gender effect
    if gender == "Female":
        score -= 0.70

    # Hobby effects
    if hobby == "Soccer":
        score += 0.25

    elif hobby == "Gaming":
        score += 0.15

    elif hobby == "Dance":
        score -= 0.15

    elif hobby == "Art":
        score -= 0.20

    # Major effects
    if major == "Engineering":
        score += 0.35

    elif major == "Computer Science":
        score += 0.30

    elif major == "Biology":
        score -= 0.05

    elif major == "Humanities":
        score -= 0.20

    # --------------------------------------------------------
    # Convert synthetic score into admission probability
    # --------------------------------------------------------

    probability = 1 / (
        1 + np.exp(-score)
    )

    admission = rng.binomial(
        1,
        probability
    )

    rows.append({
        "ID": i + 1,
        "Gender": gender,
        "SAT": round(sat, 1),
        "GPA": round(gpa, 2),
        "Hobby": hobby,
        "Major": major,
        "Admission": admission
    })


df = pd.DataFrame(rows)

df.to_csv(
    DATA_CSV_PATH,
    index=False
)


# ============================================================
# DISPLAY ORIGINAL DATA DISTRIBUTION
# ============================================================

print("=" * 70)
print("ORIGINAL DATA")
print("=" * 70)

print(f"Rows: {len(df)}")

print()

print("Gender counts:")
print(
    df["Gender"]
    .value_counts()
)

print()

print("Actual admission rates:")
print(
    df.groupby("Gender")["Admission"]
    .mean()
)

actual_rates = (
    df.groupby("Gender")["Admission"]
    .mean()
)

actual_dbr = (
    actual_rates["Male"]
    - actual_rates["Female"]
)

print()

print(
    f"Difference in base rates "
    f"(Male - Female): {actual_dbr:.4f}"
)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X = df[FEATURE_COLUMNS]

y = df[TARGET_COLUMN]

X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y
    )
)


# ============================================================
# PREPROCESSING
# ============================================================

numeric_transformer = Pipeline(
    steps=[
        (
            "scaler",
            StandardScaler()
        )
    ]
)

categorical_transformer = Pipeline(
    steps=[
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                drop="first"
            )
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_transformer,
            NUMERIC_FEATURES
        ),
        (
            "categorical",
            categorical_transformer,
            CATEGORICAL_FEATURES
        )
    ]
)


# ============================================================
# AWARE LOGISTIC REGRESSION
#
# IMPORTANT:
# Gender is included here.
# ============================================================

model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=MAX_ITER,
                random_state=RANDOM_SEED
            )
        )
    ]
)


# ============================================================
# TRAIN MODEL
# ============================================================

model.fit(
    X_train,
    y_train
)


# ============================================================
# PREDICT
# ============================================================

probabilities = model.predict_proba(
    X_test
)[:, 1]

predictions = (
    probabilities >= THRESHOLD
).astype(int)


# ============================================================
# CREATE TEST RESULTS DATAFRAME
# ============================================================

results = X_test.copy()

results["Actual_Admission"] = (
    y_test.values
)

results["Predicted_Probability"] = (
    probabilities
)

results["Predicted_Admission"] = (
    predictions
)

results["Near_Threshold"] = (
    results["Predicted_Probability"]
    .between(
        NEAR_THRESHOLD_LOW,
        NEAR_THRESHOLD_HIGH
    )
)

results.to_csv(
    PREDICTIONS_CSV_PATH,
    index=False
)


# ============================================================
# ACCURACY
# ============================================================

accuracy = accuracy_score(
    y_test,
    predictions
)


# ============================================================
# POSITIVE PREDICTION RATES
# ============================================================

male_results = results[
    results["Gender"] == "Male"
]

female_results = results[
    results["Gender"] == "Female"
]

male_positive_rate = (
    male_results["Predicted_Admission"]
    .mean()
)

female_positive_rate = (
    female_results["Predicted_Admission"]
    .mean()
)


# ============================================================
# DISPARATE IMPACT DIFFERENCE
#
# Here:
#
# DI =
# P(predicted positive | Male)
# -
# P(predicted positive | Female)
#
# Positive DI means males receive more positive predictions.
# ============================================================

disparate_impact = (
    male_positive_rate
    - female_positive_rate
)


# ============================================================
# NEAR-THRESHOLD ANALYSIS
# ============================================================

male_near = male_results[
    male_results["Near_Threshold"]
]

female_near = female_results[
    female_results["Near_Threshold"]
]


# ============================================================
# OUTPUT METRICS
# ============================================================

print()

print("=" * 70)
print("AWARE LOGISTIC REGRESSION")
print("=" * 70)

print(
    f"Features: "
    f"{', '.join(FEATURE_COLUMNS)}"
)

print()

print(
    f"Accuracy: "
    f"{accuracy:.4f}"
)

print()

print(
    f"Male positive prediction rate: "
    f"{male_positive_rate:.4f}"
)

print(
    f"Female positive prediction rate: "
    f"{female_positive_rate:.4f}"
)

print()

print(
    f"Disparate impact "
    f"(Male - Female): "
    f"{disparate_impact:.4f}"
)

print()

print(
    f"Male near-threshold points "
    f"[{NEAR_THRESHOLD_LOW}, "
    f"{NEAR_THRESHOLD_HIGH}]: "
    f"{len(male_near)}"
)

print(
    f"Female near-threshold points "
    f"[{NEAR_THRESHOLD_LOW}, "
    f"{NEAR_THRESHOLD_HIGH}]: "
    f"{len(female_near)}"
)


# ============================================================
# EXTRACT LOGISTIC REGRESSION COEFFICIENTS
# ============================================================

fitted_preprocessor = (
    model.named_steps[
        "preprocessor"
    ]
)

feature_names = (
    fitted_preprocessor
    .get_feature_names_out()
)

coefficients = (
    model.named_steps[
        "classifier"
    ]
    .coef_[0]
)

coefficient_df = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": coefficients
})


# ============================================================
# ADD FEMALE PENALTY RELATIVE TO MALE
# ============================================================

gender_male_row = coefficient_df[
    coefficient_df["Feature"]
    .str.contains("Gender_Male")
]

if not gender_male_row.empty:

    male_coefficient = (
        gender_male_row["Coefficient"]
        .iloc[0]
    )

    female_penalty_relative_to_male = (
        -male_coefficient
    )

    female_penalty_row = pd.DataFrame({
        "Feature": [
            "interpreted__Gender_Female_relative_to_Male"
        ],
        "Coefficient": [
            female_penalty_relative_to_male
        ]
    })

    coefficient_df = pd.concat(
        [
            coefficient_df,
            female_penalty_row
        ],
        ignore_index=True
    )


# ============================================================
# SORT AND SAVE
# ============================================================

coefficient_df = (
    coefficient_df
    .sort_values(
        by="Coefficient",
        ascending=False
    )
)

coefficient_df.to_csv(
    COEFFICIENTS_CSV_PATH,
    index=False
)


print()

print("=" * 70)
print("MODEL COEFFICIENTS")
print("=" * 70)

print(
    coefficient_df
    .to_string(
        index=False
    )
)


# ============================================================
# PLOT PREDICTED PROBABILITIES BY GENDER
# ============================================================

plot_df = results.copy()

rng_plot = np.random.default_rng(
    RANDOM_SEED
)

x_map = {
    "Male": 0,
    "Female": 1
}

plot_df["x"] = (
    plot_df["Gender"]
    .map(x_map)
)

plot_df["x_jitter"] = (
    plot_df["x"]
    + rng_plot.uniform(
        -0.08,
        0.08,
        len(plot_df)
    )
)

accepted = plot_df[
    plot_df["Predicted_Admission"] == 1
]

unaccepted = plot_df[
    plot_df["Predicted_Admission"] == 0
]


plt.figure(
    figsize=(7, 7)
)


# ------------------------------------------------------------
# Rejected
# ------------------------------------------------------------

plt.scatter(
    unaccepted["x_jitter"],
    unaccepted["Predicted_Probability"],
    color="red",
    s=45,
    alpha=0.65,
    label="Unaccepted"
)


# ------------------------------------------------------------
# Accepted
# ------------------------------------------------------------

plt.scatter(
    accepted["x_jitter"],
    accepted["Predicted_Probability"],
    color="blue",
    s=45,
    alpha=0.65,
    label="Accepted"
)


# ------------------------------------------------------------
# Classification threshold
# ------------------------------------------------------------

plt.axhline(
    y=THRESHOLD,
    color="black",
    linestyle="--",
    linewidth=1.5,
    label=f"Threshold = {THRESHOLD}"
)


# ------------------------------------------------------------
# Near-threshold region
# ------------------------------------------------------------

plt.axhspan(
    NEAR_THRESHOLD_LOW,
    NEAR_THRESHOLD_HIGH,
    alpha=0.08
)


# ------------------------------------------------------------
# Axes
# ------------------------------------------------------------

plt.xticks(
    [0, 1],
    ["Male", "Female"]
)

plt.yticks(
    np.arange(
        0,
        1.1,
        0.1
    )
)

plt.ylim(
    0,
    1
)

plt.xlim(
    -0.3,
    1.3
)

plt.xlabel(
    "Gender"
)

plt.ylabel(
    "Predicted Admission Probability"
)

plt.title(
    "Aware Logistic Regression\n"
    "Gender Included"
)

plt.grid(
    axis="y",
    linestyle=":",
    alpha=0.3
)

plt.legend()

plt.tight_layout()

plt.savefig(
    PLOT_PNG_PATH,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# FINAL OUTPUT
# ============================================================

print()

print("=" * 70)
print("FILES CREATED")
print("=" * 70)

print(
    f"Original data:       "
    f"{DATA_CSV_PATH}"
)

print(
    f"Predictions:         "
    f"{PREDICTIONS_CSV_PATH}"
)

print(
    f"Coefficients:        "
    f"{COEFFICIENTS_CSV_PATH}"
)

print(
    f"Probability plot:    "
    f"{PLOT_PNG_PATH}"
)