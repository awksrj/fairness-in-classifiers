import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_SEED = 42

# 400 total applicants:
#
# Male:
#   100 rejected
#   100 accepted
#
# Female:
#   100 rejected
#   100 accepted
#
N = 400

DECISION_THRESHOLD = 0.50

# Smaller C = stronger regularization.
# This keeps probabilities from becoming almost exactly 0 or 1.
MODEL_C = 0.10

OUTPUT_DIR = "fairness_through_unawareness\\di-limitation"
os.makedirs(OUTPUT_DIR, exist_ok=True)


DATA_CSV = os.path.join(
    OUTPUT_DIR,
    "2-original_data.csv"
)

AWARE_PREDICTIONS_CSV = os.path.join(
    OUTPUT_DIR,
    "3-aware_predictions.csv"
)

UNAWARE_PREDICTIONS_CSV = os.path.join(
    OUTPUT_DIR,
    "4-unaware_predictions.csv"
)


AWARE_PROBABILITY_PLOT = os.path.join(
    OUTPUT_DIR,
    "5-aware_probability_vs_gender.png"
)

AWARE_DECOMPOSITION_PLOT = os.path.join(
    OUTPUT_DIR,
    "6-aware_sat_gender.png"
)

UNAWARE_PROBABILITY_PLOT = os.path.join(
    OUTPUT_DIR,
    "7-unaware_probability_vs_gender.png"
)

UNAWARE_DECOMPOSITION_PLOT = os.path.join(
    OUTPUT_DIR,
    "8-unaware_sat_gender.png"
)


rng = np.random.default_rng(
    RANDOM_SEED
)


# ============================================================
# SYNTHETIC DATA DESIGN
#
# Goal:
#
# AWARE CASE
#
# Male:
#     Accept from SAT ≈ 1300
#
# Female:
#     Accept from SAT ≈ 1400
#
# At the same time:
#
#     50% of Male applicants accepted
#     50% of Female applicants accepted
#
# Therefore:
#
#     DI = 0
#
# even though the model treats Male and Female applicants
# differently at the same SAT.
#
#
# UNAWARE CASE
#
# Remove Gender and train on SAT only.
#
# Now:
#
#     same SAT -> same prediction
#
# but because Female applicants have a SAT distribution
# shifted upward by approximately 100 points, the group
# acceptance rates become different.
# ============================================================


MALE_THRESHOLD = 1300
FEMALE_THRESHOLD = 1400

N_PER_GENDER = N // 2
N_PER_OUTCOME = N_PER_GENDER // 2


# ============================================================
# GENERATE MALE DATA
#
# Rejected:
#     SAT 1180 - 1295
#
# Accepted:
#     SAT 1305 - 1420
#
# Small gap around 1300 helps LR learn a clean boundary.
# ============================================================

male_rejected_sat = rng.integers(
    low=1180,
    high=1296,
    size=N_PER_OUTCOME
)


male_accepted_sat = rng.integers(
    low=1305,
    high=1421,
    size=N_PER_OUTCOME
)


male_sat = np.concatenate([
    male_rejected_sat,
    male_accepted_sat
])


male_gender = np.array(
    ["Male"] * N_PER_GENDER
)


male_admission = np.array(
    ["No"] * N_PER_OUTCOME
    +
    ["Yes"] * N_PER_OUTCOME
)


# ============================================================
# GENERATE FEMALE DATA
#
# Same shape as Male distribution,
# shifted upward by approximately 100 SAT points.
#
# Rejected:
#     SAT 1280 - 1395
#
# Accepted:
#     SAT 1405 - 1520
#
# Therefore there is a substantial overlap:
#
# For example:
#
# Male SAT 1350   -> Accepted
# Female SAT 1350 -> Rejected
# ============================================================

female_rejected_sat = rng.integers(
    low=1280,
    high=1396,
    size=N_PER_OUTCOME
)


female_accepted_sat = rng.integers(
    low=1405,
    high=1521,
    size=N_PER_OUTCOME
)


female_sat = np.concatenate([
    female_rejected_sat,
    female_accepted_sat
])


female_gender = np.array(
    ["Female"] * N_PER_GENDER
)


female_admission = np.array(
    ["No"] * N_PER_OUTCOME
    +
    ["Yes"] * N_PER_OUTCOME
)


# ============================================================
# COMBINE DATA
# ============================================================

gender = np.concatenate([
    male_gender,
    female_gender
])


sat = np.concatenate([
    male_sat,
    female_sat
])


admission = np.concatenate([
    male_admission,
    female_admission
])


df = pd.DataFrame({

    "ID": np.arange(
        1,
        N + 1
    ),

    "Gender": gender,

    "SAT": sat,

    "Admission": admission

})


# Shuffle rows
df = (
    df
    .sample(
        frac=1,
        random_state=RANDOM_SEED
    )
    .reset_index(drop=True)
)


df["ID"] = np.arange(
    1,
    len(df) + 1
)


df.to_csv(
    DATA_CSV,
    index=False
)


# ============================================================
# HELPER:
# ACCEPTANCE TABLE
# ============================================================

def print_acceptance_table(
    results,
    model_name,
    outcome_column
):

    summary = (
        results
        .groupby("Gender")[outcome_column]
        .apply(
            lambda x: pd.Series({

                "Total":
                len(x),

                "Accepted":
                (x == "Yes").sum(),

                "Rate":
                (x == "Yes").mean()

            })
        )
        .unstack()
    )


    summary = summary.reindex([
        "Male",
        "Female"
    ])


    summary["Total"] = (
        summary["Total"]
        .astype(int)
    )


    summary["Accepted"] = (
        summary["Accepted"]
        .astype(int)
    )


    print("\n")
    print("=" * 70)
    print(model_name)
    print("=" * 70)


    print(
        summary
        .rename_axis(None)
        .to_string()
    )


    return summary


# ============================================================
# ORIGINAL DATA SUMMARY
#
# This should be EXACTLY:
#
#         Total  Accepted  Rate
# Male      200       100   0.5
# Female    200       100   0.5
# ============================================================

print("\n")
print("=" * 70)
print("ORIGINAL DATA")
print("=" * 70)

print(
    df.head()
)


original_summary = print_acceptance_table(
    results=df,
    model_name="ORIGINAL HISTORICAL ACCEPTANCE",
    outcome_column="Admission"
)


print("\nSAT summary by Gender:")

print(
    df
    .groupby("Gender")["SAT"]
    .agg([
        "min",
        "mean",
        "median",
        "max",
        "std"
    ])
    .round(2)
)


# ============================================================
# TRAIN / TEST DESIGN
#
# Instead of a random test split, we construct a balanced
# test set explicitly.
#
# Test set contains:
#
# Male:
#     25 rejected
#     25 accepted
#
# Female:
#     25 rejected
#     25 accepted
#
# Training set contains:
#
# Male:
#     75 rejected
#     75 accepted
#
# Female:
#     75 rejected
#     75 accepted
#
# This preserves exact 50% acceptance in each group.
# ============================================================

group_indices = {}


for gender_value in [
    "Male",
    "Female"
]:

    for admission_value in [
        "No",
        "Yes"
    ]:

        indices = df.index[
            (
                df["Gender"]
                == gender_value
            )
            &
            (
                df["Admission"]
                == admission_value
            )
        ].to_numpy()


        rng.shuffle(
            indices
        )


        group_indices[
            (
                gender_value,
                admission_value
            )
        ] = indices


# ------------------------------------------------------------
# Select 25 from each subgroup for test
# ------------------------------------------------------------

TEST_PER_SUBGROUP = 25


test_idx = np.concatenate([

    group_indices[
        ("Male", "No")
    ][
        :TEST_PER_SUBGROUP
    ],

    group_indices[
        ("Male", "Yes")
    ][
        :TEST_PER_SUBGROUP
    ],

    group_indices[
        ("Female", "No")
    ][
        :TEST_PER_SUBGROUP
    ],

    group_indices[
        ("Female", "Yes")
    ][
        :TEST_PER_SUBGROUP
    ]

])


train_idx = np.concatenate([

    group_indices[
        ("Male", "No")
    ][
        TEST_PER_SUBGROUP:
    ],

    group_indices[
        ("Male", "Yes")
    ][
        TEST_PER_SUBGROUP:
    ],

    group_indices[
        ("Female", "No")
    ][
        TEST_PER_SUBGROUP:
    ],

    group_indices[
        ("Female", "Yes")
    ][
        TEST_PER_SUBGROUP:
    ]

])


train_df = (
    df
    .loc[
        train_idx
    ]
    .copy()
)


test_df = (
    df
    .loc[
        test_idx
    ]
    .copy()
)


# Shuffle within train/test
train_df = (
    train_df
    .sample(
        frac=1,
        random_state=RANDOM_SEED
    )
)


test_df = (
    test_df
    .sample(
        frac=1,
        random_state=RANDOM_SEED
    )
)


y_train = (
    train_df["Admission"]
    == "Yes"
).astype(int)


y_test = (
    test_df["Admission"]
    == "Yes"
).astype(int)


# ============================================================
# VERIFY TEST SET BALANCE
# ============================================================

print_acceptance_table(
    results=test_df,
    model_name="TEST SET HISTORICAL ACCEPTANCE",
    outcome_column="Admission"
)


# ============================================================
# SAT SCALING
#
# We use SAT / 100 internally.
#
# This improves numerical behavior and makes regularization
# easier to control.
#
# Predictions are still interpreted in original SAT units.
# ============================================================

train_sat_scaled = (
    train_df["SAT"]
    /
    100.0
)


test_sat_scaled = (
    test_df["SAT"]
    /
    100.0
)


# ============================================================
# AWARE MODEL
#
# Features:
#
#     SAT + Gender
#
# Gender:
#
#     Male   = 0
#     Female = 1
#
# Formula:
#
# z =
#     intercept
#     + beta_sat * (SAT / 100)
#     + beta_female * Female
# ============================================================

X_train_aware = pd.DataFrame({

    "SAT_scaled":
    train_sat_scaled,

    "Female":
    (
        train_df["Gender"]
        == "Female"
    ).astype(int)

})


X_test_aware = pd.DataFrame({

    "SAT_scaled":
    test_sat_scaled,

    "Female":
    (
        test_df["Gender"]
        == "Female"
    ).astype(int)

})


aware_model = LogisticRegression(
    max_iter=5000,
    C=MODEL_C
)


aware_model.fit(
    X_train_aware,
    y_train
)


aware_probability = (
    aware_model
    .predict_proba(
        X_test_aware
    )[:, 1]
)


aware_prediction = (
    aware_probability
    >= DECISION_THRESHOLD
).astype(int)


aware_results = test_df[
    [
        "ID",
        "Gender",
        "SAT",
        "Admission"
    ]
].copy()


aware_results[
    "Predicted_Probability"
] = aware_probability


aware_results[
    "Predicted_Admission"
] = np.where(
    aware_prediction == 1,
    "Yes",
    "No"
)


aware_results.to_csv(
    AWARE_PREDICTIONS_CSV,
    index=False
)


# ============================================================
# AWARE COEFFICIENTS
# ============================================================

aware_intercept = (
    aware_model.intercept_[0]
)


aware_sat_coef = (
    aware_model.coef_[0][0]
)


aware_female_coef = (
    aware_model.coef_[0][1]
)


print("\n")
print("=" * 70)
print("AWARE MODEL")
print("=" * 70)


print(
    f"Intercept:          "
    f"{aware_intercept:.6f}"
)


print(
    f"SAT/100 coefficient: "
    f"{aware_sat_coef:.6f}"
)


print(
    f"Female coefficient: "
    f"{aware_female_coef:.6f}"
)


# ============================================================
# AWARE DECISION BOUNDARIES
#
# Male:
#
# 0 =
#     intercept
#     + beta_sat * SAT/100
#
#
# Female:
#
# 0 =
#     intercept
#     + beta_sat * SAT/100
#     + beta_female
#
# ============================================================

aware_male_boundary = (
    (
        -aware_intercept
        /
        aware_sat_coef
    )
    *
    100
)


aware_female_boundary = (
    (
        -(
            aware_intercept
            +
            aware_female_coef
        )
        /
        aware_sat_coef
    )
    *
    100
)


print(
    f"\nMale SAT boundary:   "
    f"{aware_male_boundary:.1f}"
)


print(
    f"Female SAT boundary: "
    f"{aware_female_boundary:.1f}"
)


print(
    f"Boundary gap:        "
    f"{aware_female_boundary - aware_male_boundary:.1f}"
)


# ============================================================
# AWARE FAIRNESS
# ============================================================

aware_male_rate = (
    aware_results.loc[
        aware_results["Gender"]
        == "Male",
        "Predicted_Admission"
    ]
    == "Yes"
).mean()


aware_female_rate = (
    aware_results.loc[
        aware_results["Gender"]
        == "Female",
        "Predicted_Admission"
    ]
    == "Yes"
).mean()


aware_di = (
    aware_male_rate
    -
    aware_female_rate
)


aware_accuracy = accuracy_score(
    y_test,
    aware_prediction
)


print(
    f"\nAccuracy: "
    f"{aware_accuracy:.3f}"
)


print(
    f"Male acceptance rate:   "
    f"{aware_male_rate:.3f}"
)


print(
    f"Female acceptance rate: "
    f"{aware_female_rate:.3f}"
)


print(
    f"DI gap (Male - Female): "
    f"{aware_di:.3f}"
)


aware_summary = print_acceptance_table(
    results=aware_results,
    model_name="AWARE MODEL ACCEPTANCE SUMMARY",
    outcome_column="Predicted_Admission"
)


# ============================================================
# UNAWARE MODEL
#
# Feature:
#
#     SAT only
#
# Gender is removed.
#
# Therefore:
#
#     same SAT -> same predicted probability
#
# regardless of Gender.
# ============================================================

X_train_unaware = pd.DataFrame({

    "SAT_scaled":
    train_sat_scaled

})


X_test_unaware = pd.DataFrame({

    "SAT_scaled":
    test_sat_scaled

})


unaware_model = LogisticRegression(
    max_iter=5000,
    C=MODEL_C
)


unaware_model.fit(
    X_train_unaware,
    y_train
)


unaware_probability = (
    unaware_model
    .predict_proba(
        X_test_unaware
    )[:, 1]
)


unaware_prediction = (
    unaware_probability
    >= DECISION_THRESHOLD
).astype(int)


unaware_results = test_df[
    [
        "ID",
        "Gender",
        "SAT",
        "Admission"
    ]
].copy()


unaware_results[
    "Predicted_Probability"
] = unaware_probability


unaware_results[
    "Predicted_Admission"
] = np.where(
    unaware_prediction == 1,
    "Yes",
    "No"
)


unaware_results.to_csv(
    UNAWARE_PREDICTIONS_CSV,
    index=False
)


# ============================================================
# UNAWARE COEFFICIENTS
# ============================================================

unaware_intercept = (
    unaware_model.intercept_[0]
)


unaware_sat_coef = (
    unaware_model.coef_[0][0]
)


unaware_boundary = (
    (
        -unaware_intercept
        /
        unaware_sat_coef
    )
    *
    100
)


print("\n")
print("=" * 70)
print("UNAWARE MODEL")
print("=" * 70)


print(
    f"Intercept:            "
    f"{unaware_intercept:.6f}"
)


print(
    f"SAT/100 coefficient:  "
    f"{unaware_sat_coef:.6f}"
)


print(
    f"\nCommon SAT boundary:  "
    f"{unaware_boundary:.1f}"
)


# ============================================================
# UNAWARE FAIRNESS
# ============================================================

unaware_male_rate = (
    unaware_results.loc[
        unaware_results["Gender"]
        == "Male",
        "Predicted_Admission"
    ]
    == "Yes"
).mean()


unaware_female_rate = (
    unaware_results.loc[
        unaware_results["Gender"]
        == "Female",
        "Predicted_Admission"
    ]
    == "Yes"
).mean()


unaware_di = (
    unaware_male_rate
    -
    unaware_female_rate
)


unaware_accuracy = accuracy_score(
    y_test,
    unaware_prediction
)


print(
    f"\nAccuracy: "
    f"{unaware_accuracy:.3f}"
)


print(
    f"Male acceptance rate:   "
    f"{unaware_male_rate:.3f}"
)


print(
    f"Female acceptance rate: "
    f"{unaware_female_rate:.3f}"
)


print(
    f"DI gap (Male - Female): "
    f"{unaware_di:.3f}"
)


unaware_summary = print_acceptance_table(
    results=unaware_results,
    model_name="UNAWARE MODEL ACCEPTANCE SUMMARY",
    outcome_column="Predicted_Admission"
)


# ============================================================
# PLOT 1
#
# Predicted Admission Probability vs Gender
#
# x = Gender
# y = predicted probability
#
# Blue = predicted accept
# Red  = predicted reject
#
# Grey dashed line = p = 0.5
# ============================================================

def plot_probability_vs_gender(
    results,
    title,
    output_path,
    jitter_seed
):

    plot_rng = np.random.default_rng(
        jitter_seed
    )


    fig, ax = plt.subplots(
        figsize=(7, 6)
    )


    gender_position = {
        "Male": 0,
        "Female": 1
    }


    # --------------------------------------------------------
    # Rejected
    # --------------------------------------------------------

    rejected = (
        results["Predicted_Admission"]
        == "No"
    )


    rejected_x = (
        results.loc[
            rejected,
            "Gender"
        ]
        .map(
            gender_position
        )
        .astype(float)
        .to_numpy()
    )


    rejected_x += plot_rng.normal(
        0,
        0.055,
        len(rejected_x)
    )


    ax.scatter(
        rejected_x,

        results.loc[
            rejected,
            "Predicted_Probability"
        ],

        color="red",
        s=40,
        alpha=0.70,

        label="Predicted Reject"
    )


    # --------------------------------------------------------
    # Accepted
    # --------------------------------------------------------

    accepted = (
        results["Predicted_Admission"]
        == "Yes"
    )


    accepted_x = (
        results.loc[
            accepted,
            "Gender"
        ]
        .map(
            gender_position
        )
        .astype(float)
        .to_numpy()
    )


    accepted_x += plot_rng.normal(
        0,
        0.055,
        len(accepted_x)
    )


    ax.scatter(
        accepted_x,

        results.loc[
            accepted,
            "Predicted_Probability"
        ],

        color="blue",
        s=40,
        alpha=0.70,

        label="Predicted Admit"
    )


    # --------------------------------------------------------
    # Decision threshold
    # --------------------------------------------------------

    ax.axhline(
        y=0.50,

        color="grey",
        linestyle="--",
        linewidth=2,

        label="Decision Boundary (0.50)"
    )


    # --------------------------------------------------------
    # Formatting
    # --------------------------------------------------------

    ax.set_xticks(
        [0, 1]
    )


    ax.set_xticklabels(
        [
            "Male",
            "Female"
        ]
    )


    ax.set_xlim(
        -0.35,
        1.35
    )


    ax.set_ylim(
        0,
        1
    )


    ax.set_xlabel(
        "Gender"
    )


    ax.set_ylabel(
        "Predicted Admission Probability"
    )


    ax.set_title(
        title
    )


    ax.legend()


    plt.tight_layout()


    plt.savefig(
        output_path,
        dpi=300
    )


    plt.show()


# ============================================================
# PLOT 2
#
# Gender vs SAT decomposition
#
# x = Gender
# y = SAT
#
# Blue = predicted accepted
# Red  = predicted rejected
#
# Grey dashed line = SAT decision boundary
#
# Aware:
#
#     sloped line
#
# Unaware:
#
#     flat line
# ============================================================

def plot_sat_gender_decomposition(
    results,
    male_boundary,
    female_boundary,
    title,
    output_path,
    jitter_seed
):

    plot_rng = np.random.default_rng(
        jitter_seed
    )


    fig, ax = plt.subplots(
        figsize=(7, 7)
    )


    gender_position = {
        "Male": 0,
        "Female": 1
    }


    # --------------------------------------------------------
    # Rejected
    # --------------------------------------------------------

    rejected = (
        results["Predicted_Admission"]
        == "No"
    )


    rejected_x = (
        results.loc[
            rejected,
            "Gender"
        ]
        .map(
            gender_position
        )
        .astype(float)
        .to_numpy()
    )


    rejected_x += plot_rng.normal(
        0,
        0.055,
        len(rejected_x)
    )


    ax.scatter(
        rejected_x,

        results.loc[
            rejected,
            "SAT"
        ],

        color="red",
        s=40,
        alpha=0.70,

        label="Predicted Reject"
    )


    # --------------------------------------------------------
    # Accepted
    # --------------------------------------------------------

    accepted = (
        results["Predicted_Admission"]
        == "Yes"
    )


    accepted_x = (
        results.loc[
            accepted,
            "Gender"
        ]
        .map(
            gender_position
        )
        .astype(float)
        .to_numpy()
    )


    accepted_x += plot_rng.normal(
        0,
        0.055,
        len(accepted_x)
    )


    ax.scatter(
        accepted_x,

        results.loc[
            accepted,
            "SAT"
        ],

        color="blue",
        s=40,
        alpha=0.70,

        label="Predicted Admit"
    )


    # --------------------------------------------------------
    # Decision boundary
    # --------------------------------------------------------

    ax.plot(
        [0, 1],

        [
            male_boundary,
            female_boundary
        ],

        color="grey",
        linestyle="--",
        linewidth=2.5,

        label="Decision Boundary"
    )


    # --------------------------------------------------------
    # Formatting
    # --------------------------------------------------------

    ax.set_xticks(
        [0, 1]
    )


    ax.set_xticklabels(
        [
            "Male",
            "Female"
        ]
    )


    ax.set_xlim(
        -0.35,
        1.35
    )


    ax.set_ylim(
        1150,
        1550
    )


    ax.set_xlabel(
        "Gender"
    )


    ax.set_ylabel(
        "SAT"
    )


    ax.set_title(
        title
    )


    ax.legend()


    plt.tight_layout()


    plt.savefig(
        output_path,
        dpi=300
    )


    plt.show()


# ============================================================
# AWARE PLOT 1
# ============================================================

plot_probability_vs_gender(

    results=aware_results,

    title=(
        "Aware Logistic Regression\n"
        "Predicted Admission Probability vs Gender"
    ),

    output_path=AWARE_PROBABILITY_PLOT,

    jitter_seed=101

)


# ============================================================
# AWARE PLOT 2
#
# Expected:
#
# Male threshold   ≈ 1300
# Female threshold ≈ 1400
#
# Sloped boundary:
#
# same SAT can produce different decisions.
# ============================================================

plot_sat_gender_decomposition(

    results=aware_results,

    male_boundary=aware_male_boundary,

    female_boundary=aware_female_boundary,

    title=(
        "Aware Model Decomposition\n"
        "SAT Decision Boundary by Gender"
    ),

    output_path=AWARE_DECOMPOSITION_PLOT,

    jitter_seed=102

)


# ============================================================
# UNAWARE PLOT 1
# ============================================================

plot_probability_vs_gender(

    results=unaware_results,

    title=(
        "Unaware Logistic Regression\n"
        "Predicted Admission Probability vs Gender"
    ),

    output_path=UNAWARE_PROBABILITY_PLOT,

    jitter_seed=103

)


# ============================================================
# UNAWARE PLOT 2
#
# Same SAT threshold for both genders.
#
# Flat boundary:
#
# same SAT -> same decision.
# ============================================================

plot_sat_gender_decomposition(

    results=unaware_results,

    male_boundary=unaware_boundary,

    female_boundary=unaware_boundary,

    title=(
        "Unaware Model Decomposition\n"
        "SAT Decision Boundary by Gender"
    ),

    output_path=UNAWARE_DECOMPOSITION_PLOT,

    jitter_seed=104

)


# ============================================================
# FINAL COMPARISON
# ============================================================

print("\n")
print("=" * 70)
print("FINAL FAIRNESS COMPARISON")
print("=" * 70)


comparison = pd.DataFrame({

    "Model": [
        "Aware",
        "Unaware"
    ],

    "Male Acceptance": [
        aware_male_rate,
        unaware_male_rate
    ],

    "Female Acceptance": [
        aware_female_rate,
        unaware_female_rate
    ],

    "DI Gap": [
        aware_di,
        unaware_di
    ],

    "Male SAT Boundary": [
        aware_male_boundary,
        unaware_boundary
    ],

    "Female SAT Boundary": [
        aware_female_boundary,
        unaware_boundary
    ],

    "Boundary Gap": [
        (
            aware_female_boundary
            -
            aware_male_boundary
        ),
        0.0
    ],

    "Accuracy": [
        aware_accuracy,
        unaware_accuracy
    ]

})


print(
    comparison.round(3)
)


# ============================================================
# INTERPRETATION
# ============================================================

print("\n")
print("=" * 70)
print("INTERPRETATION")
print("=" * 70)


print("\nAWARE MODEL:")

print(
    "The historical data has exactly 50% acceptance for "
    "Male and Female applicants."
)

print(
    "The aware classifier can learn different SAT thresholds "
    "for the two genders."
)

print(
    "The intended boundaries are approximately 1300 for Male "
    "and 1400 for Female."
)

print(
    "Therefore applicants with the same SAT can receive "
    "different predictions depending on Gender."
)

print(
    "=> Group parity can coexist with individual unfairness."
)


print("\nUNAWARE MODEL:")

print(
    "Gender is removed from the classifier."
)

print(
    "Male and Female applicants therefore share one common "
    "SAT decision boundary."
)

print(
    "Applicants with the same SAT receive the same prediction."
)

print(
    "However, because the Female applicant SAT distribution "
    "is shifted upward, the proportions above the common "
    "threshold differ between groups."
)

print(
    "=> Matched-SAT individual fairness can coexist with "
    "group disparity."
)


# ============================================================
# FILES SAVED
# ============================================================

print("\n")
print("=" * 70)
print("FILES SAVED")
print("=" * 70)

print(DATA_CSV)

print(AWARE_PREDICTIONS_CSV)

print(UNAWARE_PREDICTIONS_CSV)

print(AWARE_PROBABILITY_PLOT)

print(AWARE_DECOMPOSITION_PLOT)

print(UNAWARE_PROBABILITY_PLOT)

print(UNAWARE_DECOMPOSITION_PLOT)