import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_SEED = 42

TEST_SIZE = 0.25
DECISION_THRESHOLD = 0.50

# IMPORTANT:
# Reuse EXACTLY the same original dataset from aware experiment
DATA_CSV = (
    "fairness_through_unawareness\\3-features\\"
    "1-aware_experiment\\2-original_data.csv"
)

OUTPUT_DIR = (
    "fairness_through_unawareness\\3-features\\"
    "2-unaware_experiment"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

PREDICTIONS_CSV = os.path.join(
    OUTPUT_DIR,
    "3-unaware_predictions.csv"
)

COEFFICIENTS_CSV = os.path.join(
    OUTPUT_DIR,
    "4-unaware_coefficients.csv"
)

PLOT_GENDER = os.path.join(
    OUTPUT_DIR,
    "5-probability_vs_gender.png"
)

PLOT_3D = os.path.join(
    OUTPUT_DIR,
    "6-sat_gender_dept_3d.png"
)


# ============================================================
# LOAD ORIGINAL DATA
# ============================================================

df = pd.read_csv(
    DATA_CSV
)

print("=" * 70)
print("ORIGINAL DATA")
print("=" * 70)

print(df.head())

print("\nGender counts:")
print(
    df["Gender"].value_counts()
)

print("\nDepartment by Gender:")
print(
    pd.crosstab(
        df["Gender"],
        df["Dept"],
        normalize="index"
    ).round(3)
)


# ============================================================
# SAME TRAIN / TEST SPLIT
#
# Same:
#   random_state
#   test_size
#   stratification
#
# Therefore this should reproduce the same rows used in the
# aware experiment.
# ============================================================

train_idx, test_idx = train_test_split(
    df.index,
    test_size=TEST_SIZE,
    random_state=RANDOM_SEED,
    stratify=df["Admission"]
)

train_df = df.loc[
    train_idx
].copy()

test_df = df.loc[
    test_idx
].copy()


# ============================================================
# UNAWARE FEATURES
#
# Gender is REMOVED from the model.
#
# Gender is still preserved in test_df so we can evaluate
# outcomes by gender afterward.
# ============================================================

FEATURE_COLUMNS = [
    "SAT",
    "Dept"
]

TARGET_COLUMN = "Admission"


X_train = train_df[
    FEATURE_COLUMNS
]

X_test = test_df[
    FEATURE_COLUMNS
]


y_train = (
    train_df[
        TARGET_COLUMN
    ]
    == "Yes"
).astype(int)

y_test = (
    test_df[
        TARGET_COLUMN
    ]
    == "Yes"
).astype(int)


# ============================================================
# PREPROCESSING
#
# Engineering = baseline
# Humanities  = encoded category
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[

        (
            "sat",
            "passthrough",
            ["SAT"]
        ),

        (
            "categorical",
            OneHotEncoder(
                categories=[
                    [
                        "Engineering",
                        "Humanities"
                    ]
                ],
                drop="first",
                handle_unknown="ignore"
            ),
            ["Dept"]
        )

    ]
)


# ============================================================
# LOGISTIC REGRESSION
# ============================================================

model = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),

    (
        "classifier",
        LogisticRegression(
            max_iter=2000,
            C=1000
        )
    )
])


model.fit(
    X_train,
    y_train
)


# ============================================================
# PREDICTIONS
# ============================================================

predicted_probability = (
    model.predict_proba(
        X_test
    )[:, 1]
)

predicted_numeric = (
    predicted_probability
    >= DECISION_THRESHOLD
).astype(int)

predicted_admission = np.where(
    predicted_numeric == 1,
    "Yes",
    "No"
)


results = test_df[
    [
        "ID",
        "Gender",
        "SAT",
        "Dept",
        "Admission"
    ]
].copy()

results = results.rename(
    columns={
        "Admission":
        "Actual_Admission"
    }
)

results[
    "Predicted_Probability"
] = predicted_probability

results[
    "Predicted_Admission"
] = predicted_admission


results.to_csv(
    PREDICTIONS_CSV,
    index=False
)


# ============================================================
# MODEL PERFORMANCE
# ============================================================

accuracy = accuracy_score(
    y_test,
    predicted_numeric
)

print("\n")
print("=" * 70)
print("UNAWARE MODEL RESULTS")
print("=" * 70)

print(
    f"Accuracy: {accuracy:.4f}"
)


# ============================================================
# ACCEPTANCE BY GENDER
# ============================================================

print("\nAcceptance by Gender")
print("-" * 70)


gender_summary = (
    results
    .groupby(
        "Gender"
    )[
        "Predicted_Admission"
    ]
    .apply(
        lambda x: pd.Series({
            "Total":
            len(x),

            "Accepted":
            (
                x == "Yes"
            ).sum(),

            "Acceptance_Rate":
            (
                x == "Yes"
            ).mean()
        })
    )
    .unstack()
)

print(
    gender_summary
)


male_rate = (
    results.loc[
        results["Gender"]
        == "Male",
        "Predicted_Admission"
    ]
    == "Yes"
).mean()

female_rate = (
    results.loc[
        results["Gender"]
        == "Female",
        "Predicted_Admission"
    ]
    == "Yes"
).mean()

di_gap = (
    male_rate
    - female_rate
)


print(
    f"\nMale acceptance rate:   "
    f"{male_rate:.3f}"
)

print(
    f"Female acceptance rate: "
    f"{female_rate:.3f}"
)

print(
    f"DI gap (Male - Female): "
    f"{di_gap:.3f}"
)


# ============================================================
# ACCEPTANCE BY DEPARTMENT AND GENDER
# ============================================================

print("\n")
print("=" * 70)
print("DECOMPOSED ACCEPTANCE RATES")
print("=" * 70)


decomposition = (
    results
    .groupby(
        [
            "Dept",
            "Gender"
        ]
    )[
        "Predicted_Admission"
    ]
    .apply(
        lambda x:
        (
            x == "Yes"
        ).mean()
    )
)

print(
    decomposition
)


# ============================================================
# COEFFICIENTS
# ============================================================

classifier = (
    model.named_steps[
        "classifier"
    ]
)

prep = (
    model.named_steps[
        "preprocessor"
    ]
)


feature_names = (
    prep.get_feature_names_out()
)

coefficients = (
    classifier.coef_[0]
)


coef_df = pd.DataFrame({
    "Feature":
    feature_names,

    "Coefficient":
    coefficients
})


intercept_df = pd.DataFrame({
    "Feature":
    ["Intercept"],

    "Coefficient":
    [
        classifier.intercept_[0]
    ]
})


coef_df = pd.concat(
    [
        intercept_df,
        coef_df
    ],
    ignore_index=True
)


coef_df.to_csv(
    COEFFICIENTS_CSV,
    index=False
)


print("\n")
print("=" * 70)
print("LEARNED COEFFICIENTS")
print("=" * 70)

print(
    coef_df.to_string(
        index=False
    )
)


# ============================================================
# HELPER:
# ESTIMATE SAT DECISION BOUNDARY
#
# IMPORTANT:
# Gender is NOT passed to the model.
#
# Therefore within a Department:
#
#   Male boundary
#   Female boundary
#
# should be identical.
#
# We calculate by Department only.
# ============================================================

sat_grid = np.arange(
    700,
    1601
)


def get_sat_boundary(
    dept_value
):

    grid = pd.DataFrame({
        "SAT":
        sat_grid,

        "Dept":
        dept_value
    })

    probs = (
        model.predict_proba(
            grid
        )[:, 1]
    )

    accepted_indices = np.where(
        probs
        >= DECISION_THRESHOLD
    )[0]

    if (
        len(
            accepted_indices
        )
        == 0
    ):
        return None

    return sat_grid[
        accepted_indices[0]
    ]


# ============================================================
# CALCULATE DEPARTMENT BOUNDARIES
# ============================================================

boundaries = {}

for d in [
    "Engineering",
    "Humanities"
]:

    boundaries[d] = (
        get_sat_boundary(
            d
        )
    )


print("\n")
print("=" * 70)
print("ESTIMATED SAT DECISION BOUNDARIES")
print("=" * 70)


for d in [
    "Engineering",
    "Humanities"
]:

    boundary = (
        boundaries[d]
    )

    print(
        f"{d:12s} | "
        f"Male   | "
        f"SAT threshold ≈ "
        f"{boundary}"
    )

    print(
        f"{d:12s} | "
        f"Female | "
        f"SAT threshold ≈ "
        f"{boundary}"
    )


# ============================================================
# PLOT 1
#
# Predicted Admission Probability vs Gender
#
# Blue = predicted accepted
# Red  = predicted rejected
#
# Gender is used ONLY for visualization/evaluation.
# It is NOT an input to the model.
# ============================================================

rng = np.random.default_rng(
    RANDOM_SEED
)


fig, ax = plt.subplots(
    figsize=(
        7,
        6
    )
)


gender_position = {
    "Male": 0,
    "Female": 1
}


# ------------------------------------------------------------
# Rejected
# ------------------------------------------------------------

rejected = (
    results[
        "Predicted_Admission"
    ]
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


rejected_x += (
    rng.normal(
        0,
        0.055,
        len(
            rejected_x
        )
    )
)


ax.scatter(
    rejected_x,

    results.loc[
        rejected,
        "Predicted_Probability"
    ],

    color="red",
    s=35,
    alpha=0.65,

    label=(
        "Predicted Reject"
    )
)


# ------------------------------------------------------------
# Accepted
# ------------------------------------------------------------

accepted = (
    results[
        "Predicted_Admission"
    ]
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


accepted_x += (
    rng.normal(
        0,
        0.055,
        len(
            accepted_x
        )
    )
)


ax.scatter(
    accepted_x,

    results.loc[
        accepted,
        "Predicted_Probability"
    ],

    color="blue",
    s=35,
    alpha=0.65,

    label=(
        "Predicted Admit"
    )
)


# ------------------------------------------------------------
# Probability decision threshold
# ------------------------------------------------------------

ax.axhline(
    y=DECISION_THRESHOLD,

    color="grey",
    linestyle="--",
    linewidth=2,

    label=(
        "Decision Boundary (0.50)"
    )
)


# ------------------------------------------------------------
# Formatting
# ------------------------------------------------------------

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
    "Unaware Logistic Regression\n"
    "Predicted Admission Probability vs Gender"
)


ax.legend()

plt.tight_layout()


plt.savefig(
    PLOT_GENDER,
    dpi=300
)

plt.show()


# ============================================================
# PLOT 2
#
# 3D DECOMPOSITION
#
# x-axis = Gender
# y-axis = Department
# z-axis = SAT
#
# Blue = predicted accepted
# Red  = predicted rejected
#
# Since Gender is not in the model:
#
#   within Engineering:
#       Male boundary = Female boundary
#
#   within Humanities:
#       Male boundary = Female boundary
#
# So each grey dashed boundary should be FLAT across Gender.
# ============================================================

fig = plt.figure(
    figsize=(
        10,
        8
    )
)


ax = fig.add_subplot(
    111,
    projection="3d"
)


# ============================================================
# CATEGORICAL POSITIONS
# ============================================================

gender_numeric = (
    results[
        "Gender"
    ]
    .map({
        "Male": 0,
        "Female": 1
    })
    .astype(float)
)


dept_numeric = (
    results[
        "Dept"
    ]
    .map({
        "Engineering": 0,
        "Humanities": 1
    })
    .astype(float)
)


# ============================================================
# JITTER
# ============================================================

x_jitter = (
    gender_numeric
    .to_numpy()
    +
    rng.normal(
        0,
        0.035,
        len(results)
    )
)


y_jitter = (
    dept_numeric
    .to_numpy()
    +
    rng.normal(
        0,
        0.035,
        len(results)
    )
)


accepted = (
    results[
        "Predicted_Admission"
    ]
    == "Yes"
).to_numpy()

rejected = (
    ~accepted
)


# ============================================================
# REJECTED
# ============================================================

ax.scatter(
    x_jitter[
        rejected
    ],

    y_jitter[
        rejected
    ],

    results.loc[
        rejected,
        "SAT"
    ],

    color="red",
    marker="o",
    s=35,
    alpha=0.65,

    label=(
        "Predicted Reject"
    )
)


# ============================================================
# ACCEPTED
# ============================================================

ax.scatter(
    x_jitter[
        accepted
    ],

    y_jitter[
        accepted
    ],

    results.loc[
        accepted,
        "SAT"
    ],

    color="blue",
    marker="o",
    s=35,
    alpha=0.65,

    label=(
        "Predicted Admit"
    )
)


# ============================================================
# DECISION BOUNDARY LINES
#
# Since Gender is absent from the model,
# the SAT boundary for a given Department is the same
# for Male and Female.
#
# Therefore:
#
#   z(Male) = z(Female)
#
# and each boundary line should be horizontal/flat across
# the Gender axis.
# ============================================================

department_positions = {
    "Engineering": 0,
    "Humanities": 1
}


for d in [
    "Engineering",
    "Humanities"
]:

    boundary = (
        boundaries[d]
    )

    if (
        boundary
        is not None
    ):

        x_boundary = np.array([
            0,
            1
        ])

        y_boundary = np.array([
            department_positions[d],
            department_positions[d]
        ])

        z_boundary = np.array([
            boundary,
            boundary
        ])


        ax.plot(
            x_boundary,
            y_boundary,
            z_boundary,

            color="grey",
            linestyle="--",
            linewidth=2.5
        )


# ============================================================
# AXES
# ============================================================

ax.set_xlabel(
    "Gender"
)

ax.set_ylabel(
    "Department"
)

ax.set_zlabel(
    "SAT"
)


ax.set_xticks(
    [
        0,
        1
    ]
)

ax.set_xticklabels(
    [
        "Male",
        "Female"
    ]
)


ax.set_yticks(
    [
        0,
        1
    ]
)

ax.set_yticklabels(
    [
        "Engineering",
        "Humanities"
    ]
)


ax.set_zlim(
    700,
    1600
)


ax.set_title(
    "Unaware Model Decomposition\n"
    "SAT Decision Boundaries by Gender and Department"
)


# Dummy line for legend
ax.plot(
    [],
    [],
    [],

    color="grey",
    linestyle="--",
    linewidth=2.5,

    label=(
        "Decision Boundary"
    )
)


ax.legend()

plt.tight_layout()


plt.savefig(
    PLOT_3D,
    dpi=300
)

plt.show()


# ============================================================
# FILES
# ============================================================

print("\nFiles saved:")

print(
    PREDICTIONS_CSV
)

print(
    COEFFICIENTS_CSV
)

print(
    PLOT_GENDER
)

print(
    PLOT_3D
)