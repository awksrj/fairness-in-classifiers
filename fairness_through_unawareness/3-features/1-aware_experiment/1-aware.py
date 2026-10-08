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
N = 1000

TEST_SIZE = 0.25
DECISION_THRESHOLD = 0.50

OUTPUT_DIR = "fairness_through_unawareness\\3-features\\1-aware_experiment"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATA_CSV = os.path.join(OUTPUT_DIR, "2-original_data.csv")
PREDICTIONS_CSV = os.path.join(OUTPUT_DIR, "3-aware_predictions.csv")
COEFFICIENTS_CSV = os.path.join(OUTPUT_DIR, "4-aware_coefficients.csv")

PLOT_GENDER = os.path.join(
    OUTPUT_DIR,
    "5-probability_vs_gender.png"
)

PLOT_3D = os.path.join(
    OUTPUT_DIR,
    "6-sat_gender_dept_3d.png"
)


# ============================================================
# SYNTHETIC DATA GENERATION
# ============================================================

rng = np.random.default_rng(RANDOM_SEED)


# ------------------------------------------------------------
# 1. Gender
#
# Roughly equal number of Male and Female applicants.
# ------------------------------------------------------------

gender = rng.choice(
    ["Male", "Female"],
    size=N,
    p=[0.50, 0.50]
)


# ------------------------------------------------------------
# 2. SAT
#
# IMPORTANT:
# SAT has approximately the SAME distribution for both genders.
#
# This lets us focus on Gender and Department rather than
# introducing SAT as another gender proxy.
# ------------------------------------------------------------

sat = rng.normal(
    loc=1250,
    scale=170,
    size=N
)

sat = np.clip(sat, 700, 1600)
sat = np.round(sat).astype(int)


# ------------------------------------------------------------
# 3. Department
#
# Department is correlated with Gender.
#
# Male:
#   75% Engineering
#   25% Humanities
#
# Female:
#   25% Engineering
#   75% Humanities
#
# Therefore Department contains information about Gender.
# ------------------------------------------------------------

dept = []

for g in gender:

    if g == "Male":
        d = rng.choice(
            ["Engineering", "Humanities"],
            p=[0.75, 0.25]
        )

    else:
        d = rng.choice(
            ["Engineering", "Humanities"],
            p=[0.25, 0.75]
        )

    dept.append(d)

dept = np.array(dept)


# ------------------------------------------------------------
# 4. Historical Admission Process
#
# SAT is the main legitimate determinant.
#
# We intentionally add:
#
#   Female penalty
#   Humanities penalty
#
# so the historical data contains the unfair pattern we want
# to investigate.
#
# SAT is divided by 100 only to keep coefficient sizes easy
# to work with.
# ------------------------------------------------------------

sat_100 = sat / 100.0

score = (
    -8.0
    + 0.65 * sat_100
)


# Direct gender penalty
score += np.where(
    gender == "Female",
    -0.80,
    0.0
)


# Department penalty
score += np.where(
    dept == "Humanities",
    -0.55,
    0.0
)


# Convert logistic score into admission probability
true_probability = 1 / (1 + np.exp(-score))


# Sample historical admission outcome
admission_numeric = rng.binomial(
    1,
    true_probability
)

admission = np.where(
    admission_numeric == 1,
    "Yes",
    "No"
)


# ------------------------------------------------------------
# Create dataframe
# ------------------------------------------------------------

df = pd.DataFrame({
    "ID": np.arange(1, N + 1),
    "Gender": gender,
    "SAT": sat,
    "Dept": dept,
    "Admission": admission
})


df.to_csv(DATA_CSV, index=False)


print("=" * 70)
print("ORIGINAL DATA")
print("=" * 70)

print(df.head())

print("\nGender counts:")
print(df["Gender"].value_counts())

print("\nDepartment by Gender:")
print(
    pd.crosstab(
        df["Gender"],
        df["Dept"],
        normalize="index"
    ).round(3)
)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

train_idx, test_idx = train_test_split(
    df.index,
    test_size=TEST_SIZE,
    random_state=RANDOM_SEED,
    stratify=df["Admission"]
)

train_df = df.loc[train_idx].copy()
test_df = df.loc[test_idx].copy()


# ============================================================
# FEATURES
# ============================================================

FEATURE_COLUMNS = [
    "SAT",
    "Dept",
    "Gender"
]

TARGET_COLUMN = "Admission"


X_train = train_df[FEATURE_COLUMNS]
X_test = test_df[FEATURE_COLUMNS]

y_train = (
    train_df[TARGET_COLUMN] == "Yes"
).astype(int)

y_test = (
    test_df[TARGET_COLUMN] == "Yes"
).astype(int)


# ============================================================
# PREPROCESSING
#
# Explicit category order is important.
#
# Reference categories:
#
# Gender:
#   Male = baseline
#
# Department:
#   Engineering = baseline
#
# Therefore we directly obtain:
#
#   Gender_Female
#   Dept_Humanities
#
# coefficients.
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
                    ["Engineering", "Humanities"],
                    ["Male", "Female"]
                ],
                drop="first",
                handle_unknown="ignore"
            ),
            ["Dept", "Gender"]
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

predicted_probability = model.predict_proba(
    X_test
)[:, 1]

predicted_numeric = (
    predicted_probability >= DECISION_THRESHOLD
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
        "Admission": "Actual_Admission"
    }
)

results["Predicted_Probability"] = predicted_probability
results["Predicted_Admission"] = predicted_admission


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
print("AWARE MODEL RESULTS")
print("=" * 70)

print(f"Accuracy: {accuracy:.4f}")


# ============================================================
# ACCEPTANCE RATES BY GENDER
# ============================================================

print("\nAcceptance by Gender")
print("-" * 70)

gender_summary = (
    results
    .groupby("Gender")["Predicted_Admission"]
    .apply(
        lambda x: pd.Series({
            "Total": len(x),
            "Accepted": (x == "Yes").sum(),
            "Acceptance_Rate": (x == "Yes").mean()
        })
    )
    .unstack()
)

print(gender_summary)


male_rate = (
    results.loc[
        results["Gender"] == "Male",
        "Predicted_Admission"
    ] == "Yes"
).mean()

female_rate = (
    results.loc[
        results["Gender"] == "Female",
        "Predicted_Admission"
    ] == "Yes"
).mean()

di_gap = male_rate - female_rate

print(
    f"\nMale acceptance rate:   {male_rate:.3f}"
)

print(
    f"Female acceptance rate: {female_rate:.3f}"
)

print(
    f"DI gap (Male - Female): {di_gap:.3f}"
)


# ============================================================
# ACCEPTANCE RATE BY GENDER AND DEPARTMENT
# ============================================================

print("\n")
print("=" * 70)
print("DECOMPOSED ACCEPTANCE RATES")
print("=" * 70)

decomposition = (
    results
    .groupby(
        ["Dept", "Gender"]
    )["Predicted_Admission"]
    .apply(
        lambda x: (x == "Yes").mean()
    )
)

print(decomposition)


# ============================================================
# COEFFICIENTS
# ============================================================

classifier = model.named_steps["classifier"]
prep = model.named_steps["preprocessor"]


feature_names = prep.get_feature_names_out()

coefficients = classifier.coef_[0]


coef_df = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": coefficients
})

intercept_df = pd.DataFrame({
    "Feature": ["Intercept"],
    "Coefficient": [classifier.intercept_[0]]
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

print(coef_df.to_string(index=False))


# ============================================================
# HELPER: ESTIMATE SAT DECISION BOUNDARY
#
# Returns the first SAT value where predicted probability
# reaches the decision threshold for a given:
#
#   Gender × Department
#
# ============================================================

sat_grid = np.arange(
    700,
    1601
)


def get_sat_boundary(gender_value, dept_value):

    grid = pd.DataFrame({
        "SAT": sat_grid,
        "Dept": dept_value,
        "Gender": gender_value
    })

    probs = model.predict_proba(
        grid
    )[:, 1]

    accepted_indices = np.where(
        probs >= DECISION_THRESHOLD
    )[0]

    if len(accepted_indices) == 0:
        return None

    return sat_grid[
        accepted_indices[0]
    ]


# ============================================================
# CALCULATE ALL FOUR DECISION BOUNDARIES
# ============================================================

boundaries = {}

for d in [
    "Engineering",
    "Humanities"
]:

    for g in [
        "Male",
        "Female"
    ]:

        boundaries[(d, g)] = get_sat_boundary(
            gender_value=g,
            dept_value=d
        )


# Print boundaries
print("\n")
print("=" * 70)
print("ESTIMATED SAT DECISION BOUNDARIES")
print("=" * 70)

for d in [
    "Engineering",
    "Humanities"
]:

    for g in [
        "Male",
        "Female"
    ]:

        boundary = boundaries[(d, g)]

        print(
            f"{d:12s} | "
            f"{g:6s} | "
            f"SAT threshold ≈ {boundary}"
        )


# ============================================================
# PLOT 1
#
# Predicted Admission Probability vs Gender
#
# x-axis:
#   Gender
#
# y-axis:
#   Predicted Admission Probability
#
# Each applicant is one jittered point.
#
# Blue = predicted accepted
# Red  = predicted rejected
#
# Horizontal dashed line = P = 0.50 decision boundary
# ============================================================

fig, ax = plt.subplots(
    figsize=(7, 6)
)


gender_position = {
    "Male": 0,
    "Female": 1
}


# ------------------------------------------------------------
# Predicted rejected applicants
# ------------------------------------------------------------

rejected = (
    results["Predicted_Admission"] == "No"
)

rejected_x = (
    results.loc[
        rejected,
        "Gender"
    ]
    .map(gender_position)
    .astype(float)
    .to_numpy()
)

rejected_x += rng.normal(
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
    s=35,
    alpha=0.65,
    label="Predicted Reject"
)


# ------------------------------------------------------------
# Predicted accepted applicants
# ------------------------------------------------------------

accepted = (
    results["Predicted_Admission"] == "Yes"
)

accepted_x = (
    results.loc[
        accepted,
        "Gender"
    ]
    .map(gender_position)
    .astype(float)
    .to_numpy()
)

accepted_x += rng.normal(
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
    s=35,
    alpha=0.65,
    label="Predicted Admit"
)


# ------------------------------------------------------------
# Decision boundary
# ------------------------------------------------------------

ax.axhline(
    y=DECISION_THRESHOLD,
    color="black",
    linestyle="--",
    linewidth=1.5,
    label="Decision Boundary (0.50)"
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
    "Aware Logistic Regression\n"
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
# 3D DATA DECOMPOSITION
#
# x-axis = Gender
# y-axis = Department
# z-axis = SAT
#
# Each applicant is one jittered point.
#
# Blue = predicted accepted
# Red  = predicted rejected
#
# A black decision-boundary line is drawn across each
# Department group.
#
# The line connects:
#
#   Male SAT threshold
#          to
#   Female SAT threshold
#
# within that Department.
# ============================================================

fig = plt.figure(
    figsize=(10, 8)
)

ax = fig.add_subplot(
    111,
    projection="3d"
)


# ============================================================
# Convert categorical dimensions to numeric positions
# ============================================================

gender_numeric = (
    results["Gender"]
    .map({
        "Male": 0,
        "Female": 1
    })
    .astype(float)
)

dept_numeric = (
    results["Dept"]
    .map({
        "Engineering": 0,
        "Humanities": 1
    })
    .astype(float)
)


# ============================================================
# Add jitter to categorical axes
#
# This prevents applicants with similar values from sitting
# directly on top of one another.
# ============================================================

x_jitter = (
    gender_numeric.to_numpy()
    +
    rng.normal(
        0,
        0.035,
        len(results)
    )
)

y_jitter = (
    dept_numeric.to_numpy()
    +
    rng.normal(
        0,
        0.035,
        len(results)
    )
)


accepted = (
    results["Predicted_Admission"] == "Yes"
).to_numpy()

rejected = ~accepted


# ============================================================
# Rejected applicants
# ============================================================

ax.scatter(
    x_jitter[rejected],
    y_jitter[rejected],
    results.loc[
        rejected,
        "SAT"
    ],
    color="red",
    marker="o",
    s=35,
    alpha=0.65,
    label="Predicted Reject"
)


# ============================================================
# Accepted applicants
# ============================================================

ax.scatter(
    x_jitter[accepted],
    y_jitter[accepted],
    results.loc[
        accepted,
        "SAT"
    ],
    color="blue",
    marker="o",
    s=35,
    alpha=0.65,
    label="Predicted Admit"
)


# ============================================================
# DECISION BOUNDARY LINES
#
# For each Department:
#
#   x = Gender
#   y = fixed Department
#   z = corresponding SAT threshold
#
# Therefore the line directly shows how much higher/lower
# the SAT boundary is for Female vs Male within that Dept.
# ============================================================

department_positions = {
    "Engineering": 0,
    "Humanities": 1
}


for d in [
    "Engineering",
    "Humanities"
]:

    male_boundary = boundaries[
        (d, "Male")
    ]

    female_boundary = boundaries[
        (d, "Female")
    ]

    # Only draw if both boundaries exist
    if (
        male_boundary is not None
        and
        female_boundary is not None
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
            male_boundary,
            female_boundary
        ])
        ax.plot(
            x_boundary,
            y_boundary,
            z_boundary,
            color="grey",
            linestyle="--",
            linewidth=2.5
        )

        # Boundary markers
        ax.scatter(
            x_boundary,
            y_boundary,
            z_boundary,
            color="black",
            s=55,
            marker="D"
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
    [0, 1]
)

ax.set_xticklabels(
    [
        "Male",
        "Female"
    ]
)


ax.set_yticks(
    [0, 1]
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
    "Aware Model Decomposition\n"
    "SAT Decision Boundaries by Gender and Department"
)


# Add one dummy line so boundary appears in legend
ax.plot(
    [],
    [],
    [],
    color="grey",
    linestyle="--",
    linewidth=2.5,
    label="Decision Boundary"
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

print(DATA_CSV)
print(PREDICTIONS_CSV)
print(COEFFICIENTS_CSV)
print(PLOT_GENDER)
print(PLOT_3D)