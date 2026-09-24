"""Two-feature LFR thought experiment with nested model selection.

Run from the project root: python 'Pasted code(10).py'
The script generates a 240-student dataset if INPUT_CSV does not exist.
It reserves a test set, selects two models using out-of-fold predictions,
and fits the selected models on all non-test rows before testing them.
"""

from pathlib import Path
import itertools

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold, train_test_split


# ============================================================
# CONFIGURATION
# ============================================================
OUTPUT_DIR = Path("lfr2-light")
INPUT_CSV = OUTPUT_DIR / "1-data.csv"
SPLIT_CSV = OUTPUT_DIR / "2-split_assignments_2d.csv"
GRID_CSV = OUTPUT_DIR / "4-weight_grid_search_2d.csv"
SELECTED_CSV = OUTPUT_DIR / "6-selected_model_summary_2d.csv"
K = 4
RANDOM_SEED = 42
TEST_FRACTION = 0.20
N_FOLDS = 5
THRESHOLD = 0.5
NEAR_THRESHOLD_MARGIN = 0.01
SAT_MIN, SAT_MAX = 400.0, 1600.0  # Fixed before any split.

# Same 30 hyperparameter settings as the paper's search.
AX = 0.01
AY_VALUES = (0.1, 0.5, 1.0, 5.0, 10.0)
AZ_VALUES = (0.0, 0.1, 0.5, 1.0, 5.0, 10.0)
MAX_ITER = 1200
EPSILON = 1e-10


# ============================================================
# SYNTHETIC DATA AND VALIDATION
# ============================================================
def create_synthetic_data(path):
    """Make six groups of 20/80/20 per gender without duplicated rows.

    The middle groups deliberately have different observed outcomes.
    This is an experiment design, not a claim about admission fairness.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    for gender in ("M", "F"):
        groups = (
            ("High", 1550, 1600, 20, "Yes"),
            ("Middle", 800, 900, 80, "Yes") if gender == "M"
            else ("Middle", 1200, 1300, 80, "No"),
            ("Low", 400, 500, 20, "No"),
        )
        for band, lower, upper, count, admission in groups:
            # Unique within a subgroup: repeated exact SAT rows cannot leak
            # between train, validation, and test subsets.
            scores = rng.choice(np.arange(lower, upper + 1), size=count, replace=False)
            for score in scores:
                rows.append((gender, int(score), admission, band))
    rng.shuffle(rows)
    frame = pd.DataFrame(rows, columns=["Gender", "SAT", "Admission", "Band"])
    frame.insert(0, "ID", np.arange(1, len(frame) + 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame


def load_data(path):
    if not path.exists():
        print(f"Generating the synthetic dataset: {path}")
        create_synthetic_data(path)
    frame = pd.read_csv(path)
    required = {"ID", "Gender", "SAT", "Admission"}
    if required - set(frame.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(frame.columns))}")
    if not frame.Gender.isin(["M", "F"]).all():
        raise ValueError("Gender must contain only M and F.")
    if not frame.Admission.isin(["Yes", "No"]).all():
        raise ValueError("Admission must contain only Yes and No.")
    if not frame.ID.is_unique:
        raise ValueError("Every row needs a unique ID.")
    if frame[["Gender", "SAT"]].duplicated().any():
        raise ValueError(
            "Repeated (Gender, SAT) rows would leak across folds. "
            "Use distinct SAT values per gender or split repeated rows together."
        )
    sat = pd.to_numeric(frame.SAT, errors="raise")
    if sat.isna().any() or (~sat.between(SAT_MIN, SAT_MAX)).any():
        raise ValueError(f"SAT must be between {SAT_MIN:g} and {SAT_MAX:g}.")
    return frame.reset_index(drop=True)


def strata(frame):
    """Stratify by gender, outcome, and SAT band (six designed groups)."""
    if "Band" in frame.columns:
        bands = frame.Band.astype(str)
    else:
        # Also works for another dataset with no Band column.
        bands = pd.cut(
            frame.SAT, [SAT_MIN - 1, 650, 1050, 1400, SAT_MAX],
            labels=["Low", "Male-middle", "Female-middle", "High"],
        ).astype(str)
    return frame.Gender.astype(str) + "|" + frame.Admission.astype(str) + "|" + bands


def features(frame):
    """SAT and Gender are both inputs; Gender is also used for group loss."""
    sat = (frame.SAT.to_numpy(dtype=float) - SAT_MIN) / (SAT_MAX - SAT_MIN)
    gender = frame.Gender.eq("F").to_numpy(dtype=float)
    X = np.column_stack((sat, gender))
    Y = frame.Admission.eq("Yes").to_numpy(dtype=float)
    return X, Y, gender.astype(bool)


# ============================================================
# TWO-FEATURE LFR
# ============================================================
def unpack(params):
    # First K*2: prototype feature coordinates; next K: outcome scores;
    # final 2: per-feature distance weights [SAT, Gender].
    prototypes = params[:2 * K].reshape(K, 2)
    scores = params[2 * K:3 * K]
    alphas = params[3 * K:3 * K + 2]
    return prototypes, scores, alphas


def membership(X, prototypes, alphas):
    distances = np.sum(alphas[None, None, :] * (X[:, None, :] - prototypes[None, :, :]) ** 2, axis=2)
    logits = -distances
    logits -= logits.max(axis=1, keepdims=True)
    exponentials = np.exp(logits)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def loss_terms(X, Y, protected, prototypes, scores, alphas):
    M = membership(X, prototypes, alphas)
    Lz = np.abs(M[protected].mean(axis=0) - M[~protected].mean(axis=0)).sum()
    reconstructed = M @ prototypes
    squared_error = np.mean((X - reconstructed) ** 2, axis=0)
    # Each feature contributes its mean squared error; expose both terms.
    Lx_sat, Lx_gender = squared_error
    Lx = Lx_sat + Lx_gender
    predicted = np.clip(M @ scores, EPSILON, 1.0 - EPSILON)
    Ly = -np.mean(Y * np.log(predicted) + (1 - Y) * np.log(1 - predicted))
    return Lz, Lx, Ly, Lx_sat, Lx_gender


def initial_parameters(seed):
    rng = np.random.default_rng(seed)
    sat_coordinates = np.clip(np.linspace(0.05, 0.95, K) + rng.normal(0, 0.01, K), 0, 1)
    gender_coordinates = rng.uniform(0.25, 0.75, K)
    prototypes = np.column_stack((sat_coordinates, gender_coordinates))
    scores = rng.uniform(0.25, 0.75, K)
    return np.concatenate((prototypes.ravel(), scores, [10.0, 1.0]))


BOUNDS = [(0, 1)] * (3 * K) + [(0.01, 100.0)] * 2


def fit_model(X, Y, protected, az, ay, initial):
    def objective(params):
        prototypes, scores, alphas = unpack(params)
        Lz, Lx, Ly, _, _ = loss_terms(X, Y, protected, prototypes, scores, alphas)
        return az * Lz + AX * Lx + ay * Ly

    return minimize(
        objective, initial.copy(), method="L-BFGS-B", bounds=BOUNDS,
        options={"maxiter": MAX_ITER, "ftol": 1e-10, "maxls": 50},
    )


def predict(X, params):
    prototypes, scores, alphas = unpack(params)
    M = membership(X, prototypes, alphas)
    return M @ scores, M


def metrics(Y, protected, scores):
    yes = scores >= THRESHOLD
    male_rate = float(yes[~protected].mean())
    female_rate = float(yes[protected].mean())
    accuracy = float(np.mean(yes == Y))
    discrimination = abs(male_rate - female_rate)
    return {
        "Accuracy": accuracy,
        "Discrimination": discrimination,
        "Delta": accuracy - discrimination,
        "Male_Yes_Rate": male_rate,
        "Female_Yes_Rate": female_rate,
        "Mean_Score_Gap": float(abs(scores[~protected].mean() - scores[protected].mean())),
        "Near_Threshold_Count": int(np.sum(np.abs(scores - THRESHOLD) <= NEAR_THRESHOLD_MARGIN)),
    }


def score_on_sat_grid(params, criterion, path):
    # Counterfactual comparison: the SAME SAT scores for both genders.
    sat_grid = np.arange(int(SAT_MIN), int(SAT_MAX) + 1, 10)
    rows = []
    for sat in sat_grid:
        for gender, binary in (("M", 0.0), ("F", 1.0)):
            X_one = np.array([[(sat - SAT_MIN) / (SAT_MAX - SAT_MIN), binary]])
            score = float(predict(X_one, params)[0][0])
            rows.append({
                "Criterion": criterion, "SAT": sat, "Gender": gender,
                "LFR_Score": score, "Predicted_Admission": "Yes" if score >= THRESHOLD else "No",
            })
    pd.DataFrame(rows).to_csv(path, index=False)


# ============================================================
# SPLIT TRAINING / VALIDATION FOLDS / INDEPENDENT TEST
# ============================================================
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
df = load_data(INPUT_CSV)
stratum = strata(df)
counts = stratum.value_counts()
if counts.min() < 7:
    raise ValueError(
        "Every Gender x Admission x SAT band needs at least 7 distinct rows "
        "for the test split and five-fold CV. Smallest count: " + str(counts.min())
    )

pool_idx, test_idx = train_test_split(
    np.arange(len(df)), test_size=TEST_FRACTION,
    random_state=RANDOM_SEED, stratify=stratum,
)
pool = df.iloc[pool_idx].reset_index(drop=True)
test = df.iloc[test_idx].reset_index(drop=True)
X_pool, Y_pool, protected_pool = features(pool)
X_test, Y_test, protected_test = features(test)

cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
folds = list(cv.split(X_pool, strata(pool)))
split_labels = np.full(len(df), "test", dtype=object)
for fold_number, (_, valid_idx) in enumerate(folds, start=1):
    split_labels[pool_idx[valid_idx]] = f"cv_fold_{fold_number}"
split_output = df[["ID", "Gender", "SAT", "Admission"]].copy()
split_output["Split"] = split_labels
split_output.to_csv(SPLIT_CSV, index=False)

print("=" * 72)
print("TWO-FEATURE LFR: SAT + GENDER")
print("=" * 72)
print(f"Rows: {len(df)} | non-test pool: {len(pool)} | untouched test: {len(test)}")
print(f"Five CV folds: {[len(valid) for _, valid in folds]}")
print(f"Weight combinations: {len(AY_VALUES) * len(AZ_VALUES)}")


# ============================================================
# GRID SEARCH USING ONLY OUT-OF-FOLD PREDICTIONS
# ============================================================
search_rows = []
for ay, az in itertools.product(AY_VALUES, AZ_VALUES):
    out_of_fold_scores = np.empty(len(pool))
    fold_success = []
    fold_objectives = []
    for fold_number, (train_idx, valid_idx) in enumerate(folds, start=1):
        fit = fit_model(
            X_pool[train_idx], Y_pool[train_idx], protected_pool[train_idx],
            az, ay, initial_parameters(RANDOM_SEED),
        )
        if not np.isfinite(fit.fun):
            raise RuntimeError(f"Nonfinite objective: Az={az}, Ay={ay}, fold={fold_number}")
        out_of_fold_scores[valid_idx] = predict(X_pool[valid_idx], fit.x)[0]
        fold_success.append(bool(fit.success))
        fold_objectives.append(float(fit.fun))
    measurements = metrics(Y_pool, protected_pool, out_of_fold_scores)
    search_rows.append({
        "Az": az, "Ax": AX, "Ay": ay,
        **{f"CV_{name}": value for name, value in measurements.items()},
        "All_Folds_Converged": all(fold_success),
        "Mean_Fold_Objective": float(np.mean(fold_objectives)),
    })
    print(
        f"Az={az:>4}, Ay={ay:>4} | CV accuracy={measurements['Accuracy']:.4f}, "
        f"discrimination={measurements['Discrimination']:.4f}, "
        f"delta={measurements['Delta']:.4f}"
    )

search = pd.DataFrame(search_rows)
# Deterministic tie-breaking on discrete Yes/No metrics.
ranking_rules = {
    "min_discrimination": (
        ["CV_Discrimination", "CV_Accuracy", "Az", "Ay"],
        [True, False, True, True],
    ),
    "max_delta": (
        ["CV_Delta", "CV_Discrimination", "Az", "Ay"],
        [False, True, True, True],
    ),
}
selected = {}
for criterion, (columns, ascending) in ranking_rules.items():
    ranked = search.sort_values(columns, ascending=ascending, kind="stable")
    selected[criterion] = int(ranked.index[0])
    search[f"Selected_{criterion}"] = search.index == selected[criterion]
search.to_csv(GRID_CSV, index=False)


# ============================================================
# FIT SELECTED WEIGHTS ON NON-TEST POOL; EVALUATE TEST ONCE
# ============================================================
final_fits = {}
summary_rows = []
for criterion, selected_idx in selected.items():
    candidate = search.loc[selected_idx]
    az, ay = float(candidate.Az), float(candidate.Ay)
    if selected_idx not in final_fits:
        final_fits[selected_idx] = fit_model(
            X_pool, Y_pool, protected_pool, az, ay,
            initial_parameters(RANDOM_SEED),
        )
    fit = final_fits[selected_idx]
    if not np.isfinite(fit.fun):
        raise RuntimeError(f"Nonfinite final objective for {criterion}")
    prototypes, scores, alphas = unpack(fit.x)

    # Sort all prototype parameters together to preserve v1 ... vK meaning.
    order = np.argsort(prototypes[:, 0])
    prototypes, scores = prototypes[order], scores[order]
    params_sorted = np.concatenate((prototypes.ravel(), scores, alphas))

    prototype_table = pd.DataFrame({
        "Prototype": np.arange(1, K + 1),
        "SAT": prototypes[:, 0] * (SAT_MAX - SAT_MIN) + SAT_MIN,
        "Gender_Coordinate": prototypes[:, 1],
        "Admission_Score": scores,
        "Alpha_SAT": alphas[0], "Alpha_Gender": alphas[1],
        "Az": az, "Ax": AX, "Ay": ay,
    })
    prototype_path = OUTPUT_DIR / f"3-prototypes_4k_2d_{criterion}.csv"
    prototype_table.to_csv(prototype_path, index=False)

    test_scores, M_test = predict(X_test, params_sorted)
    representation = test[["ID", "Gender", "SAT", "Admission"]].copy()
    for k in range(K):
        representation[f"v{k + 1}"] = M_test[:, k]
    representation["LFR_Score"] = test_scores
    representation["Predicted_Admission"] = np.where(test_scores >= THRESHOLD, "Yes", "No")
    representation_path = OUTPUT_DIR / f"5-test_representations_4k_2d_{criterion}.csv"
    representation.to_csv(representation_path, index=False)

    grid_path = OUTPUT_DIR / f"7-same_sat_gender_grid_2d_{criterion}.csv"
    score_on_sat_grid(params_sorted, criterion, grid_path)

    cv_metrics = {key.removeprefix("CV_"): candidate[key] for key in search.columns if key.startswith("CV_")}
    test_metrics = metrics(Y_test, protected_test, test_scores)
    Lz, Lx, Ly, Lx_sat, Lx_gender = loss_terms(X_pool, Y_pool, protected_pool, prototypes, scores, alphas)
    summary_rows.append({
        "Criterion": criterion, "Az": az, "Ax": AX, "Ay": ay,
        **{f"CV_{key}": value for key, value in cv_metrics.items()},
        **{f"Test_{key}": value for key, value in test_metrics.items()},
        "Fit_Lz": Lz, "Fit_Lx": Lx, "Fit_Lx_SAT": Lx_sat,
        "Fit_Lx_Gender": Lx_gender, "Fit_Ly": Ly,
        "Weighted_Lz": az * Lz, "Weighted_Lx": AX * Lx, "Weighted_Ly": ay * Ly,
        "Final_Converged": bool(fit.success),
    })

    print("\n" + "=" * 72)
    print(f"SELECTED BY {criterion.upper()}: Az={az}, Ax={AX}, Ay={ay}")
    print("=" * 72)
    print(prototype_table.to_string(index=False))
    print("CV (out of fold):", {name: round(cv_metrics[name], 4) for name in ("Accuracy", "Discrimination", "Delta")})
    print("TEST (untouched):", {name: round(test_metrics[name], 4) for name in ("Accuracy", "Discrimination", "Delta")})
    print(f"Fit losses: Lz={Lz:.5f}, Lx={Lx:.5f} (SAT={Lx_sat:.5f}, Gender={Lx_gender:.5f}), Ly={Ly:.5f}")
    print(f"Scores near 0.5 on test (±{NEAR_THRESHOLD_MARGIN}): {test_metrics['Near_Threshold_Count']}")
    print(f"Saved: {prototype_path}, {representation_path}, {grid_path}")

pd.DataFrame(summary_rows).to_csv(SELECTED_CSV, index=False)
print(f"\nSaved split assignments: {SPLIT_CSV}")
print(f"Saved all weight results: {GRID_CSV}")
print(f"Saved selected-model summary: {SELECTED_CSV}")
