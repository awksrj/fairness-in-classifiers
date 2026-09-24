import itertools
import numpy as np
import pandas as pd
from scipy.optimize import minimize

# ============================================================
# CONFIGURATION
# ============================================================
CSV_PATH = "lfr4-weights/1-training_data.csv"
PROTOTYPE_OUTPUT_CSV = "lfr4-weights/3-prototypes_4k_{criterion}.csv"
REPRESENTATION_OUTPUT_CSV = "lfr4-weights/5-representations_4k_{criterion}.csv"
WEIGHT_SEARCH_OUTPUT_CSV = "lfr4-weights/4-weight_grid_search_4k.csv"

GENDER_COL = "Gender"
SAT_COL = "SAT"
ADMISSION_COL = "Admission"
K = 4
RANDOM_SEED = 42
PREDICTION_THRESHOLD = 0.5

# Paper's weight grid, evaluated here on the same toy data used for fitting.
AX = 0.01
AY_VALUES = [0.1, 0.5, 1.0, 5.0, 10.0]
AZ_VALUES = [0.0, 0.1, 0.5, 1.0, 5.0, 10.0]

# Run both selection rules on the same 30 fitted candidates.
SELECTION_CRITERIA = ("min_discrimination", "max_delta")
MAX_ITER = 5000
EPSILON = 1e-10


# ============================================================
# DATA HELPERS
# ============================================================
def encode_labels(series):
    encoded = series.map({"Yes": 1.0, "No": 0.0})
    if encoded.isna().any():
        bad = sorted(series[encoded.isna()].astype(str).unique())
        raise ValueError(f"Admission values must be Yes/No; found: {bad}")
    return encoded.to_numpy(dtype=float)


def normalize_sat(values, sat_min, sat_max):
    if sat_max <= sat_min:
        raise ValueError("SAT_MAX must be greater than SAT_MIN.")
    return (np.asarray(values, dtype=float) - sat_min) / (sat_max - sat_min)


# ============================================================
# LFR MODEL AND LOSSES
# ============================================================
def calculate_membership(X, prototypes, alpha):
    """M_nk = P(Z = k | x_n)."""
    distances = alpha * (X[:, None] - prototypes[None, :]) ** 2
    logits = -distances
    logits -= logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / (exp_logits.sum(axis=1, keepdims=True) + EPSILON)


def fairness_loss(M, protected):
    """Lz = sum_k |mean(M_k|protected) - mean(M_k|unprotected)|."""
    protected = np.asarray(protected, dtype=bool)
    if protected.sum() == 0 or (~protected).sum() == 0:
        raise ValueError("Fairness loss requires both protected groups.")
    return np.abs(
        M[protected].mean(axis=0) - M[~protected].mean(axis=0)
    ).sum()


def reconstruction_loss(X, M, prototypes):
    X_hat = np.sum(M * prototypes[None, :], axis=1)
    return np.mean((X - X_hat) ** 2)


def classification_loss(Y, M, prototype_scores):
    Y_hat = np.sum(M * prototype_scores[None, :], axis=1)
    Y_hat = np.clip(Y_hat, EPSILON, 1.0 - EPSILON)
    return -np.mean(Y * np.log(Y_hat) + (1.0 - Y) * np.log(1.0 - Y_hat))


def unpack_params(params):
    return params[:K], params[K:2 * K], params[-1]


def make_initial_params(seed):
    rng = np.random.default_rng(seed)
    prototypes = np.linspace(0.05, 0.95, K)
    prototypes = np.clip(prototypes + rng.normal(0, 0.01, K), 0.0, 1.0)
    scores = rng.uniform(0.25, 0.75, K)
    return np.concatenate([prototypes, scores, [10.0]])


BOUNDS = (
    [(0.0, 1.0) for _ in range(K)]
    + [(0.0, 1.0) for _ in range(K)]
    + [(0.01, 100.0)]
)


def fit_lfr(X, Y, protected, az, ax, ay, initial_params):
    """Fit one LFR model for one (Az, Ax, Ay) configuration."""
    def objective(params):
        prototypes, scores, alpha = unpack_params(params)
        M = calculate_membership(X, prototypes, alpha)
        return (
            az * fairness_loss(M, protected)
            + ax * reconstruction_loss(X, M, prototypes)
            + ay * classification_loss(Y, M, scores)
        )

    return minimize(
        objective,
        x0=initial_params.copy(),
        method="L-BFGS-B",
        bounds=BOUNDS,
        options={"maxiter": MAX_ITER, "ftol": 1e-12, "gtol": 1e-8, "maxls": 50},
    )


def predict_scores(X, params):
    prototypes, scores, alpha = unpack_params(params)
    M = calculate_membership(X, prototypes, alpha)
    return np.sum(M * scores[None, :], axis=1)


def performance_metrics(Y, protected, scores, threshold):
    predictions = (scores >= threshold).astype(float)
    accuracy = np.mean(predictions == Y)
    protected = np.asarray(protected, dtype=bool)
    protected_rate = predictions[protected].mean()
    unprotected_rate = predictions[~protected].mean()
    discrimination = abs(protected_rate - unprotected_rate)
    return accuracy, discrimination, accuracy - discrimination


# ============================================================
# LOAD ALL DATA (IN-SAMPLE THOUGHT EXPERIMENT)
# ============================================================
df = pd.read_csv(CSV_PATH)
required = {"ID", GENDER_COL, SAT_COL, ADMISSION_COL}
if required - set(df.columns):
    raise ValueError(f"Missing required columns: {sorted(required - set(df.columns))}")
if (~df[GENDER_COL].isin(["M", "F"])).any():
    bad = sorted(df.loc[~df[GENDER_COL].isin(["M", "F"]), GENDER_COL].astype(str).unique())
    raise ValueError(f"Gender values must be M/F; found: {bad}")

df = df.reset_index(drop=True)
sat_min = df[SAT_COL].astype(float).min()
sat_max = df[SAT_COL].astype(float).max()
X = normalize_sat(df[SAT_COL], sat_min, sat_max)
Y = encode_labels(df[ADMISSION_COL])
protected = df[GENDER_COL].eq("F").to_numpy()
if protected.all() or not protected.any():
    raise ValueError("The dataset must contain both Male and Female students.")


# ============================================================
# WEIGHT GRID SEARCH (FIT AND SCORE ON ALL ROWS)
# ============================================================
print("=" * 72)
print("LFR WEIGHT SEARCH — IN-SAMPLE TOY EXPERIMENT")
print("=" * 72)
print(f"Rows used for both fitting and scoring: {len(df)}")
print(f"Fixed Ax: {AX}; selection criteria: {', '.join(SELECTION_CRITERIA)}\n")

initial_params = make_initial_params(RANDOM_SEED)
search_rows, search_results = [], []

for ay, az in itertools.product(AY_VALUES, AZ_VALUES):
    search_result = fit_lfr(
        X, Y, protected, az, AX, ay, initial_params
    )
    in_sample_scores = predict_scores(X, search_result.x)
    accuracy, discrimination, delta = performance_metrics(
        Y, protected, in_sample_scores, PREDICTION_THRESHOLD
    )
    search_rows.append({
        "Az": az,
        "Ax": AX,
        "Ay": ay,
        "In_Sample_Accuracy": accuracy,
        "In_Sample_Discrimination": discrimination,
        "In_Sample_Delta": delta,
        "Training_Objective": search_result.fun,
        "Optimization_Success": search_result.success,
        "Iterations": search_result.nit,
    })
    search_results.append(search_result)
    print(
        f"Az={az:>4}, Ax={AX:.2f}, Ay={ay:>4} | accuracy={accuracy:.4f}, "
        f"discrimination={discrimination:.4f}, delta={delta:.4f}"
    )

search_df = pd.DataFrame(search_rows)

# Select two winners from the SAME grid; ties prefer the secondary metric,
# then smaller Az and Ay. Both rules may select the same configuration.
rankings = {
    "min_discrimination": (
        ["In_Sample_Discrimination", "In_Sample_Accuracy", "Az", "Ay"],
        [True, False, True, True],
    ),
    "max_delta": (
        ["In_Sample_Delta", "In_Sample_Discrimination", "Az", "Ay"],
        [False, True, True, True],
    ),
}
selected_indices = {}
for criterion in SELECTION_CRITERIA:
    columns, ascending = rankings[criterion]
    ranked = search_df.sort_values(columns, ascending=ascending, kind="stable")
    selected_indices[criterion] = ranked.index[0]
    search_df[f"Selected_{criterion}"] = search_df.index == selected_indices[criterion]

search_df.to_csv(WEIGHT_SEARCH_OUTPUT_CSV, index=False)

print("\nSELECTED WEIGHTS (IN-SAMPLE)")
for criterion, index in selected_indices.items():
    row = search_df.loc[index]
    print(
        f"{criterion}: Az={row['Az']}, Ax={row['Ax']}, Ay={row['Ay']} | "
        f"accuracy={row['In_Sample_Accuracy']:.4f}, "
        f"discrimination={row['In_Sample_Discrimination']:.4f}, "
        f"delta={row['In_Sample_Delta']:.4f}"
    )


# ============================================================
# SAVE BOTH SELECTED MODELS WITHOUT REFITTING
# ============================================================
for criterion in SELECTION_CRITERIA:
    selected_index = selected_indices[criterion]
    row = search_df.loc[selected_index]
    selected_az, selected_ax, selected_ay = (
        float(row["Az"]), float(row["Ax"]), float(row["Ay"])
    )
    result = search_results[selected_index]

    prototypes_normalized, prototype_scores, alpha = unpack_params(result.x)
    prototypes_sat = prototypes_normalized * (sat_max - sat_min) + sat_min

    # Sort parameters together so membership columns match output prototypes.
    order = np.argsort(prototypes_sat)
    prototypes_sat = prototypes_sat[order]
    prototypes_normalized = prototypes_normalized[order]
    prototype_scores = prototype_scores[order]

    M = calculate_membership(X, prototypes_normalized, alpha)
    Y_hat = np.sum(M * prototype_scores[None, :], axis=1)
    predicted_admission = np.where(Y_hat >= PREDICTION_THRESHOLD, "Yes", "No")

    Lz = fairness_loss(M, protected)
    Lx = reconstruction_loss(X, M, prototypes_normalized)
    Ly = classification_loss(Y, M, prototype_scores)
    total_loss = selected_az * Lz + selected_ax * Lx + selected_ay * Ly

    prototype_df = pd.DataFrame({
        "Prototype": np.arange(1, K + 1),
        "SAT": prototypes_sat,
        "Admission_Score": prototype_scores,
        "Alpha": alpha,
        "Az": selected_az,
        "Ax": selected_ax,
        "Ay": selected_ay,
    })
    prototype_path = PROTOTYPE_OUTPUT_CSV.format(criterion=criterion)
    prototype_df.to_csv(prototype_path, index=False)

    representation_df = df[["ID", GENDER_COL, SAT_COL, ADMISSION_COL]].copy()
    for k in range(K):
        representation_df[f"v{k + 1}"] = M[:, k]
    representation_df["LFR_Score"] = Y_hat
    representation_df["Predicted_Admission"] = predicted_admission
    representation_path = REPRESENTATION_OUTPUT_CSV.format(criterion=criterion)
    representation_df.to_csv(representation_path, index=False)

    print("\n" + "=" * 72)
    print(f"FINAL LFR MODEL: {criterion}")
    print("=" * 72)
    print(prototype_df.to_string(index=False))
    male_distribution = M[~protected].mean(axis=0)
    female_distribution = M[protected].mean(axis=0)
    print("\nGROUP PROTOTYPE DISTRIBUTIONS")
    for k in range(K):
        print(
            f"Prototype {k + 1}: Male={male_distribution[k]:.4f}, "
            f"Female={female_distribution[k]:.4f}"
        )

    final_accuracy, final_discrimination, final_delta = performance_metrics(
        Y, protected, Y_hat, PREDICTION_THRESHOLD
    )
    print("\nIN-SAMPLE LOSSES AND METRICS (ON DATA USED FOR FITTING)")
    print(f"Fairness loss Lz:       {Lz:.6f}")
    print(f"Reconstruction loss Lx: {Lx:.6f}")
    print(f"Classification loss Ly: {Ly:.6f}")
    print(f"Weighted total:         {total_loss:.6f}")
    print(f"Accuracy:               {final_accuracy:.6f}")
    print(f"Discrimination:         {final_discrimination:.6f}")
    print(f"Delta:                  {final_delta:.6f}")
    print("\nOPTIMIZATION STATUS")
    print(f"Success: {result.success}")
    print(f"Message: {result.message}")
    print(f"Iterations: {result.nit}")
    print(f"Function evaluations: {result.nfev}")
    print(f"Saved prototypes to: {prototype_path}")
    print(f"Saved representations to: {representation_path}")

print(f"\nClassification threshold: {PREDICTION_THRESHOLD:.2f}")
print(f"Saved grid-search results to: {WEIGHT_SEARCH_OUTPUT_CSV}")
