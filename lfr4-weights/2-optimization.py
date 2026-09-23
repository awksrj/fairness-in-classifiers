import itertools
import numpy as np
import pandas as pd
from scipy.optimize import minimize

# ============================================================
# CONFIGURATION
# ============================================================
CSV_PATH = "lfr4-weights/1-training_data.csv"
PROTOTYPE_OUTPUT_CSV = "lfr4-weights/3-prototypes_4k.csv"
REPRESENTATION_OUTPUT_CSV = "lfr4-weights/5-representations_4k.csv"
WEIGHT_SEARCH_OUTPUT_CSV = "lfr4-weights/4-weight_grid_search_4k.csv"

GENDER_COL = "Gender"
SAT_COL = "SAT"
ADMISSION_COL = "Admission"
K = 4
RANDOM_SEED = 42
VALIDATION_FRACTION = 0.25
PREDICTION_THRESHOLD = 0.5

# Paper's grid: Ax is fixed; every Ay/Az pair is evaluated on validation data.
AX = 0.01
AY_VALUES = [0.1, 0.5, 1.0, 5.0, 10.0]
AZ_VALUES = [0.0, 0.1, 0.5, 1.0, 5.0, 10.0]

# "min_discrimination" or "max_delta" (accuracy - discrimination)
SELECTION_CRITERION = "max_delta"
RETRAIN_ON_FULL_DATA = True
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


def stratified_train_validation_split(frame, validation_fraction, seed):
    """Split within Gender x Admission strata when a stratum has >= 2 rows."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("VALIDATION_FRACTION must be between 0 and 1.")

    rng = np.random.default_rng(seed)
    train_indices, validation_indices = [], []
    for _, group in frame.groupby([GENDER_COL, ADMISSION_COL], sort=True):
        indices = group.index.to_numpy().copy()
        rng.shuffle(indices)
        if len(indices) == 1:
            n_validation = 0
        else:
            n_validation = int(round(len(indices) * validation_fraction))
            n_validation = min(max(n_validation, 1), len(indices) - 1)
        validation_indices.extend(indices[:n_validation])
        train_indices.extend(indices[n_validation:])

    if not validation_indices:
        raise ValueError("No validation rows could be created; add more data.")

    train_df = frame.loc[sorted(train_indices)].reset_index(drop=True)
    validation_df = frame.loc[sorted(validation_indices)].reset_index(drop=True)
    if train_df[GENDER_COL].nunique() < 2:
        raise ValueError("Training split must contain both gender groups.")
    if validation_df[GENDER_COL].nunique() < 2:
        raise ValueError("Validation split must contain both gender groups.")
    return train_df, validation_df


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
# LOAD AND SPLIT DATA
# ============================================================
df = pd.read_csv(CSV_PATH)
required = {"ID", GENDER_COL, SAT_COL, ADMISSION_COL}
if required - set(df.columns):
    raise ValueError(f"Missing required columns: {sorted(required - set(df.columns))}")
if (~df[GENDER_COL].isin(["M", "F"])).any():
    bad = sorted(df.loc[~df[GENDER_COL].isin(["M", "F"]), GENDER_COL].astype(str).unique())
    raise ValueError(f"Gender values must be M/F; found: {bad}")

train_df, validation_df = stratified_train_validation_split(
    df, VALIDATION_FRACTION, RANDOM_SEED
)

# Fit normalization on training data only during model selection.
train_sat_min = train_df[SAT_COL].astype(float).min()
train_sat_max = train_df[SAT_COL].astype(float).max()
X_train = normalize_sat(train_df[SAT_COL], train_sat_min, train_sat_max)
Y_train = encode_labels(train_df[ADMISSION_COL])
protected_train = train_df[GENDER_COL].eq("F").to_numpy()
X_validation = normalize_sat(validation_df[SAT_COL], train_sat_min, train_sat_max)
Y_validation = encode_labels(validation_df[ADMISSION_COL])
protected_validation = validation_df[GENDER_COL].eq("F").to_numpy()


# ============================================================
# PAPER-STYLE WEIGHT GRID SEARCH
# ============================================================
if SELECTION_CRITERION not in {"min_discrimination", "max_delta"}:
    raise ValueError("SELECTION_CRITERION must be min_discrimination or max_delta.")

print("=" * 72)
print("PAPER-STYLE LFR WEIGHT SEARCH")
print("=" * 72)
print(f"Training rows: {len(train_df)}; validation rows: {len(validation_df)}")
print(f"Fixed Ax: {AX}; selection criterion: {SELECTION_CRITERION}\n")

initial_params = make_initial_params(RANDOM_SEED)
search_rows, search_results = [], []

for ay, az in itertools.product(AY_VALUES, AZ_VALUES):
    search_result = fit_lfr(
        X_train, Y_train, protected_train, az, AX, ay, initial_params
    )
    validation_scores = predict_scores(X_validation, search_result.x)
    accuracy, discrimination, delta = performance_metrics(
        Y_validation, protected_validation, validation_scores, PREDICTION_THRESHOLD
    )
    search_rows.append({
        "Az": az,
        "Ax": AX,
        "Ay": ay,
        "Validation_Accuracy": accuracy,
        "Validation_Discrimination": discrimination,
        "Validation_Delta": delta,
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

# Deterministic tie-breaking is necessary because binary validation decisions
# often give several configurations identical accuracy/discrimination values.
if SELECTION_CRITERION == "min_discrimination":
    ranked = search_df.sort_values(
        ["Validation_Discrimination", "Validation_Accuracy", "Az", "Ay"],
        ascending=[True, False, True, True], kind="stable"
    )
else:
    ranked = search_df.sort_values(
        ["Validation_Delta", "Validation_Discrimination", "Az", "Ay"],
        ascending=[False, True, True, True], kind="stable"
    )

selected_index = ranked.index[0]
selected_row = search_df.loc[selected_index]
selected_az = float(selected_row["Az"])
selected_ax = float(selected_row["Ax"])
selected_ay = float(selected_row["Ay"])
search_df["Selected"] = False
search_df.loc[selected_index, "Selected"] = True
search_df.to_csv(WEIGHT_SEARCH_OUTPUT_CSV, index=False)

print("\nSELECTED WEIGHTS")
print(
    f"Az={selected_az}, Ax={selected_ax}, Ay={selected_ay} | "
    f"accuracy={selected_row['Validation_Accuracy']:.4f}, "
    f"discrimination={selected_row['Validation_Discrimination']:.4f}, "
    f"delta={selected_row['Validation_Delta']:.4f}\n"
)


# ============================================================
# RETRAIN SELECTED CONFIGURATION
# ============================================================
if RETRAIN_ON_FULL_DATA:
    final_df = df.reset_index(drop=True)
    SAT_MIN = final_df[SAT_COL].astype(float).min()
    SAT_MAX = final_df[SAT_COL].astype(float).max()
    X = normalize_sat(final_df[SAT_COL], SAT_MIN, SAT_MAX)
    Y = encode_labels(final_df[ADMISSION_COL])
    protected = final_df[GENDER_COL].eq("F").to_numpy()
    print("Retraining selected weights on the full dataset...")
    result = fit_lfr(
        X, Y, protected, selected_az, selected_ax, selected_ay,
        make_initial_params(RANDOM_SEED)
    )
else:
    final_df = train_df
    SAT_MIN, SAT_MAX = train_sat_min, train_sat_max
    X, Y, protected = X_train, Y_train, protected_train
    result = search_results[selected_index]

prototypes_normalized, prototype_scores, alpha = unpack_params(result.x)
prototypes_sat = prototypes_normalized * (SAT_MAX - SAT_MIN) + SAT_MIN

# Sort all prototype parameters together, preserving membership-column meaning.
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


# ============================================================
# SAVE FINAL OUTPUTS
# ============================================================
prototype_df = pd.DataFrame({
    "Prototype": np.arange(1, K + 1),
    "SAT": prototypes_sat,
    "Admission_Score": prototype_scores,
    "Alpha": alpha,
    "Az": selected_az,
    "Ax": selected_ax,
    "Ay": selected_ay,
})
prototype_df.to_csv(PROTOTYPE_OUTPUT_CSV, index=False)

representation_df = final_df[["ID", GENDER_COL, SAT_COL, ADMISSION_COL]].copy()
for k in range(K):
    representation_df[f"v{k + 1}"] = M[:, k]
representation_df["LFR_Score"] = Y_hat
representation_df["Predicted_Admission"] = predicted_admission
representation_df.to_csv(REPRESENTATION_OUTPUT_CSV, index=False)


# ============================================================
# REPORT FINAL MODEL
# ============================================================
print("\n" + "=" * 72)
print("FINAL LFR MODEL")
print("=" * 72)
print(prototype_df.to_string(index=False))

protected_distribution = M[protected].mean(axis=0)
unprotected_distribution = M[~protected].mean(axis=0)
print("\nGROUP PROTOTYPE DISTRIBUTIONS")
for k in range(K):
    print(
        f"Prototype {k + 1}: Male={unprotected_distribution[k]:.4f}, "
        f"Female={protected_distribution[k]:.4f}"
    )

final_accuracy, final_discrimination, final_delta = performance_metrics(
    Y, protected, Y_hat, PREDICTION_THRESHOLD
)
print("\nFINAL LOSSES AND METRICS")
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

print(f"\nClassification threshold: {PREDICTION_THRESHOLD:.2f}")
print(f"Saved grid-search results to: {WEIGHT_SEARCH_OUTPUT_CSV}")
print(f"Saved prototypes to: {PROTOTYPE_OUTPUT_CSV}")
print(f"Saved representations to: {REPRESENTATION_OUTPUT_CSV}")
