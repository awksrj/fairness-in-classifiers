import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================

AWARE_PREDICTIONS_CSV = (
    "fairness_through_unawareness\\1-aware_experiment\\3-aware_predictions.csv"
)

UNAWARE_WITH_PROXIES_CSV = (
    "fairness_through_unawareness\\2-unaware_experiment\\3-unaware_predictions.csv"
)

UNAWARE_NO_PROXIES_CSV = (
    "fairness_through_unawareness\\3-unaware_noproxies_experiment\\3-unaware_no_proxies_predictions.csv"
)


# ============================================================
# FILES TO ANALYZE
# ============================================================

EXPERIMENTS = {
    "Aware": AWARE_PREDICTIONS_CSV,
    "Unaware - Proxies Retained": UNAWARE_WITH_PROXIES_CSV,
    "Unaware - Proxies Removed": UNAWARE_NO_PROXIES_CSV,
}


# ============================================================
# ANALYSIS FUNCTION
# ============================================================

def summarize_acceptance(name, csv_path):
    df = pd.read_csv(csv_path)

    male_df = df[
        df["Gender"] == "Male"
    ]

    female_df = df[
        df["Gender"] == "Female"
    ]

    male_accepted = (
        male_df["Predicted_Admission"] == 1
    ).sum()

    female_accepted = (
        female_df["Predicted_Admission"] == 1
    ).sum()

    male_total = len(male_df)
    female_total = len(female_df)

    male_acceptance_rate = (
        male_accepted / male_total
        if male_total > 0
        else 0
    )

    female_acceptance_rate = (
        female_accepted / female_total
        if female_total > 0
        else 0
    )

    print("=" * 70)
    print(name)
    print("=" * 70)

    print(
        f"Male accepted: "
        f"{male_accepted} / {male_total}"
    )

    print(
        f"Male acceptance rate: "
        f"{male_acceptance_rate:.4f} "
        f"({male_acceptance_rate * 100:.2f}%)"
    )

    print()

    print(
        f"Female accepted: "
        f"{female_accepted} / {female_total}"
    )

    print(
        f"Female acceptance rate: "
        f"{female_acceptance_rate:.4f} "
        f"({female_acceptance_rate * 100:.2f}%)"
    )

    print()

    print(
        f"Acceptance-rate gap "
        f"(Male - Female): "
        f"{male_acceptance_rate - female_acceptance_rate:.4f}"
    )

    acceptance_gap_individuals = (
        male_accepted
        - female_accepted
    )

    print(
        f"Acceptance gap by individuals "
        f"(Male - Female): "
        f"{acceptance_gap_individuals}"
    )

    print()


# ============================================================
# RUN ALL 3 EXPERIMENTS
# ============================================================

for experiment_name, csv_path in EXPERIMENTS.items():
    summarize_acceptance(
        experiment_name,
        csv_path
    )