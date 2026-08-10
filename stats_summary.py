import pandas as pd

# ============================================================
# CONFIGURATION — change these for different datasets
# ============================================================

CSV_FILE = "fairness_through_awareness_redo/training_dataset_outliner.csv"
# CSV_FILE = "fairness_through_awareness_redo/fta_training_result.csv"

ID_COL = "ID"
GENDER_COL = "Gender"
SAT_COL = "SAT"
HOBBY_COL = "Hobby"
ADMISSION_COL = "Admission"       # e.g. "Admission", "FTA_Prediction"

# Values used in the admission column
ACCEPTED_VALUE = "Yes"
REJECTED_VALUE = "No"

# Gender values
MALE_VALUE = "M"
FEMALE_VALUE = "F"

# Hobby values
SOCCER_VALUE = "Soccer"
DANCE_VALUE = "Dance"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def percentage(part, whole):
    if whole == 0:
        return 0.0
    return part / whole * 100


def accepted_sat_range(group):
    """
    Return the SAT range among accepted applicants.
    """
    accepted = group[
        group[ADMISSION_COL] == ACCEPTED_VALUE
    ][SAT_COL]

    if accepted.empty:
        return "None"

    return f"{accepted.min()}–{accepted.max()}"


def decision_boundary(group):
    """
    Calculate the decision boundary as the midpoint between:

        highest rejected SAT
        lowest accepted SAT

    Example:
        rejected: 950
        accepted: 1000

        boundary = (950 + 1000) / 2 = 975

    If one side is missing, return None.
    """

    accepted_sat = group[
        group[ADMISSION_COL] == ACCEPTED_VALUE
    ][SAT_COL]

    rejected_sat = group[
        group[ADMISSION_COL] == REJECTED_VALUE
    ][SAT_COL]

    if accepted_sat.empty or rejected_sat.empty:
        return None

    lowest_accepted = accepted_sat.min()
    highest_rejected = rejected_sat.max()

    return (lowest_accepted + highest_rejected) / 2


# ============================================================
# STATISTICS
# ============================================================

def print_stats(csv_file):

    df = pd.read_csv(csv_file)

    # --------------------------------------------------------
    # Total applicants
    # --------------------------------------------------------

    total = len(df)

    total_male = len(
        df[df[GENDER_COL] == MALE_VALUE]
    )

    total_female = len(
        df[df[GENDER_COL] == FEMALE_VALUE]
    )

    male_pct = percentage(total_male, total)
    female_pct = percentage(total_female, total)

    # --------------------------------------------------------
    # Total accepted applicants
    # --------------------------------------------------------

    accepted = df[
        df[ADMISSION_COL] == ACCEPTED_VALUE
    ]

    total_accepted = len(accepted)
    accepted_pct = percentage(total_accepted, total)

    # --------------------------------------------------------
    # Accepted applicants by gender
    # --------------------------------------------------------

    male_accepted = accepted[
        accepted[GENDER_COL] == MALE_VALUE
    ]

    female_accepted = accepted[
        accepted[GENDER_COL] == FEMALE_VALUE
    ]

    total_male_accepted = len(male_accepted)
    total_female_accepted = len(female_accepted)

    # --------------------------------------------------------
    # Accepted applicants by Gender × Hobby
    # --------------------------------------------------------

    male_soccer = len(
        male_accepted[
            male_accepted[HOBBY_COL] == SOCCER_VALUE
        ]
    )

    male_dance = len(
        male_accepted[
            male_accepted[HOBBY_COL] == DANCE_VALUE
        ]
    )

    female_soccer = len(
        female_accepted[
            female_accepted[HOBBY_COL] == SOCCER_VALUE
        ]
    )

    female_dance = len(
        female_accepted[
            female_accepted[HOBBY_COL] == DANCE_VALUE
        ]
    )

    # --------------------------------------------------------
    # Hobby statistics
    # --------------------------------------------------------

    soccer = df[
        df[HOBBY_COL] == SOCCER_VALUE
    ]

    dance = df[
        df[HOBBY_COL] == DANCE_VALUE
    ]

    soccer_total = len(soccer)
    dance_total = len(dance)

    soccer_accepted = len(
        soccer[
            soccer[ADMISSION_COL] == ACCEPTED_VALUE
        ]
    )

    dance_accepted = len(
        dance[
            dance[ADMISSION_COL] == ACCEPTED_VALUE
        ]
    )

    soccer_pct = percentage(
        soccer_total,
        total
    )

    dance_pct = percentage(
        dance_total,
        total
    )

    soccer_accepted_pct = percentage(
        soccer_accepted,
        soccer_total
    )

    dance_accepted_pct = percentage(
        dance_accepted,
        dance_total
    )

    # --------------------------------------------------------
    # Hobby × Gender distribution
    # --------------------------------------------------------

    soccer_male = len(
        soccer[
            soccer[GENDER_COL] == MALE_VALUE
        ]
    )

    soccer_female = len(
        soccer[
            soccer[GENDER_COL] == FEMALE_VALUE
        ]
    )

    dance_male = len(
        dance[
            dance[GENDER_COL] == MALE_VALUE
        ]
    )

    dance_female = len(
        dance[
            dance[GENDER_COL] == FEMALE_VALUE
        ]
    )

    # --------------------------------------------------------
    # SAT ranges of accepted applicants
    # --------------------------------------------------------

    male_sat_range = accepted_sat_range(
        df[df[GENDER_COL] == MALE_VALUE]
    )

    female_sat_range = accepted_sat_range(
        df[df[GENDER_COL] == FEMALE_VALUE]
    )

    # --------------------------------------------------------
    # Decision boundaries
    # --------------------------------------------------------

    male_soccer_boundary = decision_boundary(
        df[
            (df[GENDER_COL] == MALE_VALUE) &
            (df[HOBBY_COL] == SOCCER_VALUE)
        ]
    )

    female_soccer_boundary = decision_boundary(
        df[
            (df[GENDER_COL] == FEMALE_VALUE) &
            (df[HOBBY_COL] == SOCCER_VALUE)
        ]
    )

    male_dance_boundary = decision_boundary(
        df[
            (df[GENDER_COL] == MALE_VALUE) &
            (df[HOBBY_COL] == DANCE_VALUE)
        ]
    )

    female_dance_boundary = decision_boundary(
        df[
            (df[GENDER_COL] == FEMALE_VALUE) &
            (df[HOBBY_COL] == DANCE_VALUE)
        ]
    )

    # --------------------------------------------------------
    # Gender gaps
    #
    # Because Gender is encoded as:
    # Male = 0
    # Female = 1
    #
    # the gender gap is:
    #
    # Female boundary - Male boundary
    #
    # which is also the slope of the decision boundary.
    # --------------------------------------------------------

    if (
        male_soccer_boundary is not None
        and female_soccer_boundary is not None
    ):
        soccer_gap = (
            female_soccer_boundary
            - male_soccer_boundary
        )
    else:
        soccer_gap = None

    if (
        male_dance_boundary is not None
        and female_dance_boundary is not None
    ):
        dance_gap = (
            female_dance_boundary
            - male_dance_boundary
        )
    else:
        dance_gap = None

    # ========================================================
    # OUTPUT
    # ========================================================

    print(
        f"Number of total applicants: "
        f"{total}, "
        f"Male: {total_male} ({male_pct:.1f}%), "
        f"Female: {total_female} ({female_pct:.1f}%)"
    )

    print()

    print(
        f"Number of total accepted applicants: "
        f"{total_accepted} ({accepted_pct:.1f}%)"
    )

    print()

    print("Soccer:")

    print(
        f"- Total applicants: "
        f"{soccer_total} ({soccer_pct:.1f}%), "
        f"Accepted: {soccer_accepted} "
        f"({soccer_accepted_pct:.1f}% of Soccer)"
    )

    print(
        f"- Male accepted: "
        f"{male_soccer}, "
        f"ranging from SAT "
        f"{accepted_sat_range(soccer[soccer[GENDER_COL] == MALE_VALUE])}"
    )

    print(
        f"- Female accepted: "
        f"{female_soccer}, "
        f"ranging from SAT "
        f"{accepted_sat_range(soccer[soccer[GENDER_COL] == FEMALE_VALUE])}"
    )

    print()

    print("Dance:")

    print(
        f"- Total applicants: "
        f"{dance_total} ({dance_pct:.1f}%), "
        f"Accepted: {dance_accepted} "
        f"({dance_accepted_pct:.1f}% of Dance)"
    )

    print(
        f"- Male accepted: "
        f"{male_dance}, "
        f"ranging from SAT "
        f"{accepted_sat_range(dance[dance[GENDER_COL] == MALE_VALUE])}"
    )

    print(
        f"- Female accepted: "
        f"{female_dance}, "
        f"ranging from SAT "
        f"{accepted_sat_range(dance[dance[GENDER_COL] == FEMALE_VALUE])}"
    )

    print()

    print(
        f"Soccer distribution: "
        f"{soccer_male} male + {soccer_female} female"
    )

    print(
        f"Dance distribution: "
        f"{dance_male} male + {dance_female} female"
    )

    print()

    print("Decision boundary:")

    if male_soccer_boundary is not None:
        male_soccer_boundary = round(
            male_soccer_boundary, 1
        )

    if female_soccer_boundary is not None:
        female_soccer_boundary = round(
            female_soccer_boundary, 1
        )

    if male_dance_boundary is not None:
        male_dance_boundary = round(
            male_dance_boundary, 1
        )

    if female_dance_boundary is not None:
        female_dance_boundary = round(
            female_dance_boundary, 1
        )

    if soccer_gap is not None:
        soccer_gap = round(soccer_gap, 1)

    if dance_gap is not None:
        dance_gap = round(dance_gap, 1)

    print(
        f"- Soccer: "
        f"Male ≈ {male_soccer_boundary} SAT, "
        f"Female ≈ {female_soccer_boundary} SAT"
    )

    print(
        f"- Dance: "
        f"Male ≈ {male_dance_boundary} SAT, "
        f"Female ≈ {female_dance_boundary} SAT"
    )

    print(
        f"- Gender gap: "
        f"Soccer ≈ {soccer_gap} SAT, "
        f"Dance ≈ {dance_gap} SAT"
    )


# ============================================================
# RUN
# ============================================================

print_stats(CSV_FILE)