import pandas as pd

# Load CSV
df = pd.read_csv("lfr/1-training_data.csv")

# Define SAT ranges
bins = [799, 1199, 1399, 1499, 1600]
labels = ["800-1200", "1200-1400", "1400-1500", "1500-1600"]

df["SAT range"] = pd.cut(
    df["SAT"],
    bins=bins,
    labels=labels,
    include_lowest=True
)

# Define desired ordering
gender_order = ["M", "F"]
range_order = ["1500-1600", "1400-1500", "1200-1400", "800-1200"]
admission_order = ["Yes", "No"]

# Group and count
result = (
    df.groupby(
        ["Gender", "SAT range", "Admission"],
        observed=False
    )
    .size()
    .reset_index(name="#")
)

# Create all combinations so zero-count groups are included
all_combinations = pd.MultiIndex.from_product(
    [gender_order, range_order, admission_order],
    names=["Gender", "SAT range", "Admission"]
)

result = (
    result.set_index(["Gender", "SAT range", "Admission"])
    .reindex(all_combinations, fill_value=0)
    .reset_index()
)

# Keep only the combinations in your desired distribution
desired_combinations = [
    ("M", "1500-1600", "Yes"),
    ("M", "1400-1500", "Yes"),
    ("M", "1200-1400", "Yes"),
    ("M", "800-1200", "No"),
    ("F", "1500-1600", "Yes"),
    ("F", "1400-1500", "No"),
    ("F", "1200-1400", "No"),
    ("F", "800-1200", "No"),
]

result = result[
    result[["Gender", "SAT range", "Admission"]]
    .apply(tuple, axis=1)
    .isin(desired_combinations)
]

# Convert gender names
result["Gender"] = result["Gender"].map({
    "M": "Male",
    "F": "Female"
})

# Print Markdown table
print("| **Group** | **SAT range** | **Admission** | **#** |")
print("| --------- | ------------- | ------------- | ----- |")

for _, row in result.iterrows():
    print(
        f"| {row['Gender']:<9} "
        f"| {row['SAT range']:<13} "
        f"| {row['Admission']:<9} "
        f"| {row['#']} |"
    )