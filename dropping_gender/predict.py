import pandas as pd
import joblib

# =====================================================
# Step 1: Load trained model
# =====================================================
model = joblib.load("dropping_gender/model.pkl")

# =====================================================
# Step 2: Load evaluation dataset
# =====================================================
df = pd.read_csv("dropping_gender/evaluation_dataset.csv")

# =====================================================
# Step 3: Encode categorical variables
# =====================================================
hobby_map = {
    "Soccer": 0,
    "Dance": 1
}

# Keep original Hobby column
df["HobbyNum"] = df["Hobby"].map(hobby_map)

# =====================================================
# Step 4: Predict admissions
# =====================================================
X_test = df[["SAT", "HobbyNum"]]

predictions = model.predict(X_test)

# Convert predictions back to labels
df["Admission"] = [
    "Yes" if p == 1 else "No"
    for p in predictions
]

# Remove helper column before saving
df.drop(columns=["HobbyNum"], inplace=True)

# =====================================================
# Step 5: Save predictions
# =====================================================
df.to_csv(
    "dropping_gender/predicted_dataset.csv",
    index=False
)

print("Prediction completed.")
print("Saved to dropping_gender/predicted_dataset.csv")