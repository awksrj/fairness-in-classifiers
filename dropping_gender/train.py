import pandas as pd
import joblib
from sklearn.tree import DecisionTreeClassifier

# =====================================================
# Step 1: Load training dataset
# =====================================================
df = pd.read_csv("dropping_gender/training_dataset.csv")

# =====================================================
# Step 2: Encode categorical variables
# =====================================================
hobby_map = {
    "Soccer": 0,
    "Dance": 1
}

label_map = {
    "No": 0,
    "Yes": 1
}

# Keep original columns untouched
df["HobbyNum"] = df["Hobby"].map(hobby_map)
df["AdmissionNum"] = df["Admission"].map(label_map)

# =====================================================
# Step 3: Prepare training data
# (Gender intentionally dropped)
# =====================================================
X_train = df[["SAT", "HobbyNum"]]
y_train = df["AdmissionNum"]

# =====================================================
# Step 4: Train classifier
# =====================================================
model = DecisionTreeClassifier(
    max_depth=3,
    random_state=42
)

model.fit(X_train, y_train)

# =====================================================
# Step 5: Save trained model
# =====================================================
joblib.dump(model, "dropping_gender/model.pkl")

print("Training completed.")
print("Model saved to dropping_gender/model.pkl")

accuracy = model.score(X_train, y_train)
print(f"Training Accuracy: {accuracy:.3f}")