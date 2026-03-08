import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
data = pd.read_csv("dataset/heart.csv")

# Show first few rows (for verification)
print("Dataset Preview:")
print(data.head())

# Separate features and target
X = data.drop("target", axis=1)
y = data["target"]

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = RandomForestClassifier(n_estimators=100)

# Train model
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, predictions)

print("\nModel Accuracy:", accuracy)

# Save trained model
joblib.dump(model, "ai-model/heart_model.pkl")

print("\nModel saved as heart_model.pkl")