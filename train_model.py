
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Load student dataset
df = pd.read_csv("data/cleaned_students.csv")

# Define target: 1 means final grade below 10
y = (df["G3"] < 10).astype(int)

# Use information available before the final grade
features = [
    "G1",
    "G2",
    "studytime",
    "absences",
    "failures",
    "age"
]

X = df[features]

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Create and train the model
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000))
])

model.fit(X_train, y_train)

# Evaluate predictions
predictions = model.predict(X_test)

print("MODEL ACCURACY:", round(
    accuracy_score(y_test, predictions) * 100, 2
), "%")

print("\nCLASSIFICATION REPORT")
print(classification_report(y_test, predictions, zero_division=0))

print("\nCONFUSION MATRIX")
print(confusion_matrix(y_test, predictions))

# Save the trained model
import os
os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/risk_model.pkl")

print("\nModel saved successfully!")
