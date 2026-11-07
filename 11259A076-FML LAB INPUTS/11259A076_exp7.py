import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Create dummy data if 'data.csv' is missing
if not os.path.exists("data.csv"):
    print("'data.csv' not found. Creating a dummy dataset to resolve FileNotFoundError...")
    dummy_data = pd.DataFrame({
        'Hours_Studied': [2, 5, 8, 1, 10, 12, 4, 7, 9, 3],
        'Attendance': [60, 80, 95, 40, 99, 100, 70, 85, 90, 50],
        'Assignments': [1, 3, 5, 0, 5, 5, 2, 4, 5, 1],
        'Result': [0, 1, 1, 0, 1, 1, 0, 1, 1, 0]
    })
    dummy_data.to_csv("data.csv", index=False)

# Load dataset
data = pd.read_csv("data.csv")

# Input features
X = data[['Hours_Studied', 'Attendance', 'Assignments']]

# Target variable
y = data['Result']

# Split dataset into training and testing sets (Fixed 'Y' to 'y')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Predict test data
y_pred = model.predict(X_test)

# Evaluate performance
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature Importance Graph
importance = model.feature_importances_

plt.bar(X.columns, importance)
plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("Random Forest Feature Importance")
plt.show()