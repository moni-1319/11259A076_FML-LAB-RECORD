import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

# Load dataset
data = pd.read_csv("data.csv")

# Features and Target
X = data[['Hours_Studied', 'Attendance', 'Assignments']]
y = data['Result']

# Split dataset into Training and Testing sets
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42
)

print("Training Data Size:", len(X_train))
print("Testing Data Size:", len(X_test))

# Create KNN Model
model = KNeighborsClassifier(n_neighbors=3)


# Perform 5-Fold Cross Validation
scores = cross_val_score(model, X, y, cv=5)

print("\nCross Validation Scores:")
print(scores)

print("\nAverage Accuracy:")
print(scores.mean())

