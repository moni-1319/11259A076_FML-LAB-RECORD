import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
data = pd.read_csv("data.csv")

# Features (Input)
X = data[['Hours_Studied', 'Attendance', 'Assignments']]

# Target (Output)
y = data['Result']

# Split data into 80% Training and 20% Testing
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42
)

# Display sizes
print("Training Data:")
print(X_train)

print("\nTesting Data:")

print(X_test)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))
