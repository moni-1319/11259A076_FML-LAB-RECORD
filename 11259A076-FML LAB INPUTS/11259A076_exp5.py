import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load dataset
data = pd.read_csv("data.csv")

# Select features
X = data[['Hours_Studied', 'Marks']]

# Create K-Means model
kmeans = KMeans(n_clusters=3, random_state=42)

# Fit the model
kmeans.fit(X)

# Predict cluster labels
clusters = kmeans.labels_

# Add cluster column to dataset
data['Cluster'] = clusters

# Display clustered data
print(data)

# Plot clusters
plt.scatter(X['Hours_Studied'], X['Marks'], c=clusters)

# Plot centroids
plt.scatter(
kmeans.cluster_centers_[:, 0],
kmeans.cluster_centers_[:, 1],
marker='X',
s=200,
label='Centroids'
)

plt.xlabel("Hours Studied")
plt.ylabel("Marks")
plt.title("K-Means Clustering")
plt.legend()
plt.show()
