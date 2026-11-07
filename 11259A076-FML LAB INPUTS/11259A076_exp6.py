import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

# Load dataset
data = pd.read_csv("data.csv")

# Select features
X = data[['Hours_Studied', 'Marks']]

# Create linkage matrix
linked = linkage(X, method='ward')

# Plot Dendrogram
plt.figure(figsize=(8, 5))
dendrogram(linked)
plt.title("Dendrogram for Hierarchical Clustering")
plt.xlabel("Data Points")
plt.ylabel("Euclidean Distance")
plt.show()

# Apply Hierarchical Clustering
hc = AgglomerativeClustering(n_clusters=3)

clusters = hc.fit_predict(X)

# Add cluster labels to dataset
data['Cluster'] = clusters

print(data)

# Scatter Plot
plt.scatter(X['Hours_Studied'], X['Marks'], c=clusters)
plt.xlabel("Hours Studied")
plt.ylabel("Marks")
plt.title("Hierarchical Clustering")
plt.show()



