# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load dataset (download from: https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)
data = pd.read_csv("Mall_Customers.csv")

# Preview dataset
print("Dataset Preview:")
print(data.head())

# Select relevant features for clustering (Annual Income and Spending Score)
X = data[['Annual Income (k$)', 'Spending Score (1-100)']]

# Scale data (important for clustering)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Find optimal number of clusters using Elbow Method
wcss = []  # Within-Cluster Sum of Squares
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Plot Elbow Curve
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# Apply K-Means with optimal clusters (assume 5 clusters based on elbow curve)
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Add cluster labels to dataset
data['Cluster'] = clusters

# Visualize Clusters
plt.figure(figsize=(8, 5))
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)',
                hue='Cluster', palette='viridis', data=data, s=60)
plt.title('Customer Segments')
plt.show()

# Save clustered data (optional)
data.to_csv('clustered_customers.csv', index=False)

print("Clustering completed. File saved as clustered_customers.csv")
