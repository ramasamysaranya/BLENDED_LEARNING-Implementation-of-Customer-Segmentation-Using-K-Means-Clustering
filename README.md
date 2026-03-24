# BLENDED LEARNING
# Implementation of Customer Segmentation Using K-Means Clustering

## AIM:
To implement customer segmentation using K-Means clustering on the Mall Customers dataset to group customers based on purchasing habits.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the customer dataset and select the features **Age, Annual Income, and Spending Score** for clustering.
2. Standardize the feature values using **StandardScaler** to normalize the data.
3. Use the **Elbow Method** to determine the optimal number of clusters and train the **K-Means** model.
4. Assign cluster labels, evaluate using **Silhouette Score**, and visualize customer segments with cluster centroids. 

## Program:
```
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

data = pd.read_csv('CustomerData.csv')

print(data.head())
print(data.columns)

features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
X = data[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

inertia_values = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)  # Explicit n_init to suppress warning
    kmeans.fit(X_scaled)
    inertia_values.append(kmeans.inertia_)
    
plt.figure(figsize=(8, 4))
plt.plot(range(1, 11), inertia_values, marker='o', linestyle='-')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.show()

optimal_clusters = 4
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)  # Explicit n_init
kmeans.fit(X_scaled)

data['Cluster'] = kmeans.labels_

sil_score = silhouette_score(X_scaled, kmeans.labels_)
print(f'Silhouette Score: {sil_score}')

print("\nName: SARANYA R")
print("Reg No.: 212225040384\n")
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data,x='Annual Income (k$)',y='Spending Score (1-100)',hue='Cluster', palette='viridis',s=100,alpha=0.7)
plt.title('Customer Segmentation based on Annual Income and Spending Score')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend(title='Cluster')
plt.show()

```

## Output:
<img width="773" height="196" alt="image" src="https://github.com/user-attachments/assets/a6a4773a-c91f-4dba-8e21-c15f44fecec3" />
<img width="908" height="487" alt="image" src="https://github.com/user-attachments/assets/bc0fc8a4-acff-41a4-8c95-0d57c1767c45" />
<img width="1040" height="730" alt="image" src="https://github.com/user-attachments/assets/8f3c8574-da6a-4fc7-a6b9-729a1fbec60f" />


## Result:
Thus, customer segmentation was successfully implemented using K-Means clustering, grouping customers into distinct segments based on their annual income and spending score. 
