from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Use cost & age for anomaly detection
X_cluster = df[['age', 'cost_estimation']]

kmeans = KMeans(n_clusters=2)
df['cluster'] = kmeans.fit_predict(X_cluster)

plt.scatter(df['age'], df['cost_estimation'], c=df['cluster'])
plt.xlabel('Age')
plt.ylabel('Cost Estimation')
plt.title('Anomaly Detection via Clustering')
plt.show()
