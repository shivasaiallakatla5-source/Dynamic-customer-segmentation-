import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("Mall_Customers.csv")

# -----------------------------
# Data Cleaning
# -----------------------------
df.dropna(inplace=True)

# Remove ID
df.drop("CustomerID", axis=1, inplace=True)

# Encode Gender
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})

# -----------------------------
# Feature Selection
# -----------------------------
X = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# -----------------------------
# Scaling
# -----------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# KMeans Training
# -----------------------------
kmeans = KMeans(n_clusters=6, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

df['Cluster'] = clusters

# -----------------------------
# Evaluation
# -----------------------------
score = silhouette_score(X_scaled, clusters)
print("Silhouette Score:", score)

# -----------------------------
# Save Model
# -----------------------------
joblib.dump((kmeans, scaler), "model.pkl")

# Save clustered data
df.to_csv("clustered_customers.csv", index=False)

print("✅ Customer segmentation model saved")
