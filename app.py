import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# -----------------------------
# Load Model
# -----------------------------
kmeans, scaler = joblib.load("model.pkl")

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Segmentation Dashboard")

# -----------------------------
# Load Data
# -----------------------------
df = pd.read_csv("Mall_Customers.csv")

# Preprocess
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
X = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
X_scaled = scaler.transform(X)

# Predict clusters
df['Cluster'] = kmeans.predict(X_scaled)

# -----------------------------
# Sidebar Filters
# -----------------------------
st.sidebar.header("Filters")
selected_cluster = st.sidebar.multiselect(
    "Select Clusters",
    options=sorted(df['Cluster'].unique()),
    default=sorted(df['Cluster'].unique())
)

filtered_df = df[df['Cluster'].isin(selected_cluster)]

# -----------------------------
# Visualizations
# -----------------------------
st.subheader("Customer Clusters (Income vs Spending)")
fig1 = px.scatter(
    filtered_df,
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    color="Cluster",
    size="Age",
    hover_data=["Age"]
)
st.plotly_chart(fig1, use_container_width=True)

st.subheader("3D View of Customer Segments")
fig2 = px.scatter_3d(
    filtered_df,
    x="Age",
    y="Annual Income (k$)",
    z="Spending Score (1-100)",
    color="Cluster"
)
st.plotly_chart(fig2, use_container_width=True)

# -----------------------------
# Cluster Summary
# -----------------------------
st.subheader("Cluster Summary")
summary = filtered_df.groupby('Cluster').mean()
st.dataframe(summary)

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.caption("Customer Segmentation using K-Means Clustering")
