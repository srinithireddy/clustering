import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Customer Segmentation Model", 
                   page_icon="👕", layout="centered")
# Load data
data = pd.read_csv("customer-segmentation-data.csv")

def plot_elbow_method(data, max_clusters=10):

#features for clustering
    features = data[['Annual_Income','Spending_Score']]

#standardize the data
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

#WCSS for diff number of clusters
    wcss = []
    for i in range(1, max_clusters+1):
        kmeans = KMeans(n_clusters=i,init='k-means++', random_state=42, n_init=10)
        kmeans.fit(scaled_features)
        wcss.append(kmeans.inertia_)

#elbow plot
    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(range(1,max_clusters+1), wcss, marker = 'o', linestyle='-')
    ax.set_title('Elbow Method for Optimal K')
    ax.set_xlabel('Number of Clusters')
    ax.set_ylabel('Within-Cluster Sum of Square (WCSS)')
    ax.grid(True)

    return fig

def perform_kmeans_clustering(data, n_clusters):
    
    # Features for clustering
    features = data[['Annual_Income', 'Spending_Score']]
    
    # Standardize the data
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(scaled_features)
    
    # Add cluster labels to the original data
    data_with_clusters = data.copy()
    data_with_clusters['Cluster'] = cluster_labels
    
    # Get the cluster centers and transform back to original scale
    centers = scaler.inverse_transform(kmeans.cluster_centers_)
    
    return data_with_clusters, centers

def visualize_clusters(data_with_clusters, centers, n_clusters):

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot each cluster with a different color
    for i in range(n_clusters):
        cluster_data = data_with_clusters[data_with_clusters['Cluster'] == i]
        ax.scatter(
            cluster_data['Annual_Income'],
            cluster_data['Spending_Score'],
            s=100,  # Point size
            label=f'Cluster {i+1}'
        )
    
    # Plot the cluster centers
    ax.scatter(
        centers[:, 0],
        centers[:, 1],
        s=300,  # Center point size
        c='red',
        marker='X',
        label='Centroids'
    )
    # Set plot labels and title
    ax.set_title('Customer Segments based on Annual Income and Spending Score', fontsize=15)
    ax.set_xlabel('Annual Income (k$)', fontsize=12)
    ax.set_ylabel('Spending Score (1-100)', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig

def generate_cluster_insights(data_with_clusters, n_clusters):

    insights = []
    
    for i in range(n_clusters):
        cluster_data = data_with_clusters[data_with_clusters['Cluster'] == i]
        
        # Calculate basic statistics
        avg_income = cluster_data['Annual_Income'].mean()
        avg_spending = cluster_data['Spending_Score'].mean()
        cluster_size = len(cluster_data)
        percentage = (cluster_size / len(data_with_clusters)) * 100
        
        # Create a summary
        if avg_income > 75 and avg_spending > 65:
            profile = "High-income luxury shoppers"
        elif avg_income > 75 and avg_spending < 40:
            profile = "High-income conservative spenders"
        elif avg_income < 50 and avg_spending > 65:
            profile = "Budget-conscious enthusiastic shoppers"
        elif avg_income < 50 and avg_spending < 40:
            profile = "Low-income, low-spending customers"
        else:
            profile = "Middle-income average shoppers"
        
        insights.append({
            "cluster": i+1,
            "size": cluster_size,
            "percentage": percentage,
            "avg_income": avg_income,
            "avg_spending": avg_spending,
            "profile": profile
        })
    
    return insights


def cluster_app():
    
    st.title("K-Means Clustering: Customer Segmentation Model")
    st.write("Welcome to the customer segmentation tool using K-Means clustering!")
    
    # Display the dataframe
    st.dataframe(data, width=1000, height =300) # type: ignore
    
    # elbowplot desc
    st.subheader("Step 1: Find the optimal number of clusters")
    st.write("The elbow point is where the curve starts to flatten and at this point adding more clusters doesn't improve the model visually. To decide on the maximum number of clusters to test on the slider it would depend on if the data set is smaller (hundereds of records - max 10-15) or larger (thousands or more - max 20)")

    #slider 
    max_clusters = st.slider("Maximum number of clusters to check", 2, 15, 10)

    #display plot
    elbow_fig = plot_elbow_method(data, max_clusters) # type: ignore
    st.pyplot(elbow_fig)

    #user input of # of clusters

    n_clusters = st.number_input("Using the plot select the optimal number of clusters", min_value=2, max_value=15, value=4)
   
    #visualize k-means on app
    st.subheader("Step 2: Understand the K-Means Model")
    st.write("Below is the visualization of customer segments based on Annual Income and Spending Score.")
    
    # Apply clustering with the selected number of clusters
    data_with_clusters, centers = perform_kmeans_clustering(data, n_clusters)
    
    # Create and display cluster visualization
    cluster_fig = visualize_clusters(data_with_clusters, centers, n_clusters)
    st.pyplot(cluster_fig)
    
    # Display cluster insights
    st.subheader("Step 3: Analyze Customer Segments")
    st.write("Here are the insights for each identified customer segment:")
    
    insights = generate_cluster_insights(data_with_clusters, n_clusters)
    
    # Create columns for displaying cluster insights
    cols = st.columns(min(n_clusters, 3))  # Show up to 3 clusters per row
    
    for i, insight in enumerate(insights):
        with cols[i % 3]:
            st.markdown(f"### Cluster {insight['cluster']}")
            st.markdown(f"**Size:** {insight['size']} customers ({insight['percentage']:.1f}%)")
            st.markdown(f"**Avg Income:** ${insight['avg_income']:.2f}k")
            st.markdown(f"**Avg Spending:** {insight['avg_spending']:.1f}/100")
            st.markdown(f"**Profile:** {insight['profile']}")
            
    # Show data with cluster labels
    st.subheader("Step 4: Explore Segmented Data")
    st.write("Explore the original data with cluster labels:")
    st.dataframe(data_with_clusters)
    
    # Add a download button for the segmented data
    csv = data_with_clusters.to_csv(index=False)
    st.download_button(
        label="Download Segmented Data as CSV",
        data=csv,
        file_name="customer_segments.csv",
        mime="text/csv"
    )



 
if __name__ == "__main__":
    cluster_app()