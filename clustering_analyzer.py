import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import streamlit as st

class ClusteringAnalyzer:
    """Handles clustering analysis and optimization"""
    
    def __init__(self):
        self.kmeans_model = None
        self.pca_model = None
    
    def find_optimal_clusters(self, X, max_k=10, min_k=2):
        """
        Find optimal number of clusters using elbow method and silhouette score
        
        Args:
            X: Feature matrix
            max_k: Maximum number of clusters to test
            min_k: Minimum number of clusters to test
            
        Returns:
            Tuple of (optimal_k, elbow_scores, silhouette_scores)
        """
        inertias = []
        silhouette_scores = []
        k_range = range(min_k, max_k + 1)
        
        # Calculate inertia for elbow method
        for k in range(1, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)
        
        # Calculate silhouette scores
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
            cluster_labels = kmeans.fit_predict(X)
            silhouette_avg = silhouette_score(X, cluster_labels)
            silhouette_scores.append(silhouette_avg)
        
        # Find optimal k using elbow method
        optimal_k_elbow = self._find_elbow_point(inertias)
        
        # Find optimal k using silhouette score
        optimal_k_silhouette = k_range[np.argmax(silhouette_scores)]
        
        # Choose the optimal k (prefer silhouette score if reasonable)
        if silhouette_scores[optimal_k_silhouette - min_k] > 0.3:
            optimal_k = optimal_k_silhouette
        else:
            optimal_k = optimal_k_elbow
        
        # Ensure optimal_k is within valid range
        optimal_k = max(int(min_k), min(int(optimal_k), int(max_k)))
        
        return optimal_k, inertias, silhouette_scores
    
    def _find_elbow_point(self, inertias):
        """Find elbow point in the inertia curve"""
        # Simple elbow detection using rate of change
        if len(inertias) < 3:
            return 3
        
        # Calculate second derivatives
        second_derivatives = []
        for i in range(1, len(inertias) - 1):
            second_derivative = inertias[i-1] - 2*inertias[i] + inertias[i+1]
            second_derivatives.append(second_derivative)
        
        # Find the point with maximum second derivative
        elbow_idx = np.argmax(second_derivatives) + 1  # +1 because we started from index 1
        
        return elbow_idx + 1  # +1 because k starts from 1
    
    def perform_clustering(self, X, n_clusters=3):
        """
        Perform K-means clustering on the data
        
        Args:
            X: Feature matrix
            n_clusters: Number of clusters
            
        Returns:
            Tuple of (cluster_labels, kmeans_model, silhouette_score)
        """
        # Perform K-means clustering
        self.kmeans_model = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        cluster_labels = self.kmeans_model.fit_predict(X)
        
        # Calculate silhouette score
        if n_clusters > 1:
            silhouette_avg = silhouette_score(X, cluster_labels)
        else:
            silhouette_avg = 0.0
        
        return cluster_labels, self.kmeans_model, silhouette_avg
    
    def apply_pca(self, X, n_components=2):
        """
        Apply PCA for dimensionality reduction
        
        Args:
            X: Feature matrix
            n_components: Number of principal components
            
        Returns:
            Tuple of (X_pca, pca_model)
        """
        self.pca_model = PCA(n_components=n_components, random_state=42)
        X_pca = self.pca_model.fit_transform(X)
        
        return X_pca, self.pca_model
    
    def get_cluster_centers(self):
        """Get cluster centers from the fitted model"""
        if self.kmeans_model is not None:
            return self.kmeans_model.cluster_centers_
        return None
    
    def predict_cluster(self, X):
        """Predict cluster for new data points"""
        if self.kmeans_model is not None:
            return self.kmeans_model.predict(X)
        return None
