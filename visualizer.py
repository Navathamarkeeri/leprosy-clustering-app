import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

class Visualizer:
    """Handles all visualization for clustering analysis"""
    
    def __init__(self):
        # Define a consistent color palette
        self.colors = px.colors.qualitative.Set1
    
    def plot_elbow_method(self, k_range, inertias):
        """Create elbow method plot"""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=list(k_range),
            y=inertias,
            mode='lines+markers',
            name='Inertia',
            line=dict(width=3),
            marker=dict(size=8, symbol='circle')
        ))
        
        fig.update_layout(
            title='Elbow Method for Optimal k',
            xaxis_title='Number of Clusters (k)',
            yaxis_title='Inertia',
            showlegend=False,
            height=400
        )
        
        return fig
    
    def plot_silhouette_scores(self, k_range, silhouette_scores):
        """Create silhouette scores plot"""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=list(k_range),
            y=silhouette_scores,
            mode='lines+markers',
            name='Silhouette Score',
            line=dict(width=3, color='orange'),
            marker=dict(size=8, symbol='circle')
        ))
        
        fig.update_layout(
            title='Silhouette Scores for Different k',
            xaxis_title='Number of Clusters (k)',
            yaxis_title='Silhouette Score',
            showlegend=False,
            height=400
        )
        
        return fig
    
    def plot_cluster_distribution(self, cluster_counts):
        """Create cluster distribution bar plot"""
        fig = go.Figure(data=[
            go.Bar(
                x=[f'Cluster {i}' for i in cluster_counts.index],
                y=cluster_counts.values,
                marker_color=self.colors[:len(cluster_counts)]
            )
        ])
        
        fig.update_layout(
            title='Distribution of Records Across Clusters',
            xaxis_title='Cluster',
            yaxis_title='Number of Records',
            height=400
        )
        
        return fig
    
    def plot_pca_clusters(self, X_pca, cluster_labels, feature_names):
        """Create PCA scatter plot with clusters"""
        # Create DataFrame for plotting
        df_plot = pd.DataFrame({
            'PC1': X_pca[:, 0],
            'PC2': X_pca[:, 1],
            'Cluster': [f'Cluster {label}' for label in cluster_labels]
        })
        
        fig = px.scatter(
            df_plot,
            x='PC1',
            y='PC2',
            color='Cluster',
            title='Cluster Visualization (PCA)',
            labels={
                'PC1': 'First Principal Component',
                'PC2': 'Second Principal Component'
            },
            color_discrete_sequence=self.colors
        )
        
        fig.update_traces(marker=dict(size=8, opacity=0.7))
        fig.update_layout(height=500)
        
        return fig
    
    def plot_feature_importance_pca(self, pca_model, feature_names, n_components=2):
        """Plot PCA feature importance/loadings"""
        if len(feature_names) != pca_model.components_.shape[1]:
            # Handle case where feature names don't match
            feature_names = [f'Feature_{i}' for i in range(pca_model.components_.shape[1])]
        
        # Create subplots for each component
        fig = make_subplots(
            rows=1, cols=n_components,
            subplot_titles=[f'PC{i+1} Loadings' for i in range(n_components)]
        )
        
        for i in range(n_components):
            loadings = pca_model.components_[i]
            
            fig.add_trace(
                go.Bar(
                    x=feature_names,
                    y=loadings,
                    name=f'PC{i+1}',
                    marker_color=self.colors[i % len(self.colors)]
                ),
                row=1, col=i+1
            )
        
        fig.update_layout(
            title='PCA Feature Loadings',
            height=400,
            showlegend=False
        )
        
        # Rotate x-axis labels for better readability
        fig.update_xaxes(tickangle=45)
        
        return fig
    
    def plot_cluster_centers_heatmap(self, cluster_centers, feature_names):
        """Create heatmap of cluster centers"""
        if len(feature_names) != cluster_centers.shape[1]:
            feature_names = [f'Feature_{i}' for i in range(cluster_centers.shape[1])]
        
        fig = go.Figure(data=go.Heatmap(
            z=cluster_centers,
            x=feature_names,
            y=[f'Cluster {i}' for i in range(cluster_centers.shape[0])],
            colorscale='RdBu',
            colorbar=dict(title="Feature Value")
        ))
        
        fig.update_layout(
            title='Cluster Centers Heatmap',
            xaxis_title='Features',
            yaxis_title='Clusters',
            height=400
        )
        
        fig.update_xaxes(tickangle=45)
        
        return fig
    
    def plot_cluster_comparison(self, df, cluster_col, feature_cols):
        """Create box plots comparing clusters across different features"""
        n_features = len(feature_cols)
        cols = min(3, n_features)
        rows = (n_features + cols - 1) // cols
        
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=feature_cols,
            vertical_spacing=0.08
        )
        
        for i, feature in enumerate(feature_cols):
            row = i // cols + 1
            col = i % cols + 1
            
            for cluster_id in sorted(df[cluster_col].unique()):
                cluster_data = df[df[cluster_col] == cluster_id][feature]
                
                fig.add_trace(
                    go.Box(
                        y=cluster_data,
                        name=f'Cluster {cluster_id}',
                        boxpoints='outliers',
                        showlegend=(i == 0)  # Only show legend for first subplot
                    ),
                    row=row, col=col
                )
        
        fig.update_layout(
            title='Feature Distribution by Cluster',
            height=300 * rows
        )
        
        return fig
