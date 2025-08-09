import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import streamlit as st

class DataProcessor:
    """Handles data preprocessing for clustering analysis"""
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = None
        self.feature_names = []
    
    def preprocess_data(self, df, normalize=True, remove_outliers=False):
        """
        Preprocess the dataset for clustering analysis
        
        Args:
            df: Input DataFrame
            normalize: Whether to normalize numerical features
            remove_outliers: Whether to remove outliers using IQR method
            
        Returns:
            Tuple of (processed_features, feature_names, encoders, scaler) or None if failed
        """
        try:
            # Make a copy to avoid modifying original data
            data = df.copy()
            
            # Remove non-informative columns
            columns_to_remove = self._identify_non_informative_columns(data)
            if columns_to_remove:
                st.info(f"Removing non-informative columns: {columns_to_remove}")
                data = data.drop(columns=columns_to_remove, errors='ignore')
            
            # Identify feature types
            numerical_cols = self._identify_numerical_columns(data)
            categorical_cols = self._identify_categorical_columns(data)
            
            st.info(f"Identified {len(numerical_cols)} numerical and {len(categorical_cols)} categorical features")
            
            if len(numerical_cols) == 0 and len(categorical_cols) == 0:
                st.error("No suitable features found for clustering")
                return None
            
            # Handle missing values
            data = self._handle_missing_values(data, numerical_cols, categorical_cols)
            
            # Remove outliers if requested
            if remove_outliers and len(numerical_cols) > 0:
                data = self._remove_outliers(data, numerical_cols)
                st.info(f"Outlier removal: {len(data)} records remaining")
            
            # Encode categorical variables
            encoded_features = []
            feature_names = []
            
            # Process numerical columns
            if len(numerical_cols) > 0:
                numerical_data = data[numerical_cols].values
                encoded_features.append(numerical_data)
                feature_names.extend(numerical_cols)
            
            # Process categorical columns
            if len(categorical_cols) > 0:
                categorical_data = self._encode_categorical_features(data, categorical_cols)
                encoded_features.append(categorical_data)
                feature_names.extend([f"{col}_encoded" for col in categorical_cols])
            
            # Combine features
            if len(encoded_features) == 1:
                X = encoded_features[0]
            else:
                X = np.hstack(encoded_features)
            
            # Normalize features if requested
            if normalize:
                self.scaler = StandardScaler()
                X = self.scaler.fit_transform(X)
            
            self.feature_names = feature_names
            
            return X, feature_names, self.label_encoders, self.scaler
            
        except Exception as e:
            st.error(f"Error in data preprocessing: {str(e)}")
            return None
    
    def _identify_non_informative_columns(self, df):
        """Identify columns that should be removed for clustering"""
        columns_to_remove = []
        
        for col in df.columns:
            col_lower = col.lower()
            
            # Remove ID columns
            if any(id_pattern in col_lower for id_pattern in ['_id', 'uuid', 'index']):
                columns_to_remove.append(col)
            
            # Remove timestamp columns
            elif any(time_pattern in col_lower for time_pattern in ['timestamp', 'date', 'time', 'created', 'updated']):
                columns_to_remove.append(col)
            
            # Remove text description fields
            elif any(text_pattern in col_lower for text_pattern in ['description', 'comment', 'note', 'remark']):
                columns_to_remove.append(col)
            
            # Remove columns with too many unique values (likely IDs)
            elif df[col].dtype == 'object' and df[col].nunique() > len(df) * 0.5:
                columns_to_remove.append(col)
            
            # Remove columns with too few unique values (constant or near-constant)
            elif df[col].nunique() <= 1:
                columns_to_remove.append(col)
        
        return columns_to_remove
    
    def _identify_numerical_columns(self, df):
        """Identify numerical columns suitable for clustering"""
        numerical_cols = []
        
        for col in df.columns:
            if df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                # Additional checks for meaningful numerical data
                if df[col].nunique() > 1:  # Not constant
                    numerical_cols.append(col)
        
        return numerical_cols
    
    def _identify_categorical_columns(self, df):
        """Identify categorical columns suitable for clustering"""
        categorical_cols = []
        
        for col in df.columns:
            if df[col].dtype == 'object' or df[col].dtype.name == 'category':
                # Only include if reasonable number of categories
                unique_values = df[col].nunique()
                if 1 < unique_values <= 20:  # Between 2 and 20 unique values
                    categorical_cols.append(col)
        
        return categorical_cols
    
    def _handle_missing_values(self, df, numerical_cols, categorical_cols):
        """Handle missing values in the dataset"""
        data = df.copy()
        
        # Handle numerical missing values with median
        if len(numerical_cols) > 0:
            numerical_imputer = SimpleImputer(strategy='median')
            data[numerical_cols] = numerical_imputer.fit_transform(data[numerical_cols])
        
        # Handle categorical missing values with mode
        if len(categorical_cols) > 0:
            for col in categorical_cols:
                mode_value = data[col].mode()
                if len(mode_value) > 0:
                    data[col].fillna(mode_value[0], inplace=True)
                else:
                    data[col].fillna('Unknown', inplace=True)
        
        return data
    
    def _remove_outliers(self, df, numerical_cols):
        """Remove outliers using IQR method"""
        data = df.copy()
        
        for col in numerical_cols:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]
        
        return data
    
    def _encode_categorical_features(self, df, categorical_cols):
        """Encode categorical features using label encoding"""
        encoded_data = []
        
        for col in categorical_cols:
            le = LabelEncoder()
            # Convert to string to handle mixed types
            encoded_values = le.fit_transform(df[col].astype(str))
            if encoded_values is not None:
                encoded_data.append(encoded_values.reshape(-1, 1))
            self.label_encoders[col] = le
        
        if encoded_data:
            return np.hstack(encoded_data)
        else:
            return np.array([]).reshape(len(df), 0)
