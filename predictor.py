import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import streamlit as st

class LeprosyPredictor:
    """Handles leprosy risk prediction based on patient characteristics"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = ['Patient Age', 'Patient Gender', 'Diabeties', 'Grade']
        self.is_trained = False
        self.model_accuracy = 0.0
        
    def prepare_training_data(self, df):
        """
        Prepare training data from the uploaded dataset
        
        Args:
            df: DataFrame with patient data
            
        Returns:
            Prepared features and target variable, or None if insufficient data
        """
        try:
            # Check if we have required columns
            required_cols = self.feature_names.copy()
            
            # Try to find a target variable (leprosy indicator)
            # Look for common leprosy-related columns
            target_candidates = []
            for col in df.columns:
                col_lower = col.lower()
                if any(keyword in col_lower for keyword in ['leprosy', 'disease', 'diagnosis', 'grade', 'stage']):
                    target_candidates.append(col)
            
            # Use Grade as target if available, or create synthetic target based on patterns
            if 'Grade' in df.columns:
                # Use Grade as leprosy severity indicator
                target_col = 'Grade'
            else:
                st.warning("No clear leprosy indicator found. Creating risk assessment based on patterns.")
                # Create synthetic target based on foot measurement patterns
                target_col = self._create_synthetic_target(df)
                if target_col is None:
                    return None
            
            # Check if we have enough features
            available_features = [col for col in required_cols if col in df.columns]
            
            if len(available_features) < 2:
                st.error("Need at least age and gender information for prediction")
                return None
            
            # Prepare feature data
            X = df[available_features].copy()
            y = df[target_col].copy()
            
            # Handle missing values
            X = X.dropna()
            y = y.loc[X.index]
            
            if len(X) < 10:
                st.error("Need at least 10 complete records to train the model")
                return None
            
            # Encode categorical variables
            self.label_encoders = {}
            X_processed = X.copy()
            
            for col in X.columns:
                if X[col].dtype == 'object' or col in ['Patient Gender', 'Diabeties']:
                    le = LabelEncoder()
                    X_processed[col] = le.fit_transform(X[col].astype(str))
                    self.label_encoders[col] = le
            
            # Encode target variable if categorical
            if y.dtype == 'object':
                self.target_encoder = LabelEncoder()
                y_processed = self.target_encoder.fit_transform(y.astype(str))
            else:
                # Convert to binary classification (high risk vs low risk)
                y_processed = (y > y.median()).astype(int)
                self.target_encoder = None
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X_processed)
            
            self.feature_names = available_features
            
            return X_scaled, y_processed
            
        except Exception as e:
            st.error(f"Error preparing training data: {str(e)}")
            return None
    
    def _create_synthetic_target(self, df):
        """Create synthetic leprosy risk target based on available data"""
        try:
            # Look for foot measurement columns that might indicate severity
            foot_measurements = []
            for col in df.columns:
                col_lower = col.lower()
                if any(keyword in col_lower for keyword in ['foot', 'length', 'girth', 'step', 'malleoli']):
                    if df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                        foot_measurements.append(col)
            
            if len(foot_measurements) < 2:
                return None
            
            # Create risk score based on abnormal foot measurements
            df_work = df[foot_measurements].copy()
            
            # Calculate z-scores for foot measurements
            risk_scores = []
            for idx, row in df_work.iterrows():
                score = 0
                for col in foot_measurements:
                    if pd.notna(row[col]):
                        # Calculate how far from normal this measurement is
                        mean_val = df_work[col].mean()
                        std_val = df_work[col].std()
                        if std_val > 0:
                            z_score = abs((row[col] - mean_val) / std_val)
                            if z_score > 1.5:  # Significantly abnormal
                                score += 1
                risk_scores.append(score)
            
            # Create binary target (high risk if multiple abnormal measurements)
            threshold = np.percentile(risk_scores, 70)  # Top 30% are high risk
            df['Leprosy_Risk'] = (np.array(risk_scores) >= threshold).astype(int)
            
            return 'Leprosy_Risk'
            
        except Exception:
            return None
    
    def train_model(self, X, y):
        """
        Train the leprosy prediction model
        
        Args:
            X: Feature matrix
            y: Target variable
            
        Returns:
            Training accuracy or None if training failed
        """
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Train Random Forest model
            self.model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=10,
                min_samples_split=5
            )
            
            self.model.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = self.model.predict(X_test)
            self.model_accuracy = accuracy_score(y_test, y_pred)
            
            self.is_trained = True
            
            return self.model_accuracy
            
        except Exception as e:
            st.error(f"Error training model: {str(e)}")
            return None
    
    def predict_leprosy_risk(self, age, gender, diabetic, grade=None):
        """
        Predict leprosy risk for a patient
        
        Args:
            age: Patient age (numeric)
            gender: Patient gender (string)
            diabetic: Diabetic status (string)
            grade: Disease grade if available (string)
            
        Returns:
            Tuple of (risk_probability, risk_level, confidence)
        """
        if not self.is_trained:
            return None, "Model not trained", 0.0
        
        try:
            # Prepare input data
            input_data = {}
            
            # Add available features
            if 'Patient Age' in self.feature_names:
                input_data['Patient Age'] = float(age)
            
            if 'Patient Gender' in self.feature_names:
                if 'Patient Gender' in self.label_encoders:
                    try:
                        input_data['Patient Gender'] = self.label_encoders['Patient Gender'].transform([str(gender)])[0]
                    except ValueError:
                        # Handle unseen category
                        input_data['Patient Gender'] = 0
                else:
                    input_data['Patient Gender'] = 1 if str(gender).lower() in ['male', 'm'] else 0
            
            if 'Diabeties' in self.feature_names:
                if 'Diabeties' in self.label_encoders:
                    try:
                        input_data['Diabeties'] = self.label_encoders['Diabeties'].transform([str(diabetic)])[0]
                    except ValueError:
                        input_data['Diabeties'] = 0
                else:
                    input_data['Diabeties'] = 1 if str(diabetic).lower() in ['yes', 'y', 'true', '1'] else 0
            
            if 'Grade' in self.feature_names and grade is not None:
                if 'Grade' in self.label_encoders:
                    try:
                        input_data['Grade'] = self.label_encoders['Grade'].transform([str(grade)])[0]
                    except ValueError:
                        input_data['Grade'] = 0
                else:
                    input_data['Grade'] = float(grade) if str(grade).isdigit() else 0
            
            # Create feature array in correct order
            feature_array = []
            for feature in self.feature_names:
                if feature in input_data:
                    feature_array.append(input_data[feature])
                else:
                    feature_array.append(0)  # Default value for missing features
            
            # Scale features
            X_input = self.scaler.transform([feature_array])
            
            # Make prediction
            if self.model is not None:
                risk_proba = self.model.predict_proba(X_input)[0]
                risk_class = self.model.predict(X_input)[0]
                
                # Determine risk level and confidence
                if len(risk_proba) == 2:
                    high_risk_prob = risk_proba[1]
                    confidence = float(max(risk_proba))
                else:
                    high_risk_prob = risk_proba[risk_class]
                    confidence = float(high_risk_prob)
            else:
                return None, "Model not available", 0.0
            
            # Categorize risk level
            if high_risk_prob >= 0.7:
                risk_level = "High Risk"
            elif high_risk_prob >= 0.4:
                risk_level = "Medium Risk"
            else:
                risk_level = "Low Risk"
            
            return high_risk_prob, risk_level, confidence
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            return None, "Prediction failed", 0.0
    
    def get_feature_importance(self):
        """Get feature importance from trained model"""
        if not self.is_trained or self.model is None:
            return None
        
        try:
            importance = self.model.feature_importances_
            feature_importance = dict(zip(self.feature_names, importance))
            return feature_importance
        except Exception:
            return None
    
    def get_model_info(self):
        """Get information about the trained model"""
        if not self.is_trained:
            return None
        
        info = {
            'accuracy': self.model_accuracy,
            'features_used': self.feature_names,
            'model_type': 'Random Forest Classifier',
            'n_estimators': self.model.n_estimators if self.model else 0
        }
        
        return info