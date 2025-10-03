import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from typing import Tuple, List
import joblib

class DataPreprocessor:
    """Handle data preprocessing and feature scaling"""
    
    def __init__(self, scaler_type='standard'):
        self.scaler_type = scaler_type
        self.scaler = StandardScaler() if scaler_type == 'standard' else MinMaxScaler()
        self.feature_columns = None
        self.target_column = None
        self.label_encoder = None
        
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean data by handling missing values and outliers"""
        df = df.drop_duplicates()
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
        
        for col in numeric_columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            df[col] = df[col].clip(lower_bound, upper_bound)
            
        return df
    
    def create_sliding_windows(self, df: pd.DataFrame, window_size: int = 50, 
                             stride: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Create sliding windows for time series data"""
        features = []
        targets = []
        
        data_array = df.drop(['health_status', 'rul'], axis=1).values
        rul_values = df['rul'].values
        
        for i in range(0, len(df) - window_size, stride):
            features.append(data_array[i:i+window_size])
            targets.append(rul_values[i+window_size-1])
            
        return np.array(features), np.array(targets)
    
    def prepare_classification_data(self, df: pd.DataFrame, 
                                  target_col: str = 'health_status') -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data for classification tasks"""
        le = LabelEncoder()
        
        X = df.drop([target_col, 'rul'], axis=1)
        y = le.fit_transform(df[target_col])
        
        self.label_encoder = le
        self.feature_columns = X.columns.tolist()
        self.target_column = target_col
        
        return X, y
    
    def prepare_regression_data(self, df: pd.DataFrame, 
                               target_col: str = 'rul') -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data for regression tasks"""
        X = df.drop(['health_status', target_col], axis=1)
        y = df[target_col]
        
        self.feature_columns = X.columns.tolist()
        self.target_column = target_col
        
        return X, y
    
    def split_and_scale(self, X: pd.DataFrame, y: pd.Series, 
                       test_size: float = 0.2, random_state: int = 42) -> Tuple:
        """Split data and apply scaling"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, 
            stratify=y if len(np.unique(y)) < 20 else None
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def save_preprocessor(self, path: str):
        """Save preprocessing objects"""
        joblib.dump({
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'label_encoder': self.label_encoder
        }, path)
    
    def load_preprocessor(self, path: str):
        """Load preprocessing objects"""
        data = joblib.load(path)
        self.scaler = data['scaler']
        self.feature_columns = data['feature_columns']
        if data['label_encoder']:
            self.label_encoder = data['label_encoder']