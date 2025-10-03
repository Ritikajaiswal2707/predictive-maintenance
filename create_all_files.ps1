# PowerShell script to create all project files
Write-Host "Creating all project files..." -ForegroundColor Green

# Function to create file with content
function Create-File {
    param($Path, $Content)
    $Content | Out-File -FilePath $Path -Encoding UTF8
    Write-Host "Created $Path" -ForegroundColor Green
}

# Create models.py
$modelsContent = @'
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, SVR
import xgboost as xgb
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import joblib
from typing import Dict, Any

class ModelFactory:
    """Factory class for creating different ML models"""
    
    @staticmethod
    def create_classifier(model_type: str, **kwargs) -> Any:
        """Create classification model"""
        if model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 10),
                random_state=kwargs.get('random_state', 42),
                n_jobs=-1
            )
        elif model_type == 'xgboost':
            return xgb.XGBClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 6),
                learning_rate=kwargs.get('learning_rate', 0.1),
                random_state=kwargs.get('random_state', 42),
                use_label_encoder=False,
                eval_metric='logloss'
            )
        elif model_type == 'logistic_regression':
            return LogisticRegression(
                max_iter=kwargs.get('max_iter', 1000),
                random_state=kwargs.get('random_state', 42)
            )
        elif model_type == 'svm':
            return SVC(
                kernel=kwargs.get('kernel', 'rbf'),
                probability=True,
                random_state=kwargs.get('random_state', 42)
            )
        else:
            raise ValueError(f"Unknown classifier type: {model_type}")
    
    @staticmethod
    def create_regressor(model_type: str, **kwargs) -> Any:
        """Create regression model"""
        if model_type == 'random_forest':
            return RandomForestRegressor(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 10),
                random_state=kwargs.get('random_state', 42),
                n_jobs=-1
            )
        elif model_type == 'xgboost':
            return xgb.XGBRegressor(
                n_estimators=kwargs.get('n_estimators', 100),
                max_depth=kwargs.get('max_depth', 6),
                learning_rate=kwargs.get('learning_rate', 0.1),
                random_state=kwargs.get('random_state', 42)
            )
        elif model_type == 'svr':
            return SVR(
                kernel=kwargs.get('kernel', 'rbf')
            )
        else:
            raise ValueError(f"Unknown regressor type: {model_type}")

class DeepLearningModels:
    """Deep learning models for predictive maintenance"""
    
    @staticmethod
    def create_ann_classifier(input_dim: int, num_classes: int) -> Model:
        """Create ANN for classification"""
        model = Sequential([
            Input(shape=(input_dim,)),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    @staticmethod
    def create_ann_regressor(input_dim: int) -> Model:
        """Create ANN for regression"""
        model = Sequential([
            Input(shape=(input_dim,)),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(1, activation='linear')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mean_squared_error',
            metrics=['mae']
        )
        
        return model
    
    @staticmethod
    def create_lstm_regressor(sequence_length: int, n_features: int) -> Model:
        """Create LSTM for RUL prediction"""
        model = Sequential([
            Input(shape=(sequence_length, n_features)),
            LSTM(128, return_sequences=True),
            Dropout(0.2),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='linear')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mean_squared_error',
            metrics=['mae']
        )
        
        return model

class ModelTrainer:
    """Handle model training and saving"""
    
    def __init__(self):
        self.models = {}
        self.histories = {}
        
    def train_classifier(self, model_type: str, X_train: np.ndarray, y_train: np.ndarray,
                        X_val: np.ndarray = None, y_val: np.ndarray = None, **kwargs):
        """Train classification model"""
        
        if model_type in ['ann']:
            model = DeepLearningModels.create_ann_classifier(
                input_dim=X_train.shape[1],
                num_classes=len(np.unique(y_train))
            )
            
            callbacks = [
                EarlyStopping(patience=10, restore_best_weights=True),
                ReduceLROnPlateau(patience=5, factor=0.5)
            ]
            
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val) if X_val is not None else None,
                epochs=kwargs.get('epochs', 100),
                batch_size=kwargs.get('batch_size', 32),
                callbacks=callbacks,
                verbose=1
            )
            
            self.histories[model_type] = history
        else:
            model = ModelFactory.create_classifier(model_type, **kwargs)
            model.fit(X_train, y_train)
        
        self.models[model_type] = model
        return model
    
    def train_regressor(self, model_type: str, X_train: np.ndarray, y_train: np.ndarray,
                       X_val: np.ndarray = None, y_val: np.ndarray = None, **kwargs):
        """Train regression model"""
        
        if model_type == 'ann':
            model = DeepLearningModels.create_ann_regressor(input_dim=X_train.shape[1])
            
            callbacks = [
                EarlyStopping(patience=10, restore_best_weights=True),
                ReduceLROnPlateau(patience=5, factor=0.5)
            ]
            
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val) if X_val is not None else None,
                epochs=kwargs.get('epochs', 100),
                batch_size=kwargs.get('batch_size', 32),
                callbacks=callbacks,
                verbose=1
            )
            
            self.histories[model_type] = history
            
        elif model_type == 'lstm':
            if len(X_train.shape) == 2:
                X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
                if X_val is not None:
                    X_val = X_val.reshape(X_val.shape[0], 1, X_val.shape[1])
            
            model = DeepLearningModels.create_lstm_regressor(
                sequence_length=X_train.shape[1],
                n_features=X_train.shape[2]
            )
            
            callbacks = [
                EarlyStopping(patience=10, restore_best_weights=True),
                ReduceLROnPlateau(patience=5, factor=0.5)
            ]
            
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val) if X_val is not None else None,
                epochs=kwargs.get('epochs', 100),
                batch_size=kwargs.get('batch_size', 32),
                callbacks=callbacks,
                verbose=1
            )
            
            self.histories[model_type] = history
        else:
            model = ModelFactory.create_regressor(model_type, **kwargs)
            model.fit(X_train, y_train)
        
        self.models[model_type] = model
        return model
    
    def save_model(self, model_type: str, path: str):
        """Save trained model"""
        if model_type in self.models:
            model = self.models[model_type]
            if hasattr(model, 'save'):
                model.save(f"{path}_{model_type}.h5")
            else:
                joblib.dump(model, f"{path}_{model_type}.pkl")
'@

Create-File -Path "src\models.py" -Content $modelsContent

# Create evaluation.py
$evaluationContent = @'
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                           confusion_matrix, classification_report, roc_auc_score,
                           mean_squared_error, mean_absolute_error, r2_score)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Tuple

class ModelEvaluator:
    """Evaluate model performance"""
    
    def __init__(self):
        self.results = {}
        
    def evaluate_classifier(self, model: Any, X_test: np.ndarray, y_test: np.ndarray,
                          model_name: str) -> Dict[str, float]:
        """Evaluate classification model"""
        y_pred = model.predict(X_test)
        
        if hasattr(model, 'predict_classes'):
            y_pred = model.predict_classes(X_test)
        elif len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
            y_pred = np.argmax(y_pred, axis=1)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted')
        }
        
        if len(np.unique(y_test)) == 2:
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)[:, 1]
            else:
                y_proba = model.predict(X_test)
                if len(y_proba.shape) > 1:
                    y_proba = y_proba[:, 1]
            metrics['auc'] = roc_auc_score(y_test, y_proba)
        
        self.results[model_name] = {
            'metrics': metrics,
            'y_pred': y_pred,
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        return metrics
    
    def evaluate_regressor(self, model: Any, X_test: np.ndarray, y_test: np.ndarray,
                         model_name: str) -> Dict[str, float]:
        """Evaluate regression model"""
        y_pred = model.predict(X_test).flatten()
        
        metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred),
            'mape': np.mean(np.abs((y_test - y_pred) / (y_test + 1e-10))) * 100
        }
        
        self.results[model_name] = {
            'metrics': metrics,
            'y_pred': y_pred
        }
        
        return metrics
    
    def plot_confusion_matrix(self, model_name: str, class_names: list = None):
        """Plot confusion matrix"""
        if model_name not in self.results:
            raise ValueError(f"No results found for {model_name}")
        
        cm = self.results[model_name]['confusion_matrix']
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        return plt.gcf()
    
    def plot_regression_results(self, model_name: str, y_test: np.ndarray):
        """Plot regression predictions vs actual"""
        if model_name not in self.results:
            raise ValueError(f"No results found for {model_name}")
        
        y_pred = self.results[model_name]['y_pred']
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        axes[0].scatter(y_test, y_pred, alpha=0.5)
        axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[0].set_xlabel('Actual RUL')
        axes[0].set_ylabel('Predicted RUL')
        axes[0].set_title(f'{model_name} - Predictions vs Actual')
        
        residuals = y_test - y_pred
        axes[1].scatter(y_pred, residuals, alpha=0.5)
        axes[1].axhline(y=0, color='r', linestyle='--')
        axes[1].set_xlabel('Predicted RUL')
        axes[1].set_ylabel('Residuals')
        axes[1].set_title(f'{model_name} - Residual Plot')
        
        plt.tight_layout()
        return fig
    
    def compare_models(self) ->