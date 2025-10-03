import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

from src.nasa_data_loader import NASABearingDataLoader
from src.models import ModelTrainer
from src.evaluation import ModelEvaluator
from src.utils import setup_logging, create_directories

def prepare_nasa_data_for_training(df: pd.DataFrame, task='classification'):
    """Prepare NASA bearing data for model training"""
    
    # Select features for training
    feature_columns = [col for col in df.columns if any(
        feat in col for feat in ['rms', 'peak', 'kurtosis', 'crest', 'spectral', 'band_power', 'trend']
    )]
    
    X = df[feature_columns]
    
    if task == 'classification':
        # Binary classification: healthy vs degrading/failed
        y = (df['health_status'] != 'healthy').astype(int)
    else:
        # Regression: predict RUL
        y = df['rul_cycles']
    
    # Use GroupShuffleSplit to avoid data leakage between bearings
    groups = df['test_set'] + '_' + df['bearing']
    
    return X, y, groups

def train_nasa_classification_models():
    """Train classification models on NASA bearing data"""
    logger = setup_logging()
    logger.info("Loading NASA bearing dataset...")
    
    # Load data
    loader = NASABearingDataLoader()
    
    # Check if data is available
    if not loader.check_data_availability():
        loader.download_instructions()
        return None, None, None
    
    df = loader.load_all_test_sets()
    df_windowed = loader.create_windowed_features(df)
    
    # Prepare data
    X, y, groups = prepare_nasa_data_for_training(df_windowed, task='classification')
    
    # Split data preserving groups
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups))
    
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    trainer = ModelTrainer()
    evaluator = ModelEvaluator()
    
    models = ['random_forest', 'xgboost']
    
    for model_type in models:
        logger.info(f"Training {model_type} classifier on NASA data...")
        
        model = trainer.train_classifier(
            model_type, X_train_scaled, y_train
        )
        
        # Evaluate
        metrics = evaluator.evaluate_classifier(
            model, X_test_scaled, y_test, f"NASA_{model_type}"
        )
        logger.info(f"{model_type} metrics: {metrics}")
    
    # Save results
    comparison_df = evaluator.compare_models()
    comparison_df.to_csv('results/nasa_classification_results.csv')
    
    return trainer, evaluator, scaler

def train_nasa_rul_models():
    """Train RUL prediction models on NASA bearing data"""
    logger = setup_logging()
    logger.info("Training RUL prediction models on NASA data...")
    
    # Load data
    loader = NASABearingDataLoader()
    
    if not loader.check_data_availability():
        return None, None, None
    
    df = loader.load_all_test_sets()
    df_windowed = loader.create_windowed_features(df)
    
    # Prepare data for regression
    X, y, groups = prepare_nasa_data_for_training(df_windowed, task='regression')
    
    # Split data
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups))
    
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    trainer = ModelTrainer()
    evaluator = ModelEvaluator()
    
    models = ['random_forest', 'xgboost']
    
    for model_type in models:
        logger.info(f"Training {model_type} regressor on NASA data...")
        
        model = trainer.train_regressor(
            model_type, X_train_scaled, y_train
        )
        
        # Evaluate
        metrics = evaluator.evaluate_regressor(
            model, X_test_scaled, y_test, f"NASA_RUL_{model_type}"
        )
        logger.info(f"{model_type} RUL metrics: {metrics}")
    
    # Visualize results
    visualize_rul_predictions(evaluator, y_test)
    
    return trainer, evaluator, scaler

def visualize_rul_predictions(evaluator, y_test):
    """Visualize RUL prediction results"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    model_names = list(evaluator.results.keys())[:2]
    
    for idx, model_name in enumerate(model_names):
        y_pred = evaluator.results[model_name]['y_pred']
        
        axes[idx].scatter(y_test, y_pred, alpha=0.5)
        axes[idx].plot([y_test.min(), y_test.max()], 
                      [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[idx].set_xlabel('Actual RUL (cycles)')
        axes[idx].set_ylabel('Predicted RUL (cycles)')
        axes[idx].set_title(f'{model_name.split("_")[-1]} - R² = {r2_score(y_test, y_pred):.3f}')
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/nasa_rul_predictions.png', dpi=300)
    plt.close()

def analyze_bearing_degradation():
    """Analyze and visualize bearing degradation patterns"""
    loader = NASABearingDataLoader()
    
    if not loader.check_data_availability():
        return
    
    df = loader.load_all_test_sets()
    
    # Visualize degradation for failed bearings
    for test_set, info in loader.test_sets.items():
        failed_bearing = info['failed_bearing']
        loader.visualize_bearing_degradation(df, test_set, failed_bearing)

if __name__ == "__main__":
    create_directories()
    
    print("="*60)
    print("NASA Bearing Dataset Analysis")
    print("="*60)
    
    # Check data availability
    loader = NASABearingDataLoader()
    if not loader.check_data_availability():
        print("\nNASA bearing data not found!")
        loader.download_instructions()
        print("\nAfter downloading, run this script again.")
    else:
        print("\nNASA bearing data found! Starting analysis...")
        
        # Train classification models
        print("\n1. Training fault detection models...")
        train_nasa_classification_models()
        
        # Train RUL prediction models
        print("\n2. Training RUL prediction models...")
        train_nasa_rul_models()
        
        # Analyze degradation patterns
        print("\n3. Analyzing bearing degradation patterns...")
        analyze_bearing_degradation()
        
        print("\nAnalysis completed! Check results/ directory for outputs.")