import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

from src.models import ModelTrainer
from src.evaluation import ModelEvaluator
from src.utils import setup_logging, create_directories

def train_bearing_fault_models():
    """Train models on synthetic bearing dataset"""
    # Setup
    logger = setup_logging()
    create_directories()
    
    logger.info("Loading synthetic bearing dataset...")
    
    # Load data
    df = pd.read_csv('data/processed/synthetic_bearing_features.csv')
    logger.info(f"Loaded {len(df)} samples")
    
    # Prepare features
    feature_cols = ['rms', 'peak', 'peak_to_peak', 'crest_factor', 'kurtosis', 
                   'skewness', 'std', 'shape_factor', 'impulse_factor',
                   'dominant_frequency', 'n_peaks'] + [f'band_power_{i}' for i in range(4)]
    
    X = df[feature_cols]
    
    # Task 1: Binary Classification (Healthy vs Faulty)
    logger.info("\n=== Binary Classification: Healthy vs Faulty ===")
    
    y_binary = (df['health_status'] == 'faulty').astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    trainer_binary = ModelTrainer()
    evaluator_binary = ModelEvaluator()
    
    models = ['random_forest', 'xgboost', 'logistic_regression']
    
    for model_type in models:
        logger.info(f"Training {model_type}...")
        model = trainer_binary.train_classifier(
            model_type, X_train_scaled, y_train
        )
        
        metrics = evaluator_binary.evaluate_classifier(
            model, X_test_scaled, y_test, f"Binary_{model_type}"
        )
        logger.info(f"{model_type} metrics: {metrics}")
        
        # Save model
        trainer_binary.save_model(model_type, 'models/bearing_binary')
    
    # Save results
    binary_results = evaluator_binary.compare_models()
    binary_results.to_csv('results/bearing_binary_classification_results.csv')
    
    # Task 2: Multi-class Classification (Fault Type)
    logger.info("\n=== Multi-class Classification: Fault Types ===")
    
    # Prepare labels
    le = LabelEncoder()
    y_multi = le.fit_transform(df['condition'])
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_multi, test_size=0.2, random_state=42, stratify=y_multi
    )
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    trainer_multi = ModelTrainer()
    evaluator_multi = ModelEvaluator()
    
    for model_type in models:
        logger.info(f"Training {model_type} for fault type classification...")
        model = trainer_multi.train_classifier(
            model_type, X_train_scaled, y_train
        )
        
        metrics = evaluator_multi.evaluate_classifier(
            model, X_test_scaled, y_test, f"Multi_{model_type}"
        )
        logger.info(f"{model_type} metrics: {metrics}")
    
    # Save results
    multi_results = evaluator_multi.compare_models()
    multi_results.to_csv('results/bearing_multiclass_classification_results.csv')
    
    # Plot confusion matrix
    best_model = 'Multi_random_forest'
    fig = evaluator_multi.plot_confusion_matrix(best_model, class_names=le.classes_)
    plt.savefig('results/bearing_confusion_matrix.png', dpi=300)
    plt.close()
    
    # Task 3: Severity Classification
    logger.info("\n=== Severity Classification ===")
    
    # Filter only faulty samples
    faulty_df = df[df['health_status'] == 'faulty']
    X_severity = faulty_df[feature_cols]
    
    le_severity = LabelEncoder()
    y_severity = le_severity.fit_transform(faulty_df['severity'])
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_severity, y_severity, test_size=0.2, random_state=42, stratify=y_severity
    )
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    trainer_severity = ModelTrainer()
    evaluator_severity = ModelEvaluator()
    
    for model_type in models[:2]:  # Just RF and XGBoost
        logger.info(f"Training {model_type} for severity classification...")
        model = trainer_severity.train_classifier(
            model_type, X_train_scaled, y_train
        )
        
        metrics = evaluator_severity.evaluate_classifier(
            model, X_test_scaled, y_test, f"Severity_{model_type}"
        )
        logger.info(f"{model_type} metrics: {metrics}")
    
    # Task 4: RUL Prediction
    logger.info("\n=== RUL Prediction ===")
    
    # Use only faulty samples for RUL
    X_rul = faulty_df[feature_cols]
    y_rul = faulty_df['rul']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_rul, y_rul, test_size=0.2, random_state=42
    )
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train regression models
    trainer_rul = ModelTrainer()
    evaluator_rul = ModelEvaluator()
    
    regression_models = ['random_forest', 'xgboost']
    
    for model_type in regression_models:
        logger.info(f"Training {model_type} for RUL prediction...")
        model = trainer_rul.train_regressor(
            model_type, X_train_scaled, y_train
        )
        
        metrics = evaluator_rul.evaluate_regressor(
            model, X_test_scaled, y_test, f"RUL_{model_type}"
        )
        logger.info(f"{model_type} RUL metrics: {metrics}")
        
        # Save model
        trainer_rul.save_model(model_type, 'models/bearing_rul')
    
    # Save results
    rul_results = evaluator_rul.compare_models()
    rul_results.to_csv('results/bearing_rul_results.csv')
    
    # Visualize RUL predictions
    for model_name in ['RUL_random_forest', 'RUL_xgboost']:
        if model_name in evaluator_rul.results:
            fig = evaluator_rul.plot_regression_results(model_name, y_test)
            plt.savefig(f'results/bearing_{model_name}_predictions.png', dpi=300)
            plt.close()
    
    logger.info("\nTraining completed!")
    
    # Save preprocessor
    import joblib
    joblib.dump({
        'scaler': scaler,
        'label_encoder': le,
        'severity_encoder': le_severity,
        'feature_columns': feature_cols
    }, 'models/bearing_preprocessor.pkl')
    
    return {
        'binary_evaluator': evaluator_binary,
        'multi_evaluator': evaluator_multi,
        'severity_evaluator': evaluator_severity,
        'rul_evaluator': evaluator_rul
    }

def create_summary_report():
    """Create a summary report of all results"""
    print("\n" + "="*60)
    print("BEARING FAULT DETECTION - SUMMARY REPORT")
    print("="*60)
    
    # Load results
    try:
        binary_results = pd.read_csv('results/bearing_binary_classification_results.csv', index_col=0)
        multi_results = pd.read_csv('results/bearing_multiclass_classification_results.csv', index_col=0)
        rul_results = pd.read_csv('results/bearing_rul_results.csv', index_col=0)
        
        print("\n1. BINARY CLASSIFICATION (Healthy vs Faulty)")
        print("-"*40)
        print(binary_results[['accuracy', 'precision', 'recall', 'f1_score']])
        
        print("\n2. MULTI-CLASS CLASSIFICATION (Fault Types)")
        print("-"*40)
        print(multi_results[['accuracy', 'precision', 'recall', 'f1_score']])
        
        print("\n3. RUL PREDICTION")
        print("-"*40)
        print(rul_results[['rmse', 'mae', 'r2']])
        
        # Create comparison plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Binary classification
        binary_results[['accuracy', 'f1_score']].plot(kind='bar', ax=axes[0])
        axes[0].set_title('Binary Classification Performance')
        axes[0].set_ylim(0, 1.1)
        axes[0].legend(['Accuracy', 'F1-Score'])
        
        # Multi-class classification
        multi_results[['accuracy', 'f1_score']].plot(kind='bar', ax=axes[1])
        axes[1].set_title('Multi-class Classification Performance')
        axes[1].set_ylim(0, 1.1)
        axes[1].legend(['Accuracy', 'F1-Score'])
        
        # RUL prediction
        rul_results[['rmse', 'mae']].plot(kind='bar', ax=axes[2])
        axes[2].set_title('RUL Prediction Error')
        axes[2].set_ylabel('Error (hours)')
        axes[2].legend(['RMSE', 'MAE'])
        
        plt.tight_layout()
        plt.savefig('results/bearing_model_comparison_summary.png', dpi=300)
        plt.close()
        
        print("\nSummary plot saved to: results/bearing_model_comparison_summary.png")
        
    except Exception as e:
        print(f"Error creating summary: {e}")

if __name__ == "__main__":
    # Train models
    results = train_bearing_fault_models()
    
    # Create summary report
    create_summary_report()
    
    print("\n✅ All tasks completed successfully!")
    print("Check the 'results/' folder for detailed outputs")