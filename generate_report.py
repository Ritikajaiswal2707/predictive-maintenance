import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("Generating Performance Report...\n")

# Load results
try:
    class_results = pd.read_csv('results/classification_comparison.csv', index_col=0)
    reg_results = pd.read_csv('results/regression_comparison.csv', index_col=0)
    
    print("=== CLASSIFICATION RESULTS ===")
    print(class_results)
    print("\n=== REGRESSION RESULTS ===")
    print(reg_results)
    
    # Create visualizations
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Classification metrics
    class_results[['accuracy', 'precision', 'recall', 'f1_score']].plot(kind='bar', ax=ax1)
    ax1.set_title('Classification Model Comparison')
    ax1.set_ylabel('Score')
    ax1.legend(loc='lower right')
    
    # Regression metrics
    reg_results[['rmse', 'mae']].plot(kind='bar', ax=ax2)
    ax2.set_title('Regression Model Comparison')
    ax2.set_ylabel('Error')
    
    plt.tight_layout()
    plt.savefig('results/model_comparison.png', dpi=300)
    print("\nReport saved to results/model_comparison.png")
    
except Exception as e:
    print(f"Error: {e}")
    print("Make sure you've run the training script first!")