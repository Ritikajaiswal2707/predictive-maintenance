import pandas as pd
import numpy as np
import joblib
from src.data_loader import DataGenerator
from src.preprocessing import DataPreprocessor

print("=== Predictive Maintenance Demo ===\n")

# Load trained models
print("1. Loading trained models...")
try:
    classifier = joblib.load('models/classifier_random_forest.pkl')
    regressor = joblib.load('models/regressor_random_forest.pkl')
    preprocessor_data = joblib.load('models/preprocessor.pkl')
    scaler = preprocessor_data['scaler']
    label_encoder = preprocessor_data['label_encoder']
    print("   [OK] Models loaded successfully")
except:
    print("   [ERROR] Models not found. Run 'python train_models.py' first!")
    exit()

# Generate new test data
print("\n2. Generating new test equipment data...")
generator = DataGenerator(sample_rate=100, duration=10)

# Test different scenarios
scenarios = [
    ("Healthy Pump", generator.generate_healthy_signal()),
    ("Bearing Fault", generator.generate_faulty_signal('bearing')),
    ("Imbalance", generator.generate_faulty_signal('imbalance')),
    ("Cavitation", generator.generate_faulty_signal('cavitation'))
]

print("\n3. Making predictions:\n")
for scenario_name, data in scenarios:
    # Extract simple features that match training data
    features = {
        'vibration_rms': np.sqrt(np.mean(data['vibration']**2)),
        'temperature': np.mean(data['temperature']),
        'pressure': np.mean(data['pressure']),
        'current': np.mean(data['current'])
    }
    
    # Prepare for prediction
    feature_df = pd.DataFrame([features])
    
    # Scale features
    X_scaled = scaler.transform(feature_df)
    
    # Make predictions
    health_pred = classifier.predict(X_scaled)[0]
    health_label = label_encoder.inverse_transform([health_pred])[0]
    
    rul_pred = regressor.predict(X_scaled)[0]
    
    print(f"{scenario_name}:")
    print(f"  - Predicted Status: {health_label}")
    print(f"  - Predicted RUL: {rul_pred:.0f} hours")
    print(f"  - Actual Status: {data['health_status']}")
    print(f"  - Actual RUL: {data['rul']} hours")
    print()

print("Demo completed!")