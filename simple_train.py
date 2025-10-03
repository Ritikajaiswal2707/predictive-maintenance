import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
import joblib
import os

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

print("=== Training Predictive Maintenance Models ===")

# Generate synthetic training data
print("1. Generating synthetic training data...")
np.random.seed(42)

n_samples = 1000
data = []

for i in range(n_samples):
    # Generate features
    vibration_rms = np.random.normal(1.0, 0.3)
    temperature = np.random.normal(65, 10)
    pressure = np.random.normal(5.0, 0.5)
    current = np.random.normal(50, 5)
    
    # Determine health status based on features
    if vibration_rms > 2.0 or temperature > 75:
        health_status = 'faulty'
        rul = np.random.randint(10, 100)
    elif vibration_rms > 1.5 or temperature > 70:
        health_status = 'degrading'
        rul = np.random.randint(100, 500)
    else:
        health_status = 'healthy'
        rul = np.random.randint(500, 1000)
    
    data.append({
        'vibration_rms': vibration_rms,
        'temperature': temperature,
        'pressure': pressure,
        'current': current,
        'health_status': health_status,
        'rul': rul
    })

df = pd.DataFrame(data)
print(f"   Generated {len(df)} samples")

# Prepare features
feature_cols = ['vibration_rms', 'temperature', 'pressure', 'current']
X = df[feature_cols]
y_class = df['health_status']
y_reg = df['rul']

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_class)

# Split data
X_train, X_test, y_train_class, y_test_class, y_train_reg, y_test_reg = train_test_split(
    X, y_encoded, y_reg, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("2. Training classification model...")
# Train classifier
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train_scaled, y_train_class)

# Evaluate classifier
y_pred_class = classifier.predict(X_test_scaled)
accuracy = accuracy_score(y_test_class, y_pred_class)
print(f"   Classification accuracy: {accuracy:.3f}")

print("3. Training regression model...")
# Train regressor
regressor = RandomForestRegressor(n_estimators=100, random_state=42)
regressor.fit(X_train_scaled, y_train_reg)

# Evaluate regressor
y_pred_reg = regressor.predict(X_test_scaled)
rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_reg))
print(f"   Regression RMSE: {rmse:.1f} hours")

print("4. Saving models...")
# Save models
joblib.dump(classifier, 'models/classifier_random_forest.pkl')
joblib.dump(regressor, 'models/regressor_random_forest.pkl')

# Save preprocessor
preprocessor_data = {
    'scaler': scaler,
    'label_encoder': label_encoder
}
joblib.dump(preprocessor_data, 'models/preprocessor.pkl')

print("   Models saved successfully!")
print("\nTraining completed! You can now run:")
print("- python demo.py")
print("- streamlit run dashboard/app.py")
