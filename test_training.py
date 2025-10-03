from src.data_loader import DataLoader
from src.preprocessing import DataPreprocessor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

print("Testing complete pipeline...")

# 1. Generate data
print("\n1. Generating synthetic data...")
loader = DataLoader()
df = loader.generate_synthetic_dataset(n_samples=500)
print(f"   Generated {len(df)} samples")

# 2. Preprocess data
print("\n2. Preprocessing data...")
preprocessor = DataPreprocessor()
df_clean = preprocessor.clean_data(df)
X, y = preprocessor.prepare_classification_data(df_clean)
X_train, X_test, y_train, y_test = preprocessor.split_and_scale(X, y)
print(f"   Training samples: {len(X_train)}")
print(f"   Test samples: {len(X_test)}")

# 3. Train a simple model
print("\n3. Training Random Forest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Evaluate
print("\n4. Evaluating model...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"   Accuracy: {accuracy:.2%}")

# 5. Show classification report
print("\n5. Classification Report:")
print(classification_report(y_test, y_pred, 
                          target_names=preprocessor.label_encoder.classes_))

print("\nPipeline test successful!")