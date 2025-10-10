# 🚀 Complete Setup Guide for Predictive Maintenance System

## 📋 Prerequisites
- Python 3.8 or higher
- pip package manager
- Git (for cloning)

## 🔧 Step-by-Step Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/Ritikajaiswal2707/predictive-maintenance.git
cd predictive-maintenance
```

### 2. Install Dependencies
```bash
# Install core dependencies
pip install streamlit numpy pandas scikit-learn matplotlib plotly

# Install additional dependencies
pip install joblib pyyaml tqdm

# Optional: Install full requirements (if needed)
pip install -r requirements.txt
```

### 3. Train the Models
```bash
python simple_train.py
```

### 4. Test the System
```bash
# Test model loading
python -c "import joblib; print('Testing models...'); classifier = joblib.load('models/classifier_random_forest.pkl'); regressor = joblib.load('models/regressor_random_forest.pkl'); preprocessor = joblib.load('models/preprocessor.pkl'); print('✅ All models loaded successfully!')"
```

### 5. Run the Dashboard
```bash
# Start the simplified dashboard
python -m streamlit run simple_dashboard.py --server.port 8502

# Alternative: Start the full dashboard (if you want the complete version)
python -m streamlit run dashboard/app.py --server.port 8501
```

### 6. Access the Dashboard
Open your web browser and go to:
- **Simplified Dashboard**: http://localhost:8502
- **Full Dashboard**: http://localhost:8501

## 🎮 Quick Start Commands

### One-Line Setup (after cloning):
```bash
pip install streamlit numpy pandas scikit-learn matplotlib plotly joblib pyyaml tqdm && python simple_train.py && python -m streamlit run simple_dashboard.py --server.port 8502
```

### Test the System:
```bash
python -c "
import pandas as pd
import numpy as np
import joblib

print('=== Testing Predictive Maintenance System ===')

# Load models
classifier = joblib.load('models/classifier_random_forest.pkl')
regressor = joblib.load('models/regressor_random_forest.pkl')
preprocessor_data = joblib.load('models/preprocessor.pkl')
scaler = preprocessor_data['scaler']
label_encoder = preprocessor_data['label_encoder']

# Test scenarios
test_scenarios = [
    ('Healthy Equipment', {'vibration_rms': 0.8, 'temperature': 65, 'pressure': 5.0, 'current': 50}),
    ('Degrading Equipment', {'vibration_rms': 1.8, 'temperature': 72, 'pressure': 5.2, 'current': 55}),
    ('Faulty Equipment', {'vibration_rms': 2.5, 'temperature': 80, 'pressure': 5.5, 'current': 65})
]

print('\\nMaking predictions:')
for scenario_name, features in test_scenarios:
    feature_df = pd.DataFrame([features])
    X_scaled = scaler.transform(feature_df)
    
    health_pred = classifier.predict(X_scaled)[0]
    health_label = label_encoder.inverse_transform([health_pred])[0]
    rul_pred = regressor.predict(X_scaled)[0]
    
    print(f'{scenario_name}: {health_label} (RUL: {rul_pred:.0f} hours)')

print('\\n✅ System is working correctly!')
"
```

## 🎯 Dashboard Features

### Simplified Dashboard (simple_dashboard.py):
- ✅ Real-time equipment monitoring
- ✅ Interactive sensor controls
- ✅ Health status predictions
- ✅ RUL (Remaining Useful Life) estimates
- ✅ Live monitoring simulation
- ✅ Visual charts and metrics

### Full Dashboard (dashboard/app.py):
- ✅ Advanced equipment monitoring
- ✅ Multiple equipment types
- ✅ Comprehensive analytics
- ✅ Alert system integration
- ✅ Historical data visualization

## 🔧 Troubleshooting

### If Streamlit command not found:
```bash
# Use Python module instead
python -m streamlit run simple_dashboard.py --server.port 8502
```

### If port is already in use:
```bash
# Use a different port
python -m streamlit run simple_dashboard.py --server.port 8503
```

### If models not found:
```bash
# Retrain the models
python simple_train.py
```

### If dependencies missing:
```bash
# Install missing packages
pip install [package_name]
```

## 📊 System Capabilities

- **Equipment Health Monitoring**: Real-time status classification
- **Predictive Analytics**: RUL estimation in hours
- **Multi-Sensor Support**: Vibration, temperature, pressure, current
- **Interactive Dashboard**: User-friendly web interface
- **Machine Learning Models**: Random Forest classifier and regressor
- **Real-time Updates**: Live monitoring and alerts

## 🎉 Success Indicators

You'll know the system is working when:
1. ✅ Models load without errors
2. ✅ Dashboard opens in browser
3. ✅ Predictions show realistic results
4. ✅ Interactive controls respond
5. ✅ Health status updates in real-time

## 📞 Support

If you encounter issues:
1. Check Python version (3.8+)
2. Verify all dependencies are installed
3. Ensure models are trained
4. Check port availability
5. Review error messages in terminal

**Ready to prevent equipment failures before they happen!** 🚀
