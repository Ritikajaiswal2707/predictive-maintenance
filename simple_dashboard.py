import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="🔧",
    layout="wide"
)

st.title("🏭 Predictive Maintenance Dashboard")

# Load models
@st.cache_data
def load_models():
    try:
        classifier = joblib.load('models/classifier_random_forest.pkl')
        regressor = joblib.load('models/regressor_random_forest.pkl')
        preprocessor_data = joblib.load('models/preprocessor.pkl')
        scaler = preprocessor_data['scaler']
        label_encoder = preprocessor_data['label_encoder']
        return classifier, regressor, scaler, label_encoder
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None

# Sidebar
with st.sidebar:
    st.header("⚙️ Equipment Monitoring")
    
    st.subheader("Sensor Readings")
    vibration_rms = st.slider("Vibration RMS", 0.0, 5.0, 1.0, 0.1)
    temperature = st.slider("Temperature (°C)", 50.0, 100.0, 65.0, 1.0)
    pressure = st.slider("Pressure (bar)", 3.0, 8.0, 5.0, 0.1)
    current = st.slider("Current (A)", 30.0, 80.0, 50.0, 1.0)

# Load models
classifier, regressor, scaler, label_encoder = load_models()

if classifier is not None:
    # Prepare features
    features = {
        'vibration_rms': vibration_rms,
        'temperature': temperature,
        'pressure': pressure,
        'current': current
    }
    
    feature_df = pd.DataFrame([features])
    X_scaled = scaler.transform(feature_df)
    
    # Make predictions
    health_pred = classifier.predict(X_scaled)[0]
    health_label = label_encoder.inverse_transform([health_pred])[0]
    rul_pred = regressor.predict(X_scaled)[0]
    
    # Display results
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Health Status", health_label)
    
    with col2:
        st.metric("Remaining Useful Life", f"{rul_pred:.0f} hours")
    
    with col3:
        # Calculate health index
        if health_label == 'healthy':
            health_index = 90 + np.random.randint(-5, 10)
            color = "green"
        elif health_label == 'degrading':
            health_index = 60 + np.random.randint(-10, 10)
            color = "orange"
        else:
            health_index = 20 + np.random.randint(-10, 10)
            color = "red"
        
        st.metric("Health Index", f"{health_index}%")
    
    # Status indicator
    if health_label == 'healthy':
        st.success("✅ Equipment is operating normally")
    elif health_label == 'degrading':
        st.warning("⚠️ Equipment showing signs of degradation")
    else:
        st.error("🚨 Equipment requires immediate attention!")
    
    # Charts
    st.subheader("📊 Equipment Metrics")
    
    # Create a simple chart
    chart_data = pd.DataFrame({
        'Metric': ['Vibration RMS', 'Temperature', 'Pressure', 'Current'],
        'Value': [vibration_rms, temperature, pressure, current],
        'Normal Range': [1.0, 65.0, 5.0, 50.0]
    })
    
    st.bar_chart(chart_data.set_index('Metric'))
    
    # Real-time simulation
    st.subheader("🔄 Real-time Monitoring")
    
    if st.button("Start Live Monitoring"):
        placeholder = st.empty()
        
        for i in range(10):
            # Simulate changing values
            new_vibration = vibration_rms + np.random.normal(0, 0.1)
            new_temp = temperature + np.random.normal(0, 2)
            
            # Update features
            new_features = {
                'vibration_rms': max(0, new_vibration),
                'temperature': max(50, min(100, new_temp)),
                'pressure': pressure + np.random.normal(0, 0.1),
                'current': current + np.random.normal(0, 1)
            }
            
            new_feature_df = pd.DataFrame([new_features])
            new_X_scaled = scaler.transform(new_feature_df)
            
            new_health_pred = classifier.predict(new_X_scaled)[0]
            new_health_label = label_encoder.inverse_transform([new_health_pred])[0]
            new_rul_pred = regressor.predict(new_X_scaled)[0]
            
            with placeholder.container():
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Health Status", new_health_label)
                with col2:
                    st.metric("RUL", f"{new_rul_pred:.0f} hours")
                with col3:
                    st.metric("Vibration", f"{new_vibration:.2f}")
            
            time.sleep(1)

else:
    st.error("❌ Could not load models. Please ensure models are trained first.")
