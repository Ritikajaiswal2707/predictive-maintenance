import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import DataGenerator
from src.utils import HealthIndexCalculator

st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="🔧",
    layout="wide"
)

st.title("🏭 Predictive Maintenance Dashboard")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    equipment_type = st.selectbox(
        "Equipment Type",
        ["Centrifugal Pump", "Reciprocating Compressor", "Screw Compressor"]
    )
    
    equipment_id = st.text_input("Equipment ID", "PUMP-001")
    
    simulation_mode = st.radio(
        "Simulation Mode",
        ["Healthy", "Degrading", "Faulty"]
    )
    
    if simulation_mode == "Faulty":
        fault_type = st.selectbox(
            "Fault Type",
            ["bearing", "imbalance", "cavitation"]
        )

# Main content
col1, col2, col3, col4 = st.columns(4)

# Initialize data generator
if 'generator' not in st.session_state:
    st.session_state.generator = DataGenerator(sample_rate=100, duration=10)
    st.session_state.health_calculator = HealthIndexCalculator()

# Generate data based on mode
if simulation_mode == "Healthy":
    data = st.session_state.generator.generate_healthy_signal()
elif simulation_mode == "Faulty":
    data = st.session_state.generator.generate_faulty_signal(fault_type)
else:
    data = st.session_state.generator.generate_healthy_signal()
    # Add some degradation
    data['vibration'] = data['vibration'] * 1.5
    data['temperature'] = data['temperature'] + 10

# Extract features for display
features = {
    'vibration_rms': np.sqrt(np.mean(data['vibration']**2)),
    'temperature_mean': np.mean(data['temperature']),
    'pressure_mean': np.mean(data['pressure']),
    'current_mean': np.mean(data['current'])
}

# Calculate health index
features_df = pd.DataFrame([features])
health_index = st.session_state.health_calculator.calculate_health_index(features_df)

# Display metrics
col1.metric("Health Index", f"{health_index:.1f}%")
col2.metric("RUL (Hours)", data['rul'])
col3.metric("Vibration RMS", f"{features['vibration_rms']:.2f} mm/s")
col4.metric("Temperature", f"{features['temperature_mean']:.1f} °C")

# Plots
col1, col2 = st.columns(2)

with col1:
    st.subheader("Vibration Signal")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data['time'][:1000],
        y=data['vibration'][:1000],
        mode='lines',
        name='Vibration'
    ))
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Temperature Trend")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data['time'][:1000],
        y=data['temperature'][:1000],
        mode='lines',
        name='Temperature',
        line=dict(color='red')
    ))
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

# Status message
if health_index < 50:
    st.error("⚠️ CRITICAL: Immediate maintenance required!")
elif health_index < 70:
    st.warning("⚠️ WARNING: Schedule maintenance soon")
else:
    st.success("✅ Equipment operating normally")