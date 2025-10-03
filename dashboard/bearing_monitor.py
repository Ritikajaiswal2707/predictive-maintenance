import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generate_bearing_dataset import BearingDatasetGenerator

st.set_page_config(
    page_title="Bearing Health Monitor",
    page_icon="⚙️",
    layout="wide"
)

st.title("⚙️ Real-time Bearing Health Monitoring System")

# Load models and preprocessor
@st.cache_resource
def load_models():
    try:
        classifier = joblib.load('models/bearing_binary_random_forest.pkl')
        rul_model = joblib.load('models/bearing_rul_random_forest.pkl')
        preprocessor = joblib.load('models/bearing_preprocessor.pkl')
        return classifier, rul_model, preprocessor
    except:
        st.error("Models not found! Please run train_bearing_models.py first.")
        return None, None, None

classifier, rul_model, preprocessor = load_models()

# Sidebar
with st.sidebar:
    st.header("🔧 Bearing Configuration")
    
    bearing_id = st.text_input("Bearing ID", "BRG-001")
    
    st.subheader("Simulation Settings")
    condition = st.selectbox(
        "Bearing Condition",
        ["healthy", "inner_race", "outer_race", "ball"]
    )
    
    if condition != "healthy":
        severity = st.select_slider(
            "Fault Severity",
            ["mild", "moderate", "severe"]
        )
    else:
        severity = None
    
    st.subheader("Display Settings")
    show_frequency = st.checkbox("Show Frequency Analysis", True)
    show_trends = st.checkbox("Show Historical Trends", True)

# Main content
if classifier is not None:
    # Generate signal
    generator = BearingDatasetGenerator()
    
    # Generate appropriate signal
    if condition == "healthy":
        t, signal = generator.generate_healthy_bearing(duration=1)
    elif condition == "inner_race":
        t, signal = generator.generate_inner_race_fault(duration=1, severity=severity)
    elif condition == "outer_race":
        t, signal = generator.generate_outer_race_fault(duration=1, severity=severity)
    else:  # ball
        t, signal = generator.generate_ball_fault(duration=1, severity=severity)
    