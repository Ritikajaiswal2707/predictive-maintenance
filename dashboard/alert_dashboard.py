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
from src.alert_manager import SmartAlertManager

st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="🔧",
    layout="wide"
)

st.title("🏭 Predictive Maintenance Dashboard")

# Initialize alert manager
@st.cache_resource
def get_alert_manager():
    return SmartAlertManager()

alert_manager = get_alert_manager()

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
    
    st.header("🚨 Alert Settings")
    
    # Alert recipients
    email_recipients = st.text_area(
        "Email Recipients (one per line)",
        "maintenance@yourcompany.com\noperations@yourcompany.com",
        height=100
    )
    
    sms_recipients = st.text_area(
        "SMS Recipients (phone numbers, one per line)",
        "+1234567890\n+0987654321",
        height=100
    )
    
    # Update recipients
    if st.button("Update Alert Recipients"):
        email_list = [email.strip() for email in email_recipients.split('\n') if email.strip()]
        sms_list = [sms.strip() for sms in sms_recipients.split('\n') if sms.strip()]
        alert_manager.set_recipients(email_list, sms_list)
        st.success("Alert recipients updated!")

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
    'current_mean': np.mean(data['current']),
    'health_index': 0,  # Will be calculated
    'rul': data['rul']
}

# Calculate health index
features_df = pd.DataFrame([features])
health_index = st.session_state.health_calculator.calculate_health_index(features_df)
features['health_index'] = health_index

# Check for alerts
sensor_data = {
    'health_index': health_index,
    'rul': data['rul'],
    'vibration_rms': features['vibration_rms'],
    'temperature': features['temperature_mean'],
    'pressure': features['pressure_mean'],
    'current': features['current_mean']
}

alerts_triggered = alert_manager.check_equipment_alerts(equipment_id, sensor_data)

# Display metrics with alert indicators
health_color = "normal"
if health_index < 50:
    health_color = "inverse"
elif health_index < 70:
    health_color = "off"

col1.metric(
    "Health Index", 
    f"{health_index:.1f}%",
    delta=None,
    delta_color=health_color
)

rul_color = "normal"
if data['rul'] < 24:
    rul_color = "inverse"
elif data['rul'] < 48:
    rul_color = "off"

col2.metric(
    "RUL (Hours)", 
    data['rul'],
    delta=None,
    delta_color=rul_color
)

col3.metric("Vibration RMS", f"{features['vibration_rms']:.2f} mm/s")
col4.metric("Temperature", f"{features['temperature_mean']:.1f} °C")

# Alert notifications
if alerts_triggered:
    st.header("🚨 Active Alerts")
    
    for alert in alerts_triggered:
        if alert.severity == "critical":
            st.error(f"🔴 **{alert.severity.upper()}**: {alert.message}")
        elif alert.severity == "warning":
            st.warning(f"🟡 **{alert.severity.upper()}**: {alert.message}")
        elif alert.severity == "emergency":
            st.error(f"🚨 **{alert.severity.upper()}**: {alert.message}")
        else:
            st.info(f"🔵 **{alert.severity.upper()}**: {alert.message}")

# Alert History Section
st.header("📊 Alert History")

# Get recent alerts
recent_alerts = alert_manager.get_active_alerts(equipment_id)

if recent_alerts:
    # Convert to DataFrame for display
    alert_data = []
    for alert in recent_alerts:
        alert_data.append({
            'Time': alert.triggered_at.strftime('%Y-%m-%d %H:%M:%S'),
            'Type': alert.alert_type,
            'Severity': alert.severity,
            'Message': alert.message,
            'Value': alert.value,
            'Threshold': alert.threshold,
            'Status': 'Resolved' if alert.resolved_at else 'Active'
        })
    
    alert_df = pd.DataFrame(alert_data)
    
    # Display alerts table
    st.dataframe(alert_df, use_container_width=True)
    
    # Alert statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Alerts", len(recent_alerts))
    
    with col2:
        critical_count = len([a for a in recent_alerts if a.severity == 'critical'])
        st.metric("Critical Alerts", critical_count)
    
    with col3:
        active_count = len([a for a in recent_alerts if not a.resolved_at])
        st.metric("Active Alerts", active_count)
    
    # Alert trend chart
    if len(recent_alerts) > 1:
        st.subheader("Alert Trend")
        
        # Group alerts by hour
        alert_df['Hour'] = pd.to_datetime(alert_df['Time']).dt.floor('H')
        hourly_counts = alert_df.groupby(['Hour', 'Severity']).size().reset_index(name='Count')
        
        fig = px.bar(
            hourly_counts, 
            x='Hour', 
            y='Count', 
            color='Severity',
            title='Alerts by Hour and Severity',
            color_discrete_map={
                'info': '#36a64f',
                'warning': '#ff9800',
                'critical': '#f44336',
                'emergency': '#e91e63'
            }
        )
        
        st.plotly_chart(fig, use_container_width=True)

else:
    st.info("No recent alerts found for this equipment.")

# Alert Management Section
st.header("🔧 Alert Management")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Test Notifications")
    
    if st.button("Test Email Connection"):
        results = alert_manager.test_notifications()
        if results.get('email', False):
            st.success("✅ Email connection successful!")
        else:
            st.error("❌ Email connection failed!")
    
    if st.button("Test SMS Connection"):
        results = alert_manager.test_notifications()
        if results.get('sms', False):
            st.success("✅ SMS connection successful!")
        else:
            st.error("❌ SMS connection failed!")

with col2:
    st.subheader("Alert Statistics")
    
    if st.button("Generate Daily Summary"):
        success = alert_manager.send_daily_summary()
        if success:
            st.success("✅ Daily summary sent!")
        else:
            st.error("❌ Failed to send daily summary!")
    
    if st.button("Cleanup Old Alerts"):
        cleaned = alert_manager.cleanup_old_alerts(30)
        st.success(f"✅ Cleaned up {cleaned} old alerts!")

# Plots
st.header("📈 Equipment Monitoring")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Vibration Signal")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data['time'][:1000],
        y=data['vibration'][:1000],
        mode='lines',
        name='Vibration',
        line=dict(color='blue')
    ))
    
    # Add threshold line if vibration is high
    if features['vibration_rms'] > 2.0:
        fig.add_hline(y=2.0, line_dash="dash", line_color="red", 
                     annotation_text="Warning Threshold")
    
    fig.update_layout(height=400, title="Vibration Signal")
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
    
    # Add threshold line if temperature is high
    if features['temperature_mean'] > 70:
        fig.add_hline(y=70, line_dash="dash", line_color="orange", 
                     annotation_text="Warning Threshold")
    
    fig.update_layout(height=400, title="Temperature Trend")
    st.plotly_chart(fig, use_container_width=True)

# Status message
if health_index < 50:
    st.error("⚠️ CRITICAL: Immediate maintenance required!")
elif health_index < 70:
    st.warning("⚠️ WARNING: Schedule maintenance soon")
else:
    st.success("✅ Equipment operating normally")

# Auto-refresh
if st.checkbox("Auto-refresh (30 seconds)"):
    time.sleep(30)
    st.rerun()
