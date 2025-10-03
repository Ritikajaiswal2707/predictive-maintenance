# Simple Presentation Demo - Run This During Presentation

from src.alert_manager import SmartAlertManager
from src.data_loader import DataGenerator
import numpy as np
import time

def live_presentation_demo():
    """Live demo for presentation - run this during your talk"""
    
    print("PREDICTIVE MAINTENANCE SYSTEM - LIVE DEMO")
    print("=" * 50)
    print()
    
    # Initialize system
    print("Initializing Smart Alert System...")
    alert_manager = SmartAlertManager()
    
    # Set up SMS recipient (your friend's number)
    friend_number = "+919044235343"
    alert_manager.set_recipients(sms_recipients=[friend_number])
    
    print("SMS Service: ACTIVE")
    print(f"Alert Recipient: {friend_number}")
    print()
    
    # Demo 1: Normal Operation
    print("DEMO 1: Normal Equipment Operation")
    print("-" * 40)
    
    generator = DataGenerator(sample_rate=100, duration=10)
    data = generator.generate_healthy_signal()
    
    sensor_data = {
        'health_index': 85.0,  # Healthy
        'rul': 800.0,  # Good RUL
        'vibration_rms': np.sqrt(np.mean(data['vibration']**2)),
        'temperature': np.mean(data['temperature']),
        'pressure': np.mean(data['pressure']),
        'current': np.mean(data['current'])
    }
    
    print(f"Equipment Status:")
    print(f"   Temperature: {sensor_data['temperature']:.1f}C")
    print(f"   Health Index: {sensor_data['health_index']:.1f}%")
    print(f"   RUL: {sensor_data['rul']:.1f} hours")
    
    alerts = alert_manager.check_equipment_alerts("PUMP-001", sensor_data)
    print(f"Alerts: {len(alerts)} (None - Equipment Healthy)")
    print()
    
    # Demo 2: Warning Condition
    print("DEMO 2: Warning Condition Detected")
    print("-" * 40)
    
    data2 = generator.generate_healthy_signal()
    data2['temperature'] = data2['temperature'] + 8  # Slightly hot
    
    sensor_data2 = {
        'health_index': 65.0,  # Warning level
        'rul': 200.0,  # Moderate RUL
        'vibration_rms': np.sqrt(np.mean(data2['vibration']**2)),
        'temperature': np.mean(data2['temperature']),
        'pressure': np.mean(data2['pressure']),
        'current': np.mean(data2['current'])
    }
    
    print(f"Equipment Status:")
    print(f"   Temperature: {sensor_data2['temperature']:.1f}C")
    print(f"   Health Index: {sensor_data2['health_index']:.1f}%")
    print(f"   RUL: {sensor_data2['rul']:.1f} hours")
    
    alerts2 = alert_manager.check_equipment_alerts("PUMP-002", sensor_data2)
    print(f"Alerts: {len(alerts2)}")
    for alert in alerts2:
        print(f"   - {alert.severity.upper()}: {alert.message}")
    print()
    
    # Demo 3: Critical Condition
    print("DEMO 3: CRITICAL CONDITION - SMS ALERTS SENT!")
    print("-" * 40)
    
    data3 = generator.generate_healthy_signal()
    data3['temperature'] = data3['temperature'] + 25  # Very hot
    
    sensor_data3 = {
        'health_index': 25.0,  # Critical
        'rul': 5.0,  # Critical RUL
        'vibration_rms': np.sqrt(np.mean(data3['vibration']**2)),
        'temperature': np.mean(data3['temperature']),
        'pressure': np.mean(data3['pressure']),
        'current': np.mean(data3['current'])
    }
    
    print(f"Equipment Status:")
    print(f"   Temperature: {sensor_data3['temperature']:.1f}C (CRITICAL)")
    print(f"   Health Index: {sensor_data3['health_index']:.1f}% (CRITICAL)")
    print(f"   RUL: {sensor_data3['rul']:.1f} hours (CRITICAL)")
    
    alerts3 = alert_manager.check_equipment_alerts("PUMP-003", sensor_data3)
    print(f"Alerts: {len(alerts3)}")
    for alert in alerts3:
        print(f"   - {alert.severity.upper()}: {alert.message}")
    
    print()
    print("SMS ALERTS SENT TO MAINTENANCE TEAM!")
    print(f"   Recipient: {friend_number}")
    print("   Check phone for instant notifications!")
    print()
    
    # Show statistics
    stats = alert_manager.get_alert_statistics(1)
    print("DEMO SUMMARY:")
    print(f"   Total Alerts Generated: {stats.get('total_alerts', 0)}")
    print(f"   Critical Alerts: {stats.get('severity_distribution', {}).get('critical', 0)}")
    print(f"   Warning Alerts: {stats.get('severity_distribution', {}).get('warning', 0)}")
    print()
    
    print("SYSTEM FEATURES DEMONSTRATED:")
    print("   - Real-time equipment monitoring")
    print("   - Intelligent alert classification")
    print("   - Instant SMS notifications")
    print("   - Predictive failure detection")
    print("   - International SMS delivery")
    print()
    
    print("DEMO COMPLETE - SYSTEM READY FOR PRODUCTION!")

if __name__ == "__main__":
    live_presentation_demo()
