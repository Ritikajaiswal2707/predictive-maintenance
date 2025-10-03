# Demo SMS Alert System

from src.alert_manager import SmartAlertManager
from src.data_loader import DataGenerator
import numpy as np

def demo_sms_alerts():
    """Demo the SMS alert system with your Twilio credentials"""
    
    print("=== SMS Alert System Demo ===")
    print()
    
    # Initialize alert manager (it will load your Twilio config)
    alert_manager = SmartAlertManager()
    
    # Set SMS recipients (you can add more numbers here)
    sms_recipients = [
        "+16674463150",  # Your number (will receive alerts)
        # Add other numbers here: "+1234567890", "+0987654321"
    ]
    
    alert_manager.set_recipients(sms_recipients=sms_recipients)
    print(f"SMS recipients set: {len(sms_recipients)} numbers")
    print()
    
    # Test notification channels
    print("Testing notification channels...")
    test_results = alert_manager.test_notifications()
    
    for channel, result in test_results.items():
        status = "SUCCESS" if result else "FAILED"
        print(f"  {channel.capitalize()}: {status}")
    
    print()
    
    # Generate test scenarios that will trigger SMS alerts
    generator = DataGenerator(sample_rate=100, duration=10)
    
    print("Generating test scenarios that will trigger SMS alerts...")
    print()
    
    # Scenario 1: Critical temperature alert
    print("Scenario 1: Critical Temperature Alert")
    data = generator.generate_healthy_signal()
    data['temperature'] = data['temperature'] + 25  # Make it hot (85°C+)
    
    sensor_data = {
        'health_index': 45.0,  # Low health
        'rul': 15.0,  # Low RUL
        'vibration_rms': np.sqrt(np.mean(data['vibration']**2)),
        'temperature': np.mean(data['temperature']),  # This will be high
        'pressure': np.mean(data['pressure']),
        'current': np.mean(data['current'])
    }
    
    print(f"  Temperature: {sensor_data['temperature']:.1f}°C (threshold: 80°C)")
    print(f"  Health Index: {sensor_data['health_index']:.1f}%")
    print(f"  RUL: {sensor_data['rul']:.1f} hours")
    
    # Check for alerts
    alerts = alert_manager.check_equipment_alerts("PUMP-CRITICAL-001", sensor_data)
    
    if alerts:
        print(f"  SMS ALERTS TRIGGERED: {len(alerts)}")
        for alert in alerts:
            print(f"    - {alert.severity.upper()}: {alert.message}")
    else:
        print("  No alerts triggered")
    
    print()
    
    # Scenario 2: Low RUL alert
    print("Scenario 2: Low RUL Alert")
    data2 = generator.generate_healthy_signal()
    
    sensor_data2 = {
        'health_index': 60.0,  # Moderate health
        'rul': 10.0,  # Very low RUL
        'vibration_rms': np.sqrt(np.mean(data2['vibration']**2)),
        'temperature': np.mean(data2['temperature']),
        'pressure': np.mean(data2['pressure']),
        'current': np.mean(data2['current'])
    }
    
    print(f"  RUL: {sensor_data2['rul']:.1f} hours (threshold: 24 hours)")
    print(f"  Health Index: {sensor_data2['health_index']:.1f}%")
    
    # Check for alerts
    alerts2 = alert_manager.check_equipment_alerts("PUMP-LOW-RUL-002", sensor_data2)
    
    if alerts2:
        print(f"  SMS ALERTS TRIGGERED: {len(alerts2)}")
        for alert in alerts2:
            print(f"    - {alert.severity.upper()}: {alert.message}")
    else:
        print("  No alerts triggered")
    
    print()
    
    # Show alert statistics
    print("Alert Statistics:")
    stats = alert_manager.get_alert_statistics(1)  # Last 24 hours
    print(f"  Total alerts: {stats.get('total_alerts', 0)}")
    print(f"  Critical alerts: {stats.get('severity_distribution', {}).get('critical', 0)}")
    print(f"  Warning alerts: {stats.get('severity_distribution', {}).get('warning', 0)}")
    print(f"  Unresolved alerts: {stats.get('unresolved_alerts', 0)}")
    
    print()
    print("=== Demo Complete ===")
    print("Check your phone for SMS alerts!")
    print("Note: SMS will only be sent if alerts are triggered and severity requires SMS")

if __name__ == "__main__":
    demo_sms_alerts()
