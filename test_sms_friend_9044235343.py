# SMS Test with Your Friend's Number

from src.alert_manager import SmartAlertManager
from src.data_loader import DataGenerator
import numpy as np

def test_sms_with_friend():
    """Test SMS alerts with your friend's number"""
    
    print("=== SMS Test with Friend's Number ===")
    print()
    
    # Initialize alert manager
    alert_manager = SmartAlertManager()
    
    # Test SMS connection first
    print("Testing SMS connection...")
    test_results = alert_manager.test_notifications()
    
    if not test_results.get('sms', False):
        print("ERROR: SMS connection failed!")
        return
    
    print("SUCCESS: SMS connection working!")
    print()
    
    # Your friend's number (Indian number)
    friend_number = "+919044235343"
    
    print(f"Your Twilio number: +16674463150")
    print(f"Friend's number: {friend_number}")
    print()
    
    # Set SMS recipients
    alert_manager.set_recipients(sms_recipients=[friend_number])
    print("SMS recipient set!")
    print()
    
    # Generate critical alert scenario
    print("Generating CRITICAL alert scenario...")
    generator = DataGenerator(sample_rate=100, duration=10)
    data = generator.generate_healthy_signal()
    
    # Make it critical
    data['temperature'] = data['temperature'] + 30  # Very hot (90°C+)
    
    sensor_data = {
        'health_index': 25.0,  # Very critical health
        'rul': 3.0,  # Very critical RUL
        'vibration_rms': np.sqrt(np.mean(data['vibration']**2)),
        'temperature': np.mean(data['temperature']),
        'pressure': np.mean(data['pressure']),
        'current': np.mean(data['current'])
    }
    
    print(f"Critical Scenario:")
    print(f"  Temperature: {sensor_data['temperature']:.1f}°C (CRITICAL - threshold: 80°C)")
    print(f"  Health Index: {sensor_data['health_index']:.1f}% (CRITICAL - threshold: 50%)")
    print(f"  RUL: {sensor_data['rul']:.1f} hours (CRITICAL - threshold: 24 hours)")
    print()
    
    # Check for alerts
    alerts = alert_manager.check_equipment_alerts("CRITICAL-PUMP-TEST", sensor_data)
    
    if alerts:
        print(f"SMS ALERTS TRIGGERED: {len(alerts)}")
        for alert in alerts:
            print(f"  - {alert.severity.upper()}: {alert.message}")
        
        print()
        print("SUCCESS: SMS alerts sent to your friend!")
        print(f"SMS sent to: {friend_number}")
        print("Check your friend's phone for SMS messages.")
        print()
        
        # Show alert statistics
        stats = alert_manager.get_alert_statistics(1)
        print("Alert Statistics:")
        print(f"  Total alerts: {stats.get('total_alerts', 0)}")
        print(f"  Critical alerts: {stats.get('severity_distribution', {}).get('critical', 0)}")
        print(f"  Warning alerts: {stats.get('severity_distribution', {}).get('warning', 0)}")
        
    else:
        print("No alerts triggered")
    
    print()
    print("=== Test Complete ===")
    print("Your friend should receive SMS alerts on their phone!")

if __name__ == "__main__":
    test_sms_with_friend()
