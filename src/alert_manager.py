import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import threading
import time

from .alert_config import AlertConfiguration, AlertType, AlertSeverity
from .alert_history import AlertHistoryManager, AlertRecord
from .email_service import EmailNotificationService
from .sms_service import SMSNotificationService, SlackNotificationService

class SmartAlertManager:
    """Main alert management system"""
    
    def __init__(self, config_path: str = "configs/alert_config.yaml"):
        self.config = AlertConfiguration(config_path)
        self.history_manager = AlertHistoryManager()
        self.email_service = EmailNotificationService(
            self.config.notification_channels.get('email', {})
        )
        self.sms_service = SMSNotificationService(
            self.config.notification_channels.get('sms', {})
        )
        self.slack_service = SlackNotificationService(
            self.config.notification_channels.get('slack', {})
        )
        
        # Alert recipients
        self.recipients = {
            'email': [],
            'sms': [],
            'slack': []
        }
        
        # Alert throttling (prevent spam)
        self.throttle_cache = {}
        self.throttle_window = 300  # 5 minutes
        
        # Background monitoring
        self.monitoring_active = False
        self.monitoring_thread = None
        
        logging.info("Smart Alert Manager initialized")
    
    def set_recipients(self, email_recipients: List[str] = None, 
                      sms_recipients: List[str] = None):
        """Set alert recipients"""
        if email_recipients:
            self.recipients['email'] = email_recipients
        if sms_recipients:
            self.recipients['sms'] = sms_recipients
        
        logging.info(f"Recipients set - Email: {len(self.recipients['email'])}, SMS: {len(self.recipients['sms'])}")
    
    def check_equipment_alerts(self, equipment_id: str, sensor_data: Dict) -> List[AlertRecord]:
        """Check for alerts based on sensor data"""
        alerts_triggered = []
        
        try:
            # Check each alert rule
            for alert_type, rule in self.config.rules.items():
                if not rule.enabled:
                    continue
                
                # Get sensor value based on alert type
                sensor_value = self._get_sensor_value(alert_type, sensor_data)
                if sensor_value is None:
                    continue
                
                # Check if alert condition is met
                if self.config.check_alert_condition(alert_type, sensor_value):
                    
                    # Check throttling
                    if self._is_throttled(equipment_id, alert_type):
                        continue
                    
                    # Create alert record
                    alert_message = self.config.get_alert_message(
                        alert_type, equipment_id, sensor_value
                    )
                    
                    alert = AlertRecord(
                        equipment_id=equipment_id,
                        alert_type=alert_type.value,
                        severity=rule.severity.value,
                        message=alert_message,
                        value=sensor_value,
                        threshold=rule.threshold,
                        notification_channels=[]
                    )
                    
                    # Send notifications
                    notification_channels = self._send_notifications(alert)
                    alert.notification_channels = notification_channels
                    
                    # Add to history
                    alert_id = self.history_manager.add_alert(alert)
                    alert.id = alert_id
                    
                    # Update rule trigger info
                    self.config.update_trigger_info(alert_type)
                    
                    # Update throttle cache
                    self._update_throttle_cache(equipment_id, alert_type)
                    
                    alerts_triggered.append(alert)
                    
                    logging.info(f"Alert triggered: {equipment_id} - {alert_type.value} - {alert_message}")
            
            return alerts_triggered
            
        except Exception as e:
            logging.error(f"Error checking equipment alerts: {e}")
            return []
    
    def _get_sensor_value(self, alert_type: AlertType, sensor_data: Dict) -> Optional[float]:
        """Extract sensor value based on alert type"""
        value_map = {
            AlertType.HEALTH_DEGRADATION: sensor_data.get('health_index'),
            AlertType.RUL_THRESHOLD: sensor_data.get('rul'),
            AlertType.VIBRATION_SPIKE: sensor_data.get('vibration_rms'),
            AlertType.TEMPERATURE_HIGH: sensor_data.get('temperature'),
            AlertType.PRESSURE_ANOMALY: sensor_data.get('pressure'),
            AlertType.CURRENT_ANOMALY: sensor_data.get('current'),
            AlertType.SENSOR_FAILURE: sensor_data.get('sensor_status'),
            AlertType.MAINTENANCE_DUE: sensor_data.get('maintenance_due')
        }
        
        return value_map.get(alert_type)
    
    def _send_notifications(self, alert: AlertRecord) -> List[str]:
        """Send notifications through configured channels"""
        channels_used = []
        
        try:
            # Get channels for this severity
            severity_channels = self.config.escalation_rules.get(
                alert.severity, ['dashboard']
            )
            
            # Send email notification
            if 'email' in severity_channels and self.recipients['email']:
                success = self.email_service.send_alert_email(
                    alert.equipment_id,
                    alert.alert_type,
                    alert.severity,
                    alert.message,
                    alert.value,
                    alert.threshold,
                    self.recipients['email']
                )
                if success:
                    channels_used.append('email')
            
            # Send SMS notification
            if 'sms' in severity_channels and self.recipients['sms']:
                success = self.sms_service.send_alert_sms(
                    alert.equipment_id,
                    alert.alert_type,
                    alert.severity,
                    alert.message,
                    alert.value,
                    alert.threshold,
                    self.recipients['sms']
                )
                if success:
                    channels_used.append('sms')
            
            # Send Slack notification
            if 'slack' in severity_channels:
                success = self.slack_service.send_alert_slack(
                    alert.equipment_id,
                    alert.alert_type,
                    alert.severity,
                    alert.message,
                    alert.value,
                    alert.threshold
                )
                if success:
                    channels_used.append('slack')
            
            return channels_used
            
        except Exception as e:
            logging.error(f"Error sending notifications: {e}")
            return []
    
    def _is_throttled(self, equipment_id: str, alert_type: AlertType) -> bool:
        """Check if alert is throttled (prevent spam)"""
        key = f"{equipment_id}_{alert_type.value}"
        now = datetime.now()
        
        if key in self.throttle_cache:
            last_triggered = self.throttle_cache[key]
            if (now - last_triggered).total_seconds() < self.throttle_window:
                return True
        
        return False
    
    def _update_throttle_cache(self, equipment_id: str, alert_type: AlertType):
        """Update throttle cache"""
        key = f"{equipment_id}_{alert_type.value}"
        self.throttle_cache[key] = datetime.now()
    
    def acknowledge_alert(self, alert_id: int, acknowledged_by: str) -> bool:
        """Acknowledge an alert"""
        return self.history_manager.acknowledge_alert(alert_id, acknowledged_by)
    
    def resolve_alert(self, alert_id: int, resolution_notes: str, resolved_by: str) -> bool:
        """Resolve an alert"""
        return self.history_manager.resolve_alert(alert_id, resolution_notes, resolved_by)
    
    def get_active_alerts(self, equipment_id: Optional[str] = None) -> List[AlertRecord]:
        """Get active (unresolved) alerts"""
        return self.history_manager.get_alerts(
            equipment_id=equipment_id,
            start_date=datetime.now() - timedelta(days=7)
        )
    
    def get_alert_statistics(self, days: int = 30) -> Dict:
        """Get alert statistics"""
        return self.history_manager.get_alert_statistics(days)
    
    def send_daily_summary(self) -> bool:
        """Send daily alert summary"""
        try:
            summary_data = self.get_alert_statistics(1)  # Last 24 hours
            
            if self.recipients['email']:
                return self.email_service.send_daily_summary(
                    summary_data, self.recipients['email']
                )
            
            return False
            
        except Exception as e:
            logging.error(f"Error sending daily summary: {e}")
            return False
    
    def start_monitoring(self, equipment_data_callback):
        """Start background monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(equipment_data_callback,),
            daemon=True
        )
        self.monitoring_thread.start()
        
        logging.info("Alert monitoring started")
    
    def stop_monitoring(self):
        """Stop background monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        logging.info("Alert monitoring stopped")
    
    def _monitoring_loop(self, equipment_data_callback):
        """Background monitoring loop"""
        while self.monitoring_active:
            try:
                # Get current equipment data
                equipment_data = equipment_data_callback()
                
                # Check alerts for each equipment
                for equipment_id, sensor_data in equipment_data.items():
                    alerts = self.check_equipment_alerts(equipment_id, sensor_data)
                    
                    # Log any alerts triggered
                    for alert in alerts:
                        logging.info(f"Alert triggered in monitoring: {alert.message}")
                
                # Wait before next check
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                logging.error(f"Error in monitoring loop: {e}")
                time.sleep(30)  # Wait 30 seconds on error
    
    def test_notifications(self) -> Dict[str, bool]:
        """Test all notification channels"""
        results = {}
        
        # Test email
        results['email'] = self.email_service.test_email_connection()
        
        # Test SMS
        results['sms'] = self.sms_service.test_sms_connection()
        
        # Test Slack (if configured)
        results['slack'] = True  # Assume Slack works if webhook is configured
        
        return results
    
    def cleanup_old_alerts(self, days_to_keep: int = 90):
        """Clean up old alerts"""
        return self.history_manager.cleanup_old_alerts(days_to_keep)
