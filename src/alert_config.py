import yaml
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from enum import Enum
import logging

class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class AlertType(Enum):
    """Types of alerts"""
    HEALTH_DEGRADATION = "health_degradation"
    RUL_THRESHOLD = "rul_threshold"
    VIBRATION_SPIKE = "vibration_spike"
    TEMPERATURE_HIGH = "temperature_high"
    PRESSURE_ANOMALY = "pressure_anomaly"
    CURRENT_ANOMALY = "current_anomaly"
    SENSOR_FAILURE = "sensor_failure"
    MAINTENANCE_DUE = "maintenance_due"

class AlertRule:
    """Individual alert rule configuration"""
    
    def __init__(self, 
                 alert_type: AlertType,
                 severity: AlertSeverity,
                 threshold: float,
                 condition: str,
                 message_template: str,
                 escalation_minutes: int = 30,
                 enabled: bool = True):
        self.alert_type = alert_type
        self.severity = severity
        self.threshold = threshold
        self.condition = condition  # 'greater_than', 'less_than', 'equals', 'not_equals'
        self.message_template = message_template
        self.escalation_minutes = escalation_minutes
        self.enabled = enabled
        self.created_at = datetime.now()
        self.last_triggered = None
        self.trigger_count = 0

class AlertConfiguration:
    """Main alert configuration manager"""
    
    def __init__(self, config_path: str = "configs/alert_config.yaml"):
        self.config_path = config_path
        self.rules: Dict[AlertType, AlertRule] = {}
        self.escalation_rules: Dict[AlertSeverity, List[str]] = {}
        self.notification_channels: Dict[str, Dict] = {}
        self.load_configuration()
    
    def load_configuration(self):
        """Load alert configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Load alert rules
            for rule_config in config.get('alert_rules', []):
                rule = AlertRule(
                    alert_type=AlertType(rule_config['type']),
                    severity=AlertSeverity(rule_config['severity']),
                    threshold=rule_config['threshold'],
                    condition=rule_config['condition'],
                    message_template=rule_config['message_template'],
                    escalation_minutes=rule_config.get('escalation_minutes', 30),
                    enabled=rule_config.get('enabled', True)
                )
                self.rules[rule.alert_type] = rule
            
            # Load escalation rules
            self.escalation_rules = config.get('escalation_rules', {})
            
            # Load notification channels
            self.notification_channels = config.get('notification_channels', {})
            
            logging.info(f"Loaded {len(self.rules)} alert rules")
            
        except FileNotFoundError:
            logging.warning(f"Alert config file not found: {self.config_path}")
            self.create_default_configuration()
        except Exception as e:
            logging.error(f"Error loading alert configuration: {e}")
            self.create_default_configuration()
    
    def create_default_configuration(self):
        """Create default alert configuration"""
        default_rules = [
            {
                'type': 'health_degradation',
                'severity': 'warning',
                'threshold': 70.0,
                'condition': 'less_than',
                'message_template': 'Equipment {equipment_id} health index dropped to {value}%',
                'escalation_minutes': 60
            },
            {
                'type': 'rul_threshold',
                'severity': 'critical',
                'threshold': 24.0,
                'condition': 'less_than',
                'message_template': 'Equipment {equipment_id} RUL is only {value} hours',
                'escalation_minutes': 30
            },
            {
                'type': 'vibration_spike',
                'severity': 'warning',
                'threshold': 3.0,
                'condition': 'greater_than',
                'message_template': 'High vibration detected on {equipment_id}: {value} mm/s',
                'escalation_minutes': 45
            },
            {
                'type': 'temperature_high',
                'severity': 'critical',
                'threshold': 80.0,
                'condition': 'greater_than',
                'message_template': 'High temperature on {equipment_id}: {value}°C',
                'escalation_minutes': 15
            }
        ]
        
        for rule_config in default_rules:
            rule = AlertRule(
                alert_type=AlertType(rule_config['type']),
                severity=AlertSeverity(rule_config['severity']),
                threshold=rule_config['threshold'],
                condition=rule_config['condition'],
                message_template=rule_config['message_template'],
                escalation_minutes=rule_config['escalation_minutes']
            )
            self.rules[rule.alert_type] = rule
        
        # Default escalation rules
        self.escalation_rules = {
            'info': ['email'],
            'warning': ['email', 'dashboard'],
            'critical': ['email', 'sms', 'dashboard'],
            'emergency': ['email', 'sms', 'dashboard', 'phone_call']
        }
        
        # Default notification channels
        self.notification_channels = {
            'email': {
                'enabled': True,
                'smtp_server': 'smtp.gmail.com',
                'smtp_port': 587,
                'username': 'your-email@gmail.com',
                'password': 'your-app-password',
                'from_email': 'alerts@yourcompany.com'
            },
            'sms': {
                'enabled': False,
                'provider': 'twilio',
                'account_sid': 'your-twilio-sid',
                'auth_token': 'your-twilio-token',
                'from_number': '+1234567890'
            },
            'dashboard': {
                'enabled': True,
                'max_alerts': 100,
                'retention_days': 30
            }
        }
    
    def save_configuration(self):
        """Save current configuration to file"""
        config = {
            'alert_rules': [],
            'escalation_rules': self.escalation_rules,
            'notification_channels': self.notification_channels
        }
        
        for rule in self.rules.values():
            rule_config = {
                'type': rule.alert_type.value,
                'severity': rule.severity.value,
                'threshold': rule.threshold,
                'condition': rule.condition,
                'message_template': rule.message_template,
                'escalation_minutes': rule.escalation_minutes,
                'enabled': rule.enabled
            }
            config['alert_rules'].append(rule_config)
        
        with open(self.config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        logging.info("Alert configuration saved")
    
    def check_alert_condition(self, alert_type: AlertType, value: float) -> bool:
        """Check if alert condition is met"""
        if alert_type not in self.rules:
            return False
        
        rule = self.rules[alert_type]
        if not rule.enabled:
            return False
        
        if rule.condition == 'greater_than':
            return value > rule.threshold
        elif rule.condition == 'less_than':
            return value < rule.threshold
        elif rule.condition == 'equals':
            return abs(value - rule.threshold) < 0.001
        elif rule.condition == 'not_equals':
            return abs(value - rule.threshold) >= 0.001
        
        return False
    
    def get_alert_message(self, alert_type: AlertType, equipment_id: str, value: float) -> str:
        """Generate alert message from template"""
        if alert_type not in self.rules:
            return f"Alert for {equipment_id}: {value}"
        
        rule = self.rules[alert_type]
        return rule.message_template.format(
            equipment_id=equipment_id,
            value=value,
            threshold=rule.threshold,
            severity=rule.severity.value
        )
    
    def should_escalate(self, alert_type: AlertType) -> bool:
        """Check if alert should be escalated"""
        if alert_type not in self.rules:
            return False
        
        rule = self.rules[alert_type]
        if rule.last_triggered is None:
            return False
        
        time_since_triggered = datetime.now() - rule.last_triggered
        return time_since_triggered.total_seconds() > (rule.escalation_minutes * 60)
    
    def update_trigger_info(self, alert_type: AlertType):
        """Update trigger information for rule"""
        if alert_type in self.rules:
            self.rules[alert_type].last_triggered = datetime.now()
            self.rules[alert_type].trigger_count += 1
