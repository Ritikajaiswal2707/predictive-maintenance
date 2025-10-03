import requests
from typing import List, Dict, Optional
import logging
from datetime import datetime
import json

class SMSNotificationService:
    """SMS notification service for alerts"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.provider = config.get('provider', 'twilio')
        self.enabled = config.get('enabled', False)
        
        if self.provider == 'twilio':
            self.account_sid = config.get('account_sid', '')
            self.auth_token = config.get('auth_token', '')
            self.from_number = config.get('from_number', '')
            self.api_url = f"https://api.twilio.com/2010-04-01/Accounts/{self.account_sid}/Messages.json"
        
        # SMS templates
        self.templates = {
            'info': '[INFO] {equipment_id}: {message}',
            'warning': '[WARNING] {equipment_id}: {message} - Monitor closely',
            'critical': '[CRITICAL] {equipment_id}: {message} - Action required!',
            'emergency': '[EMERGENCY] {equipment_id}: {message} - SHUTDOWN NOW!'
        }
    
    def send_alert_sms(self, 
                      equipment_id: str,
                      alert_type: str,
                      severity: str,
                      message: str,
                      value: float,
                      threshold: float,
                      recipients: List[str]) -> bool:
        """Send alert SMS to recipients"""
        
        if not self.enabled:
            logging.warning("SMS notifications are disabled")
            return False
        
        if not recipients:
            logging.warning("No SMS recipients specified")
            return False
        
        try:
            # Get template for severity
            template = self.templates.get(severity.lower(), self.templates['info'])
            
            # Format message (keep it short for SMS)
            sms_message = template.format(
                equipment_id=equipment_id,
                message=message[:100]  # Limit message length
            )
            
            success_count = 0
            
            for recipient in recipients:
                if self.provider == 'twilio':
                    success = self._send_twilio_sms(recipient, sms_message)
                    if success:
                        success_count += 1
                else:
                    logging.warning(f"Unsupported SMS provider: {self.provider}")
            
            logging.info(f"SMS alert sent to {success_count}/{len(recipients)} recipients for {equipment_id}")
            return success_count > 0
            
        except Exception as e:
            logging.error(f"Error sending alert SMS: {e}")
            return False
    
    def _send_twilio_sms(self, to_number: str, message: str) -> bool:
        """Send SMS using Twilio API"""
        try:
            auth = (self.account_sid, self.auth_token)
            data = {
                'From': self.from_number,
                'To': to_number,
                'Body': message
            }
            
            response = requests.post(self.api_url, auth=auth, data=data)
            
            if response.status_code == 201:
                logging.info(f"SMS sent successfully to {to_number}")
                return True
            else:
                logging.error(f"Failed to send SMS to {to_number}: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logging.error(f"Error sending Twilio SMS: {e}")
            return False
    
    def test_sms_connection(self) -> bool:
        """Test SMS connection"""
        try:
            if self.provider == 'twilio':
                # Test with a simple API call
                auth = (self.account_sid, self.auth_token)
                response = requests.get(
                    f"https://api.twilio.com/2010-04-01/Accounts/{self.account_sid}.json",
                    auth=auth
                )
                
                if response.status_code == 200:
                    logging.info("SMS connection test successful")
                    return True
                else:
                    logging.error(f"SMS connection test failed: {response.status_code}")
                    return False
            else:
                logging.warning(f"SMS connection test not implemented for provider: {self.provider}")
                return False
                
        except Exception as e:
            logging.error(f"SMS connection test failed: {e}")
            return False

class SlackNotificationService:
    """Slack notification service for alerts"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.webhook_url = config.get('webhook_url', '')
        self.channel = config.get('channel', '#alerts')
        self.enabled = config.get('enabled', False)
    
    def send_alert_slack(self, 
                        equipment_id: str,
                        alert_type: str,
                        severity: str,
                        message: str,
                        value: float,
                        threshold: float) -> bool:
        """Send alert to Slack channel"""
        
        if not self.enabled or not self.webhook_url:
            return False
        
        try:
            # Color coding for severity
            color_map = {
                'info': '#36a64f',      # Green
                'warning': '#ff9800',   # Orange
                'critical': '#f44336',  # Red
                'emergency': '#e91e63'  # Pink
            }
            
            color = color_map.get(severity.lower(), '#36a64f')
            
            # Emoji for severity
            emoji_map = {
                'info': '🔵',
                'warning': '🟡',
                'critical': '🔴',
                'emergency': '🚨'
            }
            
            emoji = emoji_map.get(severity.lower(), '🔵')
            
            payload = {
                "channel": self.channel,
                "username": "Predictive Maintenance Bot",
                "icon_emoji": ":gear:",
                "attachments": [
                    {
                        "color": color,
                        "title": f"{emoji} {severity.upper()} Alert",
                        "fields": [
                            {
                                "title": "Equipment",
                                "value": equipment_id,
                                "short": True
                            },
                            {
                                "title": "Alert Type",
                                "value": alert_type,
                                "short": True
                            },
                            {
                                "title": "Message",
                                "value": message,
                                "short": False
                            },
                            {
                                "title": "Current Value",
                                "value": str(value),
                                "short": True
                            },
                            {
                                "title": "Threshold",
                                "value": str(threshold),
                                "short": True
                            },
                            {
                                "title": "Time",
                                "value": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                "short": True
                            }
                        ],
                        "footer": "Predictive Maintenance System",
                        "ts": int(datetime.now().timestamp())
                    }
                ]
            }
            
            response = requests.post(self.webhook_url, json=payload)
            
            if response.status_code == 200:
                logging.info(f"Slack alert sent for {equipment_id}")
                return True
            else:
                logging.error(f"Failed to send Slack alert: {response.status_code}")
                return False
                
        except Exception as e:
            logging.error(f"Error sending Slack alert: {e}")
            return False
