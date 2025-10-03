import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from typing import List, Dict, Optional
import logging
from datetime import datetime
import json

class EmailNotificationService:
    """Email notification service for alerts"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.smtp_server = config.get('smtp_server', 'smtp.gmail.com')
        self.smtp_port = config.get('smtp_port', 587)
        self.username = config.get('username', '')
        self.password = config.get('password', '')
        self.from_email = config.get('from_email', '')
        self.enabled = config.get('enabled', False)
        
        # Email templates
        self.templates = {
            'info': {
                'subject_template': '[INFO] Predictive Maintenance Alert - {equipment_id}',
                'body_template': '''
                <html>
                <body style="font-family: Arial, sans-serif; margin: 20px;">
                    <div style="background-color: #e3f2fd; padding: 20px; border-radius: 5px;">
                        <h2 style="color: #1976d2; margin-top: 0;">🔵 Information Alert</h2>
                        <p><strong>Equipment:</strong> {equipment_id}</p>
                        <p><strong>Alert Type:</strong> {alert_type}</p>
                        <p><strong>Message:</strong> {message}</p>
                        <p><strong>Current Value:</strong> {value}</p>
                        <p><strong>Threshold:</strong> {threshold}</p>
                        <p><strong>Time:</strong> {timestamp}</p>
                        <hr>
                        <p style="font-size: 12px; color: #666;">
                            This is an automated message from the Predictive Maintenance System.
                        </p>
                    </div>
                </body>
                </html>
                '''
            },
            'warning': {
                'subject_template': '[WARNING] Predictive Maintenance Alert - {equipment_id}',
                'body_template': '''
                <html>
                <body style="font-family: Arial, sans-serif; margin: 20px;">
                    <div style="background-color: #fff3e0; padding: 20px; border-radius: 5px; border-left: 5px solid #ff9800;">
                        <h2 style="color: #f57c00; margin-top: 0;">🟡 Warning Alert</h2>
                        <p><strong>Equipment:</strong> {equipment_id}</p>
                        <p><strong>Alert Type:</strong> {alert_type}</p>
                        <p><strong>Message:</strong> {message}</p>
                        <p><strong>Current Value:</strong> {value}</p>
                        <p><strong>Threshold:</strong> {threshold}</p>
                        <p><strong>Time:</strong> {timestamp}</p>
                        <p style="color: #f57c00; font-weight: bold;">⚠️ Please monitor this equipment closely.</p>
                        <hr>
                        <p style="font-size: 12px; color: #666;">
                            This is an automated message from the Predictive Maintenance System.
                        </p>
                    </div>
                </body>
                </html>
                '''
            },
            'critical': {
                'subject_template': '[CRITICAL] Predictive Maintenance Alert - {equipment_id}',
                'body_template': '''
                <html>
                <body style="font-family: Arial, sans-serif; margin: 20px;">
                    <div style="background-color: #ffebee; padding: 20px; border-radius: 5px; border-left: 5px solid #f44336;">
                        <h2 style="color: #d32f2f; margin-top: 0;">🔴 Critical Alert</h2>
                        <p><strong>Equipment:</strong> {equipment_id}</p>
                        <p><strong>Alert Type:</strong> {alert_type}</p>
                        <p><strong>Message:</strong> {message}</p>
                        <p><strong>Current Value:</strong> {value}</p>
                        <p><strong>Threshold:</strong> {threshold}</p>
                        <p><strong>Time:</strong> {timestamp}</p>
                        <p style="color: #d32f2f; font-weight: bold;">🚨 IMMEDIATE ACTION REQUIRED!</p>
                        <p style="color: #d32f2f; font-weight: bold;">Please investigate this equipment immediately.</p>
                        <hr>
                        <p style="font-size: 12px; color: #666;">
                            This is an automated message from the Predictive Maintenance System.
                        </p>
                    </div>
                </body>
                </html>
                '''
            },
            'emergency': {
                'subject_template': '[EMERGENCY] Predictive Maintenance Alert - {equipment_id}',
                'body_template': '''
                <html>
                <body style="font-family: Arial, sans-serif; margin: 20px;">
                    <div style="background-color: #fce4ec; padding: 20px; border-radius: 5px; border-left: 5px solid #e91e63;">
                        <h2 style="color: #c2185b; margin-top: 0;">🚨 EMERGENCY Alert</h2>
                        <p><strong>Equipment:</strong> {equipment_id}</p>
                        <p><strong>Alert Type:</strong> {alert_type}</p>
                        <p><strong>Message:</strong> {message}</p>
                        <p><strong>Current Value:</strong> {value}</p>
                        <p><strong>Threshold:</strong> {threshold}</p>
                        <p><strong>Time:</strong> {timestamp}</p>
                        <p style="color: #c2185b; font-weight: bold; font-size: 18px;">🚨 EMERGENCY SHUTDOWN REQUIRED!</p>
                        <p style="color: #c2185b; font-weight: bold;">Contact maintenance team immediately!</p>
                        <hr>
                        <p style="font-size: 12px; color: #666;">
                            This is an automated message from the Predictive Maintenance System.
                        </p>
                    </div>
                </body>
                </html>
                '''
            }
        }
    
    def send_alert_email(self, 
                        equipment_id: str,
                        alert_type: str,
                        severity: str,
                        message: str,
                        value: float,
                        threshold: float,
                        recipients: List[str]) -> bool:
        """Send alert email to recipients"""
        
        if not self.enabled:
            logging.warning("Email notifications are disabled")
            return False
        
        if not recipients:
            logging.warning("No email recipients specified")
            return False
        
        try:
            # Get template for severity
            template = self.templates.get(severity.lower(), self.templates['info'])
            
            # Format subject and body
            subject = template['subject_template'].format(equipment_id=equipment_id)
            body = template['body_template'].format(
                equipment_id=equipment_id,
                alert_type=alert_type,
                message=message,
                value=value,
                threshold=threshold,
                timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            )
            
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.from_email
            msg['To'] = ', '.join(recipients)
            
            # Add HTML body
            html_part = MIMEText(body, 'html')
            msg.attach(html_part)
            
            # Send email
            context = ssl.create_default_context()
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls(context=context)
                server.login(self.username, self.password)
                server.send_message(msg)
            
            logging.info(f"Alert email sent to {len(recipients)} recipients for {equipment_id}")
            return True
            
        except Exception as e:
            logging.error(f"Error sending alert email: {e}")
            return False
    
    def send_daily_summary(self, 
                          summary_data: Dict,
                          recipients: List[str]) -> bool:
        """Send daily alert summary email"""
        
        if not self.enabled or not recipients:
            return False
        
        try:
            subject = f"[Daily Summary] Predictive Maintenance Alerts - {datetime.now().strftime('%Y-%m-%d')}"
            
            body = f'''
            <html>
            <body style="font-family: Arial, sans-serif; margin: 20px;">
                <h2>📊 Daily Alert Summary</h2>
                <p><strong>Date:</strong> {datetime.now().strftime('%Y-%m-%d')}</p>
                
                <h3>📈 Statistics</h3>
                <ul>
                    <li><strong>Total Alerts:</strong> {summary_data.get('total_alerts', 0)}</li>
                    <li><strong>Critical Alerts:</strong> {summary_data.get('severity_distribution', {}).get('critical', 0)}</li>
                    <li><strong>Warning Alerts:</strong> {summary_data.get('severity_distribution', {}).get('warning', 0)}</li>
                    <li><strong>Unresolved Alerts:</strong> {summary_data.get('unresolved_alerts', 0)}</li>
                    <li><strong>Average Resolution Time:</strong> {summary_data.get('average_resolution_hours', 0)} hours</li>
                </ul>
                
                <h3>🔧 Equipment Status</h3>
                <p>Please check the dashboard for detailed equipment status.</p>
                
                <hr>
                <p style="font-size: 12px; color: #666;">
                    This is an automated message from the Predictive Maintenance System.
                </p>
            </body>
            </html>
            '''
            
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.from_email
            msg['To'] = ', '.join(recipients)
            
            html_part = MIMEText(body, 'html')
            msg.attach(html_part)
            
            context = ssl.create_default_context()
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls(context=context)
                server.login(self.username, self.password)
                server.send_message(msg)
            
            logging.info(f"Daily summary email sent to {len(recipients)} recipients")
            return True
            
        except Exception as e:
            logging.error(f"Error sending daily summary email: {e}")
            return False
    
    def test_email_connection(self) -> bool:
        """Test email connection"""
        try:
            context = ssl.create_default_context()
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls(context=context)
                server.login(self.username, self.password)
            
            logging.info("Email connection test successful")
            return True
            
        except Exception as e:
            logging.error(f"Email connection test failed: {e}")
            return False
