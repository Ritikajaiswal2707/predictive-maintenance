import sqlite3
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
import logging

@dataclass
class AlertRecord:
    """Alert record data structure"""
    id: Optional[int] = None
    equipment_id: str = ""
    alert_type: str = ""
    severity: str = ""
    message: str = ""
    value: float = 0.0
    threshold: float = 0.0
    triggered_at: datetime = None
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    resolution_notes: Optional[str] = None
    notification_channels: List[str] = None
    escalation_count: int = 0
    
    def __post_init__(self):
        if self.triggered_at is None:
            self.triggered_at = datetime.now()
        if self.notification_channels is None:
            self.notification_channels = []

class AlertHistoryManager:
    """Manages alert history and persistence"""
    
    def __init__(self, db_path: str = "data/alert_history.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database for alert history"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    equipment_id TEXT NOT NULL,
                    alert_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    message TEXT NOT NULL,
                    value REAL NOT NULL,
                    threshold REAL NOT NULL,
                    triggered_at TIMESTAMP NOT NULL,
                    acknowledged_at TIMESTAMP,
                    resolved_at TIMESTAMP,
                    acknowledged_by TEXT,
                    resolution_notes TEXT,
                    notification_channels TEXT,
                    escalation_count INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create indexes for better performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_equipment_id ON alerts(equipment_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_triggered_at ON alerts(triggered_at)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_severity ON alerts(severity)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_alert_type ON alerts(alert_type)')
            
            conn.commit()
            conn.close()
            
            logging.info("Alert history database initialized")
            
        except Exception as e:
            logging.error(f"Error initializing alert database: {e}")
    
    def add_alert(self, alert: AlertRecord) -> int:
        """Add new alert to history"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO alerts (
                    equipment_id, alert_type, severity, message, value, threshold,
                    triggered_at, acknowledged_at, resolved_at, acknowledged_by,
                    resolution_notes, notification_channels, escalation_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                alert.equipment_id,
                alert.alert_type,
                alert.severity,
                alert.message,
                alert.value,
                alert.threshold,
                alert.triggered_at,
                alert.acknowledged_at,
                alert.resolved_at,
                alert.acknowledged_by,
                alert.resolution_notes,
                json.dumps(alert.notification_channels),
                alert.escalation_count
            ))
            
            alert_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            logging.info(f"Alert {alert_id} added to history")
            return alert_id
            
        except Exception as e:
            logging.error(f"Error adding alert to history: {e}")
            return -1
    
    def get_alerts(self, 
                   equipment_id: Optional[str] = None,
                   severity: Optional[str] = None,
                   alert_type: Optional[str] = None,
                   start_date: Optional[datetime] = None,
                   end_date: Optional[datetime] = None,
                   limit: int = 100) -> List[AlertRecord]:
        """Retrieve alerts with filters"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            query = "SELECT * FROM alerts WHERE 1=1"
            params = []
            
            if equipment_id:
                query += " AND equipment_id = ?"
                params.append(equipment_id)
            
            if severity:
                query += " AND severity = ?"
                params.append(severity)
            
            if alert_type:
                query += " AND alert_type = ?"
                params.append(alert_type)
            
            if start_date:
                query += " AND triggered_at >= ?"
                params.append(start_date)
            
            if end_date:
                query += " AND triggered_at <= ?"
                params.append(end_date)
            
            query += " ORDER BY triggered_at DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            conn.close()
            
            alerts = []
            for row in rows:
                alert = AlertRecord(
                    id=row[0],
                    equipment_id=row[1],
                    alert_type=row[2],
                    severity=row[3],
                    message=row[4],
                    value=row[5],
                    threshold=row[6],
                    triggered_at=datetime.fromisoformat(row[7]),
                    acknowledged_at=datetime.fromisoformat(row[8]) if row[8] else None,
                    resolved_at=datetime.fromisoformat(row[9]) if row[9] else None,
                    acknowledged_by=row[10],
                    resolution_notes=row[11],
                    notification_channels=json.loads(row[12]) if row[12] else [],
                    escalation_count=row[13]
                )
                alerts.append(alert)
            
            return alerts
            
        except Exception as e:
            logging.error(f"Error retrieving alerts: {e}")
            return []
    
    def acknowledge_alert(self, alert_id: int, acknowledged_by: str) -> bool:
        """Acknowledge an alert"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE alerts 
                SET acknowledged_at = ?, acknowledged_by = ?
                WHERE id = ?
            ''', (datetime.now(), acknowledged_by, alert_id))
            
            conn.commit()
            conn.close()
            
            logging.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
            return True
            
        except Exception as e:
            logging.error(f"Error acknowledging alert: {e}")
            return False
    
    def resolve_alert(self, alert_id: int, resolution_notes: str, resolved_by: str) -> bool:
        """Resolve an alert"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE alerts 
                SET resolved_at = ?, resolution_notes = ?, acknowledged_by = ?
                WHERE id = ?
            ''', (datetime.now(), resolution_notes, resolved_by, alert_id))
            
            conn.commit()
            conn.close()
            
            logging.info(f"Alert {alert_id} resolved by {resolved_by}")
            return True
            
        except Exception as e:
            logging.error(f"Error resolving alert: {e}")
            return False
    
    def get_alert_statistics(self, days: int = 30) -> Dict:
        """Get alert statistics for the last N days"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            start_date = datetime.now() - timedelta(days=days)
            
            # Total alerts
            cursor.execute('SELECT COUNT(*) FROM alerts WHERE triggered_at >= ?', (start_date,))
            total_alerts = cursor.fetchone()[0]
            
            # Alerts by severity
            cursor.execute('''
                SELECT severity, COUNT(*) 
                FROM alerts 
                WHERE triggered_at >= ? 
                GROUP BY severity
            ''', (start_date,))
            severity_stats = dict(cursor.fetchall())
            
            # Alerts by type
            cursor.execute('''
                SELECT alert_type, COUNT(*) 
                FROM alerts 
                WHERE triggered_at >= ? 
                GROUP BY alert_type
            ''', (start_date,))
            type_stats = dict(cursor.fetchall())
            
            # Average resolution time
            cursor.execute('''
                SELECT AVG(julianday(resolved_at) - julianday(triggered_at)) * 24
                FROM alerts 
                WHERE resolved_at IS NOT NULL AND triggered_at >= ?
            ''', (start_date,))
            avg_resolution_hours = cursor.fetchone()[0] or 0
            
            # Unresolved alerts
            cursor.execute('''
                SELECT COUNT(*) 
                FROM alerts 
                WHERE resolved_at IS NULL AND triggered_at >= ?
            ''', (start_date,))
            unresolved_alerts = cursor.fetchone()[0]
            
            conn.close()
            
            return {
                'total_alerts': total_alerts,
                'severity_distribution': severity_stats,
                'type_distribution': type_stats,
                'average_resolution_hours': round(avg_resolution_hours, 2),
                'unresolved_alerts': unresolved_alerts,
                'period_days': days
            }
            
        except Exception as e:
            logging.error(f"Error getting alert statistics: {e}")
            return {}
    
    def cleanup_old_alerts(self, days_to_keep: int = 90):
        """Clean up old alerts to manage database size"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            cursor.execute('DELETE FROM alerts WHERE triggered_at < ?', (cutoff_date,))
            deleted_count = cursor.rowcount
            
            conn.commit()
            conn.close()
            
            logging.info(f"Cleaned up {deleted_count} old alerts")
            return deleted_count
            
        except Exception as e:
            logging.error(f"Error cleaning up old alerts: {e}")
            return 0
