import yaml
import json
import logging
import os
from datetime import datetime
import numpy as np
import pandas as pd

def setup_logging(log_dir: str = 'logs'):
    """Setup logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'{log_dir}/predictive_maintenance_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def save_results(results: dict, path: str):
    """Save results to JSON file"""
    with open(path, 'w') as f:
        json.dump(results, f, indent=4, default=str)

def create_directories(base_path: str = '.'):
    """Create project directory structure"""
    directories = [
        'data/raw', 'data/processed', 'data/features',
        'models', 'logs', 'results', 'configs'
    ]
    
    for directory in directories:
        os.makedirs(os.path.join(base_path, directory), exist_ok=True)

class HealthIndexCalculator:
    """Calculate equipment health index"""
    
    @staticmethod
    def calculate_health_index(features: pd.DataFrame, weights: dict = None) -> float:
        """Calculate health index from 0 (failed) to 100 (healthy)"""
        if weights is None:
            weights = {
                'vibration': 0.3,
                'temperature': 0.25,
                'pressure': 0.15,
                'current': 0.2,
                'flow_rate': 0.1
            }
        
        normal_ranges = {
            'vibration_rms': (0, 2),
            'temperature_mean': (50, 70),
            'pressure_mean': (4.5, 5.5),
            'current_mean': (45, 55),
            'flow_rate_mean': (90, 110)
        }
        
        health_scores = {}
        
        for param, (min_val, max_val) in normal_ranges.items():
            if param in features.columns:
                value = features[param].iloc[-1] if isinstance(features, pd.DataFrame) else features[param]
                
                if value < min_val:
                    score = max(0, 1 - (min_val - value) / min_val)
                elif value > max_val:
                    score = max(0, 1 - (value - max_val) / max_val)
                else:
                    score = 1.0
                
                param_type = param.split('_')[0]
                if param_type in weights:
                    health_scores[param_type] = score
        
        health_index = sum(score * weights.get(param, 0) 
                          for param, score in health_scores.items())
        
        return health_index * 100