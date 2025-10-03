import numpy as np
import pandas as pd
from scipy import signal
import os
from typing import Tuple, Dict

class DataGenerator:
    """Generate synthetic pump/compressor data for demonstration"""
    
    def __init__(self, sample_rate=1000, duration=3600):
        self.sample_rate = sample_rate
        self.duration = duration
        self.time = np.linspace(0, duration, sample_rate * duration)
        
    def generate_healthy_signal(self) -> Dict[str, np.ndarray]:
        """Generate signals for healthy equipment"""
        rotation_freq = 30
        
        vibration = (0.5 * np.sin(2 * np.pi * rotation_freq * self.time) +
                    0.2 * np.sin(2 * np.pi * 2 * rotation_freq * self.time) +
                    0.1 * np.random.randn(len(self.time)))
        
        temperature = 60 + 2 * np.sin(2 * np.pi * 0.001 * self.time) + 0.5 * np.random.randn(len(self.time))
        pressure = 5 + 0.1 * np.sin(2 * np.pi * 0.002 * self.time) + 0.05 * np.random.randn(len(self.time))
        current = 50 + 2 * np.sin(2 * np.pi * rotation_freq * self.time) + 0.5 * np.random.randn(len(self.time))
        flow_rate = 100 + 5 * np.sin(2 * np.pi * 0.001 * self.time) + 1 * np.random.randn(len(self.time))
        
        return {
            'time': self.time,
            'vibration': vibration,
            'temperature': temperature,
            'pressure': pressure,
            'current': current,
            'flow_rate': flow_rate,
            'health_status': 'healthy',
            'rul': 1000
        }
    
    def generate_faulty_signal(self, fault_type='bearing') -> Dict[str, np.ndarray]:
        """Generate signals for faulty equipment"""
        rotation_freq = 30
        
        if fault_type == 'bearing':
            fault_freq = 147
            vibration = (1.5 * np.sin(2 * np.pi * rotation_freq * self.time) +
                        0.8 * np.sin(2 * np.pi * fault_freq * self.time) +
                        0.3 * np.random.randn(len(self.time)))
            temperature = 75 + 5 * np.sin(2 * np.pi * 0.001 * self.time) + 1 * np.random.randn(len(self.time))
            current = 65 + 5 * np.sin(2 * np.pi * rotation_freq * self.time) + 1 * np.random.randn(len(self.time))
            
        elif fault_type == 'imbalance':
            vibration = (3.0 * np.sin(2 * np.pi * rotation_freq * self.time) +
                        0.5 * np.sin(2 * np.pi * 2 * rotation_freq * self.time) +
                        0.3 * np.random.randn(len(self.time)))
            temperature = 65 + 3 * np.sin(2 * np.pi * 0.001 * self.time) + 0.8 * np.random.randn(len(self.time))
            current = 55 + 3 * np.sin(2 * np.pi * rotation_freq * self.time) + 0.8 * np.random.randn(len(self.time))
            
        else:  # cavitation
            vibration = (0.8 * np.sin(2 * np.pi * rotation_freq * self.time) +
                        2.0 * np.random.randn(len(self.time)) * (np.random.rand(len(self.time)) > 0.7))
            temperature = 70 + 4 * np.sin(2 * np.pi * 0.001 * self.time) + 1.5 * np.random.randn(len(self.time))
            current = 45 + 8 * np.sin(2 * np.pi * rotation_freq * self.time) + 2 * np.random.randn(len(self.time))
        
        pressure = 4.5 + 0.3 * np.sin(2 * np.pi * 0.002 * self.time) + 0.1 * np.random.randn(len(self.time))
        flow_rate = 85 + 10 * np.sin(2 * np.pi * 0.001 * self.time) + 2 * np.random.randn(len(self.time))
        
        return {
            'time': self.time,
            'vibration': vibration,
            'temperature': temperature,
            'pressure': pressure,
            'current': current,
            'flow_rate': flow_rate,
            'health_status': f'faulty_{fault_type}',
            'rul': np.random.randint(50, 200)
        }
    
    def _extract_basic_features(self, data: Dict) -> Dict:
        """Extract basic statistical features from signals"""
        features = {}
        
        for signal_name in ['vibration', 'temperature', 'pressure', 'current', 'flow_rate']:
            signal_data = data[signal_name]
            
            features[f'{signal_name}_mean'] = np.mean(signal_data)
            features[f'{signal_name}_std'] = np.std(signal_data)
            features[f'{signal_name}_max'] = np.max(signal_data)
            features[f'{signal_name}_min'] = np.min(signal_data)
            features[f'{signal_name}_rms'] = np.sqrt(np.mean(signal_data**2))
            features[f'{signal_name}_peak'] = np.max(np.abs(signal_data))
            
        features['health_status'] = data['health_status']
        features['rul'] = data['rul']
        
        return features

class DataLoader:
    """Load and manage datasets"""
    
    def __init__(self, data_path='data/'):
        self.data_path = data_path
        
    def generate_synthetic_dataset(self, n_samples=1000) -> pd.DataFrame:
        """Generate complete synthetic dataset"""
        generator = DataGenerator(sample_rate=100, duration=10)
        
        all_data = []
        
        for _ in range(n_samples // 2):
            data = generator.generate_healthy_signal()
            features = generator._extract_basic_features(data)
            all_data.append(features)
        
        fault_types = ['bearing', 'imbalance', 'cavitation']
        for _ in range(n_samples // 2):
            fault_type = np.random.choice(fault_types)
            data = generator.generate_faulty_signal(fault_type)
            features = generator._extract_basic_features(data)
            all_data.append(features)
        
        return pd.DataFrame(all_data)
