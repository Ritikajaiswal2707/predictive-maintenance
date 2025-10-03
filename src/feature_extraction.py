import numpy as np
import pandas as pd
from scipy import signal, stats
from scipy.fft import fft, fftfreq
import pywt
from typing import Dict

class FeatureExtractor:
    """Extract time-domain, frequency-domain, and time-frequency features"""
    
    def __init__(self, sampling_rate=1000):
        self.sampling_rate = sampling_rate
        
    def extract_time_domain_features(self, signal_data: np.ndarray) -> Dict[str, float]:
        """Extract time-domain statistical features"""
        features = {
            'mean': np.mean(signal_data),
            'std': np.std(signal_data),
            'rms': np.sqrt(np.mean(signal_data**2)),
            'peak': np.max(np.abs(signal_data)),
            'peak_to_peak': np.max(signal_data) - np.min(signal_data),
            'crest_factor': np.max(np.abs(signal_data)) / np.sqrt(np.mean(signal_data**2)),
            'shape_factor': np.sqrt(np.mean(signal_data**2)) / np.mean(np.abs(signal_data)),
            'impulse_factor': np.max(np.abs(signal_data)) / np.mean(np.abs(signal_data)),
            'kurtosis': stats.kurtosis(signal_data),
            'skewness': stats.skew(signal_data),
            'clearance_factor': np.max(np.abs(signal_data)) / (np.mean(np.sqrt(np.abs(signal_data)))**2)
        }
        return features
    
    def extract_frequency_domain_features(self, signal_data: np.ndarray) -> Dict[str, float]:
        """Extract frequency-domain features using FFT"""
        n = len(signal_data)
        yf = fft(signal_data)
        xf = fftfreq(n, 1/self.sampling_rate)
        
        magnitude = 2.0/n * np.abs(yf[:n//2])
        frequencies = xf[:n//2]
        
        peak_indices = signal.find_peaks(magnitude, height=np.max(magnitude)*0.1)[0]
        dominant_freqs = frequencies[peak_indices] if len(peak_indices) > 0 else [0]
        
        features = {
            'dominant_frequency': dominant_freqs[0] if len(dominant_freqs) > 0 else 0,
            'mean_frequency': np.sum(frequencies * magnitude) / np.sum(magnitude),
            'frequency_std': np.sqrt(np.sum(((frequencies - np.sum(frequencies * magnitude) / np.sum(magnitude))**2) * magnitude) / np.sum(magnitude)),
            'spectral_entropy': -np.sum((magnitude/np.sum(magnitude)) * np.log(magnitude/np.sum(magnitude) + 1e-10)),
            'spectral_energy': np.sum(magnitude**2),
            'spectral_kurtosis': stats.kurtosis(magnitude),
            'spectral_skewness': stats.skew(magnitude)
        }
        
        bands = [(0, 50), (50, 150), (150, 300), (300, 500)]
        for i, (low, high) in enumerate(bands):
            band_mask = (frequencies >= low) & (frequencies < high)
            features[f'band_power_{i}'] = np.sum(magnitude[band_mask]**2)
            
        return features
    
    def extract_wavelet_features(self, signal_data: np.ndarray, wavelet='db4', level=4) -> Dict[str, float]:
        """Extract wavelet-based features"""
        coeffs = pywt.wavedec(signal_data, wavelet, level=level)
        
        features = {}
        for i, coeff in enumerate(coeffs):
            features[f'wavelet_energy_level_{i}'] = np.sum(coeff**2)
            features[f'wavelet_entropy_level_{i}'] = -np.sum((coeff**2/np.sum(coeff**2)) * np.log(coeff**2/np.sum(coeff**2) + 1e-10))
            features[f'wavelet_std_level_{i}'] = np.std(coeff)
            
        return features
    
    def extract_all_features(self, signal_data: np.ndarray, signal_name: str) -> Dict[str, float]:
        """Extract all features from a signal"""
        all_features = {}
        
        time_features = self.extract_time_domain_features(signal_data)
        for key, value in time_features.items():
            all_features[f'{signal_name}_{key}'] = value
            
        freq_features = self.extract_frequency_domain_features(signal_data)
        for key, value in freq_features.items():
            all_features[f'{signal_name}_{key}'] = value
            
        wavelet_features = self.extract_wavelet_features(signal_data)
        for key, value in wavelet_features.items():
            all_features[f'{signal_name}_{key}'] = value
            
        return all_features