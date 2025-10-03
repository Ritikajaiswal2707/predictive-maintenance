import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq
import os

class BearingDatasetGenerator:
    """Generate realistic bearing fault dataset similar to CWRU"""
    
    def __init__(self, sampling_rate=12000):
        self.sampling_rate = sampling_rate
        self.rpm = 1797  # Motor speed similar to CWRU
        self.bearing_freq = {
            'BPFO': 3.5848,  # Ball Pass Frequency Outer race (×RPM)
            'BPFI': 5.4152,  # Ball Pass Frequency Inner race (×RPM)
            'BSF': 2.3570,   # Ball Spin Frequency (×RPM)
            'FTF': 0.3983    # Fundamental Train Frequency (×RPM)
        }
        
    def generate_healthy_bearing(self, duration=10):
        """Generate healthy bearing signal"""
        t = np.linspace(0, duration, int(self.sampling_rate * duration))
        
        # Shaft rotation frequency
        shaft_freq = self.rpm / 60
        
        # Healthy signal components
        signal = 0.1 * np.sin(2 * np.pi * shaft_freq * t)  # 1X shaft frequency
        signal += 0.05 * np.sin(2 * np.pi * 2 * shaft_freq * t)  # 2X harmonic
        signal += 0.02 * np.sin(2 * np.pi * 3 * shaft_freq * t)  # 3X harmonic
        
        # Add small random vibrations
        signal += 0.01 * np.random.randn(len(t))
        
        return t, signal
    
    def generate_inner_race_fault(self, duration=10, severity='mild'):
        """Generate inner race fault signal"""
        t = np.linspace(0, duration, int(self.sampling_rate * duration))
        
        shaft_freq = self.rpm / 60
        bpfi = self.bearing_freq['BPFI'] * shaft_freq
        
        # Base signal
        signal = 0.1 * np.sin(2 * np.pi * shaft_freq * t)
        
        # Fault frequency components
        severity_factor = {'mild': 0.3, 'moderate': 0.7, 'severe': 1.5}[severity]
        
        # BPFI and harmonics
        for harmonic in range(1, 4):
            signal += severity_factor * 0.2 * np.sin(2 * np.pi * harmonic * bpfi * t)
        
        # Add modulation (amplitude modulation by shaft frequency)
        modulation = 1 + 0.3 * np.sin(2 * np.pi * shaft_freq * t)
        signal *= modulation
        
        # Add impacts
        impact_interval = int(self.sampling_rate / bpfi)
        for i in range(0, len(t), impact_interval):
            if i + 100 < len(t):
                impact = severity_factor * np.exp(-0.01 * np.arange(100)) * np.sin(2 * np.pi * 3000 * np.arange(100) / self.sampling_rate)
                signal[i:i+100] += impact
        
        # Add noise
        signal += 0.02 * severity_factor * np.random.randn(len(t))
        
        return t, signal
    
    def generate_outer_race_fault(self, duration=10, severity='mild'):
        """Generate outer race fault signal"""
        t = np.linspace(0, duration, int(self.sampling_rate * duration))
        
        shaft_freq = self.rpm / 60
        bpfo = self.bearing_freq['BPFO'] * shaft_freq
        
        # Base signal
        signal = 0.1 * np.sin(2 * np.pi * shaft_freq * t)
        
        # Fault frequency components
        severity_factor = {'mild': 0.3, 'moderate': 0.7, 'severe': 1.5}[severity]
        
        # BPFO and harmonics
        for harmonic in range(1, 4):
            signal += severity_factor * 0.15 * np.sin(2 * np.pi * harmonic * bpfo * t)
        
        # Outer race faults are typically not modulated by shaft frequency
        
        # Add impacts (more regular than inner race)
        impact_interval = int(self.sampling_rate / bpfo)
        for i in range(0, len(t), impact_interval):
            if i + 80 < len(t):
                impact = severity_factor * 0.8 * np.exp(-0.015 * np.arange(80)) * np.sin(2 * np.pi * 2500 * np.arange(80) / self.sampling_rate)
                signal[i:i+80] += impact
        
        # Add noise
        signal += 0.02 * severity_factor * np.random.randn(len(t))
        
        return t, signal
    
    def generate_ball_fault(self, duration=10, severity='mild'):
        """Generate ball fault signal"""
        t = np.linspace(0, duration, int(self.sampling_rate * duration))
        
        shaft_freq = self.rpm / 60
        bsf = self.bearing_freq['BSF'] * shaft_freq
        ftf = self.bearing_freq['FTF'] * shaft_freq
        
        # Base signal
        signal = 0.1 * np.sin(2 * np.pi * shaft_freq * t)
        
        # Fault frequency components
        severity_factor = {'mild': 0.3, 'moderate': 0.7, 'severe': 1.5}[severity]
        
        # BSF and harmonics (modulated by cage frequency)
        for harmonic in range(1, 3):
            bsf_signal = severity_factor * 0.1 * np.sin(2 * np.pi * harmonic * bsf * t)
            # Modulate by cage frequency
            bsf_signal *= (1 + 0.5 * np.sin(2 * np.pi * ftf * t))
            signal += bsf_signal
        
        # Add impacts (less regular than race faults)
        impact_interval = int(self.sampling_rate / bsf)
        for i in range(0, len(t), impact_interval):
            # Add some randomness to impact timing
            offset = np.random.randint(-impact_interval//10, impact_interval//10)
            pos = i + offset
            if 0 <= pos < len(t) - 60:
                impact = severity_factor * 0.6 * np.exp(-0.02 * np.arange(60)) * np.sin(2 * np.pi * 4000 * np.arange(60) / self.sampling_rate)
                signal[pos:pos+60] += impact
        
        # Add noise
        signal += 0.03 * severity_factor * np.random.randn(len(t))
        
        return t, signal
    
    def extract_features(self, signal_data):
        """Extract features from signal"""
        features = {}
        
        # Time domain features
        features['rms'] = np.sqrt(np.mean(signal_data**2))
        features['peak'] = np.max(np.abs(signal_data))
        features['peak_to_peak'] = np.max(signal_data) - np.min(signal_data)
        features['crest_factor'] = features['peak'] / features['rms']
        features['kurtosis'] = pd.Series(signal_data).kurtosis()
        features['skewness'] = pd.Series(signal_data).skew()
        features['std'] = np.std(signal_data)
        features['shape_factor'] = features['rms'] / np.mean(np.abs(signal_data))
        features['impulse_factor'] = features['peak'] / np.mean(np.abs(signal_data))
        
        # Frequency domain features
        freqs, psd = signal.welch(signal_data, fs=self.sampling_rate, nperseg=1024)
        
        # Find peaks
        peaks, _ = signal.find_peaks(psd, height=np.max(psd)*0.1)
        if len(peaks) > 0:
            features['dominant_frequency'] = freqs[peaks[0]]
            features['n_peaks'] = len(peaks)
        else:
            features['dominant_frequency'] = 0
            features['n_peaks'] = 0
        
        # Band powers
        bands = [(0, 1000), (1000, 3000), (3000, 5000), (5000, 6000)]
        for i, (low, high) in enumerate(bands):
            mask = (freqs >= low) & (freqs < high)
            features[f'band_power_{i}'] = np.sum(psd[mask])
        
        return features
    
    def generate_dataset(self, n_samples_per_class=100):
        """Generate complete dataset"""
        all_data = []
        
        conditions = [
            ('healthy', None),
            ('inner_race', 'mild'),
            ('inner_race', 'moderate'),
            ('inner_race', 'severe'),
            ('outer_race', 'mild'),
            ('outer_race', 'moderate'),
            ('outer_race', 'severe'),
            ('ball', 'mild'),
            ('ball', 'moderate'),
            ('ball', 'severe')
        ]
        
        for condition, severity in conditions:
            print(f"Generating {condition} {severity if severity else ''} samples...")
            
            for i in range(n_samples_per_class):
                # Generate signal
                if condition == 'healthy':
                    t, signal_data = self.generate_healthy_bearing(duration=1)
                elif condition == 'inner_race':
                    t, signal_data = self.generate_inner_race_fault(duration=1, severity=severity)
                elif condition == 'outer_race':
                    t, signal_data = self.generate_outer_race_fault(duration=1, severity=severity)
                elif condition == 'ball':
                    t, signal_data = self.generate_ball_fault(duration=1, severity=severity)
                
                # Extract features
                features = self.extract_features(signal_data)
                
                # Add metadata
                features['condition'] = condition
                features['severity'] = severity if severity else 'none'
                features['health_status'] = 'healthy' if condition == 'healthy' else 'faulty'
                
                # Estimate RUL based on severity
                if condition == 'healthy':
                    features['rul'] = 1000
                else:
                    severity_rul = {'mild': 500, 'moderate': 200, 'severe': 50}
                    features['rul'] = severity_rul[severity] + np.random.randint(-20, 20)
                
                all_data.append(features)
        
        df = pd.DataFrame(all_data)
        return df
    
    def visualize_samples(self, save_path='results/'):
        """Generate and visualize sample signals"""
        fig, axes = plt.subplots(4, 2, figsize=(15, 12))
        
        # Healthy
        t, signal_data = self.generate_healthy_bearing(duration=0.1)
        axes[0, 0].plot(t, signal_data)
        axes[0, 0].set_title('Healthy Bearing - Time Domain')
        axes[0, 0].set_xlabel('Time (s)')
        
        freqs, psd = signal.welch(signal_data, fs=self.sampling_rate)
        axes[0, 1].semilogy(freqs[:1000], psd[:1000])
        axes[0, 1].set_title('Healthy Bearing - Frequency Domain')
        axes[0, 1].set_xlabel('Frequency (Hz)')
        
        # Inner Race Fault
        t, signal_data = self.generate_inner_race_fault(duration=0.1, severity='moderate')
        axes[1, 0].plot(t, signal_data)
        axes[1, 0].set_title('Inner Race Fault - Time Domain')
        axes[1, 0].set_xlabel('Time (s)')
        
        freqs, psd = signal.welch(signal_data, fs=self.sampling_rate)
        axes[1, 1].semilogy(freqs[:1000], psd[:1000])
        axes[1, 1].set_title('Inner Race Fault - Frequency Domain')
        axes[1, 1].set_xlabel('Frequency (Hz)')
        
        # Outer Race Fault
        t, signal_data = self.generate_outer_race_fault(duration=0.1, severity='moderate')
        axes[2, 0].plot(t, signal_data)
        axes[2, 0].set_title('Outer Race Fault - Time Domain')
        axes[2, 0].set_xlabel('Time (s)')
        
        freqs, psd = signal.welch(signal_data, fs=self.sampling_rate)
        axes[2, 1].semilogy(freqs[:1000], psd[:1000])
        axes[2, 1].set_title('Outer Race Fault - Frequency Domain')
        axes[2, 1].set_xlabel('Frequency (Hz)')
        
        # Ball Fault
        t, signal_data = self.generate_ball_fault(duration=0.1, severity='moderate')
        axes[3, 0].plot(t, signal_data)
        axes[3, 0].set_title('Ball Fault - Time Domain')
        axes[3, 0].set_xlabel('Time (s)')
        
        freqs, psd = signal.welch(signal_data, fs=self.sampling_rate)
        axes[3, 1].semilogy(freqs[:1000], psd[:1000])
        axes[3, 1].set_title('Ball Fault - Frequency Domain')
        axes[3, 1].set_xlabel('Frequency (Hz)')
        axes[3, 1].set_ylabel('PSD')
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Generate dataset
    generator = BearingDatasetGenerator()
    
    print("Generating synthetic bearing dataset...")
    df = generator.generate_dataset(n_samples_per_class=200)
    
    # Save dataset
    os.makedirs('data/processed', exist_ok=True)
    df.to_csv('data/processed/synthetic_bearing_features.csv', index=False)
    
    print(f"\nGenerated {len(df)} samples")
    print("\nDataset summary:")
    print(df.groupby(['condition', 'severity']).size())
    
    # Visualize samples
    print("\nGenerating visualizations...")
    generator.visualize_samples()
    plt.tight_layout()
    plt.savefig('results/bearing_signal_samples.png', dpi=300)
    plt.close()
    
    print("\nDataset saved to: data/processed/synthetic_bearing_features.csv")
    print("Visualizations saved to: results/bearing_signal_samples.png")