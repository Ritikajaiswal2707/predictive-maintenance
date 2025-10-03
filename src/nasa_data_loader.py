import os
import numpy as np
import pandas as pd
import scipy.io
from scipy import signal
import requests
import zipfile
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

class NASABearingDataLoader:
    """Load and process NASA IMS Bearing Dataset"""
    
    def __init__(self, data_path='data/raw/nasa_bearing/'):
        self.data_path = data_path
        self.sampling_rate = 20480  # 20.48 kHz for IMS dataset
        
        # Dataset structure
        self.test_sets = {
            '1st_test': {
                'bearings': ['Bearing 1', 'Bearing 2', 'Bearing 3', 'Bearing 4'],
                'failed_bearing': 'Bearing 3',  # Inner race defect
                'failure_time': '2004.02.19.06.22.39'
            },
            '2nd_test': {
                'bearings': ['Bearing 1', 'Bearing 2', 'Bearing 3', 'Bearing 4'],
                'failed_bearing': 'Bearing 1',  # Outer race failure
                'failure_time': '2004.02.19.01.02.39'
            },
            '3rd_test': {
                'bearings': ['Bearing 1', 'Bearing 2', 'Bearing 3', 'Bearing 4'],
                'failed_bearing': 'Bearing 3',  # Outer race failure
                'failure_time': '2004.04.18.17.47.00'
            }
        }
    
    def download_instructions(self):
        """Print instructions for downloading NASA dataset"""
        print("="*60)
        print("NASA IMS Bearing Dataset Download Instructions")
        print("="*60)
        print("\n1. Visit: https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/")
        print("\n2. Look for 'Bearing Data Set' or 'IMS Bearing Data'")
        print("\n3. Download the following files:")
        print("   - IMS Bearing Data Set (1st test)")
        print("   - IMS Bearing Data Set (2nd test)")
        print("   - IMS Bearing Data Set (3rd test)")
        print("\n4. Extract each ZIP file to separate folders:")
        print(f"   - {self.data_path}1st_test/")
        print(f"   - {self.data_path}2nd_test/")
        print(f"   - {self.data_path}3rd_test/")
        print("\n5. Each folder should contain multiple .txt files")
        print("   (e.g., 2003.10.22.12.06.24, 2003.10.22.12.36.56, etc.)")
        print("="*60)
    
    def check_data_availability(self) -> bool:
        """Check if NASA data is available"""
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path, exist_ok=True)
            return False
        
        # Check for test folders
        test_folders = ['1st_test', '2nd_test', '3rd_test']
        available = []
        
        for folder in test_folders:
            folder_path = os.path.join(self.data_path, folder)
            if os.path.exists(folder_path):
                files = [f for f in os.listdir(folder_path) if not f.startswith('.')]
                if len(files) > 0:
                    available.append(folder)
                    print(f"✓ Found {folder} with {len(files)} files")
                else:
                    print(f"✗ {folder} exists but is empty")
            else:
                print(f"✗ {folder} not found")
        
        return len(available) > 0
    
    def load_bearing_file(self, filepath: str) -> np.ndarray:
        """Load a single bearing data file"""
        try:
            # IMS dataset files are in ASCII format with tab separation
            data = pd.read_csv(filepath, sep='\t', header=None)
            return data.values
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None
    
    def extract_features_from_signal(self, signal_data: np.ndarray) -> Dict[str, float]:
        """Extract comprehensive features from vibration signal"""
        features = {}
        
        # Time domain features
        features['rms'] = np.sqrt(np.mean(signal_data**2))
        features['peak'] = np.max(np.abs(signal_data))
        features['peak_to_peak'] = np.max(signal_data) - np.min(signal_data)
        features['crest_factor'] = features['peak'] / features['rms']
        features['kurtosis'] = pd.Series(signal_data).kurtosis()
        features['skewness'] = pd.Series(signal_data).skew()
        features['std'] = np.std(signal_data)
        features['variance'] = np.var(signal_data)
        
        # Shape factor
        features['shape_factor'] = features['rms'] / np.mean(np.abs(signal_data))
        
        # Impulse factor
        features['impulse_factor'] = features['peak'] / np.mean(np.abs(signal_data))
        
        # Clearance factor
        features['clearance_factor'] = features['peak'] / (np.mean(np.sqrt(np.abs(signal_data)))**2)
        
        # Frequency domain features
        freqs, psd = signal.welch(signal_data, fs=self.sampling_rate, nperseg=1024)
        
        # Find peak frequency
        peak_idx = np.argmax(psd)
        features['peak_frequency'] = freqs[peak_idx]
        features['peak_amplitude'] = psd[peak_idx]
        
        # Spectral features
        features['spectral_centroid'] = np.sum(freqs * psd) / np.sum(psd)
        features['spectral_spread'] = np.sqrt(np.sum(((freqs - features['spectral_centroid'])**2) * psd) / np.sum(psd))
        features['spectral_entropy'] = -np.sum(psd * np.log(psd + 1e-10))
        
        # Band power features (relevant for bearing faults)
        bands = {
            'low': (0, 1000),
            'mid': (1000, 4000),
            'high': (4000, 8000),
            'very_high': (8000, 10000)
        }
        
        for band_name, (low, high) in bands.items():
            band_mask = (freqs >= low) & (freqs < high)
            features[f'band_power_{band_name}'] = np.sum(psd[band_mask])
        
        return features
    
    def process_test_set(self, test_name: str) -> pd.DataFrame:
        """Process a complete test set (1st_test, 2nd_test, or 3rd_test)"""
        test_path = os.path.join(self.data_path, test_name)
        
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"Test set {test_name} not found at {test_path}")
        
        all_data = []
        test_info = self.test_sets[test_name]
        
        # Get all data files sorted by timestamp
        data_files = sorted([f for f in os.listdir(test_path) if not f.startswith('.')])
        
        if len(data_files) == 0:
            raise ValueError(f"No data files found in {test_path}")
        
        print(f"Processing {len(data_files)} files from {test_name}...")
        
        # Calculate total files for RUL estimation
        total_files = len(data_files)
        
        for idx, filename in enumerate(data_files):
            if idx % 50 == 0:  # Progress update
                print(f"  Processing file {idx+1}/{total_files}...")
            
            filepath = os.path.join(test_path, filename)
            
            # Extract timestamp from filename
            timestamp = filename
            
            # Load data (4 channels for 4 bearings)
            raw_data = self.load_bearing_file(filepath)
            
            if raw_data is None:
                continue
            
            # Process each bearing
            for bearing_idx, bearing_name in enumerate(test_info['bearings']):
                if bearing_idx < raw_data.shape[1]:
                    signal_data = raw_data[:, bearing_idx]
                    
                    # Extract features
                    features = self.extract_features_from_signal(signal_data)
                    
                    # Add metadata
                    features['test_set'] = test_name
                    features['bearing'] = bearing_name
                    features['timestamp'] = timestamp
                    features['file_index'] = idx
                    
                    # Determine health status and RUL
                    if bearing_name == test_info['failed_bearing']:
                        # This bearing will fail
                        rul_cycles = total_files - idx
                        
                        # Health status based on position in lifecycle
                        if idx < total_files * 0.6:
                            features['health_status'] = 'healthy'
                        elif idx < total_files * 0.85:
                            features['health_status'] = 'degrading'
                        else:
                            features['health_status'] = 'faulty'
                    else:
                        # Healthy bearing
                        rul_cycles = total_files + 1000
                        features['health_status'] = 'healthy'
                    
                    features['rul_cycles'] = rul_cycles
                    features['rul_percentage'] = (rul_cycles / total_files) * 100
                    
                    all_data.append(features)
        
        print(f"Completed processing {test_name}")
        return pd.DataFrame(all_data)
    
    def load_all_test_sets(self) -> pd.DataFrame:
        """Load and process all three test sets"""
        all_data = []
        
        for test_name in self.test_sets.keys():
            try:
                print(f"\nProcessing {test_name}...")
                test_data = self.process_test_set(test_name)
                all_data.append(test_data)
                print(f"✓ Processed {len(test_data)} samples from {test_name}")
            except Exception as e:
                print(f"✗ Error processing {test_name}: {e}")
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            print(f"\nTotal samples loaded: {len(combined_df)}")
            return combined_df
        else:
            raise ValueError("No data could be loaded")
    
    def create_windowed_features(self, df: pd.DataFrame, window_size: int = 10) -> pd.DataFrame:
        """Create features using sliding windows for better RUL prediction"""
        windowed_data = []
        
        for (test_set, bearing), group in df.groupby(['test_set', 'bearing']):
            group = group.sort_values('file_index')
            
            for i in range(len(group) - window_size + 1):
                window = group.iloc[i:i+window_size]
                
                # Calculate statistical features over the window
                window_features = {}
                
                # Trend features
                for col in ['rms', 'peak', 'kurtosis', 'peak_frequency']:
                    if col in window.columns:
                        values = window[col].values
                        window_features[f'{col}_mean'] = np.mean(values)
                        window_features[f'{col}_std'] = np.std(values)
                        window_features[f'{col}_trend'] = np.polyfit(range(len(values)), values, 1)[0]
                        window_features[f'{col}_max'] = np.max(values)
                        window_features[f'{col}_min'] = np.min(values)
                
                # Metadata from last sample in window
                window_features['test_set'] = test_set
                window_features['bearing'] = bearing
                window_features['health_status'] = window.iloc[-1]['health_status']
                window_features['rul_cycles'] = window.iloc[-1]['rul_cycles']
                window_features['timestamp'] = window.iloc[-1]['timestamp']
                
                windowed_data.append(window_features)
        
        return pd.DataFrame(windowed_data)
    
    def visualize_bearing_degradation(self, df: pd.DataFrame, test_set: str, bearing: str):
        """Visualize bearing degradation over time"""
        bearing_data = df[(df['test_set'] == test_set) & (df['bearing'] == bearing)].sort_values('file_index')
        
        if len(bearing_data) == 0:
            print(f"No data found for {test_set} - {bearing}")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Bearing Degradation: {test_set} - {bearing}', fontsize=16)
        
        # RMS trend
        axes[0, 0].plot(bearing_data['file_index'], bearing_data['rms'])
        axes[0, 0].set_title('RMS Vibration')
        axes[0, 0].set_xlabel('Time Index')
        axes[0, 0].set_ylabel('RMS Value')
        axes[0, 0].grid(True)
        
        # Kurtosis trend
        axes[0, 1].plot(bearing_data['file_index'], bearing_data['kurtosis'])
        axes[0, 1].set_title('Kurtosis')
        axes[0, 1].set_xlabel('Time Index')
        axes[0, 1].set_ylabel('Kurtosis Value')
        axes[0, 1].grid(True)
        
        # Peak frequency
        axes[1, 0].plot(bearing_data['file_index'], bearing_data['peak_frequency'])
        axes[1, 0].set_title('Peak Frequency')
        axes[1, 0].set_xlabel('Time Index')
        axes[1, 0].set_ylabel('Frequency (Hz)')
        axes[1, 0].grid(True)
        
        # Crest factor
        axes[1, 1].plot(bearing_data['file_index'], bearing_data['crest_factor'])
        axes[1, 1].set_title('Crest Factor')
        axes[1, 1].set_xlabel('Time Index')
        axes[1, 1].set_ylabel('Crest Factor')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        # Save plot
        save_path = f'results/nasa_bearing_degradation_{test_set}_{bearing.replace(" ", "_")}.png'
        plt.savefig(save_path, dpi=300)
        print(f"Saved plot to {save_path}")
        
        return fig