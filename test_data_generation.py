from src.data_loader import DataGenerator, DataLoader
import pandas as pd

print("Testing data generation...")
try:
    generator = DataGenerator(sample_rate=100, duration=10)
    healthy_data = generator.generate_healthy_signal()
    print(f"Generated healthy signal with {len(healthy_data['vibration'])} points")
    
    loader = DataLoader()
    df = loader.generate_synthetic_dataset(n_samples=100)
    print(f"Generated dataset with shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print("\nSample data:")
    print(df.head())
    print("\nData generation successful!")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()