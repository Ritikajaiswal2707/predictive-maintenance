print("Testing imports...")

try:
    import numpy as np
    print("✓ NumPy imported successfully")
    
    import pandas as pd
    print("✓ Pandas imported successfully")
    
    import sklearn
    print("✓ Scikit-learn imported successfully")
    
    import tensorflow as tf
    print("✓ TensorFlow imported successfully")
    print(f"  TensorFlow version: {tf.__version__}")
    
    import streamlit
    print("✓ Streamlit imported successfully")
    
    print("\nAll core packages imported successfully!")
    
except ImportError as e:
    print(f"✗ Import error: {e}")