import sys
print(f"Python version: {sys.version}")
print("\nTesting imports...")

packages = [
    'numpy',
    'pandas', 
    'sklearn',
    'matplotlib',
    'seaborn',
    'plotly',
    'xgboost',
    'scipy',
    'pywavelets',
    'streamlit',
    'joblib',
    'yaml',
    'tqdm'
]

failed = []
for package in packages:
    try:
        if package == 'sklearn':
            import sklearn
        elif package == 'yaml':
            import yaml
        else:
            __import__(package)
        print(f"✓ {package} imported successfully")
    except ImportError as e:
        print(f"✗ {package} import failed: {e}")
        failed.append(package)

if not failed:
    print("\n✅ All packages imported successfully!")
else:
    print(f"\n❌ Failed packages: {', '.join(failed)}")

# Test TensorFlow separately (it's large and sometimes problematic)
try:
    import tensorflow as tf
    print(f"\n✓ TensorFlow {tf.__version__} imported successfully")
except ImportError as e:
    print(f"\n✗ TensorFlow import failed: {e}")
    print("  (This is optional - the project can work without it)")
