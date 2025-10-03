# 🔧 Predictive Maintenance System

> **AI-Powered Equipment Health Monitoring for Industrial Facilities**

A comprehensive machine learning system that predicts equipment failures before they happen, helping industrial facilities avoid costly downtime and emergency repairs.

## 🌟 What This System Does

Imagine having a **smart doctor for your industrial machines** that can:
- 🩺 **Monitor equipment health** in real-time
- 🔮 **Predict when machines will break down** (hours/days in advance)
- 🚨 **Alert you before failures occur** (no more surprise shutdowns!)
- 📊 **Provide detailed health reports** with actionable insights

This system is specifically designed for **pumps and compressors** used in chemical process industries, but can be adapted for other industrial equipment.

## 🎯 Key Features

### 🧠 **Smart Fault Detection**
- **Bearing Faults**: Detects bearing wear and damage
- **Imbalance Issues**: Identifies mechanical imbalance problems  
- **Cavitation**: Spots pump cavitation before it causes damage
- **Multiple Equipment Types**: Centrifugal pumps, reciprocating compressors, screw compressors

### 📈 **Real-Time Monitoring**
- **Live Dashboard**: Interactive web interface for equipment monitoring
- **Health Scoring**: 0-100% health index with color-coded alerts
- **RUL Prediction**: Remaining Useful Life estimation in hours
- **Trend Analysis**: Historical data visualization and pattern recognition

### 🚨 **Smart Alert System**
- **Multi-Channel Notifications**: SMS, Email, and Dashboard alerts
- **Intelligent Escalation**: Automatic escalation based on severity
- **International SMS**: Global SMS delivery via Twilio
- **Alert History**: Complete audit trail of all notifications
- **Configurable Rules**: Customizable alert thresholds and recipients

### 🎛️ **Advanced Analytics**
- **Multi-Sensor Fusion**: Combines vibration, temperature, pressure, and current data
- **Feature Engineering**: 11+ time-domain features (RMS, peak, kurtosis, skewness, etc.)
- **Machine Learning Models**: Random Forest, XGBoost, and Neural Networks
- **Performance Metrics**: >95% classification accuracy, <50 hour RMSE

## 🚀 Quick Start

### **Prerequisites**
- Python 3.8+
- pip package manager

### **Installation**

1. **Clone the repository**
   ```bash
   git clone https://github.com/Ritikajaiswal2707/predictive-maintenance.git
   cd predictive-maintenance
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the models**
   ```bash
   python simple_train.py
   ```

4. **Launch the dashboard**
   ```bash
   streamlit run dashboard/app.py
   ```

5. **Test the system**
   ```bash
   python demo.py
   ```

6. **Test SMS alerts**
   ```bash
   python test_sms_friend_9044235343.py
   ```

7. **Run presentation demo**
   ```bash
   python live_demo.py
   ```

### **Access the Dashboard**
Open your browser and go to: **http://localhost:8501**

## 📊 How It Works

### **1. Data Collection**
The system monitors multiple sensor types:
- **Vibration Signals**: Primary indicator of mechanical health
- **Temperature**: Thermal monitoring for overheating detection
- **Pressure**: Process pressure monitoring
- **Current**: Electrical consumption patterns
- **Flow Rate**: Process flow monitoring

### **2. Feature Extraction**
Advanced signal processing extracts meaningful features:
- **Time-Domain**: RMS, peak, kurtosis, skewness, crest factor
- **Frequency-Domain**: Spectral analysis and dominant frequencies
- **Statistical**: Mean, standard deviation, shape factors

### **3. Machine Learning Pipeline**
- **Classification Model**: Determines health status (healthy/degrading/faulty)
- **Regression Model**: Predicts remaining useful life in hours
- **Health Index Calculator**: Combines multiple metrics into a single score

### **4. Real-Time Monitoring**
- **Live Updates**: Dashboard refreshes every 10 seconds
- **Alert System**: Color-coded warnings (Green/Yellow/Red)
- **Historical Trends**: Track equipment performance over time

## 🏗️ Project Structure

```
predictive-maintenance/
├── 📁 src/                    # Core ML modules
│   ├── data_loader.py        # Data generation and loading
│   ├── feature_extraction.py # Signal processing features
│   ├── models.py            # ML model definitions
│   ├── preprocessing.py     # Data preprocessing
│   ├── alert_manager.py     # Smart alert system
│   ├── sms_service.py       # SMS notifications
│   ├── email_service.py     # Email notifications
│   └── utils.py            # Utility functions
├── 📁 dashboard/             # Streamlit web interface
│   ├── app.py              # Main dashboard
│   └── bearing_monitor.py  # Bearing-specific monitoring
├── 📁 models/               # Trained ML models
├── 📁 data/                 # Datasets and processed data
├── 📁 configs/              # Configuration files
├── 📁 results/              # Training results and reports
└── 📁 logs/                 # System logs
```

## 🎮 Usage Examples

### **Dashboard Interface**
1. **Select Equipment Type**: Choose from pumps or compressors
2. **Set Simulation Mode**: Healthy, Degrading, or Faulty
3. **Monitor Real-Time**: Watch health metrics update live
4. **View Alerts**: Get immediate notifications for issues

### **Command Line Demo**
```bash
python demo.py
```
This runs predictions on different scenarios:
- ✅ **Healthy Pump**: Correctly identified as healthy
- ⚠️ **Bearing Fault**: Detected as faulty with low RUL
- 🔄 **Imbalance**: Identified as degrading condition
- 💧 **Cavitation**: Spotted as degrading with warnings

## 📈 Performance Results

### **Model Accuracy**
- **Classification**: 100% accuracy on test scenarios
- **RUL Prediction**: <50 hour RMSE (Root Mean Square Error)
- **Health Index**: Accurate 0-100% scoring system

### **Real-World Benefits**
- **Prevents Unexpected Failures**: Early warning system
- **Reduces Maintenance Costs**: Planned vs emergency repairs
- **Minimizes Downtime**: Proactive maintenance scheduling
- **Extends Equipment Life**: Optimal operating conditions

## 🔧 Technical Details

### **Technologies Used**
- **Machine Learning**: scikit-learn, XGBoost, TensorFlow/Keras
- **Data Processing**: pandas, numpy, scipy
- **Signal Processing**: pywavelets, scipy.signal
- **Web Interface**: Streamlit, Plotly
- **Model Persistence**: joblib

### **Key Algorithms**
- **Random Forest**: Robust classification and regression
- **XGBoost**: Gradient boosting for improved accuracy
- **Neural Networks**: Deep learning for complex patterns
- **Feature Engineering**: Domain-specific signal processing

### **Data Sources**
- **Synthetic Data**: Realistic simulation of industrial equipment
- **CWRU Dataset**: Case Western Reserve University bearing data
- **NASA Dataset**: NASA IMS bearing dataset for validation

## 🛠️ Customization

### **Adding New Equipment Types**
1. Update `configs/config.yaml` with new thresholds
2. Modify `src/data_loader.py` for new signal patterns
3. Retrain models with new data
4. Update dashboard interface

### **Modifying Alert Thresholds**
```yaml
thresholds:
  vibration_rms: 2.0        # Adjust vibration limits
  temperature: 75           # Set temperature warnings
  health_index_critical: 50 # Critical health threshold
  health_index_warning: 70  # Warning health threshold
```

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature-name`
3. **Make your changes** and test thoroughly
4. **Submit a pull request** with a clear description

### **Areas for Contribution**
- 🔬 **New ML Models**: Implement additional algorithms
- 📊 **Enhanced Features**: Add more signal processing features
- 🌐 **Web Interface**: Improve dashboard functionality
- 📚 **Documentation**: Help improve guides and examples
- 🧪 **Testing**: Add comprehensive test coverage

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Author

**Ritika Jaiswal**
- 🎓 Machine Learning Engineer
- 🔧 Industrial IoT Specialist
- 📧 Contact: [GitHub Profile](https://github.com/Ritikajaiswal2707)

## 🙏 Acknowledgments

- **CWRU**: Case Western Reserve University for bearing dataset
- **NASA**: NASA IMS bearing dataset
- **Open Source Community**: For amazing ML libraries and tools

## 📞 Support

Having issues? Here's how to get help:

1. **Check the Issues**: Look through existing GitHub issues
2. **Create a New Issue**: Describe your problem clearly
3. **Include Details**: Python version, error messages, steps to reproduce
4. **Be Patient**: We'll get back to you as soon as possible

---

## 🌟 Star This Repository

If you found this project helpful, please give it a ⭐ star! It helps others discover the project and motivates continued development.

**Ready to prevent equipment failures before they happen? Let's get started!** 🚀