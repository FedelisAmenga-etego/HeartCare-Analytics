# ❤️ HeartCare Analytics Dashboard

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://heartcare-analytics.streamlit.app)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![GitHub](https://img.shields.io/badge/github-FedelisAmenga--etego-black.svg)](https://github.com/FedelisAmenga-etego)

A **professional-grade medical analytics platform** built with Streamlit that leverages machine learning to assess heart disease risk. The dashboard provides real-time patient insights, statistical analysis, and predictive modeling with an intuitive, modern interface.

## 🎯 Features

### 📊 **Overview Dashboard**
- Real-time dataset statistics and distribution analysis
- Interactive visualizations of key health metrics
- Disease prevalence by demographics (age, sex, chest pain type)
- Filtered data exploration with dynamic updates

### 🎯 **Patient Risk Assessment**
- Comprehensive patient data input interface
- ML-powered risk probability prediction
- Visual risk gauge with color-coded indicators
- Automated risk factor identification and reporting

### 📉 **Advanced Analytics**
- Age-group stratified disease prevalence analysis
- Cholesterol vs Heart Rate correlation studies
- Heart rate reserve distribution analysis
- Combined multivariate risk scoring
- Detailed demographic comparisons

### 🔍 **Statistical Insights**
- T-tests for continuous variables (cholesterol, blood pressure)
- Chi-square tests for categorical associations
- Correlation matrix analysis with heatmaps
- Clinical pattern recognition across patient cohorts

### ⚙️ **Model Performance**
- Random Forest classifier with 200 decision trees
- 80/20 train-test split with stratified sampling
- Comprehensive performance metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
- Confusion matrix visualization
- Feature importance analysis via permutation importance

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip or conda package manager
- 200MB free disk space

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/FedelisAmenga-etego/heartcare-dashboard.git
   cd heartcare-dashboard
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare your data**
   ```bash
   mkdir data
   # Place your heart.csv file in the data folder
   ```

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

6. **Access the dashboard**
   Open your browser to `http://localhost:8501`

## 📋 Project Structure

```
heartcare-dashboard/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── .streamlit/
│   └── config.toml                # Streamlit configuration
├── data/
│   └── heart.csv                  # Heart disease dataset
├── heart_disease_rf_model.pkl     # Trained ML model (auto-generated)
├── README.md                       # Project documentation
└── LICENSE                         # MIT License
```

## 📊 Dataset Format

The application expects a CSV file with the following columns:

| Column | Type | Description |
|--------|------|-------------|
| age | int | Patient age in years |
| sex | int | 1=Male, 0=Female |
| cp | int | Chest pain type (0-3) |
| trestbps | int | Resting blood pressure (mmHg) |
| chol | int | Serum cholesterol (mg/dl) |
| fbs | int | Fasting blood sugar > 120 mg/dl (1=yes, 0=no) |
| restecg | int | Resting ECG results (0-2) |
| thalach | int | Maximum heart rate achieved (bpm) |
| exang | int | Exercise induced angina (1=yes, 0=no) |
| oldpeak | float | ST depression induced by exercise |
| slope | int | Slope of ST segment (0-2) |
| ca | int | Number of major vessels (0-4) |
| thal | int | Thalassemia type (0-3) |
| target | int | Heart disease presence (1=yes, 0=no) |

**Example datasets:**
- [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)
- [Kaggle Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)

## 🎨 Dashboard Tabs

### 1. **📈 Overview**
- Key metrics with disease statistics
- Age and cholesterol distributions
- Chest pain type analysis
- Sex-based prevalence comparison

### 2. **🎯 Prediction**
- Patient information entry form
- Clinical measurements input
- Real-time risk assessment
- Risk factor enumeration

### 3. **📉 Analytics**
- Demographic comparisons
- Age-group risk stratification
- Scatter plot analysis
- Distribution visualizations

### 4. **🔍 Insights**
- Statistical hypothesis testing
- Correlation analysis
- P-value interpretations
- Clinical significance assessment

### 5. **⚙️ Model**
- Performance metrics display
- Model configuration details
- Confusion matrix
- Training data statistics

## 🤖 Machine Learning Pipeline

### Model Architecture
- **Algorithm:** Random Forest Classifier
- **Trees:** 200 decision trees
- **Class Weight:** Balanced to handle imbalanced data
- **Random State:** 42 (reproducibility)

### Preprocessing Pipeline
```
Input Data
    ↓
StandardScaler (numeric features)
    ↓
OneHotEncoder (categorical features)
    ↓
Random Forest Classifier
    ↓
Risk Probability (0-100%)
```

### Feature Engineering
- **chol_age_ratio:** Cholesterol normalized by age
- **predicted_max_hr:** Estimated max heart rate (220 - age or Tanaka formula)
- **heart_rate_reserve:** Difference between predicted and actual max HR
- **age_trestbps_interaction:** Interactive effect of age and blood pressure

## 📈 Performance Metrics

The model is evaluated on a held-out 20% test set using:

- **Accuracy:** Overall correctness of predictions
- **Precision:** True positives / (true positives + false positives)
- **Recall:** True positives / (true positives + false negatives)
- **F1 Score:** Harmonic mean of precision and recall
- **ROC-AUC:** Area under the receiver operating characteristic curve

## ⚙️ Configuration

### Streamlit Settings (`config.toml`)
```toml
[theme]
primaryColor = "#7C3AED"
backgroundColor = "#0F172A"
secondaryBackgroundColor = "#1E293B"
textColor = "#FFFFFF"

[server]
port = 8501
maxUploadSize = 200
```

### Customization Options
- **Tanaka Formula:** Toggle between 220-age or 208-0.7*age for max HR estimation
- **Filters:** Age range, sex, and chest pain type filtering
- **Model Retraining:** Button to retrain on latest data

## 🔧 Advanced Features

### Risk Factor Analysis
- Automatic detection of cardiovascular risk indicators
- Color-coded risk warnings (🔴 High, 🟡 Borderline, ✅ Low)
- Personalized risk factor summaries

### Statistical Testing
- T-tests for continuous variable differences
- Chi-square tests for categorical associations
- P-value interpretation with significance indicators

### Data Exploration
- Correlation heatmaps
- Distribution analysis (histograms, box plots, violin plots)
- Scatter plots with interactive hover details

## 📦 Dependencies

```
streamlit>=1.28.0
pandas>=1.5.0
numpy>=1.24.0
plotly>=5.17.0
scikit-learn>=1.3.0
scipy>=1.11.0
joblib>=1.3.0
```

For full requirements, see `requirements.txt`

## 🚀 Deployment

### Local Deployment
```bash
streamlit run app.py
```

### Streamlit Cloud
1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Select `app.py` as the main file
5. Deploy!

### Docker
```bash
docker build -t heartcare-dashboard .
docker run -p 8501:8501 heartcare-dashboard
```

## ⚠️ Important Disclaimer

**This application is for educational and informational purposes only.** It should not be used for:
- Clinical diagnosis
- Treatment decisions
- Medical emergencies
- Replacing professional medical consultation

Always consult with qualified healthcare professionals for medical advice and diagnosis.

## 📊 Data Privacy

- No data is stored or transmitted to external servers
- All processing happens locally on your machine
- Models are trained and saved locally
- HIPAA compliance depends on your deployment method

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Fedelis Amenga-etego**
- GitHub: [@FedelisAmenga-etego](https://github.com/FedelisAmenga-etego)
- Email: [Your Email]

## 🙏 Acknowledgments

- UCI Machine Learning Repository for the heart disease dataset
- Streamlit for the amazing web framework
- Scikit-learn for machine learning algorithms
- Plotly for interactive visualizations

## 📚 References

- [Heart Disease on Wikipedia](https://en.wikipedia.org/wiki/Cardiovascular_disease)
- [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Scikit-learn Guide](https://scikit-learn.org/)

## 📞 Support

For issues, questions, or suggestions:
1. Check existing [GitHub Issues](https://github.com/FedelisAmenga-etego/heartcare-dashboard/issues)
2. Create a new issue with detailed description
3. Include steps to reproduce any bugs

---

**Built by Fedelis Amenga-etego** | *Advanced Medical Intelligence Platform*