# Term Deposit Prediction - Frontend Applications

This project includes two Streamlit frontend applications for the Term Deposit Subscription Predictor:

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Virtual environment activated
- All dependencies installed: `pip install -r requirements.txt`

### Running the Applications

#### 1. Simple Prediction App (`app.py`)
A basic prediction interface with form inputs and results.

```bash
streamlit run app.py
```

**Features:**
- Simple form-based input
- Basic prediction results
- Business insights

#### 2. Enhanced Analytics Dashboard (`enhanced_app.py`)
A comprehensive dashboard with multiple pages and advanced visualizations.

```bash
streamlit run enhanced_app.py
```

**Features:**
- 📊 **Dashboard**: Key metrics and overview visualizations
- 📈 **Data Exploration**: Interactive feature analysis
- 🎯 **Prediction Tool**: Advanced prediction interface with insights
- 📊 **Model Insights**: Model performance and feature importance
- 📋 **About**: Project information and documentation

## 🎯 Dashboard Pages

### 🏠 Dashboard
- Key performance metrics
- Subscription distribution charts
- Success rates by demographics
- Age and duration analysis

### 📈 Data Exploration
- Dataset overview and statistics
- Interactive feature analysis
- Correlation matrix
- Custom feature selection

### 🎯 Prediction Tool
- Comprehensive input form
- Real-time prediction results
- Probability visualization
- Business insights and recommendations

### 📊 Model Insights
- Model performance metrics
- Feature importance analysis
- Business recommendations
- Technical details

### 📋 About
- Project overview
- Dataset information
- Technical stack
- Key findings

## 🎨 Features

### Interactive Visualizations
- Plotly charts for dynamic exploration
- Responsive design for all screen sizes
- Color-coded success/failure indicators

### Smart Insights
- Real-time business recommendations
- Feature-based insights
- Confidence level indicators

### User Experience
- Intuitive navigation
- Clean, modern interface
- Mobile-responsive design
- Fast loading with caching

## 🔧 Technical Details

### Dependencies
- **Streamlit**: Web application framework
- **Plotly**: Interactive visualizations
- **Pandas**: Data manipulation
- **Scikit-learn**: Machine learning
- **Joblib**: Model persistence

### Performance
- Data caching for fast loading
- Efficient model loading
- Optimized visualizations

## 📱 Usage Tips

1. **Start with Dashboard**: Get an overview of your data
2. **Explore Features**: Use Data Exploration to understand patterns
3. **Make Predictions**: Use the Prediction Tool for individual clients
4. **Review Insights**: Check Model Insights for business recommendations

## 🚨 Troubleshooting

### Model Not Found
If you see "Model file not found" error:
1. Run `analysis_notebook.py` first to train the model
2. Ensure `simple_term_deposit_model.pkl` exists in the root directory

### Missing Dependencies
If you get import errors:
```bash
pip install -r requirements.txt
```

### Port Issues
If port 8501 is busy:
```bash
streamlit run enhanced_app.py --server.port 8502
```

## 📊 Data Sources

The applications use the bank marketing dataset:
- **File**: `data/bank-additional-full.csv`
- **Records**: 45,211
- **Features**: 16 input variables + 1 target
- **Period**: May 2008 - November 2010

## 🎯 Business Value

This frontend provides:
- **Data-Driven Insights**: Visual patterns and trends
- **Predictive Analytics**: Individual client predictions
- **Strategic Recommendations**: Actionable business advice
- **Performance Monitoring**: Model and campaign metrics

---

**Happy Analyzing! 🎉** 