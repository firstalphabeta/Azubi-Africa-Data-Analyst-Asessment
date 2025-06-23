# Term Deposit Subscription Predictor

A comprehensive machine learning solution to predict whether bank clients will subscribe to term deposits based on marketing campaign data.

## 📊 Project Overview

This project analyzes direct marketing campaign data from a banking institution to build predictive models that can identify clients likely to subscribe to term deposits. The analysis includes thorough EDA, feature engineering, multiple model comparisons, and actionable business insights.

## 🎯 Business Objective

Predict whether a client will subscribe to a term deposit (y = "yes" or "no") to:
- Optimize marketing campaign efficiency
- Reduce costs by targeting high-probability prospects
- Improve conversion rates through data-driven insights

## 📁 Dataset Information

The project includes 4 datasets in the `data/` directory:

- **bank-additional-full.csv**: Complete dataset (41,188 examples, 20 features)
- **bank-additional.csv**: 10% sample (4,119 examples, 20 features)  
- **bank-full.csv**: Complete dataset (45,211 examples, 17 features)
- **bank.csv**: 10% sample (4,521 examples, 17 features)

### Key Features:
- **Client data**: age, job, marital status, education, default, balance, housing, loan
- **Campaign data**: contact type, month, day, duration, campaign count
- **Previous campaigns**: pdays, previous contacts, outcome
- **Economic indicators**: employment rate, price index, confidence index, euribor3m
- **Target**: y (subscription: yes/no)

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Simple Analysis
```bash
python analysis_notebook.py
```

### 3. Run Comprehensive Analysis
```bash
python term_deposit_predictor.py
```

## 📈 Analysis Components

### 1. Exploratory Data Analysis (EDA)
- Target variable distribution analysis
- Feature distributions and correlations
- Success rate analysis by demographic segments
- Campaign effectiveness patterns
- Economic indicator impacts

### 2. Feature Engineering
- Age group categorization
- Call duration categories
- Campaign intensity levels
- Economic score combinations
- Previous success indicators
- Contact frequency metrics

### 3. Model Development
- **Logistic Regression**: Interpretable baseline model
- **Random Forest**: Ensemble method with feature importance
- **Gradient Boosting**: Advanced ensemble technique
- **SVM**: Support Vector Machine for complex patterns

### 4. Model Optimization
- Cross-validation for robust evaluation
- Hyperparameter tuning via GridSearch
- Class imbalance handling with SMOTE
- Performance metrics: Accuracy, Precision, Recall, F1, ROC-AUC

### 5. Business Insights
- Feature importance analysis
- Customer segment profiling
- Campaign optimization recommendations
- ROI improvement strategies

## 📊 Key Findings

### Top Success Factors:
1. **Call Duration**: Longer calls (>500 seconds) have significantly higher success rates
2. **Education Level**: Tertiary education clients show 2x higher conversion
3. **Job Type**: Students and retirees have highest subscription rates
4. **Campaign Frequency**: Success decreases with repeated contacts

### Business Recommendations:
- **Quality over Quantity**: Focus on meaningful conversations rather than call volume
- **Targeted Approach**: Prioritize educated professionals and specific job segments
- **Campaign Limits**: Avoid excessive contact attempts to prevent negative impact
- **Seasonal Timing**: Leverage monthly patterns for optimal campaign timing

## 🎯 Model Performance

The best performing model achieves:
- **F1 Score**: ~65-70%
- **Precision**: ~60-65% (accurate targeting)
- **Recall**: ~70-75% (captures most opportunities)
- **ROC-AUC**: ~85-90% (excellent discrimination)

## 📁 Project Structure

```
Term-Deposit-Subscription-Predictor/
├── data/                          # Dataset files
├── requirements.txt               # Python dependencies
├── analysis_notebook.py          # Simple analysis script
├── term_deposit_predictor.py     # Comprehensive analysis class
├── README.md                     # This file
├── eda_analysis.png              # EDA visualizations
├── model_evaluation.png          # Model comparison plots
├── basic_eda.png                 # Basic analysis charts
└── simple_term_deposit_model.pkl # Saved model
```

## 🔧 Usage Examples

### Load and Use Trained Model
```python
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('simple_term_deposit_model.pkl')

# Make predictions on new data
# new_data should have the same features as training data
predictions = model.predict(new_data)
probabilities = model.predict_proba(new_data)[:, 1]
```

### Interactive Analysis
```python
from term_deposit_predictor import TermDepositPredictor

# Run complete analysis
predictor = TermDepositPredictor()
predictor.run_complete_analysis()

# Or run step by step
predictor.load_data()
predictor.exploratory_data_analysis()
predictor.feature_engineering()
# ... continue with other methods
```

## 📋 Evaluation Metrics

The models are evaluated using multiple metrics to ensure robust performance:

- **Accuracy**: Overall correctness
- **Precision**: Positive prediction accuracy (minimize false positives)
- **Recall**: Sensitivity to positive cases (minimize false negatives)
- **F1 Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the receiver operating characteristic curve

## 🎪 Model Interpretability

### Feature Importance Analysis:
- Duration and economic indicators are top predictors
- Previous campaign outcomes strongly influence decisions
- Demographic factors provide valuable segmentation insights
- Contact type and timing affect success probability

### Business Impact Metrics:
- **Cost Reduction**: Target high-probability prospects (60%+ precision)
- **Revenue Optimization**: Capture majority of opportunities (70%+ recall)
- **Efficiency Gains**: Reduce unnecessary calls while maintaining conversion rates

## 🚀 Future Enhancements

1. **Advanced Models**: XGBoost, LightGBM, Neural Networks
2. **Time Series Analysis**: Seasonal patterns and trend analysis
3. **Customer Lifetime Value**: Integrate CLV predictions
4. **Real-time Scoring**: Deploy model for live campaign optimization
5. **A/B Testing Framework**: Continuous model improvement

## 📞 Model Deployment Ready

The trained model is production-ready and can be deployed for:
- Real-time customer scoring during calls
- Batch processing for campaign planning
- Integration with CRM systems
- Marketing automation platforms

## 🤝 Contributing

Feel free to contribute by:
- Adding new feature engineering techniques
- Implementing additional models
- Improving visualizations
- Enhancing documentation

## 📄 License

This project is available for educational and commercial use. Please credit the contributors when using the code.

---

**Ready to boost your marketing campaign effectiveness? Run the analysis and start making data-driven decisions!** 🎯 