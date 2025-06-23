import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Term Deposit Analytics Dashboard",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-metric {
        border-left-color: #28a745;
    }
    .warning-metric {
        border-left-color: #ffc107;
    }
</style>
""", unsafe_allow_html=True)

# Load data and model
@st.cache_data
def load_data():
    """Load and cache the dataset"""
    try:
        df = pd.read_csv('data/bank-additional-full.csv', sep=';')
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_resource
def load_model():
    """Load and cache the trained model"""
    try:
        return joblib.load('simple_term_deposit_model.pkl')
    except Exception as e:
        st.error(f"Model file not found: {e}")
        st.info("Please ensure the model file exists in the repository.")
        return None

# Load data and model
df = load_data()
model = load_model()

# Sidebar navigation
st.sidebar.title("📊 Navigation")
page = st.sidebar.selectbox(
    "Choose a page:",
    ["🏠 Dashboard", "📈 Data Exploration", "🎯 Prediction Tool", "📊 Model Insights", "📋 About"]
)

# Feature options for prediction
job_options = [
    "admin.", "unknown", "unemployed", "management", "housemaid", "entrepreneur",
    "student", "blue-collar", "self-employed", "retired", "technician", "services"
]
marital_options = ["married", "divorced", "single"]
education_options = ["unknown", "secondary", "primary", "tertiary"]
default_options = ["yes", "no"]
housing_options = ["yes", "no"]
loan_options = ["yes", "no"]
contact_options = ["unknown", "telephone", "cellular"]
month_options = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
poutcome_options = ["unknown", "other", "failure", "success"]

# Dashboard Page
if page == "🏠 Dashboard":
    if df is not None:
        st.markdown('<h1 class="main-header">💰 Term Deposit Analytics Dashboard</h1>', unsafe_allow_html=True)
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3>Total Clients</h3>
                <h2>{:,}</h2>
            </div>
            """.format(len(df)), unsafe_allow_html=True)
        
        with col2:
            subscription_rate = (df['y'] == 'yes').mean() * 100
            st.markdown("""
            <div class="metric-card success-metric">
                <h3>Subscription Rate</h3>
                <h2>{:.1f}%</h2>
            </div>
            """.format(subscription_rate), unsafe_allow_html=True)
        
        with col3:
            avg_age = df['age'].mean()
            st.markdown("""
            <div class="metric-card">
                <h3>Average Age</h3>
                <h2>{:.1f}</h2>
            </div>
            """.format(avg_age), unsafe_allow_html=True)
        
        with col4:
            avg_duration = df['duration'].mean()
            st.markdown("""
            <div class="metric-card">
                <h3>Avg Call Duration</h3>
                <h2>{:.0f}s</h2>
            </div>
            """.format(avg_duration), unsafe_allow_html=True)
        
        # Main visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Subscription Distribution")
            fig = px.pie(
                df, 
                names='y', 
                title="Client Subscription Status",
                color_discrete_map={'yes': '#28a745', 'no': '#dc3545'}
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📈 Age Distribution by Subscription")
            fig = px.histogram(
                df, 
                x='age', 
                color='y',
                barmode='overlay',
                opacity=0.7,
                title="Age Distribution",
                color_discrete_map={'yes': '#28a745', 'no': '#dc3545'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Success rates by key features
        st.subheader("🎯 Success Rates by Key Features")
        col1, col2 = st.columns(2)
        
        with col1:
            # Job success rates
            job_success = df.groupby('job')['y'].apply(lambda x: (x == 'yes').mean() * 100).sort_values(ascending=False)
            fig = px.bar(
                x=job_success.index, 
                y=job_success.values,
                title="Success Rate by Job",
                labels={'x': 'Job', 'y': 'Success Rate (%)'}
            )
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Education success rates
            edu_success = df.groupby('education')['y'].apply(lambda x: (x == 'yes').mean() * 100).sort_values(ascending=False)
            fig = px.bar(
                x=edu_success.index, 
                y=edu_success.values,
                title="Success Rate by Education",
                labels={'x': 'Education', 'y': 'Success Rate (%)'}
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Data could not be loaded. Please check the data file.")

# Data Exploration Page
elif page == "📈 Data Exploration":
    if df is not None:
        st.title("📈 Data Exploration")
        
        # Dataset overview
        st.subheader("📋 Dataset Overview")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Dataset Shape:** {df.shape}")
            st.write(f"**Features:** {len(df.columns)}")
            st.write(f"**Target Variable:** y (subscription)")
        
        with col2:
            st.write("**Data Types:**")
            st.write(df.dtypes.value_counts())
        
        # Feature analysis
        st.subheader("🔍 Feature Analysis")
        
        # Select feature to analyze
        feature_to_analyze = st.selectbox(
            "Select a feature to analyze:",
            [col for col in df.columns if col != 'y']
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            if df[feature_to_analyze].dtype in ['int64', 'float64']:
                # Numerical feature
                fig = px.histogram(
                    df, 
                    x=feature_to_analyze, 
                    color='y',
                    barmode='overlay',
                    opacity=0.7,
                    title=f"{feature_to_analyze.title()} Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Categorical feature
                fig = px.bar(
                    df[feature_to_analyze].value_counts(),
                    title=f"{feature_to_analyze.title()} Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Success rate by feature
            if df[feature_to_analyze].dtype in ['int64', 'float64']:
                # For numerical features, create bins
                df_binned = df.copy()
                df_binned[f'{feature_to_analyze}_binned'] = pd.cut(df[feature_to_analyze], bins=10)
                success_rate = df_binned.groupby(f'{feature_to_analyze}_binned')['y'].apply(
                    lambda x: (x == 'yes').mean() * 100
                )
                fig = px.bar(
                    x=success_rate.index.astype(str), 
                    y=success_rate.values,
                    title=f"Success Rate by {feature_to_analyze.title()}",
                    labels={'x': feature_to_analyze.title(), 'y': 'Success Rate (%)'}
                )
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Categorical feature success rate
                success_rate = df.groupby(feature_to_analyze)['y'].apply(
                    lambda x: (x == 'yes').mean() * 100
                ).sort_values(ascending=False)
                fig = px.bar(
                    x=success_rate.index, 
                    y=success_rate.values,
                    title=f"Success Rate by {feature_to_analyze.title()}",
                    labels={'x': feature_to_analyze.title(), 'y': 'Success Rate (%)'}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Correlation analysis
        st.subheader("🔗 Correlation Analysis")
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        correlation_matrix = df[numerical_cols].corr()
        
        fig = px.imshow(
            correlation_matrix,
            title="Correlation Matrix",
            color_continuous_scale='RdBu',
            aspect='auto'
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Data could not be loaded. Please check the data file.")

# Prediction Tool Page
elif page == "🎯 Prediction Tool":
    st.title("🎯 Term Deposit Prediction Tool")
    st.write("Enter client information to predict subscription likelihood.")
    
    if model is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("👤 Client Information")
            age = st.number_input("Age", min_value=18, max_value=100, value=30)
            job = st.selectbox("Job", job_options)
            marital = st.selectbox("Marital Status", marital_options)
            education = st.selectbox("Education", education_options)
            default = st.selectbox("Has credit in default?", default_options)
            balance = st.number_input("Average yearly balance (euros)", value=1000)
            housing = st.selectbox("Has housing loan?", housing_options)
            loan = st.selectbox("Has personal loan?", loan_options)
        
        with col2:
            st.subheader("📞 Contact Information")
            contact = st.selectbox("Contact communication type", contact_options)
            day = st.number_input("Last contact day of the month", min_value=1, max_value=31, value=15)
            month = st.selectbox("Last contact month", month_options)
            duration = st.number_input("Last contact duration (seconds)", min_value=0, value=100)
            
            st.subheader("📊 Campaign Information")
            campaign = st.number_input("Number of contacts during this campaign", min_value=1, value=1)
            pdays = st.number_input("Days since last contact (-1 if not previously contacted)", value=-1)
            previous = st.number_input("Number of contacts before this campaign", min_value=0, value=0)
            poutcome = st.selectbox("Outcome of previous campaign", poutcome_options)
        
        if st.button("🔮 Predict Subscription", type="primary"):
            # Prepare input
            input_dict = {
                'age': age, 'job': job, 'marital': marital, 'education': education,
                'default': default, 'balance': balance, 'housing': housing, 'loan': loan,
                'contact': contact, 'day': day, 'month': month, 'duration': duration,
                'campaign': campaign, 'pdays': pdays, 'previous': previous, 'poutcome': poutcome
            }
            input_df = pd.DataFrame([input_dict])
            
            # Make prediction
            pred = model.predict(input_df)[0]
            pred_proba = model.predict_proba(input_df)[0][1]
            
            # Display results
            st.markdown("---")
            st.subheader("📊 Prediction Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if pred == 1:
                    st.success(f"✅ **LIKELY TO SUBSCRIBE**")
                    st.metric("Probability", f"{pred_proba:.1%}")
                else:
                    st.error(f"❌ **NOT LIKELY TO SUBSCRIBE**")
                    st.metric("Probability", f"{pred_proba:.1%}")
            
            with col2:
                # Progress bar for probability
                st.write("**Subscription Probability:**")
                st.progress(pred_proba)
                
                # Confidence level
                if pred_proba > 0.8:
                    confidence = "Very High"
                elif pred_proba > 0.6:
                    confidence = "High"
                elif pred_proba > 0.4:
                    confidence = "Medium"
                else:
                    confidence = "Low"
                st.write(f"**Confidence Level:** {confidence}")
            
            # Business insights
            st.subheader("💡 Business Insights")
            insights = []
            
            if job in ['student', 'retired']:
                insights.append("✅ High success rate for this job category")
            elif job in ['blue-collar', 'unemployed']:
                insights.append("⚠️ Lower success rate for this job category")
            
            if education == 'tertiary':
                insights.append("✅ Higher education clients are more likely to subscribe")
            
            if duration > 300:
                insights.append("✅ Longer call duration increases success probability")
            
            if campaign > 3:
                insights.append("⚠️ Multiple campaigns may reduce success rate")
            
            for insight in insights:
                st.write(insight)
    else:
        st.error("Model not available. Please ensure the model file exists.")

# Model Insights Page
elif page == "📊 Model Insights":
    st.title("📊 Model Performance & Insights")
    
    if model is not None:
        # Model information
        st.subheader("🤖 Model Information")
        model_type = type(model.named_steps['classifier']).__name__
        st.write(f"**Model Type:** {model_type}")
        
        # Feature importance (if available)
        if hasattr(model.named_steps['classifier'], 'feature_importances_'):
            st.subheader("🎯 Feature Importance")
            
            feature_names = model.named_steps['preprocessor'].get_feature_names_out()
            importances = model.named_steps['classifier'].feature_importances_
            
            # Create feature importance dataframe
            feature_importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            # Top 10 features
            fig = px.bar(
                feature_importance_df.head(10),
                x='importance',
                y='feature',
                orientation='h',
                title="Top 10 Most Important Features"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Business recommendations
        st.subheader("💼 Business Recommendations")
        
        recommendations = [
            "🎯 **Target High-Value Segments:** Focus on students, retirees, and tertiary-educated clients",
            "⏱️ **Optimize Call Duration:** Aim for meaningful conversations (300+ seconds)",
            "📞 **Limit Campaign Frequency:** Avoid excessive contact attempts",
            "👥 **Train Agents:** Focus on building rapport and understanding client needs",
            "📊 **Monitor Performance:** Track success rates by job category and education level"
        ]
        
        for rec in recommendations:
            st.write(rec)
    
    else:
        st.error("Model not available. Please ensure the model file exists.")

# About Page
elif page == "📋 About":
    st.title("📋 About This Project")
    
    st.subheader("🎯 Project Overview")
    st.write("""
    This Term Deposit Subscription Predictor analyzes banking data to predict whether clients will subscribe to term deposits.
    The project uses machine learning to identify patterns and provide actionable business insights.
    """)
    
    st.subheader("📊 Dataset Information")
    st.write("""
    - **Source:** Bank marketing campaigns data
    - **Size:** 45,211 records
    - **Features:** 16 input variables + 1 target variable
    - **Time Period:** May 2008 to November 2010
    """)
    
    st.subheader("🔧 Technical Details")
    st.write("""
    - **Framework:** Streamlit
    - **Machine Learning:** Scikit-learn
    - **Visualization:** Plotly, Matplotlib, Seaborn
    - **Data Processing:** Pandas, NumPy
    """)
    
    st.subheader("📈 Key Findings")
    st.write("""
    1. **Demographics Matter:** Students and retirees show higher subscription rates
    2. **Education Impact:** Tertiary education clients are more likely to subscribe
    3. **Call Quality:** Longer, meaningful conversations lead to better outcomes
    4. **Campaign Strategy:** Quality over quantity - avoid excessive contact attempts
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>💰 Term Deposit Analytics Dashboard | Built with Streamlit</p>
</div>
""", unsafe_allow_html=True) 