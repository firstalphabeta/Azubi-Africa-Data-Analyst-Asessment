"""
Term Deposit Subscription Predictor
====================================

This script performs comprehensive analysis and builds a predictive model
to determine if a client will subscribe to a term deposit.

Author: Data Analytics Team
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')

class TermDepositPredictor:
    """
    A comprehensive class for analyzing and predicting term deposit subscriptions.
    """
    
    def __init__(self, data_path='data/bank-additional-full.csv'):
        """Initialize the predictor with data loading."""
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.preprocessor = None
        self.models = {}
        self.results = {}
        
    def load_data(self):
        """Load the dataset and perform initial inspection."""
        print("Loading data...")
        self.df = pd.read_csv(self.data_path, sep=';')
        print(f"Dataset shape: {self.df.shape}")
        print(f"Columns: {list(self.df.columns)}")
        return self
    
    def exploratory_data_analysis(self):
        """Perform comprehensive EDA."""
        print("\n" + "="*50)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*50)
        
        # Basic info
        print("\n1. Dataset Overview:")
        print(f"Shape: {self.df.shape}")
        print(f"Missing values per column:")
        print(self.df.isnull().sum())
        
        print(f"\nData types:")
        print(self.df.dtypes)
        
        # Target variable analysis
        print(f"\n2. Target Variable Analysis:")
        target_counts = self.df['y'].value_counts()
        print(target_counts)
        print(f"Class distribution: {target_counts / len(self.df) * 100}")
        
        # Create visualizations
        self._create_eda_plots()
        
        # Statistical summary
        print(f"\n3. Statistical Summary (Numerical Features):")
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        print(self.df[numerical_cols].describe())
        
        # Categorical analysis
        print(f"\n4. Categorical Features Analysis:")
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col != 'y':
                print(f"\n{col}:")
                print(self.df[col].value_counts().head(10))
        
        return self
    
    def _create_eda_plots(self):
        """Create comprehensive EDA visualizations."""
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Target distribution
        plt.subplot(3, 4, 1)
        target_counts = self.df['y'].value_counts()
        plt.pie(target_counts.values, labels=target_counts.index, autopct='%1.1f%%')
        plt.title('Target Distribution')
        
        # 2. Age distribution
        plt.subplot(3, 4, 2)
        plt.hist(self.df['age'], bins=30, alpha=0.7, edgecolor='black')
        plt.title('Age Distribution')
        plt.xlabel('Age')
        plt.ylabel('Frequency')
        
        # 3. Job distribution
        plt.subplot(3, 4, 3)
        job_counts = self.df['job'].value_counts()
        plt.bar(range(len(job_counts)), job_counts.values)
        plt.title('Job Distribution')
        plt.xticks(range(len(job_counts)), job_counts.index, rotation=45, ha='right')
        
        # 4. Education vs Target
        plt.subplot(3, 4, 4)
        education_target = pd.crosstab(self.df['education'], self.df['y'], normalize='index')
        education_target.plot(kind='bar', stacked=True, ax=plt.gca())
        plt.title('Education vs Target')
        plt.xticks(rotation=45)
        
        # 5. Marital status vs Target
        plt.subplot(3, 4, 5)
        marital_target = pd.crosstab(self.df['marital'], self.df['y'], normalize='index')
        marital_target.plot(kind='bar', ax=plt.gca())
        plt.title('Marital Status vs Target')
        plt.xticks(rotation=45)
        
        # 6. Duration distribution
        plt.subplot(3, 4, 6)
        plt.hist(self.df['duration'], bins=50, alpha=0.7, edgecolor='black')
        plt.title('Call Duration Distribution')
        plt.xlabel('Duration (seconds)')
        
        # 7. Campaign vs Target
        plt.subplot(3, 4, 7)
        campaign_stats = self.df.groupby('campaign')['y'].apply(lambda x: (x == 'yes').sum() / len(x))
        plt.plot(campaign_stats.index, campaign_stats.values, marker='o')
        plt.title('Success Rate by Campaign Count')
        plt.xlabel('Number of Campaigns')
        plt.ylabel('Success Rate')
        
        # 8. Monthly trends
        plt.subplot(3, 4, 8)
        month_order = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 
                      'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
        month_stats = self.df.groupby('month')['y'].apply(lambda x: (x == 'yes').sum() / len(x))
        month_stats = month_stats.reindex(month_order, fill_value=0)
        plt.bar(range(len(month_stats)), month_stats.values)
        plt.title('Success Rate by Month')
        plt.xticks(range(len(month_stats)), month_stats.index, rotation=45)
        
        # 9. Correlation heatmap for numerical features
        plt.subplot(3, 4, 9)
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        correlation_matrix = self.df[numerical_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, ax=plt.gca())
        plt.title('Numerical Features Correlation')
        
        # 10. Balance distribution
        plt.subplot(3, 4, 10)
        plt.hist(self.df['balance'], bins=50, alpha=0.7, edgecolor='black')
        plt.title('Balance Distribution')
        plt.xlabel('Balance (euros)')
        
        # 11. Previous outcome vs Target
        plt.subplot(3, 4, 11)
        poutcome_target = pd.crosstab(self.df['poutcome'], self.df['y'], normalize='index')
        poutcome_target.plot(kind='bar', ax=plt.gca())
        plt.title('Previous Outcome vs Target')
        plt.xticks(rotation=45)
        
        # 12. Contact type vs Target
        plt.subplot(3, 4, 12)
        contact_target = pd.crosstab(self.df['contact'], self.df['y'], normalize='index')
        contact_target.plot(kind='bar', ax=plt.gca())
        plt.title('Contact Type vs Target')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('eda_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def feature_engineering(self):
        """Perform feature engineering and preprocessing."""
        print("\n" + "="*50)
        print("FEATURE ENGINEERING")
        print("="*50)
        
        # Create new features
        print("Creating new features...")
        
        # Age groups
        self.df['age_group'] = pd.cut(self.df['age'], bins=[0, 25, 35, 45, 55, 100], 
                                     labels=['young', 'young_adult', 'middle_aged', 'senior', 'elderly'])
        
        # Duration categories
        self.df['duration_category'] = pd.cut(self.df['duration'], 
                                            bins=[0, 60, 180, 300, float('inf')], 
                                            labels=['very_short', 'short', 'medium', 'long'])
        
        # Campaign intensity
        self.df['campaign_intensity'] = self.df['campaign'].apply(
            lambda x: 'low' if x <= 2 else 'medium' if x <= 5 else 'high'
        )
        
        # Previous contact success
        self.df['previous_success'] = (self.df['poutcome'] == 'success').astype(int)
        
        # Economic indicators combination
        self.df['economic_score'] = (
            self.df['emp.var.rate'] + 
            self.df['cons.price.idx'] / 100 + 
            self.df['cons.conf.idx'] / 100 + 
            self.df['euribor3m']
        )
        
        # Contact frequency
        self.df['contact_frequency'] = self.df['campaign'] + self.df['previous']
        
        # Balance categories
        self.df['balance_category'] = pd.cut(self.df['balance'], 
                                           bins=[-float('inf'), 0, 1000, 5000, float('inf')],
                                           labels=['negative', 'low', 'medium', 'high'])
        
        print("New features created:")
        new_features = ['age_group', 'duration_category', 'campaign_intensity', 
                       'previous_success', 'economic_score', 'contact_frequency', 'balance_category']
        for feature in new_features:
            print(f"- {feature}")
        
        return self
    
    def prepare_data(self):
        """Prepare data for modeling."""
        print("\n" + "="*50)
        print("DATA PREPARATION")
        print("="*50)
        
        # Separate features and target
        X = self.df.drop('y', axis=1)
        y = self.df['y'].map({'yes': 1, 'no': 0})
        
        # Identify column types
        numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object']).columns.tolist()
        
        print(f"Numerical features: {numerical_features}")
        print(f"Categorical features: {categorical_features}")
        
        # Create preprocessing pipelines
        numerical_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(drop='first', handle_unknown='ignore')
        
        # Combine preprocessing steps
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set shape: {self.X_train.shape}")
        print(f"Test set shape: {self.X_test.shape}")
        print(f"Training set target distribution:")
        print(self.y_train.value_counts(normalize=True))
        
        return self
    
    def train_models(self):
        """Train multiple models and compare performance."""
        print("\n" + "="*50)
        print("MODEL TRAINING")
        print("="*50)
        
        # Define models
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'SVM': SVC(random_state=42, probability=True)
        }
        
        # Train models with cross-validation
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Create pipeline with preprocessing
            pipeline = Pipeline([
                ('preprocessor', self.preprocessor),
                ('classifier', model)
            ])
            
            # Cross-validation
            cv_scores = cross_val_score(pipeline, self.X_train, self.y_train, 
                                      cv=5, scoring='f1')
            
            # Fit the model
            pipeline.fit(self.X_train, self.y_train)
            
            # Make predictions
            y_pred = pipeline.predict(self.X_test)
            y_pred_proba = pipeline.predict_proba(self.X_test)[:, 1]
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(self.y_test, y_pred),
                'precision': precision_score(self.y_test, y_pred),
                'recall': recall_score(self.y_test, y_pred),
                'f1': f1_score(self.y_test, y_pred),
                'roc_auc': roc_auc_score(self.y_test, y_pred_proba),
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            self.models[name] = pipeline
            self.results[name] = {
                'model': pipeline,
                'predictions': y_pred,
                'probabilities': y_pred_proba,
                'metrics': metrics
            }
            
            print(f"CV F1 Score: {metrics['cv_mean']:.4f} (+/- {metrics['cv_std']*2:.4f})")
            print(f"Test Accuracy: {metrics['accuracy']:.4f}")
            print(f"Test F1 Score: {metrics['f1']:.4f}")
            print(f"Test ROC AUC: {metrics['roc_auc']:.4f}")
        
        return self
    
    def handle_class_imbalance(self):
        """Handle class imbalance using SMOTE."""
        print("\n" + "="*50)
        print("HANDLING CLASS IMBALANCE")
        print("="*50)
        
        # Apply SMOTE
        smote = SMOTE(random_state=42)
        
        # Best performing model from previous training
        best_model_name = max(self.results.keys(), 
                            key=lambda k: self.results[k]['metrics']['f1'])
        best_model = self.models[best_model_name].named_steps['classifier']
        
        print(f"Applying SMOTE to improve {best_model_name}...")
        
        # Create pipeline with SMOTE
        smote_pipeline = ImbPipeline([
            ('preprocessor', self.preprocessor),
            ('smote', smote),
            ('classifier', best_model)
        ])
        
        # Train with SMOTE
        smote_pipeline.fit(self.X_train, self.y_train)
        
        # Make predictions
        y_pred_smote = smote_pipeline.predict(self.X_test)
        y_pred_proba_smote = smote_pipeline.predict_proba(self.X_test)[:, 1]
        
        # Calculate metrics
        smote_metrics = {
            'accuracy': accuracy_score(self.y_test, y_pred_smote),
            'precision': precision_score(self.y_test, y_pred_smote),
            'recall': recall_score(self.y_test, y_pred_smote),
            'f1': f1_score(self.y_test, y_pred_smote),
            'roc_auc': roc_auc_score(self.y_test, y_pred_proba_smote)
        }
        
        self.results[f'{best_model_name} + SMOTE'] = {
            'model': smote_pipeline,
            'predictions': y_pred_smote,
            'probabilities': y_pred_proba_smote,
            'metrics': smote_metrics
        }
        
        print(f"SMOTE Results:")
        print(f"Accuracy: {smote_metrics['accuracy']:.4f}")
        print(f"Precision: {smote_metrics['precision']:.4f}")
        print(f"Recall: {smote_metrics['recall']:.4f}")
        print(f"F1 Score: {smote_metrics['f1']:.4f}")
        print(f"ROC AUC: {smote_metrics['roc_auc']:.4f}")
        
        return self
    
    def hyperparameter_tuning(self):
        """Perform hyperparameter tuning for the best model."""
        print("\n" + "="*50)
        print("HYPERPARAMETER TUNING")
        print("="*50)
        
        # Best model
        best_model_name = max(self.results.keys(), 
                            key=lambda k: self.results[k]['metrics']['f1'])
        
        print(f"Tuning hyperparameters for {best_model_name}...")
        
        if 'Random Forest' in best_model_name:
            param_grid = {
                'classifier__n_estimators': [100, 200],
                'classifier__max_depth': [10, 20, None],
                'classifier__min_samples_split': [2, 5],
                'classifier__min_samples_leaf': [1, 2]
            }
        elif 'Gradient Boosting' in best_model_name:
            param_grid = {
                'classifier__n_estimators': [100, 200],
                'classifier__learning_rate': [0.05, 0.1, 0.2],
                'classifier__max_depth': [3, 5, 7]
            }
        else:
            print("Skipping hyperparameter tuning for this model.")
            return self
        
        # Create base pipeline
        base_pipeline = Pipeline([
            ('preprocessor', self.preprocessor),
            ('classifier', RandomForestClassifier(random_state=42) if 'Random Forest' in best_model_name 
             else GradientBoostingClassifier(random_state=42))
        ])
        
        # Grid search
        grid_search = GridSearchCV(
            base_pipeline, param_grid, cv=3, scoring='f1', n_jobs=-1, verbose=1
        )
        
        grid_search.fit(self.X_train, self.y_train)
        
        # Best model predictions
        y_pred_tuned = grid_search.predict(self.X_test)
        y_pred_proba_tuned = grid_search.predict_proba(self.X_test)[:, 1]
        
        # Calculate metrics
        tuned_metrics = {
            'accuracy': accuracy_score(self.y_test, y_pred_tuned),
            'precision': precision_score(self.y_test, y_pred_tuned),
            'recall': recall_score(self.y_test, y_pred_tuned),
            'f1': f1_score(self.y_test, y_pred_tuned),
            'roc_auc': roc_auc_score(self.y_test, y_pred_proba_tuned)
        }
        
        self.results[f'{best_model_name} (Tuned)'] = {
            'model': grid_search.best_estimator_,
            'predictions': y_pred_tuned,
            'probabilities': y_pred_proba_tuned,
            'metrics': tuned_metrics
        }
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Tuned F1 Score: {tuned_metrics['f1']:.4f}")
        
        return self
    
    def evaluate_models(self):
        """Comprehensive model evaluation."""
        print("\n" + "="*50)
        print("MODEL EVALUATION")
        print("="*50)
        
        # Create comparison dataframe
        comparison_data = []
        for name, result in self.results.items():
            metrics = result['metrics']
            comparison_data.append({
                'Model': name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1 Score': metrics['f1'],
                'ROC AUC': metrics['roc_auc']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        print("\nModel Comparison:")
        print(comparison_df.round(4))
        
        # Find best model
        best_model_name = comparison_df.loc[comparison_df['F1 Score'].idxmax(), 'Model']
        print(f"\nBest Model: {best_model_name}")
        
        # Detailed evaluation for best model
        best_result = self.results[best_model_name]
        
        print(f"\nDetailed Evaluation for {best_model_name}:")
        print(classification_report(self.y_test, best_result['predictions']))
        
        # Confusion Matrix
        cm = confusion_matrix(self.y_test, best_result['predictions'])
        print(f"\nConfusion Matrix:")
        print(cm)
        
        # Visualization
        self._create_evaluation_plots()
        
        return best_model_name, best_result
    
    def _create_evaluation_plots(self):
        """Create evaluation visualizations."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Model comparison
        comparison_data = []
        for name, result in self.results.items():
            metrics = result['metrics']
            comparison_data.append({
                'Model': name,
                'F1 Score': metrics['f1'],
                'ROC AUC': metrics['roc_auc']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        axes[0, 0].bar(range(len(comparison_df)), comparison_df['F1 Score'])
        axes[0, 0].set_title('F1 Score Comparison')
        axes[0, 0].set_xticks(range(len(comparison_df)))
        axes[0, 0].set_xticklabels(comparison_df['Model'], rotation=45, ha='right')
        
        # 2. ROC Curves
        for name, result in self.results.items():
            fpr, tpr, _ = roc_curve(self.y_test, result['probabilities'])
            axes[0, 1].plot(fpr, tpr, label=f"{name} (AUC: {result['metrics']['roc_auc']:.3f})")
        
        axes[0, 1].plot([0, 1], [0, 1], 'k--', label='Random')
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title('ROC Curves')
        axes[0, 1].legend()
        
        # 3. Best model confusion matrix
        best_model_name = max(self.results.keys(), 
                            key=lambda k: self.results[k]['metrics']['f1'])
        cm = confusion_matrix(self.y_test, self.results[best_model_name]['predictions'])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
        axes[1, 0].set_title(f'Confusion Matrix - {best_model_name}')
        axes[1, 0].set_xlabel('Predicted')
        axes[1, 0].set_ylabel('Actual')
        
        # 4. Feature importance (if available)
        best_model = self.results[best_model_name]['model']
        if hasattr(best_model.named_steps['classifier'], 'feature_importances_'):
            # Get feature names after preprocessing
            feature_names = (self.preprocessor.named_transformers_['num'].get_feature_names_out().tolist() +
                           self.preprocessor.named_transformers_['cat'].get_feature_names_out().tolist())
            
            importances = best_model.named_steps['classifier'].feature_importances_
            indices = np.argsort(importances)[-10:]  # Top 10 features
            
            axes[1, 1].barh(range(len(indices)), importances[indices])
            axes[1, 1].set_yticks(range(len(indices)))
            axes[1, 1].set_yticklabels([feature_names[i] for i in indices])
            axes[1, 1].set_title('Top 10 Feature Importances')
        
        plt.tight_layout()
        plt.savefig('model_evaluation.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def get_insights_and_recommendations(self):
        """Generate business insights and recommendations."""
        print("\n" + "="*50)
        print("INSIGHTS AND RECOMMENDATIONS")
        print("="*50)
        
        # Best model analysis
        best_model_name = max(self.results.keys(), 
                            key=lambda k: self.results[k]['metrics']['f1'])
        best_model = self.results[best_model_name]['model']
        best_metrics = self.results[best_model_name]['metrics']
        
        print(f"1. MODEL PERFORMANCE INSIGHTS:")
        print(f"   - Best performing model: {best_model_name}")
        print(f"   - Achieved F1 Score: {best_metrics['f1']:.4f}")
        print(f"   - Precision: {best_metrics['precision']:.4f} (% of predicted yes that are actually yes)")
        print(f"   - Recall: {best_metrics['recall']:.4f} (% of actual yes that were predicted)")
        print(f"   - ROC AUC: {best_metrics['roc_auc']:.4f} (ability to distinguish classes)")
        
        # Feature importance insights
        if hasattr(best_model.named_steps['classifier'], 'feature_importances_'):
            feature_names = (self.preprocessor.named_transformers_['num'].get_feature_names_out().tolist() +
                           self.preprocessor.named_transformers_['cat'].get_feature_names_out().tolist())
            importances = best_model.named_steps['classifier'].feature_importances_
            
            # Top features
            top_indices = np.argsort(importances)[-5:]
            print(f"\n2. MOST IMPORTANT FEATURES:")
            for i, idx in enumerate(reversed(top_indices)):
                print(f"   {i+1}. {feature_names[idx]}: {importances[idx]:.4f}")
        
        # Business insights from EDA
        print(f"\n3. BUSINESS INSIGHTS FROM DATA:")
        
        # Success rate by education
        education_success = self.df.groupby('education')['y'].apply(lambda x: (x == 'yes').sum() / len(x))
        print(f"   - Success rate by education:")
        for edu, rate in education_success.items():
            print(f"     • {edu}: {rate:.3f}")
        
        # Success rate by job
        job_success = self.df.groupby('job')['y'].apply(lambda x: (x == 'yes').sum() / len(x))
        top_jobs = job_success.nlargest(3)
        print(f"   - Top 3 job types for success:")
        for job, rate in top_jobs.items():
            print(f"     • {job}: {rate:.3f}")
        
        # Duration insights
        avg_duration_yes = self.df[self.df['y'] == 'yes']['duration'].mean()
        avg_duration_no = self.df[self.df['y'] == 'no']['duration'].mean()
        print(f"   - Average call duration:")
        print(f"     • Successful calls: {avg_duration_yes:.0f} seconds")
        print(f"     • Unsuccessful calls: {avg_duration_no:.0f} seconds")
        
        print(f"\n4. ACTIONABLE RECOMMENDATIONS:")
        print(f"   📞 CALL STRATEGY:")
        print(f"   - Focus on longer calls (>{avg_duration_yes:.0f} seconds) for better conversion")
        print(f"   - Limit campaign frequency - high contact counts reduce success")
        print(f"   - Optimal timing appears to be specific months - analyze seasonal patterns")
        
        print(f"\n   🎯 TARGET AUDIENCE:")
        print(f"   - Prioritize clients with tertiary education (higher success rates)")
        print(f"   - Focus on specific job types: {', '.join(top_jobs.head(3).index)}")
        print(f"   - Consider previous campaign outcomes when targeting")
        
        print(f"\n   📊 PERFORMANCE MONITORING:")
        print(f"   - Track precision to minimize wasted calls on unlikely prospects")
        print(f"   - Monitor recall to ensure not missing potential subscribers")
        print(f"   - Current model can identify {best_metrics['recall']*100:.1f}% of actual subscribers")
        
        print(f"\n   💡 BUSINESS IMPACT:")
        print(f"   - Model can help prioritize {best_metrics['precision']*100:.1f}% likely prospects")
        print(f"   - Reduce call volume while maintaining subscription rates")
        print(f"   - Focus marketing budget on high-probability segments")
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline."""
        print("TERM DEPOSIT SUBSCRIPTION PREDICTION ANALYSIS")
        print("=" * 60)
        
        try:
            self.load_data()
            self.exploratory_data_analysis()
            self.feature_engineering()
            self.prepare_data()
            self.train_models()
            self.handle_class_imbalance()
            self.hyperparameter_tuning()
            best_model_name, best_result = self.evaluate_models()
            self.get_insights_and_recommendations()
            
            print(f"\n{'='*60}")
            print("ANALYSIS COMPLETED SUCCESSFULLY!")
            print(f"Best Model: {best_model_name}")
            print(f"F1 Score: {best_result['metrics']['f1']:.4f}")
            print(f"Model saved as: term_deposit_model.pkl")
            print(f"{'='*60}")
            
            # Save the best model
            import joblib
            joblib.dump(best_result['model'], 'term_deposit_model.pkl')
            
        except Exception as e:
            print(f"Error in analysis: {str(e)}")
            import traceback
            traceback.print_exc()

def main():
    """Main function to run the analysis."""
    # Create and run the predictor
    predictor = TermDepositPredictor()
    predictor.run_complete_analysis()

if __name__ == "__main__":
    main() 