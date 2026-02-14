import streamlit as st
import pandas as pd
import joblib
import warnings
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, classification_report, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
import subprocess
import sys
import os
import numpy as np

import datetime
import time

# Add current directory to path so we can import utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.data_preprocessing import (
    engineer_features, impute_missing_values,
    split_features_target, split_data
)

# Suppress sklearn pickle version mismatch warning
warnings.filterwarnings("ignore", message="Trying to unpickle estimator")

# --- Page Configuration ---
st.set_page_config(page_title="Credit Risk Classification", layout="wide")

# --- SVG Icons (Restored from Original App) ---
ICON_DATASET = """<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-database"><ellipse cx="12" cy="5" rx="9" ry="3"></ellipse><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"></path><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"></path></svg>"""
ICON_PREVIEW = """<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-eye"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"></path><circle cx="12" cy="12" r="3"></circle></svg>"""
ICON_MODEL = """<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-bar-chart-2"><line x1="18" y1="20" x2="18" y2="10"></line><line x1="12" y1="20" x2="12" y2="4"></line><line x1="6" y1="20" x2="6" y2="14"></line></svg>"""
ICON_VISUALIZATION = """<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-pie-chart"><path d="M21.21 15.89A10 10 0 1 1 8 2.83"></path><path d="M22 12A10 10 0 0 0 12 2v10z"></path></svg>"""
ICON_REPORT = """<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-file-text"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline></svg>"""
ICON_DEEP_DIVE = """<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-search"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>"""
ICON_WARNING = """<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-alert-triangle"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>"""
ICON_INSIGHTS = """<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-message-square"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path></svg>"""

# --- Artifact Loading Logic ---
@st.cache_resource
def load_artifacts(model_name):
    """Loads model, features, scaler, encoders, threshold from disk."""
    base_path = 'models/pkl_files'
    artifacts = {}
    try:
        model_path = f'{base_path}/{model_name}.pkl'
        artifacts['model'] = joblib.load(model_path)
        
        # Get timestamp
        mtime = os.path.getmtime(model_path)
        artifacts['timestamp'] = datetime.datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
        
        artifacts['features'] = joblib.load(f'{base_path}/{model_name}_features.pkl')
        artifacts['threshold'] = joblib.load(f'{base_path}/{model_name}_threshold.pkl')
        
        if os.path.exists(f'{base_path}/{model_name}_scaler.pkl'):
            artifacts['scaler'] = joblib.load(f'{base_path}/{model_name}_scaler.pkl')
        if os.path.exists(f'{base_path}/{model_name}_encoders.pkl'):
            artifacts['encoders'] = joblib.load(f'{base_path}/{model_name}_encoders.pkl')
        return artifacts
    except FileNotFoundError:
        return None

@st.cache_data
def preprocess_common(df):
    """Runs model-agnostic feature engineering and imputation once."""
    df = engineer_features(df.copy())
    df = impute_missing_values(df)
    return df

def process_input_data(df, artifacts):
    """Detailed preprocessing pipeline (Encoding/Scaling only). Assumes FE/Imputation done."""
    # 3. Encoding & 4. Alignment
    feature_cols = artifacts['features']
    
    if 'encoders' in artifacts:
        # Label Encoding
        for col, le in artifacts['encoders'].items():
            if col in df.columns:
                df[col] = df[col].astype(str)
                known_classes = set(le.classes_)
                # Handle unseen labels
                df[col] = df[col].apply(lambda x: x if x in known_classes else list(known_classes)[0])
                df[col] = le.transform(df[col])
        X = df[feature_cols]
    else:
        # One-Hot Encoding
        df_encoded = pd.get_dummies(df)
        X = df_encoded.reindex(columns=feature_cols, fill_value=0)
    
    # 5. Scaling
    if 'scaler' in artifacts:
        X = pd.DataFrame(
            artifacts['scaler'].transform(X),
            columns=feature_cols
        )
    return X

# ... (Training Logic skipped) ...

# --- Main App Logic ---
# ...


def train_models():
    st.info("Training models... This may take a few minutes. Please see the console for progress.")
    scripts = [
        "models/logistic_regression.py", "models/decision_tree.py", "models/knn.py",
        "models/naive_bayes.py", "models/random_forest.py", "models/train_xgboost.py"
    ]

    progress_bar = st.progress(0)
    log_area = st.empty()
    log_content = ""

    for i, script in enumerate(scripts):
        log_content += f"--- Running {script} ---\n"
        log_area.code(log_content, height=300)

        try:
            # Use sys.executable to run in same environment
            result = subprocess.run(
                [sys.executable, script],
                capture_output=True, text=True, check=True, cwd=os.path.dirname(os.path.abspath(__file__))
            )
            log_content += result.stdout + "\n"
        except subprocess.CalledProcessError as e:
            log_content += f"ERROR in {script}:\n{e.stderr}\n"
            log_area.code(log_content, height=300)
            st.error(f"Failed to train models. See logs above for details.")
            return

        log_area.code(log_content, height=300)
        progress_bar.progress((i + 1) / len(scripts))

    st.success("All models trained successfully!")
    st.cache_resource.clear()

# --- Main App Logic ---
def main():
    if 'page' not in st.session_state:
        st.session_state.page = 'landing'

    if st.session_state.page == 'landing':
        landing_page()
    elif st.session_state.page == 'dashboard':
        dashboard_page()

def landing_page():
    st.sidebar.title("")
    st.title("Credit Risk Classification")
    st.markdown("---")
    st.markdown("""
        This application analyzes customer data to predict the risk of credit default.

        It provides a comprehensive platform to evaluate and compare various machine learning models. Users can upload their own datasets or use the pre-loaded one to see how different models like Logistic Regression, Decision Trees, and XGBoost perform. The dashboard offers a detailed view of model metrics, feature importances, and visualizations to help in making informed decisions.

        The primary goal is to identify customers who are likely to default on their loans, allowing financial institutions to mitigate risk. By understanding the key factors that contribute to default, lenders can improve their lending criteria and strategies. This interactive tool aims to make the process of model evaluation more transparent and accessible.
    """)

    st.header("Start Analysis")

    with st.expander("Analyse Existing Dataset and Upload New Dataset", expanded=True):
        if st.button("Analyze Existing Dataset"):
            if os.path.exists('data/application_train.csv'):
                with st.spinner("Loading and splitting data (Test Set)..."):
                    # Load full data
                    full_df = pd.read_csv('data/application_train.csv')
                    
                    # Split exactly as Trainer does to get the Validation Set
                    # We need to ensure we use the SAME random_state=42
                    X, y = split_features_target(full_df)
                    
                    # We only care about the test set
                    # Note: engineer_features happens inside process_input_data for the app
                    # BUT split_data in trainer happens on Encoded data. 
                    # Here we split Raw data. 
                    # As long as rows are independent, StratifiedShuffleSplit (train_test_split) 
                    # with same seed should yield same Indices.
                    _, X_test, _, y_test = split_data(X, y, random_state=42)
                    
                    # Recombine for App State
                    test_df = X_test.copy()
                    test_df['TARGET'] = y_test
                    
                    # Sample from this Test Set to be manageable
                    # Increased to 10,000 per user request for stability
                    if len(test_df) > 10000:
                        test_df = test_df.sample(10000, random_state=42)
                        
                    st.session_state.data = test_df
                    st.session_state.page = 'dashboard'
                    st.rerun()

        # Download Button for Test Data
        st.markdown("---")
        st.markdown("### Generate Test Data for Download")
        if st.checkbox("Generate & Download Test Data CSV"):
            if os.path.exists('data/application_train.csv'):
                 with st.spinner("Generating Test Set CSV..."):
                    full_df = pd.read_csv('data/application_train.csv')
                    X, y = split_features_target(full_df)
                    _, X_test, _, y_test = split_data(X, y, random_state=42)
                    test_df_export = X_test.copy()
                    test_df_export['TARGET'] = y_test
                    
                    csv = test_df_export.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Test Data CSV (Full)",
                        data=csv,
                        file_name="test_data_credit_risk.csv",
                        mime="text/csv"
                    )
            else:
                st.error("Default data file not found.")

        st.markdown("<div style='text-align:center; font-weight:bold; margin:0.5rem 0;'>OR</div>", unsafe_allow_html=True)

        uploaded_file = st.file_uploader("Upload a new CSV file", type="csv")
        if uploaded_file:
            st.session_state.data = pd.read_csv(uploaded_file)
            st.session_state.page = 'dashboard'
            st.rerun()

    st.markdown("---")
    st.markdown("**If you encounter errors, please click the 'Train / Retrain All Models' button.**")

    with st.expander("Train / Retrain Models", expanded=False):
        st.markdown("Use this only when the underlying data or model code has changed.")
        if st.button("Train / Retrain All Models"):
            train_models()

def dashboard_page():
    st.title("Model Performance Dashboard")
    st.markdown("---")

    data = st.session_state.data
    y = data['TARGET'] if 'TARGET' in data.columns else None

    # Sidebar
    # Sidebar Navigation
    st.sidebar.header("Data Source")
    data_source = st.sidebar.radio(
        "Select Data Source:",
        ("Use Test Data Set", "Upload a new CSV file"),
        index=0  # Default to Test Data since we are on the dashboard
    )
    
    if data_source == "Upload a new CSV file":
         if 'data' in st.session_state:
            del st.session_state['data']
         st.session_state.page = 'landing'
         st.rerun()
    st.sidebar.markdown("---")
    st.sidebar.title("Model Selection")
    model_mapping = {
        'Logistic Regression': 'logistic_regression',
        'Decision Tree': 'decision_tree',
        'K-Nearest Neighbors': 'knn',
        'Naive Bayes': 'naive_bayes',
        'Random Forest': 'random_forest',
        'XGBoost': 'xgboost'
    }
    all_model_names = list(model_mapping.keys())
    selected_models = st.sidebar.multiselect("Select models to compare", options=all_model_names, default=all_model_names)
    
    # Default fallback if artifact threshold missing
    prob_threshold = 0.5 

    # --- Dataset Overview (Restored) ---
    st.markdown(f"<h3>{ICON_DATASET} Dataset Overview</h3>", unsafe_allow_html=True)
    with st.expander("Show detailed statistics", expanded=True):
        n_rows = len(data)
        raw_missing_pct = (data.isnull().sum().sum() / data.size) * 100
        raw_cat_features = len(data.select_dtypes(include=['object']).columns)
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Rows", f"{n_rows:,}")
        c2.metric("Missing Values", f"{raw_missing_pct:.1f}%")
        c3.metric("Categorical Feats", raw_cat_features)
        if y is not None:
            c4.metric("Defaults", f"{y.sum():,}")

    st.markdown("---")
    
    # --- Dataset Preview (RESTORED REQUEST) ---
    st.markdown(f"<h3>{ICON_PREVIEW} Dataset Preview (Evaluation/Test Subset)</h3>", unsafe_allow_html=True)
    st.dataframe(data.head())
    st.markdown("---")

    if y is not None:
        st.markdown(f"<h3>{ICON_WARNING} Class Imbalance & Metric Interpretation</h3>", unsafe_allow_html=True)
        
        with st.expander("Show imbalance statistics", expanded=True):
            default_rate = y.mean()
            non_default_rate = 1 - default_rate
            imbalance_ratio = int(non_default_rate / default_rate) if default_rate > 0 else float('inf')
            baseline_acc = max(default_rate, non_default_rate)

            c1, c2, c3 = st.columns(3)
            c1.metric("Default Rate (Positive)", f"{default_rate:.2%}")
            c2.metric("Majority Baseline Acc", f"{baseline_acc:.2%}")
            c3.metric("Imbalance Ratio", f"1 : {imbalance_ratio}" if default_rate > 0 else "N/A")

        colA, colB = st.columns(2)

        with colA:
            st.subheader("Class Distribution")
            fig, ax = plt.subplots(figsize=(6, 4))
            y.value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax, colors=['#66b3ff','#ff9999'])
            ax.set_ylabel("")
            ax.axis('equal')
            st.pyplot(fig)
            plt.close()

        with colB:
            st.subheader("Class Counts")
            fig, ax = plt.subplots(figsize=(6, 4))
            y.value_counts().plot(kind='bar', ax=ax, color=['#66b3ff','#ff9999'])
            ax.set_xlabel("Class")
            ax.set_ylabel("Count")
            st.pyplot(fig)
            plt.close()

        st.info(
            """
            ### 📌 Imbalance & Metric Interpretation
            This dataset is **highly imbalanced** (~8% defaults).
            
            - **Baseline Accuracy**: Predicting "No Default" for everyone gives ~92% accuracy but is useless for risk detection.
            - **Recall (Sensitivity)**: This is critical. We must catch as many actual defaults as possible to mitigate risk.
            - **Comprehensive Metrics**: AUC and MCC provide a more reliable measure of model performance than accuracy.
            """
        )
        st.markdown("---")

        # Optimized Model Evaluation Block
        results = None
        
        if selected_models:
            st.markdown(f"<h3>{ICON_MODEL} Model Comparison</h3>", unsafe_allow_html=True)
            
            # Check if we need to re-run evaluation
            # Re-run if:
            # 1. 'eval_results' not in session_state
            # 2. Selected models changed
            # 3. Input data changed (handled by session_state.data check usually)
            
            # Generate a cache key based on data ID and selected models
            cache_key = f"{id(data)}_{sorted(selected_models)}"
            
            if 'eval_cache_key' not in st.session_state or st.session_state.eval_cache_key != cache_key:
                
                with st.spinner("Processing data & Evaluating models..."):
                    # Run Common Preprocessing ONCE
                    data_common = preprocess_common(data)
                    
                    calc_results = []
                    model_predictions = {}
                    model_probas = {}
        
                    progress = st.progress(0)
                    
                    for i, name in enumerate(selected_models):
                        file_name = model_mapping[name]
                        artifacts = load_artifacts(file_name)
                        
                        if artifacts:
                            try:
                                # PER-MODEL PREPROCESSING (Now faster)
                                # Pass data_common (already FE/Imputed)
                                X_processed = process_input_data(data_common.copy(), artifacts)
                                
                                model = artifacts['model']
                                
                                if hasattr(model, "predict_proba"):
                                    y_prob = model.predict_proba(X_processed)[:, 1]
                                else:
                                    y_prob = model.predict(X_processed)
            
                                # Use saved threshold or fallback
                                thresh = artifacts['threshold'] if 'threshold' in artifacts else prob_threshold
                                y_pred = (y_prob >= thresh).astype(int)
                                
                                model_predictions[name] = y_pred
                                model_probas[name] = y_prob
                                
                                calc_results.append({
                                    "Model": name,
                                    "Acc": accuracy_score(y, y_pred),
                                    "AUC": roc_auc_score(y, y_prob),
                                    "Precision": precision_score(y, y_pred, zero_division=0),
                                    "Recall": recall_score(y, y_pred),
                                    "F1": f1_score(y, y_pred),
                                    "MCC": matthews_corrcoef(y, y_pred),
                                    "Last Trained": artifacts.get('timestamp', 'N/A')
                                })
                            except Exception as e:
                                st.error(f"Error in {name}: {e}")
                        
                        progress.progress((i + 1) / len(selected_models))
                    
                    # Cache the results
                    st.session_state.eval_results = calc_results
                    st.session_state.eval_predictions = model_predictions
                    st.session_state.eval_probas = model_probas
                    st.session_state.eval_cache_key = cache_key
                    
            # Retrieve from Session State
            results = st.session_state.eval_results
            model_predictions = st.session_state.eval_predictions
            model_probas = st.session_state.eval_probas

        if results:
            results_df = pd.DataFrame(results)
            st.dataframe(results_df.style.highlight_max(axis=0, color='lightgreen'), use_container_width=True)
            st.markdown("---")
            
            # --- Visualizations (Restored 6 metrics) ---
            st.markdown(f"<h3>{ICON_VISUALIZATION} Performance Visualization</h3>", unsafe_allow_html=True)
            metrics_to_plot = ['Acc', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']
            cols = 3
            rows = 2
            fig, axes = plt.subplots(rows, cols, figsize=(15, 10))
            axes = axes.flatten()
            
            for idx, metric in enumerate(metrics_to_plot):
                ax = axes[idx]
                if metric in results_df.columns:
                    sns.barplot(x='Model', y=metric, data=results_df, ax=ax, palette='viridis')
                    ax.set_title(metric)
                    ax.tick_params(axis='x', rotation=45)
                    ax.set_ylim((-0.05, 1.05) if metric == 'MCC' else (0, 1.05))
                    for container in ax.containers:
                        ax.bar_label(container, fmt='%.2f')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            st.markdown("---")

            # --- Confusion Matrix & Report (Separate Section) ---
            st.markdown(f"<h3>{ICON_REPORT} Confusion Matrix & Classification Report</h3>", unsafe_allow_html=True)
            detail_model = st.selectbox("Select model for detailed report:", [r['Model'] for r in results])

            if detail_model:
                y_p = model_predictions[detail_model]
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Confusion Matrix")
                    cm = confusion_matrix(y, y_p)
                    fig, ax = plt.subplots(figsize=(5, 4))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                                xticklabels=['No Default', 'Default'], yticklabels=['No Default', 'Default'])
                    ax.set_ylabel('Actual')
                    ax.set_xlabel('Predicted')
                    st.pyplot(fig)
                    plt.close()
                
                with col2:
                    st.subheader("Classification Report")
                    report = classification_report(y, y_p, target_names=['No Default', 'Default'])
                    st.code(report)
            st.markdown("---")

            # --- Deep Dive (ROC Only) ---
            st.markdown(f"<h3>{ICON_DEEP_DIVE} Deep Dive Analysis</h3>", unsafe_allow_html=True)
            deep_dive_model = st.selectbox("Select model for deep dive analysis:", [r['Model'] for r in results])
            
            if deep_dive_model:
                y_pb = model_probas[deep_dive_model]
                
                c1, c2 = st.columns(2)
                with c1:
                    st.subheader("ROC Curve")
                    fpr, tpr, _ = roc_curve(y, y_pb)
                    fig, ax = plt.subplots()
                    ax.plot(fpr, tpr, label=f"AUC={roc_auc_score(y, y_pb):.2f}")
                    ax.plot([0,1],[0,1], 'k--')
                    ax.legend()
                    st.pyplot(fig)
                    plt.close()
                
                with c2:
                    # Precision-Recall Curve
                    st.subheader("Precision-Recall Curve")
                    precision, recall, _ = precision_recall_curve(y, y_pb)
                    fig, ax = plt.subplots()
                    ax.plot(recall, precision, color='purple')
                    ax.set_xlabel('Recall')
                    ax.set_ylabel('Precision')
                    st.pyplot(fig)
                    plt.close()

    # --- Insights ---
    st.markdown("---")
    st.markdown(f"<h3>{ICON_INSIGHTS} Model Insights</h3>", unsafe_allow_html=True)
    st.markdown("""
| ML Model Name                | Observation about model performance                                                                                                                     |
| :--------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Logistic Regression**      | Provides a strong baseline. With class weighting, it achieves balanced recall and precision. Performs well considering its linear nature.               |
| **Decision Tree**            | Captures non-linear patterns and achieves high recall but suffers from lower precision and generalization compared to ensemble models.                  |
| **kNN**                      | Performs poorly due to high dimensionality and sparse features. Distance-based methods struggle on large structured financial datasets.                 |
| **Naive Bayes**              | Predicts most cases as default, resulting in extremely high recall but very low precision and accuracy. Assumption of feature independence is violated. |
| **Random Forest (Ensemble)** | Improves stability over a single tree but does not outperform boosting methods. Moderate AUC and MCC.                                                   |
| **XGBoost (Ensemble)**       | **Best performing model.** Achieves highest AUC, F1-score, and MCC. Effectively handles class imbalance using `scale_pos_weight` and generalizes well.  |
""")

if __name__ == "__main__":
    main()
