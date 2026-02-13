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

# Suppress sklearn pickle version mismatch warning (models may be from different sklearn version)
warnings.filterwarnings("ignore", message="Trying to unpickle estimator")

# --- Page Configuration ---
st.set_page_config(layout="wide")

# Default max rows for model comparison
DEFAULT_EVAL_ROWS = 10000

# --- SVG Icons ---
# Using simple, open-source SVG icons from Feather Icons (https://feathericons.com/)
# Encoded for direct use in markdown.
ICON_DATASET = """
<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-database"><ellipse cx="12" cy="5" rx="9" ry="3"></ellipse><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"></path><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"></path></svg>
"""
ICON_PREVIEW = """
<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-eye"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"></path><circle cx="12" cy="12" r="3"></circle></svg>
"""
ICON_MODEL = """
<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-bar-chart-2"><line x1="18" y1="20" x2="18" y2="10"></line><line x1="12" y1="20" x2="12" y2="4"></line><line x1="6" y1="20" x2="6" y2="14"></line></svg>
"""
ICON_VISUALIZATION = """
<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-pie-chart"><path d="M21.21 15.89A10 10 0 1 1 8 2.83"></path><path d="M22 12A10 10 0 0 0 12 2v10z"></path></svg>
"""
ICON_REPORT = """
<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-file-text"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline></svg>
"""
ICON_DEEP_DIVE = """
<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-search"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>
"""
ICON_WARNING = """
<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-alert-triangle"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>
"""
ICON_INSIGHTS = """
<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-message-square"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path></svg>
"""

# --- Model and Scaler Loading ---
def _patch_sklearn_model(model):
    """Patch models pickled with older sklearn versions (missing multi_class, etc.)."""
    if model is None:
        return model
    # LogisticRegression: older pickles may lack multi_class (required for predict_proba in sklearn 1.2+)
    if hasattr(model, '__class__') and model.__class__.__name__ == 'LogisticRegression':
        if not hasattr(model, 'multi_class') or model.multi_class is None:
            model.multi_class = 'ovr'
    return model

@st.cache_data
def load_model(model_name):
    model_path = f'models/pkl_files/{model_name}.pkl'
    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}. Please train the models first.")
        return None
    model = joblib.load(model_path)
    return _patch_sklearn_model(model)

@st.cache_data
def load_scaler(model_name):
    scaler_path = f'models/pkl_files/{model_name}_scaler.pkl'
    if not os.path.exists(scaler_path):
        return None
    return joblib.load(scaler_path)

@st.cache_data
def load_features(model_name):
    """Load saved feature columns for model alignment (fallback when feature_names_in_ not available)."""
    features_path = f'models/pkl_files/{model_name}_features.pkl'
    if not os.path.exists(features_path):
        return None
    return joblib.load(features_path)

# --- Data Preprocessing ---
def preprocess_data(df):
    X = df.drop('TARGET', axis=1, errors='ignore')
    y = df['TARGET'] if 'TARGET' in df.columns else None

    for col in X.columns:
        if X[col].dtype == 'object':
            mode_vals = X[col].mode()
            X[col] = X[col].fillna(mode_vals[0] if len(mode_vals) > 0 else 'Unknown')
        else:
            X[col] = X[col].fillna(X[col].mean())
    
    X = pd.get_dummies(X)
    return X, y

# --- Model Training Function ---
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
            script_dir = os.path.dirname(os.path.abspath(__file__))
            result = subprocess.run(
                [sys.executable, script],
                capture_output=True, text=True, check=True, cwd=script_dir
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
    st.cache_data.clear()
    st.info("Cache cleared. Models will be reloaded on the next run.")

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
            st.session_state.data = pd.read_csv('data/application_train.csv')
            st.session_state.page = 'dashboard'
            st.rerun()

        st.markdown("<div style='text-align:center; font-weight:bold; margin:0.5rem 0;'>OR</div>", unsafe_allow_html=True)

        uploaded_file = st.file_uploader("Upload a new CSV file", type="csv")
        if uploaded_file:
            st.session_state.data = pd.read_csv(uploaded_file)
            st.session_state.page = 'dashboard'
            st.rerun()

    st.markdown("---")
    
    st.markdown("""
        **If you encounter errors, please click the 'Train / Retrain All Models' button.** This ensures all models are up-to-date with the latest code.
    """)

    with st.expander("Train / Retrain Models", expanded=False):
        st.markdown("Use this only when the underlying data or model code has changed.")
        if st.button("Train / Retrain All Models"):
            train_models()

def dashboard_page():
    st.title("Model Performance Dashboard")
    st.markdown("---")

    data = st.session_state.data
    X, y = preprocess_data(data)

    model_mapping = {
        'Logistic Regression': 'logistic_regression',
        'Decision Tree': 'decision_tree',
        'K-Nearest Neighbors': 'knn',
        'Naive Bayes': 'naive_bayes',
        'Random Forest': 'random_forest',
        'XGBoost': 'xgboost'
    }
    all_model_names = list(model_mapping.keys())

    
    st.sidebar.title("Sample Size")
    n_rows = len(X)
    max_sample = min(n_rows, 50000) if n_rows > 0 else 10000
    min_sample = min(500, max_sample) if max_sample > 0 else 0
    default_val = min(DEFAULT_EVAL_ROWS, max_sample) if max_sample > 0 else 0
    slider_step = max(1, min(1000, max(1, max_sample // 10))) if max_sample > 0 else 1
    sample_size = st.sidebar.slider(
        "Records for model evaluation",
        min_value=min_sample,
        max_value=max_sample,
        value=default_val,
        step=slider_step,
        help="Larger samples give more accurate metrics but run slower (especially KNN & Random Forest)."
    ) if max_sample > 0 else 0

    st.sidebar.markdown("---")
    st.sidebar.title("Model Selection")
    selected_models = st.sidebar.multiselect(
        "Select models to compare",
        options=all_model_names,
        default=all_model_names
    )

    st.sidebar.markdown("---")
    st.sidebar.title("Evaluation Settings")
    prob_threshold = st.sidebar.slider(
        "Probability Threshold",
        min_value=0.0, max_value=1.0, value=0.5, step=0.05,
        help="Set the threshold for classifying a loan as 'Default'. Affects all metrics except AUC."
    )

    all_metrics = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1 Score', 'MCC']
    selected_metrics = st.sidebar.multiselect(
        "Select metrics to visualize",
        options=all_metrics,
        default=['Accuracy', 'AUC', 'Precision', 'Recall', 'F1 Score', 'MCC']
    )

    st.markdown(f"<h3>{ICON_DATASET} Dataset Overview</h3>", unsafe_allow_html=True)
    with st.expander("", expanded=True):
        if n_rows == 0:
            sample_label = "N/A"
        elif sample_size and sample_size < n_rows:
            sample_label = f"{sample_size:,}"
        else:
            sample_label = f"{n_rows:,} (all rows)"
        
        raw_missing_pct = (data.isnull().sum().sum() / data.size) * 100
        raw_cat_features = len(data.select_dtypes(include=['object']).columns)

        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("Total Instances", f"{n_rows:,}")
        col_b.metric("Sample Used", sample_label)
        col_c.metric("Missing Values % (Raw)", f"{raw_missing_pct:.2f}%")
        col_d.metric("Categorical Features (Raw)", f"{raw_cat_features:,}")
        
        scaled_models = "Logistic Regression, K-Nearest Neighbors, Naive Bayes"
        if y is not None:
            default_rate = y.mean()
            st.caption(f"Defaults (TARGET=1): {default_rate:.2%}. Features scaled via StandardScaler for: {scaled_models}.")
        else:
            st.caption(f"No TARGET column detected. Features scaled via StandardScaler for: {scaled_models}.")

    st.markdown("---")
    st.markdown(f"<h3>{ICON_PREVIEW} Dataset Preview</h3>", unsafe_allow_html=True)
    st.dataframe(data.head())
    st.markdown("---")
    
    if y is not None:
        st.markdown(f"<h3>{ICON_WARNING} Class Imbalance & Metric Interpretation</h3>", unsafe_allow_html=True)
    
        default_rate = y.mean()
        non_default_rate = 1 - default_rate
        imbalance_ratio = int(non_default_rate / default_rate) if default_rate > 0 else float('inf')
        baseline_acc = max(default_rate, non_default_rate)
    
        c1, c2, c3 = st.columns(3)
    
        c1.metric("Default Rate (Positive Class)", f"{default_rate:.2%}")
        c2.metric("Majority Baseline Accuracy", f"{baseline_acc:.2%}")
        c3.metric("Imbalance Ratio", f"1 : {imbalance_ratio}" if default_rate > 0 else "N/A")
    
        colA, colB = st.columns(2)
    
        with colA:
            st.subheader("Class Distribution")
            fig, ax = plt.subplots(figsize=(6, 4))
            y.value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax)
            ax.set_ylabel("")
            ax.axis('equal')
            st.pyplot(fig)
            plt.close()

        with colB:
            st.subheader("Class Counts")
            fig, ax = plt.subplots(figsize=(6, 4))
            y.value_counts().plot(kind='bar', ax=ax)
            ax.set_xlabel("Class")
            ax.set_ylabel("Count")
            st.pyplot(fig)
            plt.close()
    
        st.info(
            """
    ### 📌 Why are Precision / F1 lower?
    
    This dataset is **highly imbalanced (~8% defaults)**.
    
    • Predicting all loans as *No Default* already gives ~92% accuracy  
    • Very few positive samples → precision drops  
    • Small mistakes strongly affect F1  
    • Accuracy becomes misleading  
    
    👉 For credit risk problems, **AUC, Recall, and MCC are better metrics than Accuracy**
            """
        )
        st.markdown("---")

    if y is not None:
        st.markdown(f"<h3>{ICON_MODEL} Model Comparison</h3>", unsafe_allow_html=True)
        if sample_size and n_rows > sample_size:
            sample_idx = X.sample(n=sample_size, random_state=42).index
            X_eval = X.loc[sample_idx]
            y_eval = y.loc[sample_idx]
            st.info(f"Dataset has {n_rows:,} rows. Using a random sample of {sample_size:,} for model comparison.")
        else:
            X_eval, y_eval = X, y

        with st.spinner("Loading models and computing metrics... (this may take 1–2 minutes on first load)"):
            results = []
            model_predictions = {}
            model_probas = {}
            
            for name in selected_models:
                file = model_mapping[name]
                model = load_model(file)
                if not model:
                    continue

                feature_cols = None
                if hasattr(model, 'feature_names_in_'):
                    feature_cols = model.feature_names_in_
                else:
                    feature_cols = load_features(file)
                
                if feature_cols is None:
                    st.warning(f"Could not get feature columns for {name}. Skipping.")
                    continue

                X_aligned = X_eval.reindex(columns=feature_cols, fill_value=0)
                
                scaler = load_scaler(file)
                if scaler:
                    X_scaled = scaler.transform(X_aligned)
                    X_scaled = pd.DataFrame(X_scaled, columns=X_aligned.columns)
                else:
                    X_scaled = X_aligned

                X_scaled = X_scaled[list(feature_cols)]

                y_pred_proba = model.predict_proba(X_scaled)[:, 1]
                y_pred = (y_pred_proba >= prob_threshold).astype(int)

                model_predictions[name] = y_pred
                model_probas[name] = y_pred_proba
                results.append({
                    "Model": name, "Accuracy": accuracy_score(y_eval, y_pred), "AUC": roc_auc_score(y_eval, y_pred_proba),
                    "Precision": precision_score(y_eval, y_pred, zero_division=0), "Recall": recall_score(y_eval, y_pred, zero_division=0),
                    "F1 Score": f1_score(y_eval, y_pred, zero_division=0), "MCC": matthews_corrcoef(y_eval, y_pred)
                })

        results_df = pd.DataFrame(results)
        st.dataframe(results_df, use_container_width=True)
        st.markdown("---")

        st.markdown(f"<h3>{ICON_VISUALIZATION} Model Performance Visualization</h3>", unsafe_allow_html=True)
        if len(results_df) > 0 and len(selected_metrics) > 0:
            num_metrics = len(selected_metrics)
            cols = 3
            rows = (num_metrics + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(14, 4 * rows))
            axes = axes.flatten()
            colors = plt.cm.Set3.colors
            for idx, metric in enumerate(selected_metrics):
                ax = axes[idx]
                df_plot = results_df[['Model', metric]].dropna()
                if len(df_plot) > 0:
                    bars = ax.bar(df_plot['Model'], df_plot[metric], color=colors[idx % len(colors)])
                    ax.set_title(metric, fontsize=11)
                    ax.set_ylim((-1.05, 1.05) if metric == 'MCC' else (0, 1.05))
                    ax.tick_params(axis='x', rotation=45, labelsize=8)
                    for bar in bars:
                        h = bar.get_height()
                        offset = 0.02 if h >= 0 else -0.05
                        ax.text(bar.get_x() + bar.get_width()/2, h + offset,
                                f'{h:.2f}', ha='center', va='bottom' if h >= 0 else 'top', fontsize=8)
            
            for i in range(num_metrics, len(axes)):
                axes[i].set_visible(False)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        st.markdown("---")

        st.markdown(f"<h3>{ICON_REPORT} Confusion Matrix & Classification Report</h3>", unsafe_allow_html=True)
        if selected_models:
            detail_model_report = st.selectbox(
                "Select model for detailed report",
                options=selected_models,
                key='detail_report_select'
            )

            if detail_model_report and detail_model_report in model_predictions:
                y_pred_display = model_predictions[detail_model_report]
                col_cm, col_report = st.columns(2)
                with col_cm:
                    st.subheader("Confusion Matrix")
                    cm = confusion_matrix(y_eval, y_pred_display)
                    fig, ax = plt.subplots(figsize=(6, 5))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                                xticklabels=['No Default', 'Default'], yticklabels=['No Default', 'Default'])
                    ax.set_xlabel('Predicted')
                    ax.set_ylabel('Actual')
                    st.pyplot(fig)
                    plt.close()
                with col_report:
                    st.subheader("Classification Report")
                    report = classification_report(y_eval, y_pred_display, target_names=['No Default', 'Default'])
                    st.code(report)
        st.markdown("---")

        st.markdown(f"<h3>{ICON_DEEP_DIVE} Deep Dive Analysis</h3>", unsafe_allow_html=True)
        if selected_models:
            deep_dive_model_name = st.selectbox(
                "Select a model for deep dive analysis",
                options=selected_models,
                key='deep_dive_select'
            )
            
            if deep_dive_model_name:
                model_file = model_mapping[deep_dive_model_name]
                model = load_model(model_file)
                y_pred_proba = model_probas[deep_dive_model_name]

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("ROC and Precision-Recall Curves")
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                    
                    # ROC Curve
                    fpr, tpr, _ = roc_curve(y_eval, y_pred_proba)
                    ax1.plot(fpr, tpr, color='blue', lw=2, label=f'AUC = {roc_auc_score(y_eval, y_pred_proba):.2f}')
                    ax1.plot([0, 1], [0, 1], color='gray', linestyle='--')
                    ax1.set_title('ROC Curve')
                    ax1.set_xlabel('False Positive Rate')
                    ax1.set_ylabel('True Positive Rate')
                    ax1.legend(loc='lower right')

                    # Precision-Recall Curve
                    precision, recall, _ = precision_recall_curve(y_eval, y_pred_proba)
                    ax2.plot(recall, precision, color='purple', lw=2)
                    ax2.set_title('Precision-Recall Curve')
                    ax2.set_xlabel('Recall')
                    ax2.set_ylabel('Precision')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

                with col2:
                    st.subheader("Feature Importance")
                    
                    feature_importances = None
                    if hasattr(model, 'feature_importances_'):
                        feature_importances = model.feature_importances_
                    elif hasattr(model, 'coef_'):
                        feature_importances = np.abs(model.coef_[0])

                    if feature_importances is not None:
                        feature_cols = None
                        if hasattr(model, 'feature_names_in_'):
                            feature_cols = model.feature_names_in_
                        else:
                            feature_cols = load_features(model_file)
                        
                        if feature_cols is not None:
                            imp_df = pd.DataFrame({'feature': feature_cols, 'importance': feature_importances})
                            imp_df = imp_df.sort_values('importance', ascending=False).head(15)
                            
                            fig, ax = plt.subplots(figsize=(10, 6))
                            sns.barplot(x='importance', y='feature', data=imp_df, ax=ax)
                            ax.set_title('Top 15 Features')
                            st.pyplot(fig)
                            plt.close()
                        else:
                            st.info("Could not determine feature names for importance plot.")
                    else:
                        st.info("Feature importance is not available for this model type (e.g., K-Nearest Neighbors).")
        
        st.markdown("---")
        st.markdown(f"<h3>{ICON_INSIGHTS} Model Insights</h3>", unsafe_allow_html=True)
        st.markdown('''
| ML Model Name                 | Observation about Model Performance                                                                                                                                                                                                                    |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Logistic Regression**       | Strong baseline model. Handles imbalance well using `class_weight='balanced'`. Produces stable AUC and good recall. Interpretable and fast. One of the most reliable models for this dataset.                                                          |
| **Decision Tree**             | Captures non-linear patterns but tends to overfit. High recall but low precision, resulting in many false positives. Less stable than ensemble methods.                                                                                                |
| **k-Nearest Neighbors (kNN)** | Performs poorly due to high dimensional sparse features and class imbalance. Distance-based methods struggle when minority class is small. Accuracy appears high but predicts mostly majority class → near zero recall. Not suitable for this dataset. |
| **Naive Bayes**               | Assumption of feature independence is violated. Predicts many positives causing extremely high recall but very low precision and accuracy. Weak overall performance.                                                                                   |
| **Random Forest (Ensemble)**  | Much better than single tree. Reduces overfitting and improves stability. Good balance between precision and recall. Strong MCC. Reliable ensemble baseline.                                                                                           |
| **XGBoost (Ensemble)**        | Best overall performer. Highest AUC and strong generalization. Handles imbalance well with boosting and class weighting. Captures complex patterns effectively. Final recommended model.                                                               |
''')

    else:
        st.warning("No 'TARGET' column found in the uploaded data. Cannot display model performance.")

    if st.button("Back to Home"):
        if 'data' in st.session_state:
            del st.session_state.data
        st.session_state.page = 'landing'
        st.rerun()

if __name__ == '__main__':
    main()
