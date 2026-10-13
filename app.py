# streamlit_app.py - Streamlit Web Interface for Fraud Detection
"""
Credit Card Fraud Detection - Interactive Web Interface
Built with Streamlit for easy visualization and prediction.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 0rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD MODEL AND SCALER
# ============================================================================

@st.cache_resource
def load_model_and_scaler(model_path):
    """Load trained model and scaler from files."""
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open('model/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

# ============================================================================
# SIDEBAR NAVIGATION
# ============================================================================
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "🔮 Make Prediction", "📊 Model Performance", "📚 About"])

# ============================================================================
# PAGE 1: HOME
# ============================================================================

if page == "🏠 Home":
    st.title("🏠 Credit Card Fraud Detection System")
    st.markdown("Welcome to the Fraud Detection Dashboard")

# ============================================================================
# PAGE 2: MAKE PREDICTION
# ============================================================================

elif page == "🔮 Make Prediction":
    st.title("🔮 Fraud Detection Predictor")
    st.markdown("Enter transaction details to predict if it's fraudulent")

    model_options = {
        "Logistic Regression": "logistic",
        "Random Forest": "rf",
        "XGBoost": "xgboost",
        "SVM (LinearSVC)": "svm",
    }
    selected_model = st.selectbox("Choose model", list(model_options))
    selected_variant = st.selectbox("Training data", ["SMOTE", "No SMOTE"])
    variant_key = "smote" if selected_variant == "SMOTE" else "no_smote"
    model_path = Path("model") / f"{model_options[selected_model]}_{variant_key}.pkl"

    if not model_path.is_file() or not Path("model/scaler.pkl").is_file():
        st.error("Selected model is not trained yet. Run all cells in `credit_card_fraud_detection.ipynb` first.")
    else:
        model, scaler = load_model_and_scaler(str(model_path))
        st.markdown("---")
        
        # Two tabs: Quick Prediction vs Detailed Input
        tab1, tab2 = st.tabs(["⚡ Quick Prediction", "🔧 Detailed Input"])
        
        # TAB 1: Quick Prediction
        with tab1:
            st.write("### Quick Prediction with Sample Data")
            
            col1, col2 = st.columns(2)
            
            with col1:
                amount = st.slider("Transaction Amount ($)", 0, 2500, 100)
                time = st.slider("Time (hours since first transaction)", 0, 48, 12)
            
            with col2:
                st.write("### Sample Features")
                st.write("V1-V28: PCA-transformed features (default values)")
            
            # Create transaction with default values
            sample_transaction = {
                'Time': time, 
                'V1': np.random.normal(0, 2), 
                'V2': np.random.normal(0, 2),
                'V3': np.random.normal(0, 2), 
                'V4': np.random.normal(0, 2),
                'V5': np.random.normal(0, 2), 
                'V6': np.random.normal(0, 2),
                'V7': np.random.normal(0, 2), 
                'V8': np.random.normal(0, 2),
                'V9': np.random.normal(0, 2), 
                'V10': np.random.normal(0, 2),
                'V11': np.random.normal(0, 2), 
                'V12': np.random.normal(0, 2),
                'V13': np.random.normal(0, 2), 
                'V14': np.random.normal(0, 2),
                'V15': np.random.normal(0, 2), 
                'V16': np.random.normal(0, 2),
                'V17': np.random.normal(0, 2), 
                'V18': np.random.normal(0, 2),
                'V19': np.random.normal(0, 2), 
                'V20': np.random.normal(0, 2),
                'V21': np.random.normal(0, 2), 
                'V22': np.random.normal(0, 2),
                'V23': np.random.normal(0, 2), 
                'V24': np.random.normal(0, 2),
                'V25': np.random.normal(0, 2), 
                'V26': np.random.normal(0, 2),
                'V27': np.random.normal(0, 2), 
                'V28': np.random.normal(0, 2),
                'Amount': amount
            }
            
            if st.button("🔮 Predict", key="quick_predict", use_container_width=True):
                # Make prediction
                df = pd.DataFrame([sample_transaction])
                df_scaled = scaler.transform(df)
                
                prediction = model.predict(df_scaled)[0]
                probability = model.predict_proba(df_scaled)[0] if hasattr(model, "predict_proba") else None
                decision_score = model.decision_function(df_scaled)[0] if probability is None else None
                
                # Display results
                st.markdown("---")
                st.subheader("📊 Prediction Result")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if prediction == 1:
                        st.error("🚨 FRAUDULENT")
                    else:
                        st.success("✅ LEGITIMATE")
                
                with col2:
                    if probability is not None:
                        st.metric("Fraud Probability", f"{probability[1]*100:.2f}%")
                    else:
                        st.metric("SVM decision score", f"{decision_score:.3f}")
                
                with col3:
                    if probability is not None:
                        st.metric("Confidence", f"{max(probability)*100:.2f}%")
                
                st.markdown("---")
                
                # Detailed breakdown
                st.write("### Confidence Breakdown")
                
                if probability is not None:
                    fig, ax = plt.subplots(figsize=(10, 4))
                    categories = ['Legitimate', 'Fraudulent']
                    values = [probability[0]*100, probability[1]*100]
                    colors = ['#2ecc71', '#e74c3c']

                    bars = ax.bar(categories, values, color=colors, alpha=0.7, width=0.5)
                    ax.set_ylabel('Probability (%)', fontsize=12)
                    ax.set_ylim([0, 100])

                    for bar, value in zip(bars, values):
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{value:.2f}%',
                               ha='center', va='bottom', fontsize=12, fontweight='bold')

                    plt.tight_layout()
                    st.pyplot(fig)
                else:
                    st.info("Linear SVM returns a decision score, not a probability estimate.")
                
                st.markdown("---")
                st.info(f"""
                **Transaction Details:**
                - Amount: ${amount:,.2f}
                - Time: {time} hours
                - Prediction: {'Fraudulent' if prediction == 1 else 'Legitimate'}
                - Fraud Probability: {f"{probability[1]*100:.2f}%" if probability is not None else "Not available for Linear SVM"}
                """)
        
        # TAB 2: Detailed Input
        with tab2:
            st.write("### Enter Transaction Details")
            st.info("Note: For demonstration, enter your own values or use defaults")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                amount = st.number_input("Amount ($)", value=100.0, min_value=0.0)
                time = st.number_input("Time (hours)", value=12, min_value=0, max_value=48)
            
            with col2:
                v1 = st.slider("V1", -5.0, 5.0, 0.0)
                v2 = st.slider("V2", -5.0, 5.0, 0.0)
                v3 = st.slider("V3", -5.0, 5.0, 0.0)
            
            with col3:
                v4 = st.slider("V4", -5.0, 5.0, 0.0)
                v5 = st.slider("V5", -5.0, 5.0, 0.0)
                v6 = st.slider("V6", -5.0, 5.0, 0.0)
            
            st.write("*(Use defaults for other V7-V28 features)*")
            
            if st.button("🔮 Predict", key="detailed_predict", use_container_width=True):
                # Create transaction
                transaction = {
                    'Time': time, 'V1': v1, 'V2': v2, 'V3': v3, 'V4': v4, 'V5': v5, 'V6': v6,
                    'V7': 0, 'V8': 0, 'V9': 0, 'V10': 0, 'V11': 0, 'V12': 0, 'V13': 0,
                    'V14': 0, 'V15': 0, 'V16': 0, 'V17': 0, 'V18': 0, 'V19': 0, 'V20': 0,
                    'V21': 0, 'V22': 0, 'V23': 0, 'V24': 0, 'V25': 0, 'V26': 0, 'V27': 0,
                    'V28': 0, 'Amount': amount
                }
                
                df = pd.DataFrame([transaction])
                df_scaled = scaler.transform(df)
                
                prediction = model.predict(df_scaled)[0]
                probability = model.predict_proba(df_scaled)[0] if hasattr(model, "predict_proba") else None
                decision_score = model.decision_function(df_scaled)[0] if probability is None else None
                
                st.markdown("---")
                st.subheader("📊 Prediction Result")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if prediction == 1:
                        st.error("🚨 FRAUDULENT")
                    else:
                        st.success("✅ LEGITIMATE")
                
                with col2:
                    if probability is not None:
                        st.metric("Fraud Probability", f"{probability[1]*100:.2f}%")
                    else:
                        st.metric("SVM decision score", f"{decision_score:.3f}")
                
                with col3:
                    if probability is not None:
                        st.metric("Confidence", f"{max(probability)*100:.2f}%")

# ============================================================================
# PAGE 3: MODEL PERFORMANCE
# ============================================================================

elif page == "📊 Model Performance":
    st.title("📊 Model Performance Analysis")
    results_path = Path(__file__).resolve().parent / "smote_vs_baseline_results.csv"
    if not results_path.is_file():
        st.error("Training results are not available. Run all cells in `credit_card_fraud_detection.ipynb` first.")
    else:
        results = pd.read_csv(results_path)
        if results.empty:
            st.error("The training results file is empty. Run all cells in `credit_card_fraud_detection.ipynb` again.")
        else:
            results["accuracy"] = (results["tp"] + results["tn"]) / (
                results["tp"] + results["tn"] + results["fp"] + results["fn"]
            )
            st.caption(
                "Metrics below are from the held-out test split in the latest training run. "
                "They refresh when this page reruns."
            )

            st.subheader("All models: test-set comparison")
            comparison = results[
                ["model", "variant", "recall", "precision", "specificity", "auc", "accuracy", "seconds"]
            ].copy()
            comparison[["recall", "precision", "specificity", "auc", "accuracy"]] *= 100
            comparison = comparison.rename(columns={
                "model": "Model",
                "variant": "Training data",
                "recall": "Recall (%)",
                "precision": "Precision (%)",
                "specificity": "Specificity (%)",
                "auc": "AUC-ROC (%)",
                "accuracy": "Accuracy (%)",
                "seconds": "Training time (s)",
            })
            st.dataframe(comparison.round(2), use_container_width=True, hide_index=True)

            col1, col2 = st.columns(2)
            with col1:
                selected_model = st.selectbox("Model details", results["model"].drop_duplicates().tolist())
            model_results = results[results["model"] == selected_model]
            with col2:
                selected_variant = st.selectbox("Training data details", model_results["variant"].tolist())
            selected = model_results[model_results["variant"] == selected_variant].iloc[0]

            total = selected["tp"] + selected["tn"] + selected["fp"] + selected["fn"]
            metric_columns = st.columns(5)
            metric_columns[0].metric("Recall", f"{selected['recall']:.2%}")
            metric_columns[1].metric("Precision", f"{selected['precision']:.2%}")
            metric_columns[2].metric("Specificity", f"{selected['specificity']:.2%}")
            metric_columns[3].metric("AUC-ROC", f"{selected['auc']:.4f}")
            metric_columns[4].metric("Accuracy", f"{selected['accuracy']:.2%}")
            st.caption(f"Training time: {selected['seconds']:.1f}s · Test transactions: {total:,}")

            confusion = np.array([[selected["tn"], selected["fp"]], [selected["fn"], selected["tp"]]])
            fig, ax = plt.subplots(figsize=(7, 4))
            sns.heatmap(
                confusion,
                annot=True,
                fmt=".0f",
                cmap="Blues",
                xticklabels=["Predicted Legitimate", "Predicted Fraud"],
                yticklabels=["Actual Legitimate", "Actual Fraud"],
                cbar_kws={"label": "Transactions"},
                ax=ax,
            )
            ax.set_title(f"{selected_model} — {selected_variant}")
            plt.tight_layout()
            st.pyplot(fig)

# ============================================================================
# PAGE 4: ABOUT
# ============================================================================

elif page == "📚 About":
    st.title("📚 About This Project")
    
    st.markdown("---")
    
    st.subheader("🎯 Project Overview")
    
    st.markdown("""
    This is a **Credit Card Fraud Detection System** built with machine learning.
    
    It uses a dataset of 284,807 real credit card transactions from Kaggle to detect 
    fraudulent transactions with high accuracy.
    """)
    
    st.markdown("---")
    
    st.subheader("🔑 Key Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ✅ **Data Handling**
        - 284,807 transactions analyzed
        - 30 features (PCA-transformed)
        - Extreme class imbalance (99.83% legitimate)
        
        ✅ **Class Imbalance Solution**
        - SMOTE (Synthetic Minority Over-sampling)
        - Creates synthetic fraud examples
        - Prevents model bias
        """)
    
    with col2:
        st.markdown("""
        ✅ **Model Comparison**
        - Logistic Regression
        - Random Forest
        - XGBoost
        - Linear SVM
        
        ✅ **Evaluation Metrics**
        - AUC-ROC (overall quality)
        - Sensitivity (fraud detection)
        - Specificity (false alarm rate)
        - Confusion Matrix (detailed breakdown)
        """)
    
    st.markdown("---")
    
    st.subheader("🔬 Technical Details")
    
    st.markdown("""
    **Technologies Used:**
    - Python, Scikit-learn, XGBoost
    - Pandas, NumPy (data processing)
    - Matplotlib, Seaborn (visualization)
    - Imbalanced-learn (SMOTE)
    - Streamlit (web interface)
    
    **Data Processing:**
    1. Load 284,807 credit card transactions
    2. Check for missing values (none found)
    3. Split 70% train, 30% test (stratified)
    4. Apply StandardScaler normalization
    5. Use SMOTE for class balance
    
    **Model Training:**
    1. Train 4 algorithms with and without SMOTE
    2. Compare results on the held-out test split
    3. Review recall, precision, specificity, and AUC-ROC
    4. Save each trained model for predictions
    """)
    
    st.markdown("---")
    
    st.subheader("📊 Dataset Information")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Transactions", "284,807")
    
    with col2:
        st.metric("Fraudulent Cases", "492 (0.17%)")
    
    with col3:
        st.metric("Features", "30 (PCA)")
    
    st.markdown("---")
    
    
    st.success("""
    ✅ **Project Status**: Complete and Production-Ready
    
    - Model trained and evaluated
    - Interactive web interface
    - Professional documentation
    - Ready for deployment
    """)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 20px;'>
    <p>💳 Credit Card Fraud Detection System</p>
    <p>Built with Streamlit • Compare Logistic Regression, Random Forest, XGBoost, and SVM</p>
</div>
""", unsafe_allow_html=True)