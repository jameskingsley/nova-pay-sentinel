import streamlit as st
import pandas as pd
import joblib
import os
import shap
import matplotlib.pyplot as plt
import warnings
from datetime import datetime

# Suppressing scikit-learn version mismatch warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ROBUST PATH
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.normpath(os.path.join(BASE_DIR, '..', 'notebooks', 'models'))

st.set_page_config(
    page_title="Nova Pay | Sentinel AI", 
    page_icon=" ",
    layout="wide"
)

@st.cache_resource(show_spinner="Initializing Sentinel AI Engine...")
def load_assets():
    model_file = os.path.join(MODEL_PATH, 'nova_pay_fraud_model.pkl')
    scaler_file = os.path.join(MODEL_PATH, 'nova_pay_scaler.pkl')
    features_file = os.path.join(MODEL_PATH, 'model_features.pkl')
    
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Asset directory mismatch. Looked in: {MODEL_PATH}")
    
    model = joblib.load(model_file)
    scaler = joblib.load(scaler_file)
    features = joblib.load(features_file)
    
    background_data = scaler.transform(pd.DataFrame(0, index=[0], columns=features))
    masker = shap.maskers.Independent(data=background_data)
    explainer = shap.LinearExplainer(model, masker=masker) 
    
    return model, scaler, features, explainer

try:
    model, scaler, model_features, explainer = load_assets()
except Exception as e:
    st.error("**System Asset Load Error**")
    st.info(f"Technical Detail: {e}")
    st.stop()

# Sidebar Navigation 
st.sidebar.image("https://img.icons8.com/fluency/96/shield.png", width=80)
st.sidebar.title("Sentinel Dashboard")
st.sidebar.markdown("---")
app_mode = st.sidebar.radio("Select Workflow", ["Single Transaction Audit", "Batch Fraud Analysis"])

# SINGLE TRANSACTION AUDIT 
if app_mode == "Single Transaction Audit":
    st.title("Single Transaction Audit")
    st.markdown("Enter transaction details below to evaluate risk in real-time.")
    
    with st.form("input_form"):
        col1, col2 = st.columns(2)
        with col1:
            amount = st.number_input("Amount (USD)", value=150.0, step=10.0)
            ip_score = st.slider("IP Risk Score", 0.0, 1.0, 0.2)
            velocity = st.number_input("1h Transaction Velocity", value=1, min_value=0)
        with col2:
            acc_age = st.number_input("Account Age (Days)", value=45, min_value=0)
            internal_score = st.slider("Internal Risk Score", 0.0, 1.0, 0.2)
            kyc = st.selectbox("KYC Tier", ["Standard", "Low", "Unknown"])
        
        submit = st.form_submit_button("Run Analysis")

    if submit:
        # Feature Engineering & Mapping
        input_df = pd.DataFrame(0, index=[0], columns=model_features)
        
        input_df['amount_usd'] = amount
        input_df['amount_src'] = amount 
        input_df['ip_risk_score'] = ip_score
        input_df['txn_velocity_1h'] = velocity
        input_df['account_age_days'] = acc_age
        input_df['risk_score_internal'] = internal_score
        
        kyc_col = f"kyc_tier_{kyc.lower()}"
        if kyc_col in input_df.columns:
            input_df[kyc_col] = 1

        current_hour = datetime.now().hour
        for h_col in ['hour', 'hour_of_day']:
            if h_col in input_df.columns:
                input_df[h_col] = current_hour
        
        if 'exchange_rate_src_to_dest' in input_df.columns:
            input_df['exchange_rate_src_to_dest'] = 1.0

        # 2. Prediction Pipeline
        input_scaled = scaler.transform(input_df)
        
        # Scaling returns an array
        input_scaled_df = pd.DataFrame(input_scaled, columns=model_features)
        
        prob = model.predict_proba(input_scaled)[0][1]
        
        # Dynamic Results Display
        st.markdown("---")
        st.subheader("Model Evaluation")
        
        if prob > 0.8:
            st.error(f"### **CRITICAL RISK: FRAUDULENT ({prob:.2%})**")
            st.warning("Recommended Action: Freeze Transaction & Flag Account.")
        elif prob > 0.5:
            st.warning(f"### **ELEVATED RISK: SUSPICIOUS ({prob:.2%})**")
            st.info("Recommended Action: Send to Manual Review Queue.")
        else:
            st.success(f"### **LOW RISK: LEGITIMATE ({prob:.2%})**")
            st.info("Recommended Action: Automatic Approval.")

        # Explainability Section
        st.write("### Decision Logic (SHAP Explanation)")
        
        # Pass the named DataFrame to the explainer
        explanation = explainer(input_scaled_df)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        # max_display=10 keeps the chart clean by showing only top impact features
        shap.plots.bar(explanation[0], max_display=10, show=False)
        plt.title("Top 10 Feature Contributions to Risk Score")
        st.pyplot(plt.gcf())
        plt.clf()

# BATCH FRAUD ANALYSIS 
elif app_mode == "Batch Fraud Analysis":
    st.title("Batch Fraud Audit")
    st.markdown("Upload a CSV file containing transaction data for high-volume screening.")
    
    uploaded_file = st.file_uploader("Upload CSV File", type="csv")
    
    if uploaded_file:
        batch_df = pd.read_csv(uploaded_file)
        proc_df = pd.DataFrame(0, index=batch_df.index, columns=model_features)
        for f in [c for c in model_features if c in batch_df.columns]:
            proc_df[f] = batch_df[f]
            
        batch_scaled = scaler.transform(proc_df)
        batch_df['fraud_probability'] = model.predict_proba(batch_scaled)[:, 1]
        batch_df['prediction'] = (batch_df['fraud_probability'] > 0.5).map({True: 'Fraud', False: 'Legit'})
        
        st.write("Processed Data Preview (Top 5):", batch_df.head())
        
        csv = batch_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Scored Results CSV",
            data=csv,
            file_name=f"sentinel_audit_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )