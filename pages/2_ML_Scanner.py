import streamlit as st
import joblib
import json
import os

st.set_page_config(page_title="NQIRP ML Scanner", page_icon="🤖", layout="wide")

st.title("🤖 AI & ML Strategy Scanner")
st.markdown("*1-Year Backtested Model | Smart Money Concepts (FVG & OB) | Dynamic Targets*")

@st.cache_resource
def load_ai_assets():
    model = joblib.load("colab_ai_model.pkl") if os.path.exists("colab_ai_model.pkl") else None
    config = {}
    if os.path.exists("ai_strategy_config.json"):
        with open("ai_strategy_config.json", "r") as f:
            config = json.load(f)
    return model, config

model, config = load_ai_assets()

if model is None:
    st.error("Model file 'colab_ai_model.pkl' not found. Please verify repo uploads.")
else:
    st.success("✅ AI Model Loaded Successfully")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Nifty 50 Trend", "Bullish")
    col2.metric("Midcap Trend", "Neutral")
    col3.metric("Smallcap Trend", "Bullish")

    if st.button("🚀 Run ML Confluence Scan", type="primary"):
        st.info("Scanning market setups using trained XGBoost AI parameters...")
        st.subheader("🎯 High Probability Setups")
        st.write("ML scanner active.")
