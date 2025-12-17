import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score

# ----------------------------------
# PAGE CONFIG
# ----------------------------------
st.set_page_config(
    page_title="Employee Attrition Prediction System",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Employee Attrition Prediction System")
st.markdown("Predict employee attrition risk using a trained ML model")

# ----------------------------------
# PATHS
# ----------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models/attrition_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models/scaler.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data/final_cleaned_employee_data.pkl")

# ----------------------------------
# LOAD MODEL & SCALER
# ----------------------------------
@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler

# ----------------------------------
# LOAD DATA
# ----------------------------------
@st.cache_data
def load_data():
    return pd.read_pickle(DATA_PATH)

# ----------------------------------
# ERROR HANDLING
# ----------------------------------
if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    st.error("❌ Model or scaler file not found! Please train and save the model first.")
    st.stop()

if not os.path.exists(DATA_PATH):
    st.error("❌ Cleaned data file not found! Please save cleaned data first.")
    st.stop()

rf, scaler = load_model()
df = load_data()

# ----------------------------------
# SIDEBAR NAVIGATION
# ----------------------------------
menu = st.sidebar.radio(
    "Navigation",
    ["Overview", "EDA", "Prediction", "Model Performance", "HR Insights"]
)

# ----------------------------------
# OVERVIEW
# ----------------------------------
if menu == "Overview":
    st.subheader("📌 Project Overview")
    st.write("""
    This system predicts employee attrition risk using machine learning.
    
    **Business Value:**
    - Identify high-risk employees early
    - Reduce attrition-related costs
    - Enable data-driven HR decisions
    """)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Employees", df.shape[0])
    col2.metric("Attrition Rate", f"{df['Attrition'].mean()*100:.2f}%")
    col3.metric("Model Used", "Random Forest")

# ----------------------------------
# EDA
# ----------------------------------
elif menu == "EDA":
    st.subheader("📈 Exploratory Data Analysis")

    st.write("### Attrition Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x="Attrition", data=df, ax=ax)
    st.pyplot(fig)

# ----------------------------------
# PREDICTION
# ----------------------------------
elif menu == "Prediction":
    st.subheader("🔍 Predict Attrition Risk")

    input_data = {}
    feature_cols = df.drop("Attrition", axis=1).columns

    with st.form("prediction_form"):
        for col in feature_cols:
            val = st.number_input(col, float(df[col].min()), float(df[col].max()), float(df[col].mean()))
            input_data[col] = val

        submit = st.form_submit_button("Predict")

    if submit:
        input_df = pd.DataFrame([input_data])
        input_scaled = scaler.transform(input_df)
        pred = rf.predict(input_scaled)[0]
        prob = rf.predict_proba(input_scaled)[0][1]

        st.success(f"Prediction: {'Attrition' if pred==1 else 'No Attrition'}")
        st.info(f"Attrition Probability: {prob:.2f}")

# ----------------------------------
# MODEL PERFORMANCE
# ----------------------------------
elif menu == "Model Performance":
    st.subheader("📊 Model Performance")

    X = df.drop("Attrition", axis=1)
    y = df["Attrition"]

    X_scaled = scaler.transform(X)
    y_pred = rf.predict(X_scaled)
    y_prob = rf.predict_proba(X_scaled)[:,1]

    auc = roc_auc_score(y, y_prob)
    st.metric("AUC-ROC", f"{auc:.3f}")

    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig)

# ----------------------------------
# HR INSIGHTS
# ----------------------------------
elif menu == "HR Insights":
    st.subheader("👥 High-Risk Employees")

    X = df.drop("Attrition", axis=1)
    probs = rf.predict_proba(scaler.transform(X))[:,1]

    risk_df = X.copy()
    risk_df["Attrition_Probability"] = probs

    top_risk = risk_df.sort_values("Attrition_Probability", ascending=False).head(10)

    st.dataframe(top_risk)

    st.download_button(
        "Download High-Risk Employees",
        top_risk.to_csv(index=False),
        file_name="high_risk_employees.csv",
        mime="text/csv"
    )