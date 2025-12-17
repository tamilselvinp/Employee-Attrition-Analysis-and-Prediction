📊 Employee Attrition Prediction System

Predict employee attrition risk using Machine Learning to support data-driven HR retention strategies.
🚀 Project Overview

Employee attrition is a major challenge for organizations, leading to increased hiring costs, productivity loss, and knowledge drain.

This project builds an end-to-end Machine Learning solution that:

Predicts which employees are at risk of leaving

Identifies key drivers of attrition

Presents insights through an interactive Streamlit dashboard

Supports HR teams with actionable business insights
🎯 Business Objectives

Reduce employee turnover

Identify high-risk employees early

Optimize HR retention strategies

Estimate potential cost savings from prevented attrition
🧠 Machine Learning Approach

Model Used: Random Forest Classifier (Balanced)

Problem Type: Binary Classification

Target Variable: Attrition (0 = No, 1 = Yes)

🧹 Data Preprocessing

✔ Dropped irrelevant columns
✔ Handled missing values safely
✔ Encoded categorical variables
✔ Removed duplicates
✔ Scaled numerical features
✔ Verified target variable integrity

📈 Exploratory Data Analysis (EDA)

Attrition distribution analysis

Feature relationships with attrition

Key workforce trends

📊 Model Performance
Metric	Score
Accuracy	~83%
Precision	High
Recall	Optimized for attrition detection
F1-Score	Balanced
AUC-ROC	Strong class separation

🔍 Confusion Matrix Interpretation

True Positives: Employees correctly predicted to leave

False Negatives: Employees at risk but missed (critical for HR)

False Positives: Employees incorrectly flagged as high risk

This helps HR focus on preventable attrition.

🔑 Feature Importance (Key Drivers)

Top drivers of attrition include:

Overtime

Monthly Income

Job Role

Years at Company

Work-Life Balance

Age

💼 Business Impact Metrics
📉 Attrition Rate Comparison

Actual Attrition (Test Data): ~16%

Predicted Attrition (Model): ~9%

💰 Estimated Cost Savings

Using HR assumptions:

Cost per employee attrition ≈ ₹2,00,000

Prevented attrition leadsTMs estimated via True Positives

➡️ Significant potential savings for HR teams


🖥️ Streamlit Dashboard Features


📊 Attrition prediction for individual employees

📈 Visual analytics & KPIs

📉 Confusion matrix visualization

🔍 Feature importance insights

💡 HR-friendly explanations


🛠️ Tech Stack

Python

Pandas, NumPy

Scikit-learn

Matplotlib, Seaborn

Streamlit

Joblib

Future Enhancements

Threshold tuning for better recall

SHAP explainability

Department-wise attrition analysis

Cloud deployment (AWS / Azure)

👩‍💼 HR-Friendly Summary

This system helps HR teams identify at-risk employees early, understand why they may leave, and take proactive retention actions, reducing turnover costs and improving workforce stability.



👩‍💻 Author

Tamilselvi Nataraja
🎓 MSc Software Engineering
🤖 Aspiring Machine Learning Engineer
📊 Passionate about building end-to-end ML projects with real-world business impact
💡 Interested in Data Science, Machine Learning, and AI-driven solutions
