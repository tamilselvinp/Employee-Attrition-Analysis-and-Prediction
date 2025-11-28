# Employee-Attrition-Analysis-and-Prediction
Project Overview  Employee attrition (or turnover) is a critical concern for organizations because it affects productivity, morale, and operational costs. This project aims to analyze factors contributing to employee attrition and predict employees at high risk of leaving using machine learning models.
Features of the Project

Exploratory Data Analysis (EDA):

Visualizes trends, distributions, and correlations of features.

Highlights patterns in attrition and performance metrics.

Data Preprocessing:

Handles numeric and categorical variables.

Scales numeric features and encodes categorical features using ColumnTransformer pipelines.

Supports multiple target variables, including attrition, promotion likelihood, performance rating, and job satisfaction.

Machine Learning Models:

Trains Random Forest models for predicting attrition and other HR-related outcomes.

Uses pipelines to combine preprocessing and modeling for reproducibility.

Can be extended to other tree-based models like XGBoost.

Model Evaluation:

Computes performance metrics: Accuracy, F1-score, Precision, Recall, ROC-AUC.

Identifies top features influencing employee attrition (feature importance).
