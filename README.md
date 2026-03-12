# Sentinel AI: End-to-End Fraud Detection Pipeline

Sentinel AI (by Nova Pay) is a production-grade MLOps system designed to identify and mitigate fraudulent financial transactions in real-time. This project demonstrates a complete machine learning lifecycle, from rigorous data preprocessing to an interactive, explainable dashboard for fraud analysts.

### Project Overview
In the digital payments landscape, security must be balanced with transparency. Sentinel AI provides:

Evidence-Based Model Selection: A comparative study of linear and ensemble architectures.

Explainable AI (XAI): Integrating SHAP to provide "human-readable" logic for every decision.

Scalable MLOps: A robust folder structure separating notebooks, serialized models, and the front-end application.

##### The Pipeline
###### Data Cleaning & Feature Engineering
I engineered 44 distinct features to capture subtle fraud signals, including behavioral velocity, temporal patterns (hour_of_day), and categorical risk factors (kyc_tier).

######  Experimental Design & Model Selection
To find the optimal classifier, three distinct architectures were evaluated:

Logistic Regression (Baseline)

Random Forest (Balanced via SMOTE)

XGBoost (Balanced via SMOTE)

###### The Selection Process:

* Imbalance Handling: Synthetic Minority Over-sampling Technique (SMOTE) was applied to the training sets of the Random Forest and XGBoost models to address class imbalance.

* Hyperparameter Optimization: Extensive tuning was performed using RandomizedSearchCV for the ensemble models to maximize their predictive power.

* Final Verdict: Despite the complexity of the tuned ensembles, Logistic Regression consistently outperformed the others in terms of generalization and stability on the test set. Given its superior performance and inherent interpretability, it was selected as the production model.

###### MLOps & Serialization
* Persistence: The final model and StandardScaler were serialized using joblib for deployment.

* Feature Consistency: A dedicated model_features.pkl ensures the production app maintains strict feature alignment with the training environment.

###### Sentinel Dashboard (Streamlit)
* Single Transaction Audit: Real-time risk scoring for individual transactions.

* Batch Processing: High-volume CSV screening for historical audit reporting.

* Dashboard link: https://nova-pay-sentinel-euix2lpusszimewjvdentk.streamlit.app/


###### Explainability with SHAP
Because I chose an interpretable linear model, we utilize the SHAP LinearExplainer to decompose the fraud probability into individual feature contributions.

Why this matters: In financial services, "Black Box" models are a liability. Sentinel provides transparency, allowing compliance officers to see exactly why a transaction was flagged; be it high IP risk, unusual velocity, or account age.

##### Author
James Kingsley Philip Senior Data Scientist & ML Engineer