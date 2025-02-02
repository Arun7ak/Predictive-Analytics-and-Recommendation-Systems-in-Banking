# Predictive-Analytics-and-Recommendation-Systems-in-Banking

This repository contains the implementation of a project aimed at solving three critical problems in banking using data analytics and machine learning techniques:

1.Loan Default Prediction (Supervised Learning)
2.Customer Segmentation (Unsupervised Learning)
3.Product Recommendations (Recommendation Engine)

Table of Contents:
Project Overview
Dataset
Streamlit Application
Approach
Loan Default Prediction
Customer Segmentation
Product Recommendations
Results
Model Evaluation
Usage

Project Overview:
Banks deal with large volumes of customer data and transactions daily. This project leverages this data to enhance customer experience, reduce risk, and optimize product offerings. The following business use cases are addressed:

Loan Default Prediction: Minimize financial risk by predicting customers likely to default on loans.
Customer Segmentation: Understand customer behavior to tailor marketing strategies and improve customer satisfaction.
Product Recommendations: Recommend suitable banking products to customers based on their past behavior.

Dataset
The project uses synthetic data created for the following use cases:

Loan Default Prediction: Historical data including customer demographics, credit scores, income, loan details, and repayment history.
Customer Segmentation: Transactional data such as transaction frequency, amount, and types of transactions.
Product Recommendations: Customer interaction data, including product purchase history and interaction types.
Synthetic data is generated using Python libraries like faker, and further feature engineering is applied to create new derived attributes that enhance model performance.

Streamlit Application
The project also includes a Streamlit web application to interact with the models and visualize the results for each problem.

Features:
Loan Default Prediction: Users can input customer data (e.g., age, income, credit score) to predict the likelihood of loan default using the trained XGBoost model.
Customer Segmentation: Visualizes customer segments based on transaction behavior using K-Means clustering. Users can explore the segments through interactive plots.
Product Recommendations: Users can enter their customer ID to receive personalized product recommendations based on collaborative filtering and matrix factorization.

Approach:
Loan Default Prediction (Supervised Learning)
Data Source: Historical loan data with customer demographics and loan details.
Algorithm: XGBoost (Chosen for high prediction accuracy)
Steps:
Data preprocessing and cleaning.
Feature engineering and selection.
Model training and hyperparameter tuning.
Model evaluation using accuracy, precision, recall, F1-score, and ROC-AUC.

Customer Segmentation (Unsupervised Learning)
Data Source: Transactional data with details on transaction amount, frequency, and type.
Algorithm: K-Means Clustering (Chosen for high clustering performance)
Steps:
Data preprocessing and standardization.
Applying K-Means clustering to group customers.
Cluster evaluation using silhouette score, Davies-Bouldin index, and PCA-based visualization.

Product Recommendations (Recommendation Engine)
Data Source: Customer-product interaction data.
Algorithms: Collaborative Filtering and Matrix Factorization (Chosen for personalized recommendations)
Steps:
Data preprocessing, handling sparsity in the customer-product interaction matrix.
Implementing collaborative filtering and matrix factorization models.
Evaluation of recommendation effectiveness using Precision and Recall.

Results:
Loan Default Prediction: XGBoost achieved the highest accuracy in predicting loan defaults, enabling effective risk management for banks.
Customer Segmentation: K-Means clustering effectively segmented customers based on transaction behavior, allowing for more targeted marketing strategies.
Product Recommendations: Collaborative filtering and matrix factorization provided personalized product recommendations, enhancing customer satisfaction and increasing sales.

Model Evaluation:
The models were evaluated using the following metrics:
Loan Default Prediction: Accuracy, Precision, Recall, F1-Score, ROC-AUC.
Customer Segmentation: Silhouette Score, Davies-Bouldin Index, PCA Visualization.
Product Recommendations: Precision, Recall.

Usage
To use the project, run the respective Python scripts for each problem. The scripts will:

Load and preprocess the synthetic data.
Train the appropriate model for each task.
Provide evaluation metrics for each model.

Each section of the project is organized into individual Python files:

LOAN_DEFAULT_STREAMLIT_APP.py
CUSTOMER_SEGMENTATION_STREAMLIT_APP.py
PRODUCT_RECOMMENDATION_STREAMLIT_APP.py
Simply execute the scripts to see the results and models in action.

