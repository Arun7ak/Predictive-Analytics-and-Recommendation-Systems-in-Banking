# Predictive-Analytics-and-Recommendation-Systems-in-Banking

Overview
This project leverages machine learning techniques to solve critical problems for banks dealing with large volumes of customer data and transactions. The main objectives of this project are:

Predicting Loan Defaults: Using supervised learning techniques to predict whether a customer will default on a loan based on various features.
Segmenting Customers Based on Transaction Behavior: Applying unsupervised learning algorithms to segment customers into different groups based on their transaction patterns.
Recommending Suitable Banking Products: Developing a recommendation engine that suggests relevant banking products to customers based on their past interactions and transaction behavior.

Technologies Used
Python: The primary programming language.
Scikit-learn: For implementing machine learning algorithms.
XGBoost: For training the loan default prediction model.
Streamlit: To create an interactive web app for showcasing the models.
Matplotlib & Seaborn: For data visualization.
Pandas & Numpy: For data manipulation.

Data Preprocessing
Loan Default Prediction: Data cleaning (handling missing values, encoding categorical variables, feature scaling) was applied before training the models.
Customer Segmentation: Transaction data was normalized and standardized to prepare it for clustering algorithms.
Recommendation Engine: Created an interaction matrix for customers and banking products.

Modeling
Loan Default Prediction (Supervised Learning):
Algorithms Used:
1)Logistic Regression
2)Random Forest Classifier
3)XGBoost
Evaluation Metrics:
1.Accuracy
2.Precision
3.Recall
4.F1-Score
5.ROC-AUC Score

Customer Segmentation (Unsupervised Learning):
Algorithms Used:
1.K-Means Clustering
2.DB scan 
3.Hierarchical Clustering
Evaluation Metrics:
1.Silhouette Score
2.Davies-Bouldin Index
3.Cluster Visualization (e.g., PCA plot)

Bank Product Recommendation (Recommendation Engine)
Algorithms Used:
1.Collaborative Filtering
2.Matrix Factorization
Evaluation Metrics:
1.Precision
2.Recall

Evaluation Metrics
Loan Default Prediction:

Streamlit Application:
A Streamlit web app is created to interact with the models. The app allows users to:

Predict the likelihood of loan default for a customer.
Segment customers into groups based on their transaction behaviors.
Recommend suitable banking products to customers based on their transaction history.
