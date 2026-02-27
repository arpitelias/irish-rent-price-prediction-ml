Project Title: Predicting Irish Rental Prices Using Machine Learning
Author: Arpit Joshua Elias

Project Summary:
In this project, I worked with real rental data from the Residential Tenancies Board (RTB) covering the years 2008–2024. The main goal was to build a model that could predict monthly rent using features like the year, county, property type (apartment, house, etc.), and a few extra ones I created myself things like a simple Year_Trend variable and a county_cluster label that groups areas into high-rent and lower-rent regions.
I tested three models in total: started with a simple Linear Regression just to set a baseline, then moved on to Random Forest, and finally an MLP Neural Network. When I put the results side by side, the MLP was the clear winner it gave me the highest R² and the lowest error metrics by a decent margin. Because of that, I went ahead and pickled the MLP as my final model.
Looking back, I’m really glad I picked this project. Taking a genuinely messy, real-world dataset and turning it into a working predictor that performs pretty solidly across all of Ireland felt like proper end-to-end data science, and it was satisfying to see it all come together.

Folder Structure:

dublin_housing_project/
│
├── Data_Analytics_Final_Report.pdf
│     Final written report containing the full analysis,
│     visualisations, modelling results, and conclusions.
│
│
├── notebooks/
│     └── 1_data_inspection.ipynb
│         Main project notebook containing data cleaning,
│         feature engineering, visualisation, clustering,
│         and model training.
│
├── data/
│     ├── raw_data.csv
│     │     Original dataset before any processing.
│     │
│     ├── processed_data.csv
│     │     First-stage cleaned dataset used during analysis.
│     │
│     ├── processed_data_with_cluster.csv
│     │     Final processed dataset including engineered
│     │     features and county_cluster column.
│     │
│     ├── county_features.csv
│     └── county_features_with_clusters.csv
│           Intermediate files created during county-level
│           clustering and feature extraction.
│
├── models/
│     ├── mlp_rent_pipeline.pkl
│     │     Initial trained MLP model.
│     │
│     └── mlp_final_all_data_with_cluster.pkl
│           Final model pipeline including preprocessing,
│           one-hot encoding, cluster features, and the MLP.
│
│
├── DataAnalytics Video
│
└── README.txt
      Overview of all project files and structure.



How to Run:

1. Open 1_data_inspection.ipynb

2. Run Kernel → Restart & Run All

3. Outputs (processed data + trained model) will be saved automatically.



Load Final Model:

import joblib
model = joblib.load("models/mlp_final_all_data_with_cluster.pkl")
