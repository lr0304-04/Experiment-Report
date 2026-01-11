# Experiment-Report
Introduction to the Experiment
This is a complete machine learning project template for processing structured data, conducting feature engineering, training multiple machine learning models and evaluating them. The project is specifically designed for binary classification problems and encompasses the entire process from data exploration, visualization, feature engineering, model training to evaluation.

File structure
Project Catalogue /
├── 机器学习.py        
├── train.csv     
├── test.csv  
├── requirements.txt  
│
├── visualizations/    
│   ├── Training_Set_missing_values.png
│   ├── Test_Set_missing_values.png
│   ├── target_distribution.png
│   ├── numeric_features_distribution.png
│   ├── categorical_features_distribution.png
│   ├── feature_target_relationship.png
│   ├── spending_features_distribution.png
│   └── feature_importance.png
│
├── train_processed.csv    
├── test_processed.csv    
├── selected_features.json 
└── README.md          

Environmental Requirement
Python >= 3.7
pandas >= 1.3.0
numpy >= 1.21.0
scikit-learn >= 1.0.0
matplotlib >= 3.5.0
seaborn >= 0.11.0
plotly >= 5.6.0

npm install
pip install pandas numpy matplotlib seaborn plotly scikit-learn xgboost lightgbm joblib

Instructions for Use
1. Prepare the data
Save the training data as train.csv
Save the test data as test.csv
Make sure that the data file and the script are in the same directory

2. Run the complete process
python
# Run the main script directly
python 机器学习.py

