# Healthcare Admission Type Prediction 🏥
## 📌 Project Overview

This project analyzes a healthcare dataset and builds a Machine Learning model to predict patient admission type based on demographic, medical, and billing information. It includes data preprocessing, visualization, feature engineering, model training, evaluation, and saving the trained model.

## 📂 Dataset

The dataset contains healthcare-related information such as:

Patient demographics (Age, Gender)

Admission details (Type, Date, Length of Stay)

Medical condition

Billing amount

Insurance provider

Test results

## ⚙️ Technologies Used

Python

Pandas & NumPy — Data processing

Matplotlib & Seaborn — Visualization

Scikit-learn — Machine learning models

Joblib — Model saving

## 🔎 Workflow
### 1️⃣ Data Loading

Dataset is loaded using Pandas:

df = pd.read_csv("healthcare_dataset.csv")
### 2️⃣ Data Preprocessing

Converted date columns to datetime

Created Length of Stay

Handled missing values

Created new features:

Age Group

Billing Category

Admission Day

### 3️⃣ Exploratory Data Analysis

Visualizations created:

Gender distribution

Top medical conditions

Billing vs admission type

Length of stay comparison

Correlation heatmap

### 4️⃣ Feature Engineering

Removed irrelevant columns (Name, Doctor, Hospital, etc.)

Converted categorical variables using Label Encoding

Converted time-based columns to numeric format

### 5️⃣ Model Training

Model used:

RandomForestClassifier

Dataset split:

80% Training

20% Testing

### 6️⃣ Evaluation

Performance metrics:

Accuracy score

Classification report (Precision, Recall, F1-Score)

### 7️⃣ Model Saving

The trained model and encoders are saved for deployment:

joblib.dump(model, "healthcare_model.pkl")
joblib.dump(encoders, "encoders.pkl")

## 🚀 How to Run

Install dependencies

pip install pandas numpy matplotlib seaborn scikit-learn joblib

Place dataset file:

healthcare_dataset.csv

Run notebook:

health1.ipynb

## 🎯 Project Goal

To demonstrate an end-to-end data science workflow including:

Data cleaning

Feature engineering

Visualization

ML modeling

Model persistence

## 📈 Future Improvements

Hyperparameter tuning

Try multiple models (XGBoost, SVM, Logistic Regression)

Deploy as web app

Add real-time prediction API

# 👨‍💻 Author

Aryan Narendra Harke
