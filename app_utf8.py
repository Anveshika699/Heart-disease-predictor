# -*- coding: utf-8 -*-

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import os
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

os.chdir('D:\\vsproject\\pythondataanalysis')

heart_data = pd.read_csv(r'D:\vsproject\pythondataanalysis\heart.csv')

# splitting the features and target
X = heart_data.drop(columns='target', axis= 1)

Y = heart_data['target']

#splitting the data into training and test data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# model training

model = LogisticRegression(max_iter=500)

# training the LogisticRegression model with training data
model.fit(X_train, Y_train)

# accuracy on training data
X_train_prediction = model.predict(X_train)

training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

# accuracy on test data
X_test_prediction = model.predict(X_test)

test_data_accuracy = accuracy_score (X_test_prediction, Y_test)

st.title("💓 Heart Disease Prediction App")
st.write("Enter the details below to check the likelihood of heart disease.")

# User input fields
age = st.number_input("Enter age:", min_value=1, max_value=120, step=1)
sex = st.selectbox("Sex:", ["Male", "Female"])
cp = st.slider("Chest Pain Type (0-3):", 0, 3)
trestbps = st.number_input("Resting Blood Pressure:", min_value=80, max_value=200)
chol = st.number_input("Cholesterol Level:", min_value=100, max_value=600)
fbs = st.radio("Fasting Blood Sugar > 120 mg/dL?", ["Yes", "No"])
restecg = st.selectbox("Resting ECG Results:", [0, 1, 2])
thalach = st.number_input("Max Heart Rate Achieved:", min_value=50, max_value=250)
exang = st.radio("Exercise Induced Angina?", ["Yes", "No"])
oldpeak = st.number_input("ST Depression:", min_value=0.0, max_value=6.0, step=0.1)
slope = st.selectbox("Slope of Peak:", [0, 1, 2])
ca = st.slider("Number of Major Vessels:", 0, 4)
thal = st.selectbox("Thalassemia Type:", [0, 1, 2, 3])

# Convert categorical inputs
sex = 1 if sex == "Male" else 0
fbs = 1 if fbs == "Yes" else 0
exang = 1 if exang == "Yes" else 0

input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])

# Store inputs in a tuple
input_data = (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)

# Convert input data to a NumPy array and reshape
input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)

# Prediction button
if st.button("Predict"):
    prediction = model.predict(input_data_as_numpy_array)
    result = "🔴 **High likelihood of heart disease!** Consult a doctor." if prediction[0] == 1 else "🟢 **Low likelihood of heart disease. Stay healthy!**"
    st.success(result)



