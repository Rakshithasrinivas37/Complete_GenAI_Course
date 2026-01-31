import streamlit as st
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

## Load the model
model = tf.keras.models.load_model('model.h5')

with open("label_encoder_gender.pkl", 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open("onehot_encoder_geo.pkl", 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open("scaler.pkl", 'rb') as file:
    scaler = pickle.load(file)

## Streamlit App
st.title("Data-Driven Customer Churn Prediction")

geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
credit_score = st.number_input('Credit score')
estimated_salary = st.number_input("Salary")
tenure = st.slider('Tenure', 0, 10)
balance = st.number_input('Balance')
num_of_products = st.slider('Numer of Products', 1, 4)
has_cr_card = st.selectbox('Has credit card?', [0, 1])
is_active_member = st.selectbox("Is active customer?", [0,1])

input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure':[tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

#Scale the input data
scaled_input = scaler.transform(input_data)

#Prdiction
prediction = model.predict(scaled_input)
prediction_prob = prediction[0][0]

st.write("The predicted value: ", prediction_prob)

if prediction_prob > 0.5:
    st.write("The customer is likely to churn.")
else:
    st.write("The customer is not likely to churn.")

