import streamlit as st
import tensorflow 
# from tensorflow.keras.models import load_model
import pickle
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
import pandas as pd
import numpy as np

## Load the ANN trained model, scalar, one hot, label encoders pickle file
model=tensorflow.keras.models.load_model('model.h5')

## Load the encoder and scalar
with open('one_hot_encoder_geography.pkl','rb') as file:
    onehot_encoder_geo=pickle.load(file)


with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender=pickle.load(file)

with open('Scalar.pkl','rb') as file:
    scalar=pickle.load(file)



## Streamlit App
st.title('Customer Churn Prediction')


## User Input
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])


## Prepare the input data
input_data = pd.DataFrame({
    'CreditScore'     : [credit_score],
    'Gender'          : [label_encoder_gender.transform([gender])[0]],
    'Age'             : [age],
    'Tenure'          : [tenure],
    'Balance'         : [balance],
    'NumOfProducts'   : [num_of_products],
    'HasCrCard'       : [has_cr_card],
    'IsActiveMember'  : [is_active_member],
    'EstimatedSalary' : [estimated_salary]
})


## One Hot encode 'Geography'
geo_encoded=onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df=pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))


## Combine one-hot encoding with input data
input_data=pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)


## Scale the input data
input_data_scale=scalar.transform(input_data)


## Predict Churn
prediction=model.predict(input_data_scale)
prediction_probaility=prediction[0][0]

st.write(f'Churn Probability: {prediction_probaility:.2f}')

if prediction_probaility>0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is not likely to churn')


