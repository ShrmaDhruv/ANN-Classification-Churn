import streamlit as st
import pickle
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import numpy as np

model = tf.keras.models.load_model('model.h5')
# load scaler and encoder
with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender = pickle.load(file)
with open('Scaler.pkl','rb') as sc:
    scaler = pickle.load(sc)
with open('OneHotEncoder_Geography.pkl','rb') as f:
    Geo = pickle.load(f)

# streamlit app

st.title('Customer Churn Prediction')

geopgraphy = st.selectbox('Geography',Geo.categories_[0])
gender = st.selectbox('Gender',label_encoder_gender.classes_)
age = st.slider('Age',18,92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('tenure',0,10)
num_of_products = st.slider('NUmber of Products',1,4)
has_cr_card = st.selectbox('Has Credit Card',[0,1])
is_active_member = st.selectbox('Is Active Member',[0,1])

df  = pd.DataFrame({
    "CreditScore": [credit_score],
    "Gender": [label_encoder_gender.transform([gender])],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [num_of_products],
    "HasCrCard": [has_cr_card],
    "IsActiveMember": [is_active_member],
    "EstimatedSalary": [estimated_salary]
})

geo_encoded = Geo.transform([[geopgraphy]])
geo_df = pd.DataFrame(geo_encoded,columns=Geo.get_feature_names_out())
df = pd.concat([df,geo_df],axis=1)

scaled_df = scaler.transform(df)

prediction = model.predict(scaled_df)
pred_pro = prediction[0][0]
st.write(pred_pro)
if pred_pro < 0.5:
    st.write("The Customer is not Likely to Churn")
else:
    st.write("The Customer Is Likely to Churn")