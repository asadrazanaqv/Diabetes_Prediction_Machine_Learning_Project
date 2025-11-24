# -*- coding: utf-8 -*-
"""
Created on Mon Nov 24 03:51:23 2025

@author: Yousuf Traders
"""

import numpy as np
import pickle
import streamlit as st


loaded_model = pickle.load(open('diabetes_prediction_trained_model.pkl', 'rb'))
loaded_scaler = pickle.load(open('scaler.pkl','rb'))


# Creating a function for Prediciton

def diabetes_prediction(input_data):
    
    # Changing The Data into Numpy Array
    input_data_as_numpy_array = np.asarray(input_data)

    # Reshape The Array as we are Predicting
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    
    # Scale the input using loaded scaler
    scaled_input = loaded_scaler.transform(input_data_reshaped)

    prediction = loaded_model.predict(scaled_input)

    if (prediction[0] == 0):
      return("The Person is Non-Diabetic")
    else:
      return("The Person is Diabetic")
  
    
  
    
def main():
    
    # Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age
    
    # Giving a title
    st.title('Diabetes Prediction Web App')
    
    # Getting the Data as input from user
    
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input('Blood Pressure Value')
    SkinThickness = st.text_input('Skin Thickness Value')
    Insulin = st.text_input('Insulin Level')
    BMI = st.text_input('Body Mass Index Value')
    Diabetes_Pedigree_Function = st.text_input('Diabetes Pedigree Function Value')
    Age = st.text_input('Age of the Person')
    
    
    # Code for Prediction
    diagnosis = ''
    
    # Creating a Button
    
        
    if st.button('Diabetes Test Result:'):
        # Convert all inputs to float
        try:
            input_data = [float(Pregnancies), float(Glucose), float(BloodPressure), float(SkinThickness),
                          float(Insulin), float(BMI), float(Diabetes_Pedigree_Function), float(Age)]
            
            diagnosis = diabetes_prediction(input_data)
        except ValueError:
            diagnosis = "Please enter valid numeric values for all fields."
        
    st.success(diagnosis)
    
    
if __name__ == '__main__':
    main()

    
    
    
    
    
    

    
