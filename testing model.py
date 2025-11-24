# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pickle




loaded_model = pickle.load(open('D:/coding/Machine Learning/Diabetes Data Project/Diabetes_Prediction_Machine_Learning_Project/diabetes_prediction_trained_model', 'rb'))




input_data = (3,158,76,36,245,31.6,0.851,28)

# Changing The Data into Numpy Array
input_data_as_numpy_array = np.asarray(input_data)

# Reshape The Array as we are Predicting
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

prediction = loaded_model.predict(input_data_reshaped)

if (prediction[0] == 0):
  print("The Person is Non-Diabetic")
else:
  print("The Person is Diabetic")