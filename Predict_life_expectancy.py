
# coding: utf-8

# In[34]:

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

#load the datasets
bmi_life_data=pd.read_csv('bmi_and_life_expectancy.csv')

#set x and y values
x_values,y_values=bmi_life_data['BMI'],bmi_life_data['Life expectancy']

#build the regression model
bmi_life_model=LinearRegression()

bmi_life_model.fit(x_values.reshape(-1,1),y_values.reshape(-1,1))

#predict using the model
laos_life_exp=bmi_life_model.predict(np.array([21.07931]).reshape(-1,1))

