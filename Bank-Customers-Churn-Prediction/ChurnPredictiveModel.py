#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 23:58:12 2022

@author: iqrabismi
"""

import pickle 
import streamlit
import pandas as pd
import numpy as np 
import statistics as stat
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')
import sklearn.model_selection
import sklearn.preprocessing
import sklearn.compose
import sklearn.pipeline

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler,OneHotEncoder,LabelBinarizer
from sklearn.compose import ColumnTransformer,make_column_transformer

from sklearn.pipeline import make_pipeline



from imblearn.pipeline import make_pipeline as imbl_pipe
from imblearn.over_sampling import SMOTE

#hyper-parameter tuning
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV

#importing models
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import PowerTransformer
from sklearn.linear_model import Perceptron
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
import xgboost as xgb





loaded_model = pickle.load(open(r'/Users/iqrabismi/Desktop/machine_learning_files/deploy_model.sav', 'rb'))


def churn(input_data):
    
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
      return 'Customer will not churn'
    else:
      return 'Customer will churn'
  
    

def main():
    streamlit.title('Churn Prediction Model')
    
    
    
    CreditScore	= streamlit.text_input('Credit Score')
    
    Geography	= streamlit.text_input('Country')
    
    Gender	= streamlit.text_input('Gender')
    
    Age= streamlit.text_input('Age of Employee')
    
    Tenure	= streamlit.text_input('Tenure in Bank')
    
    Balance	= streamlit.text_input('Balance')
    
    NumOfProducts	= streamlit.text_input('Number of Products')
    
    IsActiveMember	= streamlit.text_input('Member is active or not')
    
    EstimatedSalary	= streamlit.text_input('Salary of Employee')
    
    
    churned=''
        
        
    if streamlit.button('Churn Model Result'):
        churned= churn([CreditScore,Geography,Gender,Age,Tenure,Balance,NumOfProducts,
                           IsActiveMember, EstimatedSalary])
        
    streamlit.success(churned)
        
    
    
if __name__=='__main__':
    main()
        
        
        
        
        
        
        
        
