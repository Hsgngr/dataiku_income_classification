# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 23:55:15 2021

@author: Ege
"""
###############################################################################
def categorize_education(df):
    temp= df.copy()
    di={
        ' Children': 0,
        ' Less than 1st grade': 31,
        ' 1st 2nd 3rd or 4th grade': 32,
        ' 5th or 6th grade':33,
        ' 7th and 8th grade':34,
        ' 9th grade': 35,
        ' 10th grade': 36,
        ' 11th grade': 37,
        ' 12th grade no diploma': 38,
        ' High school graduate' : 39,
        ' Some college but no degree': 40,
        ' Associates degree-occup /vocational': 41,
        ' Associates degree-academic program': 42,
        ' Bachelors degree(BA AB BS)':43,
        ' Masters degree(MA MS MEng MEd MSW MBA)': 44,
        ' Prof school degree (MD DDS DVM LLB JD)': 45,
        ' Doctorate degree(PhD EdD)': 46,    
        }
    temp['education'] = temp['education'].map(di).fillna(df['education'])
    return temp


X_train = categorize_columns(nominal_columns, label_encoding=True)
X_train = categorize_education(X_train)
y_train= X_train.pop('y')

X_test = categorize_columns(nominal_columns, label_encoding=True)
X_test = categorize_education(X_test)
y_test= X_test.pop('y')

